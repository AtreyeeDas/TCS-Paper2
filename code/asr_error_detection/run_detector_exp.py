"""
Master Experiment Orchestration Script:
1. Performs forward inference for Voice-NLU (cached) and Text-NLU (dynamic datasets).
2. Computes Semantic Inconsistency Diagnostics.
3. Constructs in-memory feature representations for Detector A and Detector B.
4. Trains and evaluates Rule-based, Detector A, and Detector B models.
5. Saves all results uniquely based on the input dataset name to prevent overwrites.
"""

import json
import os
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import (
    CONTROLLED_ERROR_CSV,
    HEAD_WEIGHTS,
    HEADS,
    LOGISTIC_REGRESSION_MAX_ITER,
    OUTPUT_DIR,
    RANDOM_SEED,
    VOICE_LABEL_ENCODERS,
)
from feature_extractor import extract_sample_features
from text_nlu_forward import TextNLUForwardInference
from voice_nlu_forward import VoiceNLUForwardInference


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    """Computes comprehensive binary classification metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    auc = (
        roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.50
    )

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(auc),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def main():
    parser = argparse.ArgumentParser(description="ASR Error Detection via Semantic Inconsistency")
    parser.add_argument("--input_csv", type=str, default=str(CONTROLLED_ERROR_CSV),
                        help="Path to the paired dataset CSV to evaluate.")
    parser.add_argument("--force_voice_recompute", action="store_true",
                        help="Force recomputation of Voice NLU instead of using cache.")
    args = parser.parse_args()

    # Determine prefix to save files uniquely without overwriting
    dataset_name = Path(args.input_csv).stem
    
    print("=" * 80)
    print(f"ASR -> NLU SEMANTIC INCONSISTENCY ERROR DETECTION EXPERIMENT")
    print(f"Target Dataset: {dataset_name}")
    print("=" * 80)

    # -----------------------------------------------------------------
    # STEP 0: LOAD DATASET & VERIFY ARTIFACTS
    # -----------------------------------------------------------------
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(
            f"Dataset not found at: {args.input_csv}"
        )

    print(f"\n[+] Loading dataset from: {args.input_csv}")
    dataset_df = pd.read_csv(args.input_csv)
    dataset_df["sample_id"] = dataset_df["sample_id"].astype(str)

    # -----------------------------------------------------------------
    # STEP 0b: RUN OR LOAD VOICE-NLU CACHE
    # -----------------------------------------------------------------
    voice_cache_path = OUTPUT_DIR / "voice_nlu_cache.joblib"
    
    if voice_cache_path.exists() and not args.force_voice_recompute:
        print(f"\n[+] Loading CACHED Voice-NLU Inference from: {voice_cache_path}")
        voice_inf_df, voice_posteriors = joblib.load(voice_cache_path)
        voice_encoders = joblib.load(VOICE_LABEL_ENCODERS)
    else:
        print("\n[+] Initializing Voice-NLU Forward Inference...")
        voice_engine = VoiceNLUForwardInference()
        voice_inf_df, voice_posteriors = voice_engine.run_inference()
        voice_encoders = voice_engine.encoders
        
        # Cache for future runs on different text datasets
        joblib.dump((voice_inf_df, voice_posteriors), voice_cache_path)
        print(f"[✓] Cached Voice-NLU Inference & Posteriors to: {voice_cache_path}")

    # Align Voice-NLU inference strictly by sample_id
    voice_df = pd.merge(
        dataset_df[["sample_id"]],
        voice_inf_df,
        on="sample_id",
        how="left",
    )

    print("\n================================================================================")
    print("SCIENTIFIC CAVEAT:")
    print("The Voice-NLU semantic hypothesis is obtained by forward inference using the trained")
    print("Voice-NLU models on the same semantic embedding set used during Voice-NLU training.")
    print("This makes the audio-side reference potentially optimistic and is treated as an")
    print("audio-grounded reference hypothesis, not as an independent ground-truth oracle.")
    print("================================================================================")

    # -----------------------------------------------------------------
    # STEP 0c: RUN TEXT-NLU FORWARD INFERENCE (CLEAN & ERRONEOUS)
    # -----------------------------------------------------------------
    print("\n[+] Initializing Text-NLU Forward Inference...")
    text_engine = TextNLUForwardInference()

    print("\n[+] Running Text-NLU on Clean Reference Transcripts (A)...")
    clean_texts = dataset_df["reference_transcript"].astype(str).tolist()
    text_clean_df, text_clean_posteriors = (
        text_engine.run_inference_on_transcripts(clean_texts, desc="Clean Text")
    )
    text_clean_df["sample_id"] = dataset_df["sample_id"].values

    print("\n[+] Running Text-NLU on Erroneous/Target Transcripts (B)...")
    err_texts = dataset_df["controlled_transcript"].astype(str).tolist()
    text_err_df, text_err_posteriors = text_engine.run_inference_on_transcripts(
        err_texts, desc="Target Text"
    )
    text_err_df["sample_id"] = dataset_df["sample_id"].values

    # -----------------------------------------------------------------
    # STEP 1: SEMANTIC INCONSISTENCY DIAGNOSTIC
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: SEMANTIC INCONSISTENCY DIAGNOSTIC")
    print("=" * 80)

    diag_rows = []
    for i in range(len(dataset_df)):
        sample_id = dataset_df.iloc[i]["sample_id"]
        v_dom = voice_df.iloc[i]["voice_domain"]
        v_sub = voice_df.iloc[i]["voice_subdomain"]
        v_top = voice_df.iloc[i]["voice_topic"]
        v_doc = voice_df.iloc[i]["voice_document_type"]

        tc_dom = text_clean_df.iloc[i]["text_domain"]
        tc_sub = text_clean_df.iloc[i]["text_subdomain"]
        tc_top = text_clean_df.iloc[i]["text_topic"]
        tc_doc = text_clean_df.iloc[i]["text_document_type"]

        te_dom = text_err_df.iloc[i]["text_domain"]
        te_sub = text_err_df.iloc[i]["text_subdomain"]
        te_top = text_err_df.iloc[i]["text_topic"]
        te_doc = text_err_df.iloc[i]["text_document_type"]

        diag_rows.append(
            {
                "sample_id": sample_id,
                "domain": dataset_df.iloc[i].get("domain_label", dataset_df.iloc[i].get("domain", "unknown")),
                "error_type": dataset_df.iloc[i].get("error_type", "unknown"),
                "split": dataset_df.iloc[i]["split"],
                "clean_domain_mismatch": int(v_dom != tc_dom),
                "clean_subdomain_mismatch": int(v_sub != tc_sub),
                "clean_topic_mismatch": int(v_top != tc_top),
                "clean_document_mismatch": int(v_doc != tc_doc),
                "clean_total_mismatches": ((v_dom != tc_dom) + (v_sub != tc_sub) + (v_top != tc_top) + (v_doc != tc_doc)),
                "err_domain_mismatch": int(v_dom != te_dom),
                "err_subdomain_mismatch": int(v_sub != te_sub),
                "err_topic_mismatch": int(v_top != te_top),
                "err_document_mismatch": int(v_doc != te_doc),
                "err_total_mismatches": ((v_dom != te_dom) + (v_sub != te_sub) + (v_top != te_top) + (v_doc != te_doc)),
            }
        )

    df_diag = pd.DataFrame(diag_rows)
    # Using dynamic naming so datasets are not overwritten
    diag_csv_out = OUTPUT_DIR / f"{dataset_name}_semantic_inconsistency_results.csv"
    df_diag.to_csv(diag_csv_out, index=False)

    print("\n--- Target Transcripts vs Voice-NLU ---")
    print(f"Domain Mismatch Rate        : {df_diag['err_domain_mismatch'].mean()*100:.2f}%")
    print(f"Subdomain Mismatch Rate     : {df_diag['err_subdomain_mismatch'].mean()*100:.2f}%")
    print(f"Topic Mismatch Rate         : {df_diag['err_topic_mismatch'].mean()*100:.2f}%")
    print(f"Document-Type Mismatch Rate : {df_diag['err_document_mismatch'].mean()*100:.2f}%")
    print(f"At Least 1 Mismatch         : {(df_diag['err_total_mismatches'] >= 1).mean()*100:.2f}%")
    
    # -----------------------------------------------------------------
    # STEPS 2, 3, 4: BUILD IN-MEMORY FEATURE MATRICES
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEPS 2-4: EXTRACTING IN-MEMORY DETECTOR FEATURES")
    print("=" * 80)

    rows_meta, rows_feat_a, rows_feat_b, targets = [], [], [], []

    for i in tqdm(range(len(dataset_df)), desc="Building In-Memory Features"):
        sample_id = dataset_df.iloc[i]["sample_id"]
        scenario_id = dataset_df.iloc[i]["scenario_id"]
        split = dataset_df.iloc[i]["split"]

        v_row = voice_df.iloc[i]
        v_post_sample = {h: voice_posteriors[h][i] for h in HEADS}

        # CLEAN INSTANCE (error_label = 0)
        tc_row = text_clean_df.iloc[i]
        tc_post_sample = {h: text_clean_posteriors[h][i] for h in HEADS}

        fa_clean, fb_clean = extract_sample_features(
            v_row, tc_row, v_post_sample, tc_post_sample, voice_encoders, text_engine.encoders
        )

        rows_meta.append({"sample_id": sample_id, "scenario_id": scenario_id, "split": split, "is_error": 0})
        rows_feat_a.append(fa_clean)
        rows_feat_b.append(fb_clean)
        targets.append(0)

        # ERRONEOUS INSTANCE (error_label = 1)
        te_row = text_err_df.iloc[i]
        te_post_sample = {h: text_err_posteriors[h][i] for h in HEADS}

        fa_err, fb_err = extract_sample_features(
            v_row, te_row, v_post_sample, te_post_sample, voice_encoders, text_engine.encoders
        )

        rows_meta.append({"sample_id": sample_id, "scenario_id": scenario_id, "split": split, "is_error": 1})
        rows_feat_a.append(fa_err)
        rows_feat_b.append(fb_err)
        targets.append(1)

    df_meta = pd.DataFrame(rows_meta)
    df_feat_a = pd.DataFrame(rows_feat_a)
    df_feat_b = pd.DataFrame(rows_feat_b)
    y_all = np.array(targets, dtype=int)

    train_mask = (df_meta["split"] == "train").values
    val_mask = (df_meta["split"] == "validation").values
    unseen_mask = (df_meta["split"] == "unseen").values

    # -----------------------------------------------------------------
    # STEP 7: RULE-BASED DETECTOR (BASELINE)
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 7: TUNING RULE-BASED BASELINE ON VALIDATION")
    print("=" * 80)

    val_weighted_dis = df_feat_a.loc[val_mask, "weighted_disagreement"].values
    y_val = y_all[val_mask]

    best_rb_threshold, best_rb_f1 = 0.50, -1.0
    for thresh in np.arange(0.05, 0.95, 0.05):
        pred_val = (val_weighted_dis >= thresh).astype(int)
        f1 = f1_score(y_val, pred_val, zero_division=0)
        if f1 > best_rb_f1:
            best_rb_f1 = f1
            best_rb_threshold = float(thresh)

    print(f"[Rule-Based] Optimal threshold on Validation: {best_rb_threshold:.2f} (Val F1 = {best_rb_f1:.4f})")

    # -----------------------------------------------------------------
    # STEP 5: TRAIN DETECTOR A & DETECTOR B CLASSIFIERS
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: TRAINING DETECTOR A & DETECTOR B")
    print("=" * 80)

    X_train_a = df_feat_a[train_mask].values
    X_val_a = df_feat_a[val_mask].values
    X_unseen_a = df_feat_a[unseen_mask].values

    detector_a = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=LOGISTIC_REGRESSION_MAX_ITER, random_state=RANDOM_SEED)),
    ])
    detector_a.fit(X_train_a, y_all[train_mask])

    X_train_b = df_feat_b[train_mask].values
    X_val_b = df_feat_b[val_mask].values
    X_unseen_b = df_feat_b[unseen_mask].values

    detector_b = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=LOGISTIC_REGRESSION_MAX_ITER, random_state=RANDOM_SEED)),
    ])
    detector_b.fit(X_train_b, y_all[train_mask])

    # -----------------------------------------------------------------
    # THRESHOLD CALIBRATION
    # -----------------------------------------------------------------
    val_prob_a = detector_a.predict_proba(X_val_a)[:, 1]
    val_prob_b = detector_b.predict_proba(X_val_b)[:, 1]

    def find_best_threshold(y_true, y_prob):
        best_t, best_f = 0.50, -1.0
        for t in np.arange(0.10, 0.90, 0.02):
            f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
            if f > best_f: best_f, best_t = f, float(t)
        return best_t, best_f

    thresh_a, val_f1_a = find_best_threshold(y_val, val_prob_a)
    thresh_b, val_f1_b = find_best_threshold(y_val, val_prob_b)

    # -----------------------------------------------------------------
    # STEPS 8 & 9: EVALUATION & POSTERIOR ABLATION
    # -----------------------------------------------------------------
    y_unseen = y_all[unseen_mask]
    unseen_weighted_dis = df_feat_a.loc[unseen_mask, "weighted_disagreement"].values
    unseen_prob_a = detector_a.predict_proba(X_unseen_a)[:, 1]
    unseen_prob_b = detector_b.predict_proba(X_unseen_b)[:, 1]

    pred_rb_unseen = (unseen_weighted_dis >= best_rb_threshold).astype(int)
    pred_a_unseen = (unseen_prob_a >= thresh_a).astype(int)
    pred_b_unseen = (unseen_prob_b >= thresh_b).astype(int)

    eval_records = []
    eval_records.append({"Split": "Unseen (Test)", "Method": "Rule-based Disagreement", **compute_metrics(y_unseen, pred_rb_unseen, unseen_weighted_dis)})
    eval_records.append({"Split": "Unseen (Test)", "Method": "Detector A (No Posterior)", **compute_metrics(y_unseen, pred_a_unseen, unseen_prob_a)})
    eval_records.append({"Split": "Unseen (Test)", "Method": "Detector B (With Posterior)", **compute_metrics(y_unseen, pred_b_unseen, unseen_prob_b)})

    df_results = pd.DataFrame(eval_records)
    res_csv_out = OUTPUT_DIR / f"{dataset_name}_error_detector_results.csv"
    df_results.to_csv(res_csv_out, index=False)

    print("\n" + df_results.to_string(index=False))

    # -----------------------------------------------------------------
    # STEP 12: SAVE MODELS AND CONFIGURATION (Dynamic Naming)
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 12: SAVING TRAINED DETECTOR MODELS & THRESHOLDS")
    print("=" * 80)

    model_a_path = OUTPUT_DIR / f"{dataset_name}_detector_without_posterior.joblib"
    model_b_path = OUTPUT_DIR / f"{dataset_name}_detector_with_posterior.joblib"
    thresh_path = OUTPUT_DIR / f"{dataset_name}_detector_thresholds.json"
    
    joblib.dump(detector_a, model_a_path)
    joblib.dump(detector_b, model_b_path)

    with open(thresh_path, "w") as f:
        json.dump({
            "dataset_evaluated": str(args.input_csv),
            "rule_based_disagreement_threshold": best_rb_threshold,
            "detector_A_optimal_threshold": thresh_a,
            "detector_B_optimal_threshold": thresh_b
        }, f, indent=4)

    print(f"[✓] Saved Detector A Pipeline : {model_a_path}")
    print(f"[✓] Saved Detector B Pipeline : {model_b_path}")
    print(f"[✓] Saved Detector Thresholds : {thresh_path}")
    print("\n[✓] ALL DONE! Voice-NLU cache generated successfully and outputs saved distinctly.")


if __name__ == "__main__":
    main()
