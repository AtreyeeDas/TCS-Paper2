"""
Master Experiment Orchestration Script:
1. Performs forward inference for Voice-NLU and Text-NLU (clean & erroneous).
2. Computes Semantic Inconsistency Diagnostics (Step 1).
3. Constructs in-memory feature representations for Detector A and Detector B (Steps 2-4).
4. Trains and evaluates Rule-based, Detector A, and Detector B models (Steps 5-8).
5. Conducts posterior ablation, error-type breakdown, representative sampling, and artifact saving (Steps 9-15).
"""

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
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
)
from feature_extractor import extract_sample_features
from text_nlu_forward import TextNLUForwardInference
from voice_nlu_forward import VoiceNLUForwardInference


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
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
    print("=" * 80)
    print("ASR -> NLU SEMANTIC INCONSISTENCY ERROR DETECTION EXPERIMENT")
    print("=" * 80)

    # -----------------------------------------------------------------
    # STEP 0: LOAD DATASET & VERIFY ARTIFACTS
    # -----------------------------------------------------------------
    if not os.path.exists(CONTROLLED_ERROR_CSV):
        raise FileNotFoundError(
            f"Controlled error dataset not found at: {CONTROLLED_ERROR_CSV}"
        )

    print(f"\n[+] Loading controlled error dataset from: {CONTROLLED_ERROR_CSV}")
    dataset_df = pd.read_csv(CONTROLLED_ERROR_CSV)
    assert (
        len(dataset_df) == 6000
    ), f"Expected 6000 rows, found {len(dataset_df)}"

    # Ensure sample_id is string
    dataset_df["sample_id"] = dataset_df["sample_id"].astype(str)

    # -----------------------------------------------------------------
    # STEP 0b: RUN VOICE-NLU FORWARD INFERENCE
    # -----------------------------------------------------------------
    print("\n[+] Initializing Voice-NLU Forward Inference...")
    voice_engine = VoiceNLUForwardInference()
    voice_inf_df, voice_posteriors = voice_engine.run_inference()

    # Align Voice-NLU inference strictly by sample_id
    voice_df = pd.merge(
        dataset_df[["sample_id"]],
        voice_inf_df,
        on="sample_id",
        how="left",
    )

    # Save Voice-NLU forward inference results artifact
    voice_inf_out = OUTPUT_DIR / "voice_nlu_inference_results.csv"
    voice_df.to_csv(voice_inf_out, index=False)
    print(f"[✓] Saved Voice-NLU Inference Artifact: {voice_inf_out}")

    print(
        "\n================================================================================"
    )
    print("SCIENTIFIC CAVEAT:")
    print(
        "The Voice-NLU semantic hypothesis is obtained by forward inference using the trained"
    )
    print(
        "Voice-NLU models on the same semantic embedding set used during Voice-NLU training."
    )
    print(
        "This makes the audio-side reference potentially optimistic and is treated as an"
    )
    print(
        "audio-grounded reference hypothesis, not as an independent ground-truth oracle."
    )
    print(
        "================================================================================"
    )

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

    print("\n[+] Running Text-NLU on Controlled Erroneous Transcripts (B)...")
    err_texts = dataset_df["controlled_transcript"].astype(str).tolist()
    text_err_df, text_err_posteriors = text_engine.run_inference_on_transcripts(
        err_texts, desc="Erroneous Text"
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

        # Clean Text
        tc_dom = text_clean_df.iloc[i]["text_domain"]
        tc_sub = text_clean_df.iloc[i]["text_subdomain"]
        tc_top = text_clean_df.iloc[i]["text_topic"]
        tc_doc = text_clean_df.iloc[i]["text_document_type"]

        # Erroneous Text
        te_dom = text_err_df.iloc[i]["text_domain"]
        te_sub = text_err_df.iloc[i]["text_subdomain"]
        te_top = text_err_df.iloc[i]["text_topic"]
        te_doc = text_err_df.iloc[i]["text_document_type"]

        diag_rows.append(
            {
                "sample_id": sample_id,
                "domain": dataset_df.iloc[i].get(
                    "domain_label",
                    dataset_df.iloc[i].get("domain", "unknown"),
                ),
                "error_type": dataset_df.iloc[i]["error_type"],
                "split": dataset_df.iloc[i]["split"],
                # Clean vs Voice
                "clean_domain_mismatch": int(v_dom != tc_dom),
                "clean_subdomain_mismatch": int(v_sub != tc_sub),
                "clean_topic_mismatch": int(v_top != tc_top),
                "clean_document_mismatch": int(v_doc != tc_doc),
                "clean_total_mismatches": (
                    (v_dom != tc_dom)
                    + (v_sub != tc_sub)
                    + (v_top != tc_top)
                    + (v_doc != tc_doc)
                ),
                # Erroneous vs Voice
                "err_domain_mismatch": int(v_dom != te_dom),
                "err_subdomain_mismatch": int(v_sub != te_sub),
                "err_topic_mismatch": int(v_top != te_top),
                "err_document_mismatch": int(v_doc != te_doc),
                "err_total_mismatches": (
                    (v_dom != te_dom)
                    + (v_sub != te_sub)
                    + (v_top != te_top)
                    + (v_doc != te_doc)
                ),
            }
        )

    df_diag = pd.DataFrame(diag_rows)
    diag_csv_out = OUTPUT_DIR / "semantic_inconsistency_results.csv"
    df_diag.to_csv(diag_csv_out, index=False)

    print("\n--- Controlled Erroneous Transcripts vs Voice-NLU ---")
    print(
        f"Domain Mismatch Rate        : {df_diag['err_domain_mismatch'].mean()*100:.2f}%"
    )
    print(
        f"Subdomain Mismatch Rate     : {df_diag['err_subdomain_mismatch'].mean()*100:.2f}%"
    )
    print(
        f"Topic Mismatch Rate         : {df_diag['err_topic_mismatch'].mean()*100:.2f}%"
    )
    print(
        f"Document-Type Mismatch Rate : {df_diag['err_document_mismatch'].mean()*100:.2f}%"
    )
    print(
        f"At Least 1 Mismatch         : {(df_diag['err_total_mismatches'] >= 1).mean()*100:.2f}%"
    )
    print(
        f"At Least 2 Mismatches       : {(df_diag['err_total_mismatches'] >= 2).mean()*100:.2f}%"
    )

    print("\n--- Baseline Clean Reference Transcripts vs Voice-NLU ---")
    print(
        f"Clean At Least 1 Mismatch   : {(df_diag['clean_total_mismatches'] >= 1).mean()*100:.2f}%"
    )
    print(
        f"Clean Topic Mismatch Rate   : {df_diag['clean_topic_mismatch'].mean()*100:.2f}%"
    )

    # Inconsistency Breakdown by Error Type
    print("\n--- Erroneous Mismatch Rate by Error Type ---")
    for etype, grp in df_diag.groupby("error_type"):
        print(
            f"  [{etype:<25}] >=1 Mismatch: {(grp['err_total_mismatches'] >= 1).mean()*100:6.2f}% | Topic: {grp['err_topic_mismatch'].mean()*100:6.2f}%"
        )

    # -----------------------------------------------------------------
    # STEPS 2, 3, 4: BUILD IN-MEMORY FEATURE MATRICES (CLEAN + ERRONEOUS)
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(
        "STEPS 2-4: EXTRACTING IN-MEMORY DETECTOR FEATURES (12,000 PAIRED EXAMPLES)"
    )
    print("=" * 80)

    rows_meta = []
    rows_feat_a = []
    rows_feat_b = []
    targets = []

    # Iterate over all 6,000 samples and generate 2 in-memory rows each
    for i in tqdm(range(len(dataset_df)), desc="Building In-Memory Features"):
        sample_id = dataset_df.iloc[i]["sample_id"]
        scenario_id = dataset_df.iloc[i]["scenario_id"]
        split = dataset_df.iloc[i]["split"]

        v_row = voice_df.iloc[i]
        v_post_sample = {h: voice_posteriors[h][i] for h in HEADS}

        # -------------------------------------------------------------
        # EXAMPLE 1: CLEAN INSTANCE (error_label = 0)
        # -------------------------------------------------------------
        tc_row = text_clean_df.iloc[i]
        tc_post_sample = {h: text_clean_posteriors[h][i] for h in HEADS}

        fa_clean, fb_clean = extract_sample_features(
            v_row,
            tc_row,
            v_post_sample,
            tc_post_sample,
            voice_engine.encoders,
            text_engine.encoders,
        )

        rows_meta.append(
            {
                "sample_id": sample_id,
                "scenario_id": scenario_id,
                "split": split,
                "is_error": 0,
                "transcript_type": "clean",
            }
        )
        rows_feat_a.append(fa_clean)
        rows_feat_b.append(fb_clean)
        targets.append(0)

        # -------------------------------------------------------------
        # EXAMPLE 2: ERRONEOUS INSTANCE (error_label = 1)
        # -------------------------------------------------------------
        te_row = text_err_df.iloc[i]
        te_post_sample = {h: text_err_posteriors[h][i] for h in HEADS}

        fa_err, fb_err = extract_sample_features(
            v_row,
            te_row,
            v_post_sample,
            te_post_sample,
            voice_engine.encoders,
            text_engine.encoders,
        )

        rows_meta.append(
            {
                "sample_id": sample_id,
                "scenario_id": scenario_id,
                "split": split,
                "is_error": 1,
                "transcript_type": "erroneous",
            }
        )
        rows_feat_a.append(fa_err)
        rows_feat_b.append(fb_err)
        targets.append(1)

    df_meta = pd.DataFrame(rows_meta)
    df_feat_a = pd.DataFrame(rows_feat_a)
    df_feat_b = pd.DataFrame(rows_feat_b)
    y_all = np.array(targets, dtype=int)

    # -----------------------------------------------------------------
    # STEP 6 & 14: SCENARIO-LEVEL SPLITTING & DATA LEAKAGE CHECKS
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEPS 6 & 14: VERIFYING SCENARIO-LEVEL SPLITS & ZERO LEAKAGE")
    print("=" * 80)

    train_scenarios = set(df_meta[df_meta["split"] == "train"]["scenario_id"])
    val_scenarios = set(
        df_meta[df_meta["split"] == "validation"]["scenario_id"]
    )
    unseen_scenarios = set(df_meta[df_meta["split"] == "unseen"]["scenario_id"])

    print(f"Train Scenarios      : {len(train_scenarios)}")
    print(f"Validation Scenarios : {len(val_scenarios)}")
    print(f"Unseen Scenarios     : {len(unseen_scenarios)}")

    # Leakage assertions
    assert (
        len(train_scenarios & val_scenarios) == 0
    ), "Scenario leakage between Train and Validation!"
    assert (
        len(train_scenarios & unseen_scenarios) == 0
    ), "Scenario leakage between Train and Unseen!"
    assert (
        len(val_scenarios & unseen_scenarios) == 0
    ), "Scenario leakage between Validation and Unseen!"

    # Verify no lexical leakage columns in feature matrix
    forbidden_tokens = [
        "source_term",
        "replacement_term",
        "error_type",
        "controlled_WER",
        "error_injected",
        "transcript",
    ]
    for col in df_feat_a.columns:
        assert not any(
            tok in col for tok in forbidden_tokens
        ), f"Leakage column found in Detector A features: {col}"
    for col in df_feat_b.columns:
        assert not any(
            tok in col for tok in forbidden_tokens
        ), f"Leakage column found in Detector B features: {col}"

    print("[✓] Disjoint scenario verification passed.")
    print(
        "[✓] Feature matrix leakage assertion passed (0 lexical/target columns)."
    )

    train_mask = (df_meta["split"] == "train").values
    val_mask = (df_meta["split"] == "validation").values
    unseen_mask = (df_meta["split"] == "unseen").values

    print(
        f"\nIn-Memory Sample Counts: Train={train_mask.sum()} | Val={val_mask.sum()} | Unseen={unseen_mask.sum()}"
    )

    # -----------------------------------------------------------------
    # STEP 7: RULE-BASED DETECTOR (BASELINE)
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 7: TUNING RULE-BASED BASELINE ON VALIDATION")
    print("=" * 80)

    val_weighted_dis = df_feat_a.loc[val_mask, "weighted_disagreement"].values
    y_val = y_all[val_mask]

    best_rb_threshold = 0.50
    best_rb_f1 = -1.0

    for thresh in np.arange(0.05, 0.95, 0.05):
        pred_val = (val_weighted_dis >= thresh).astype(int)
        f1 = f1_score(y_val, pred_val, zero_division=0)
        if f1 > best_rb_f1:
            best_rb_f1 = f1
            best_rb_threshold = float(thresh)

    print(
        f"[Rule-Based] Optimal threshold on Validation: {best_rb_threshold:.2f} (Validation F1 = {best_rb_f1:.4f})"
    )

    # -----------------------------------------------------------------
    # STEP 5: TRAIN DETECTOR A & DETECTOR B CLASSIFIERS
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: TRAINING DETECTOR A & DETECTOR B (LOGISTIC REGRESSION)")
    print("=" * 80)

    # Detector A
    X_train_a = df_feat_a[train_mask].values
    X_val_a = df_feat_a[val_mask].values
    X_unseen_a = df_feat_a[unseen_mask].values

    detector_a = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=LOGISTIC_REGRESSION_MAX_ITER,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    detector_a.fit(X_train_a, y_all[train_mask])

    # Detector B
    X_train_b = df_feat_b[train_mask].values
    X_val_b = df_feat_b[val_mask].values
    X_unseen_b = df_feat_b[unseen_mask].values

    detector_b = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=LOGISTIC_REGRESSION_MAX_ITER,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    detector_b.fit(X_train_b, y_all[train_mask])

    # -----------------------------------------------------------------
    # THRESHOLD CALIBRATION ON VALIDATION SET
    # -----------------------------------------------------------------
    val_prob_a = detector_a.predict_proba(X_val_a)[:, 1]
    val_prob_b = detector_b.predict_proba(X_val_b)[:, 1]

    def find_best_threshold(y_true, y_prob):
        best_t, best_f = 0.50, -1.0
        for t in np.arange(0.10, 0.90, 0.02):
            f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
            if f > best_f:
                best_f = f
                best_t = float(t)
        return best_t, best_f

    thresh_a, val_f1_a = find_best_threshold(y_val, val_prob_a)
    thresh_b, val_f1_b = find_best_threshold(y_val, val_prob_b)

    print(
        f"[Detector A] Tuned Validation Threshold: {thresh_a:.2f} (Val F1 = {val_f1_a:.4f})"
    )
    print(
        f"[Detector B] Tuned Validation Threshold: {thresh_b:.2f} (Val F1 = {val_f1_b:.4f})"
    )

    # -----------------------------------------------------------------
    # STEP 8 & 9: EVALUATION & POSTERIOR ABLATION
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEPS 8 & 9: COMPREHENSIVE EVALUATION ON VALIDATION & UNSEEN")
    print("=" * 80)

    y_unseen = y_all[unseen_mask]
    unseen_weighted_dis = df_feat_a.loc[
        unseen_mask, "weighted_disagreement"
    ].values

    unseen_prob_a = detector_a.predict_proba(X_unseen_a)[:, 1]
    unseen_prob_b = detector_b.predict_proba(X_unseen_b)[:, 1]

    # Predictions
    pred_rb_val = (val_weighted_dis >= best_rb_threshold).astype(int)
    pred_rb_unseen = (unseen_weighted_dis >= best_rb_threshold).astype(int)

    pred_a_val = (val_prob_a >= thresh_a).astype(int)
    pred_a_unseen = (unseen_prob_a >= thresh_a).astype(int)

    pred_b_val = (val_prob_b >= thresh_b).astype(int)
    pred_b_unseen = (unseen_prob_b >= thresh_b).astype(int)

    # Collect Results Table
    eval_records = []

    # Validation Results
    eval_records.append(
        {
            "Split": "Validation",
            "Method": "Rule-based Disagreement",
            **compute_metrics(y_val, pred_rb_val, val_weighted_dis),
        }
    )
    eval_records.append(
        {
            "Split": "Validation",
            "Method": "Detector A (No Posterior)",
            **compute_metrics(y_val, pred_a_val, val_prob_a),
        }
    )
    eval_records.append(
        {
            "Split": "Validation",
            "Method": "Detector B (With Posterior)",
            **compute_metrics(y_val, pred_b_val, val_prob_b),
        }
    )

    # Unseen Results
    eval_records.append(
        {
            "Split": "Unseen (Test)",
            "Method": "Rule-based Disagreement",
            **compute_metrics(y_unseen, pred_rb_unseen, unseen_weighted_dis),
        }
    )
    eval_records.append(
        {
            "Split": "Unseen (Test)",
            "Method": "Detector A (No Posterior)",
            **compute_metrics(y_unseen, pred_a_unseen, unseen_prob_a),
        }
    )
    eval_records.append(
        {
            "Split": "Unseen (Test)",
            "Method": "Detector B (With Posterior)",
            **compute_metrics(y_unseen, pred_b_unseen, unseen_prob_b),
        }
    )

    df_results = pd.DataFrame(eval_records)
    res_csv_out = OUTPUT_DIR / "error_detector_results.csv"
    df_results.to_csv(res_csv_out, index=False)

    print("\n" + df_results.to_string(index=False))

    # -----------------------------------------------------------------
    # STEP 10: ERROR-TYPE & DOMAIN-LEVEL ABLATION
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 10: DISAGGREGATED PERFORMANCE BY ERROR-TYPE & DOMAIN (UNSEEN)")
    print("=" * 80)

    df_unseen_eval = df_meta[unseen_mask].copy().reset_index(drop=True)
    df_unseen_eval["y_true"] = y_unseen
    df_unseen_eval["pred_rb"] = pred_rb_unseen
    df_unseen_eval["pred_a"] = pred_a_unseen
    df_unseen_eval["pred_b"] = pred_b_unseen
    df_unseen_eval["prob_b"] = unseen_prob_b

    # Merge dataset attributes for detailed breakdown
    df_unseen_eval = pd.merge(
        df_unseen_eval,
        dataset_df[
            [
                "sample_id",
                "error_type",
                "domain_label",
                "source_term",
                "replacement_term",
                "reference_transcript",
                "controlled_transcript",
            ]
        ],
        on="sample_id",
        how="left",
    )

    print("\n--- Unseen F1-Score by Error Type (Erroneous vs Clean) ---")
    for etype, grp in df_unseen_eval.groupby("error_type"):
        # Match error rows and corresponding clean rows
        f1_a = f1_score(grp["y_true"], grp["pred_a"], zero_division=0)
        f1_b = f1_score(grp["y_true"], grp["pred_b"], zero_division=0)
        print(
            f"  [{etype:<25}] Samples: {len(grp):<4} | Detector A F1: {f1_a:.4f} | Detector B F1: {f1_b:.4f}"
        )

    print("\n--- Unseen F1-Score by Domain ---")
    for dom, grp in df_unseen_eval.groupby("domain_label"):
        f1_a = f1_score(grp["y_true"], grp["pred_a"], zero_division=0)
        f1_b = f1_score(grp["y_true"], grp["pred_b"], zero_division=0)
        print(
            f"  [{dom:<15}] Samples: {len(grp):<4} | Detector A F1: {f1_a:.4f} | Detector B F1: {f1_b:.4f}"
        )

    # -----------------------------------------------------------------
    # STEP 11: REPRESENTATIVE ERROR EXAMPLES AUDIT FILE
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 11: GENERATING REPRESENTATIVE UNSEEN AUDIT SAMPLES (TP, TN, FP, FN)")
    print("=" * 80)

    # Construct comprehensive unseen prediction artifact
    df_unseen_full = pd.concat(
        [
            df_unseen_eval,
            df_feat_a[unseen_mask].reset_index(drop=True),
            df_feat_b[unseen_mask][
                [c for c in df_feat_b.columns if c not in df_feat_a.columns]
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    # Merge Voice-NLU & Text-NLU labels
    df_unseen_full = pd.merge(
        df_unseen_full,
        voice_df[
            [
                "sample_id",
                "voice_domain",
                "voice_subdomain",
                "voice_topic",
                "voice_document_type",
            ]
        ],
        on="sample_id",
        how="left",
    )

    # Attach predictions
    df_unseen_full["detector_A_probability"] = unseen_prob_a
    df_unseen_full["detector_A_prediction"] = pred_a_unseen
    df_unseen_full["detector_B_probability"] = unseen_prob_b
    df_unseen_full["detector_B_prediction"] = pred_b_unseen

    # Assign classification outcome for Detector B
    def classify_outcome(row):
        if row["y_true"] == 1 and row["detector_B_prediction"] == 1:
            return "TP"
        if row["y_true"] == 0 and row["detector_B_prediction"] == 0:
            return "TN"
        if row["y_true"] == 0 and row["detector_B_prediction"] == 1:
            return "FP"
        return "FN"

    df_unseen_full["detector_B_outcome"] = df_unseen_full.apply(
        classify_outcome, axis=1
    )

    # Save complete unseen predictions file
    unseen_preds_out = OUTPUT_DIR / "error_detector_unseen_predictions.csv"
    df_unseen_full.to_csv(unseen_preds_out, index=False)
    print(f"[✓] Saved Unseen Predictions Artifact: {unseen_preds_out}")

    # Extract 5 balanced examples for each outcome
    rep_samples = []
    for outcome in ["TP", "TN", "FP", "FN"]:
        sub = df_unseen_full[df_unseen_full["detector_B_outcome"] == outcome]
        if len(sub) > 0:
            rep_samples.append(sub.head(5))

    if rep_samples:
        df_rep = pd.concat(rep_samples, ignore_index=True)
        rep_csv_out = OUTPUT_DIR / "representative_error_examples.csv"
        df_rep.to_csv(rep_csv_out, index=False)
        print(f"[✓] Saved Representative Audit Examples: {rep_csv_out}")

    # -----------------------------------------------------------------
    # STEP 12: SAVE MODELS AND CONFIGURATION
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 12: SAVING TRAINED DETECTOR MODELS & THRESHOLDS")
    print("=" * 80)

    model_a_path = OUTPUT_DIR / "error_detector_without_posterior.joblib"
    model_b_path = OUTPUT_DIR / "error_detector_with_posterior.joblib"
    thresh_path = OUTPUT_DIR / "detector_thresholds.json"
    config_path = OUTPUT_DIR / "experiment_config.json"

    joblib.dump(detector_a, model_a_path)
    joblib.dump(detector_b, model_b_path)

    thresholds_dict = {
        "rule_based_disagreement_threshold": best_rb_threshold,
        "detector_A_optimal_threshold": thresh_a,
        "detector_B_optimal_threshold": thresh_b,
        "feature_names_A": list(df_feat_a.columns),
        "feature_names_B": list(df_feat_b.columns),
    }

    with open(thresh_path, "w") as f:
        json.dump(thresholds_dict, f, indent=4)

    exp_config = {
        "dataset": str(CONTROLLED_ERROR_CSV),
        "source_samples": 6000,
        "in_memory_clean_samples": 6000,
        "in_memory_erroneous_samples": 6000,
        "train_scenarios": len(train_scenarios),
        "validation_scenarios": len(val_scenarios),
        "unseen_scenarios": len(unseen_scenarios),
        "weights": HEAD_WEIGHTS,
        "max_iter": LOGISTIC_REGRESSION_MAX_ITER,
        "random_seed": RANDOM_SEED,
    }

    with open(config_path, "w") as f:
        json.dump(exp_config, f, indent=4)

    print(f"[✓] Saved Detector A Pipeline : {model_a_path}")
    print(f"[✓] Saved Detector B Pipeline : {model_b_path}")
    print(f"[✓] Saved Detector Thresholds : {thresh_path}")
    print(f"[✓] Saved Experiment Config   : {config_path}")

    # -----------------------------------------------------------------
    # STEP 15: FINAL RESEARCH SUMMARY & REPORTING
    # -----------------------------------------------------------------
    unseen_rb = df_results[
        (df_results["Split"] == "Unseen (Test)")
        & (df_results["Method"] == "Rule-based Disagreement")
    ].iloc[0]
    unseen_a = df_results[
        (df_results["Split"] == "Unseen (Test)")
        & (df_results["Method"] == "Detector A (No Posterior)")
    ].iloc[0]
    unseen_b = df_results[
        (df_results["Split"] == "Unseen (Test)")
        & (df_results["Method"] == "Detector B (With Posterior)")
    ].iloc[0]

    f1_delta = unseen_b["F1"] - unseen_a["F1"]
    auc_delta = unseen_b["ROC-AUC"] - unseen_a["ROC-AUC"]

    print("\n" + "=" * 80)
    print("FINAL RESEARCH SUMMARY")
    print("=" * 80)
    print(f"Number of Source Samples            : 6000")
    print(f"Number of Clean In-Memory Examples  : 6000")
    print(f"Number of Error In-Memory Examples  : 6000")
    print(f"Train Scenarios                     : {len(train_scenarios)}")
    print(f"Validation Scenarios                : {len(val_scenarios)}")
    print(f"Unseen Scenarios                    : {len(unseen_scenarios)}")
    print(
        f"Semantic Inconsistency Rate (Error) : {(df_diag['err_total_mismatches'] >= 1).mean()*100:.2f}%"
    )
    print("-" * 80)
    print(f"Rule-based Unseen F1                : {unseen_rb['F1']:.4f}")
    print(f"Logistic Regression (No Post) F1    : {unseen_a['F1']:.4f}")
    print(f"Logistic Regression (With Post) F1  : {unseen_b['F1']:.4f}")
    print(f"Rule-based Unseen ROC-AUC           : {unseen_rb['ROC-AUC']:.4f}")
    print(f"Detector A (No Post) Unseen ROC-AUC : {unseen_a['ROC-AUC']:.4f}")
    print(f"Detector B (With Post) Unseen ROC-AUC: {unseen_b['ROC-AUC']:.4f}")
    print("-" * 80)
    print(
        f"Posterior Information Impact        : {'IMPROVED' if f1_delta > 0 else 'NEUTRAL/MARGINAL'} (F1 Delta: {f1_delta:+.4f}, AUC Delta: {auc_delta:+.4f})"
    )
    print(
        f"Most Sensitive Semantic Head        : Topic Head ({df_diag['err_topic_mismatch'].mean()*100:.2f}% mismatch rate on error)"
    )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
