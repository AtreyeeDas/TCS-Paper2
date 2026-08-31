"""
run_full_pipeline.py
FINAL Inference and Objective Evaluation Pipeline for NLU_Robust_Experiment

Execution: python run_full_pipeline.py
"""

import os
import sys
import json
import time
import psutil
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import whisper
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, auc, precision_recall_curve, balanced_accuracy_score
)
from scipy.spatial.distance import jensenshannon
import evaluate

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. DIRECTORY STRUCTURE & CONFIGURATION
# ==============================================================================
PROJECT_ROOT = Path("/home/spark2/users/intern/Atreyee-Das/NLU_Robust_Experiment")
MODELS_DIR = PROJECT_ROOT / "models"
DETECTOR_DIR = PROJECT_ROOT / "detector"
DATASET_DIR = PROJECT_ROOT / "dataset"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
AUDIO_DIR = PROJECT_ROOT / "audio"
RESULTS_DIR = PROJECT_ROOT / "runtime_results"

# Specific File Paths
WHISPER_PATH = "/home/spark2/Models/base.en.pt"
MINILM_PATH = MODELS_DIR / "all-MiniLM-L6-v2"
GEMMA_PATH = "/home/spark2/Models/gemma-3-1b-it"  # Assumed local path based on specs
DATASET_CSV = DATASET_DIR / "nlu_robust_6000_scenario_paraphrase_FINAL_70_10_20.csv"
WHISPER_EMBEDDINGS_NPY = EMBEDDINGS_DIR / "whisper_embeddings_FINAL_70_10_20.npy"
WHISPER_EMBEDDINGS_META = EMBEDDINGS_DIR / "whisper_embedding_metadata_FINAL_70_10_20.csv"

HEADS = ["domain", "subdomain", "topic", "document_type"]
HEAD_WEIGHTS = {"domain": 0.20, "subdomain": 0.25, "topic": 0.40, "document_type": 0.15}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 2. ENVIRONMENT VERIFICATION
# ==============================================================================
def record_environment():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    
    env_info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "cpu_ram_gb": psutil.virtual_memory().total / (1024**3)
    }
    
    print("\n--- ENVIRONMENT SPECIFICATIONS ---")
    for k, v in env_info.items():
        print(f"{k}: {v}")
        
    with open(RESULTS_DIR / "environment_info.json", "w") as f:
        json.dump(env_info, f, indent=4)

def validate_artifacts():
    required_files = [
        WHISPER_PATH, MINILM_PATH, DATASET_CSV, WHISPER_EMBEDDINGS_NPY, WHISPER_EMBEDDINGS_META,
        MODELS_DIR / "best_voice_projection_FINAL_70_10_20.pt",
        MODELS_DIR / "best_text_projection_FINAL_70_10_20.pt",
        DETECTOR_DIR / "Detector_A_STRICT_ASR_INDUCED.joblib",
        DETECTOR_DIR / "Detector_B_STRICT_ASR_INDUCED.joblib",
        DETECTOR_DIR / "strict_detector_thresholds.json"
    ]
    for head in HEADS:
        required_files.extend([
            MODELS_DIR / f"voice_{head}_label_mlp_FINAL_70_10_20.joblib",
            MODELS_DIR / f"text_{head}_label_mlp_FINAL_70_10_20.joblib"
        ])
    
    missing = [str(f) for f in required_files if not Path(f).exists()]
    if missing:
        print("CRITICAL ERROR: Missing artifacts:")
        for m in missing: print(f" - {m}")
        sys.exit(1)
        
    print("\nARTIFACT VALIDATION PASSED")

# ==============================================================================
# 3. ARCHITECTURE DEFINITIONS
# ==============================================================================
class VoiceHierarchicalProjection(nn.Module):
    """Transforms 512-D Whisper encoder output to 128-D L2-normalized vector."""
    def __init__(self, input_dim=512, projection_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, projection_dim),
        )
    def forward(self, x):
        return F.normalize(self.projector(x), p=2, dim=1)

class TextHierarchicalProjection(nn.Module):
    """Transforms 384-D MiniLM output to 128-D L2-normalized vector[cite: 4, 5]."""
    def __init__(self, input_dim=384, projection_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, projection_dim),
        )
    def forward(self, x):
        return F.normalize(self.projector(x), p=2, dim=1)

# ==============================================================================
# 4. DETECTOR FEATURE RECONSTRUCTION
# ==============================================================================
def align_and_js(v_probs, v_classes, t_probs, t_classes):
    """Aligns distributions to the same vocabulary union before JS divergence."""
    u_cls = sorted(list(set(v_classes) | set(t_classes)))
    v_a = np.array([v_probs[np.where(v_classes == c)[0][0]] if c in v_classes else 1e-12 for c in u_cls])
    t_a = np.array([t_probs[np.where(t_classes == c)[0][0]] if c in t_classes else 1e-12 for c in u_cls])
    v_a = np.clip(v_a / v_a.sum(), 1e-12, 1.0)
    t_a = np.clip(t_a / t_a.sum(), 1e-12, 1.0)
    return float(jensenshannon(v_a, t_a) ** 2)

def extract_detector_features(voice_preds, text_preds, voice_probs, text_probs, encoders):
    """Exact feature construction for Detector A (Semantic) and B (Semantic + Posteriors)[cite: 2]."""
    feat_a, feat_b = {}, {}
    tot_dis, w_dis = 0.0, 0.0
    v_confs, t_confs, cross_supp = [], [], []
    
    for h in HEADS:
        dis = 1.0 if voice_preds[h] != text_preds[h] else 0.0
        feat_a[f"{h}_disagreement"] = dis
        tot_dis += dis
        w_dis += HEAD_WEIGHTS[h] * dis
        
        v_p, t_p = voice_probs[h], text_probs[h]
        v_cls, t_cls = encoders[f"{h}_label"].classes_, encoders[f"{h}_label"].classes_
        
        v_top1, t_top1 = float(np.max(v_p)), float(np.max(t_p))
        v_confs.append(v_top1); t_confs.append(t_top1)
        
        v_idx_t = np.where(v_cls == text_preds[h])[0]
        t_idx_v = np.where(t_cls == voice_preds[h])[0]
        v_prob_t = float(v_p[v_idx_t[0]]) if len(v_idx_t) > 0 else 0.0
        t_prob_v = float(t_p[t_idx_v[0]]) if len(t_idx_v) > 0 else 0.0
        cross_supp.extend([v_prob_t, t_prob_v])
        
        feat_b[f"{h}_voice_top1_confidence"] = v_top1
        feat_b[f"{h}_text_top1_confidence"] = t_top1
        feat_b[f"{h}_confidence_gap"] = abs(v_top1 - t_top1)
        feat_b[f"{h}_text_prob_of_voice_label"] = t_prob_v
        feat_b[f"{h}_voice_prob_of_text_label"] = v_prob_t
        feat_b[f"{h}_js_divergence"] = align_and_js(v_p, v_cls, t_p, t_cls)
        feat_b[f"{h}_voice_entropy"] = float(-np.sum(v_p * np.log2(np.clip(v_p, 1e-12, 1.0))))
        feat_b[f"{h}_text_entropy"] = float(-np.sum(t_p * np.log2(np.clip(t_p, 1e-12, 1.0))))
        
        v_s, t_s = np.sort(v_p)[::-1], np.sort(t_p)[::-1]
        feat_b[f"{h}_voice_margin"] = float(v_s[0] - (v_s[1] if len(v_s) > 1 else 0))
        feat_b[f"{h}_text_margin"] = float(t_s[0] - (t_s[1] if len(t_s) > 1 else 0))
        
    feat_a["total_disagreements"] = tot_dis
    feat_a["weighted_disagreement"] = w_dis
    feat_b.update(feat_a)
    
    feat_b["mean_voice_confidence"] = float(np.mean(v_confs))
    feat_b["mean_text_confidence"] = float(np.mean(t_confs))
    feat_b["mean_cross_model_support"] = float(np.mean(cross_supp))
    
    return feat_a, feat_b

# ==============================================================================
# 5. MAIN PIPELINE EXECUTION
# ==============================================================================
def run_pipeline():
    record_environment()
    validate_artifacts()
    
    # 1. Load Models & Scalers
    print("\nLoading Foundation Models...")
    whisper_model = whisper.load_model(WHISPER_PATH, device=DEVICE)
    minilm_model = SentenceTransformer(MINILM_PATH, device=DEVICE)
    gemma_tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, local_files_only=True)
    gemma_model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, local_files_only=True, torch_dtype=torch.float16).to(DEVICE)
    
    print("Loading NLU Projections & MLPs...")
    v_scaler = joblib.load(MODELS_DIR / "voice_whisper_scaler_FINAL_70_10_20.joblib")
    t_scaler = joblib.load(MODELS_DIR / "text_scaler_FINAL_70_10_20.joblib")
    v_proj = VoiceHierarchicalProjection().to(DEVICE)
    v_proj.load_state_dict(torch.load(MODELS_DIR / "best_voice_projection_FINAL_70_10_20.pt", map_location=DEVICE))
    v_proj.eval()
    t_proj = TextHierarchicalProjection().to(DEVICE)
    t_proj.load_state_dict(torch.load(MODELS_DIR / "best_text_projection_FINAL_70_10_20.pt", map_location=DEVICE))
    t_proj.eval()
    
    shared_encoders = joblib.load(MODELS_DIR / "shared_label_encoders_FINAL_70_10_20.joblib")
    v_mlps = {h: joblib.load(MODELS_DIR / f"voice_{h}_label_mlp_FINAL_70_10_20.joblib") for h in HEADS}
    t_mlps = {h: joblib.load(MODELS_DIR / f"text_{h}_label_mlp_FINAL_70_10_20.joblib") for h in HEADS}
    
    detector_A = joblib.load(DETECTOR_DIR / "Detector_A_STRICT_ASR_INDUCED.joblib")
    detector_B = joblib.load(DETECTOR_DIR / "Detector_B_STRICT_ASR_INDUCED.joblib")
    with open(DETECTOR_DIR / "strict_detector_thresholds.json", "r") as f:
        thresholds = json.load(f)
        thresh_A = float(thresholds["threshold_A"])
        thresh_B = float(thresholds["threshold_B"])

    # 2. Dataset Preparation
    df = pd.read_csv(DATASET_CSV)
    
    # Validation of Splits
    split_counts = df['split'].value_counts()
    print(f"\nDataset Splits:\n{split_counts}")
    
    # 3. Execution & Latency Profiling
    inference_results = []
    latencies = []
    
    # For latency warm-up
    warmup_runs = 20
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running E2E Inference"):
        sample_id = str(row['sample_id'])
        split = row['split']
        gt_target = str(row.get('target_terms', ''))
        query = str(row.get('user_query', f"What is the {row.get('topic', 'subject')}?"))
        
        timers = {}
        
        # Audio / Whisper
        audio_file = AUDIO_DIR / f"{sample_id}.wav"
        if not audio_file.exists(): continue
        
        t0 = time.perf_counter()
        audio = whisper.load_audio(str(audio_file))
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(DEVICE)
        if DEVICE == "cuda": torch.cuda.synchronize()
        timers['audio_load_ms'] = (time.perf_counter() - t0) * 1000
        
        # Whisper Encoder
        t0 = time.perf_counter()
        with torch.no_grad():
            enc_out = whisper_model.encoder(mel.unsqueeze(0))
            emb_512 = enc_out.mean(dim=1).cpu().numpy().astype(np.float32)
        if DEVICE == "cuda": torch.cuda.synchronize()
        timers['whisper_encoder_ms'] = (time.perf_counter() - t0) * 1000
        
        # Voice NLU Path
        t0 = time.perf_counter()
        v_scaled = v_scaler.transform(emb_512)
        with torch.no_grad():
            v_128 = v_proj(torch.tensor(v_scaled, device=DEVICE)).cpu().numpy()
        
        v_preds, v_probs = {}, {}
        for h in HEADS:
            probs = v_mlps[h].predict_proba(v_128)[0]
            v_probs[h] = probs
            v_preds[h] = shared_encoders[f"{h}_label"].inverse_transform([np.argmax(probs)])[0]
        if DEVICE == "cuda": torch.cuda.synchronize()
        timers['voice_nlu_ms'] = (time.perf_counter() - t0) * 1000
        
        # Whisper Decoder
        t0 = time.perf_counter()
        with torch.no_grad():
            dec_res = whisper.decode(whisper_model, mel, whisper.DecodingOptions(fp16=False))
            decoded_transcript = dec_res.text.strip()
        if DEVICE == "cuda": torch.cuda.synchronize()
        timers['whisper_decoder_ms'] = (time.perf_counter() - t0) * 1000
        
        # Text NLU Path
        t0 = time.perf_counter()
        emb_384 = minilm_model.encode([decoded_transcript], convert_to_numpy=True).astype(np.float32)
        t_scaled = t_scaler.transform(emb_384)
        with torch.no_grad():
            t_128 = t_proj(torch.tensor(t_scaled, device=DEVICE)).cpu().numpy()
            
        t_preds, t_probs = {}, {}
        for h in HEADS:
            probs = t_mlps[h].predict_proba(t_128)[0]
            t_probs[h] = probs
            t_preds[h] = shared_encoders[f"{h}_label"].inverse_transform([np.argmax(probs)])[0]
        if DEVICE == "cuda": torch.cuda.synchronize()
        timers['text_nlu_ms'] = (time.perf_counter() - t0) * 1000
        
        # Detectors
        t0 = time.perf_counter()
        feat_a, feat_b = extract_detector_features(v_preds, t_preds, v_probs, t_probs, shared_encoders)
        
        df_a = pd.DataFrame([feat_a])[detector_A.feature_names_in_]
        df_b = pd.DataFrame([feat_b])[detector_B.feature_names_in_]
        
        prob_a = detector_A.predict_proba(df_a)[0][1]
        prob_b = detector_B.predict_proba(df_b)[0][1]
        pred_a = 1 if prob_a >= thresh_A else 0
        pred_b = 1 if prob_b >= thresh_B else 0
        if DEVICE == "cuda": torch.cuda.synchronize()
        timers['detector_ms'] = (time.perf_counter() - t0) * 1000
        
        # Gemma Reasoning
        # Baseline Prompt
        base_prompt = f"You are answering the user's question using the transcript. Do not assume that any word is wrong unless the transcript itself provides evidence. Do not invent information. Answer very briefly.\nTranscript: {decoded_transcript}\nUser query: {query}"
        
        # Detector-Gated Prompt (Only uses Voice NLU if Detector B fires)
        if pred_b == 1:
            gated_prompt = f"You are answering the user's question using the transcript. The transcript has been flagged as potentially containing an ASR error. You have independent semantic evidence derived from the acoustic speech representation: Domain: {v_preds['domain']}, Topic: {v_preds['topic']}. Use that evidence only as supporting evidence when resolving ambiguity. Correct the interpretation only when the supplied evidence supports doing so. Do not invent details. Answer very briefly.\nTranscript: {decoded_transcript}\nUser query: {query}"
        else:
            gated_prompt = base_prompt
            
        # Voice-NLU Prompt (Always uses Voice NLU)
        voice_prompt = f"You are answering the user's question using the transcript. You also have independent semantic evidence derived from the acoustic speech representation: Domain: {v_preds['domain']}, Topic: {v_preds['topic']}. Use that evidence only as supporting evidence when resolving ambiguity. Do not blindly override the transcript. Do not invent details. Answer very briefly.\nTranscript: {decoded_transcript}\nUser query: {query}"
        
        t0 = time.perf_counter()
        inputs = gemma_tokenizer([base_prompt, gated_prompt, voice_prompt], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = gemma_model.generate(**inputs, max_new_tokens=30, temperature=0.0)
        answers = gemma_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if DEVICE == "cuda": torch.cuda.synchronize()
        timers['gemma_ms'] = (time.perf_counter() - t0) * 1000
        
        timers['total_pipeline_ms'] = sum(timers.values())
        
        if warmup_runs > 0:
            warmup_runs -= 1
        else:
            latencies.append(timers)
            
        # Ground Truth & Semantic Checks
        # ---------------------------------------------------------
# Strict ASR-induced semantic error
# clean text must be correct on ALL four heads,
# decoded text must be wrong on >=1 head
# ----------------------------------------------------------

gt_labels = {
    h: str(row.get(f"{h}_label", ""))
    for h in HEADS
}

clean_text_correct = int(
    all(str(text_clean_preds[h]) == gt_labels[h] for h in HEADS ))

        decoded_text_correct = int(all( str(t_preds[h]) == gt_labels[h] for h in HEADS ))

        strict_asr_error = int(clean_text_correct == 1 and decoded_text_correct == 0) 

        res_dict = {
            "sample_id": sample_id,
            "split": split,
            "ground_truth": row.get('reference_transcript', ''),
            "whisper_transcript": decoded_transcript,
            "strict_asr_induced_error": strict_asr_error,
            "detector_A_prob": prob_a,
            "detector_A_pred": pred_a,
            "detector_B_prob": prob_b,
            "detector_B_pred": pred_b,
            "gemma_baseline_ans": answers[0].replace(base_prompt, "").strip(),
            "gemma_gated_ans": answers[1].replace(gated_prompt, "").strip(),
            "gemma_voice_ans": answers[2].replace(voice_prompt, "").strip()
        }
        for h in HEADS:
            res_dict[f"gt_{h}"] = str(row.get(f"{h}_label", ''))
            res_dict[f"voice_pred_{h}"] = v_preds[h]
            res_dict[f"text_pred_{h}"] = t_preds[h]
            
        inference_results.append(res_dict)
        
    # ==============================================================================
    # 6. METRICS & SAVING
    # ==============================================================================
    print("\nCalculating Final Metrics...")
    df_res = pd.DataFrame(inference_results)
    df_res.to_csv(RESULTS_DIR / "inference_results.csv", index=False)
    
    df_lat = pd.DataFrame(latencies)
    df_lat.to_csv(RESULTS_DIR / "LATENCY_BREAKDOWN.csv", index=False)
    
    lat_summary = df_lat.describe(percentiles=[.50, .90, .95, .99]).T
    lat_summary.to_csv(RESULTS_DIR / "LATENCY_SUMMARY.csv")
    
    # Split isolation for primary results
    df_unseen = df_res[df_res['split'] == 'unseen']
    
    # Detector Metrics
    y_true = df_unseen['strict_asr_induced_error']
    y_prob_a = df_unseen['detector_A_prob']
    y_pred_a = df_unseen['detector_A_pred']
    y_prob_b = df_unseen['detector_B_prob']
    y_pred_b = df_unseen['detector_B_pred']
    
    det_metrics = []
    for name, pred, prob in [("Detector_A", y_pred_a, y_prob_a), ("Detector_B", y_pred_b, y_prob_b)]:
        try:
            roc = roc_auc_score(y_true, prob)
            pr_curve = precision_recall_curve(y_true, prob)
            pr = auc(pr_curve[1], pr_curve[0])
        except:
            roc, pr = np.nan, np.nan
        
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel() if len(np.unique(y_true)) > 1 else (0,0,0,0)
        det_metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, pred),
            "F1": f1_score(y_true, pred, zero_division=0),
            "ROC-AUC": roc,
            "PR-AUC": pr,
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn
        })
    pd.DataFrame(det_metrics).to_csv(RESULTS_DIR / "DETECTOR_RUNTIME_RESULTS.csv", index=False)
    
    # Reasoning Metrics (Simple proxy evaluation)
    # Using exact match presence for demo purposes, normally BERTScore
    reasoning_metrics = []
    for cond, col in [("Baseline", "gemma_baseline_ans"), ("Detector-Gated", "gemma_gated_ans"), ("Voice-Grounded", "gemma_voice_ans")]:
        acc = df_unseen.apply(lambda r: str(r['gt_topic']).lower() in str(r[col]).lower(), axis=1).mean()
        reasoning_metrics.append({"Condition": cond, "Proxy_Accuracy": acc})
        
    pd.DataFrame(reasoning_metrics).to_csv(RESULTS_DIR / "REASONING_RESULTS.csv", index=False)
    
    # NLU Runtime Results
    nlu_metrics = []
    for h in HEADS:
        v_f1 = f1_score(df_unseen[f"gt_{h}"], df_unseen[f"voice_pred_{h}"], average='macro', zero_division=0)
        t_f1 = f1_score(df_unseen[f"gt_{h}"], df_unseen[f"text_pred_{h}"], average='macro', zero_division=0)
        nlu_metrics.append({"Head": h, "Voice_Macro_F1": v_f1, "Text_Macro_F1": t_f1})
    pd.DataFrame(nlu_metrics).to_csv(RESULTS_DIR / "NLU_RUNTIME_RESULTS.csv", index=False)
    
    # ==============================================================================
    # 7. FINAL SUMMARY PRINT
    # ==============================================================================
    print("\n" + "="*50)
    print("FINAL RESEARCH SUMMARY (UNSEEN SPLIT)")
    print("="*50)
    print(f"A. Voice-NLU Unseen Topic F1: {nlu_metrics[2]['Voice_Macro_F1']:.3f}")
    print(f"B. Text-NLU Unseen Topic F1:  {nlu_metrics[2]['Text_Macro_F1']:.3f}")
    print(f"D. Strict ASR-Induced Error Rate: {y_true.mean():.1%}")
    print(f"E. Detector A Unseen ROC-AUC: {det_metrics[0]['ROC-AUC']:.3f} | F1: {det_metrics[0]['F1']:.3f}")
    print(f"F. Detector B Unseen ROC-AUC: {det_metrics[1]['ROC-AUC']:.3f} | F1: {det_metrics[1]['F1']:.3f}")
    
    if not np.isnan(det_metrics[1]['ROC-AUC']) and not np.isnan(det_metrics[0]['ROC-AUC']):
        print(f"G. Detector B Improvement over A: {det_metrics[1]['ROC-AUC'] - det_metrics[0]['ROC-AUC']:+.3f} ROC-AUC")
        
    print(f"H. Reasoning Proxy Accuracy - Baseline: {reasoning_metrics[0]['Proxy_Accuracy']:.1%}")
    print(f"                              Gated:    {reasoning_metrics[1]['Proxy_Accuracy']:.1%}")
    print(f"                              Voice:    {reasoning_metrics[2]['Proxy_Accuracy']:.1%}")
    
    print(f"J. End-to-End Latency (Mean): {lat_summary.loc['total_pipeline_ms', 'mean']:.1f} ms")
    print(f"K. P95 End-to-End Latency:    {lat_summary.loc['total_pipeline_ms', '95%']:.1f} ms")
    print(f"L. Peak GPU Memory Reserved:  {torch.cuda.max_memory_reserved(DEVICE)/(1024**3):.2f} GB")
    print("="*50)
    
    with open(RESULTS_DIR / "FINAL_RUNTIME_SUMMARY.txt", "w") as f:
        f.write("FINAL RESEARCH SUMMARY (UNSEEN SPLIT)\n")
        f.write(f"Detector B F1: {det_metrics[1]['F1']:.3f}\n")
        f.write(f"Latency P95: {lat_summary.loc['total_pipeline_ms', '95%']:.1f} ms\n")

if __name__ == "__main__":
    run_pipeline()
