"""
run_full_pipeline.py
FINAL Inference and Objective Evaluation Pipeline for NLU_Robust_Experiment

LIVE / DEPLOYABLE PATH:
Audio → Whisper encoder/decoder → Voice/Text NLU → Detector → Gemma

OFFLINE EVALUATION ONLY:
Master transcript → clean Text-NLU
Master labels → determine whether live Text-NLU is semantically wrong
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
import jiwer
from pathlib import Path
from tqdm import tqdm
import whisper
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, auc, precision_recall_curve
)
from scipy.spatial.distance import jensenshannon

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

WHISPER_PATH = "/home/spark2/Models/base.en.pt"
MINILM_PATH = "/home/spark2/Models/all-MiniLM-L6-v2"
GEMMA_PATH = "/home/spark2/Models/gemma_2_models/gemma-3-1b-it" 

# Canonical Master Dataset
DATASET_CSV = DATASET_DIR / "nlu_robust_6000_scenario_paraphrase_FINAL_70_10_20.csv"

HEADS = ["domain", "subdomain", "topic", "document_type"]
HEAD_WEIGHTS = {"domain": 0.20, "subdomain": 0.25, "topic": 0.40, "document_type": 0.15}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WARMUP_RUNS = 20
MEASURED_RUNS = 200
EPS = 1e-12

# ==============================================================================
# 2. ENVIRONMENT VERIFICATION
# ==============================================================================
def record_environment():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    
    import sklearn
    import scipy
    import sentence_transformers
    import transformers
    
    env_info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "cpu_ram_gb": psutil.virtual_memory().total / (1024**3),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__
    }
    
    print("\n--- ENVIRONMENT SPECIFICATIONS ---")
    for k, v in env_info.items():
        print(f"{k}: {v}")
        
    with open(RESULTS_DIR / "environment_info.json", "w") as f:
        json.dump(env_info, f, indent=4)

def validate_artifacts():
    required_files = [
        WHISPER_PATH, MINILM_PATH, GEMMA_PATH, DATASET_CSV,
        MODELS_DIR / "best_voice_projection_FINAL_70_10_20.pt",
        MODELS_DIR / "best_text_projection_FINAL_70_10_20.pt",
        MODELS_DIR / "voice_whisper_scaler_FINAL_70_10_20.joblib",
        MODELS_DIR / "text_scaler_FINAL_70_10_20.joblib",
        MODELS_DIR / "shared_label_encoders_FINAL_70_10_20.joblib",
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
        raise RuntimeError(f"CRITICAL ERROR: Missing artifacts: {missing}")
        
    print("\nARTIFACT VALIDATION PASSED")

# ==============================================================================
# 3. ARCHITECTURE DEFINITIONS
# ==============================================================================
class VoiceHierarchicalProjection(nn.Module):
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
# 4. DETECTOR FEATURE RECONSTRUCTION (COLAB EXACT MATCH)
# ==============================================================================
def entropy(p):
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))

def margin(p):
    if len(p) < 2:
        return 1.0
    s = np.sort(p)[::-1]
    return float(s[0] - s[1])

def aligned_js(vp, vc, tp, tc):
    classes = sorted(set(vc) | set(tc))
    vmap = {c: p for c, p in zip(vc, vp)}
    tmap = {c: p for c, p in zip(tc, tp)}

    va = np.array([vmap.get(c, EPS) for c in classes])
    ta = np.array([tmap.get(c, EPS) for c in classes])

    va /= va.sum()
    ta /= ta.sum()

    d = jensenshannon(va, ta, base=2)
    if np.isnan(d):
        return 0.0
    return float(d ** 2)

def extract_detector_features(v_preds, t_preds, v_probs, t_probs, v_mlps, t_mlps, encoders):
    fa = {}
    total_disagreement = 0.0
    weighted_disagreement = 0.0

    # Detector A - Hard Disagreement
    for head in HEADS:
        v_label = v_preds[head]
        t_label = t_preds[head]
        disagreement = float(v_label != t_label)
        
        short = head.replace("_label", "")
        fa[f"{short}_disagreement"] = disagreement
        total_disagreement += disagreement
        weighted_disagreement += HEAD_WEIGHTS[head] * disagreement

    fa["total_disagreements"] = total_disagreement
    fa["weighted_disagreement"] = weighted_disagreement

    # Detector B - Posteriors
    fb = dict(fa)
    voice_confs, text_confs = [], []
    cross_supports = []
    js_values = []

    for head in HEADS:
        vp = v_probs[head]
        tp = t_probs[head]
        
        v_classes = encoders[f"{head}_label"].inverse_transform(v_mlps[head].classes_)
        t_classes = encoders[f"{head}_label"].inverse_transform(t_mlps[head].classes_)

        v_idx = int(np.argmax(vp))
        t_idx = int(np.argmax(tp))
        
        v_label = v_classes[v_idx]
        t_label = t_classes[t_idx]

        v_conf = float(vp[v_idx])
        t_conf = float(tp[t_idx])

        v_class_map = {c: i for i, c in enumerate(v_classes)}
        t_class_map = {c: i for i, c in enumerate(t_classes)}

        text_prob_voice = float(tp[t_class_map[v_label]]) if v_label in t_class_map else 0.0
        voice_prob_text = float(vp[v_class_map[t_label]]) if t_label in v_class_map else 0.0

        js = aligned_js(vp, v_classes, tp, t_classes)
        v_ent = entropy(vp)
        t_ent = entropy(tp)
        v_margin = margin(vp)
        t_margin = margin(tp)

        short = head.replace("_label", "")
        
        fb[f"{short}_voice_top1_confidence"] = v_conf
        fb[f"{short}_text_top1_confidence"] = t_conf
        fb[f"{short}_confidence_gap"] = abs(v_conf - t_conf)
        fb[f"{short}_text_prob_of_voice_label"] = text_prob_voice
        fb[f"{short}_voice_prob_of_text_label"] = voice_prob_text
        fb[f"{short}_js_divergence"] = js
        fb[f"{short}_voice_entropy"] = v_ent
        fb[f"{short}_text_entropy"] = t_ent
        fb[f"{short}_voice_margin"] = v_margin
        fb[f"{short}_text_margin"] = t_margin

        voice_confs.append(v_conf)
        text_confs.append(t_conf)
        cross_supports.extend([text_prob_voice, voice_prob_text])
        js_values.append(js)

    mean_v_conf = float(np.mean(voice_confs))
    mean_t_conf = float(np.mean(text_confs))

    fb["mean_voice_confidence"] = mean_v_conf
    fb["mean_text_confidence"] = mean_t_conf
    fb["mean_cross_model_support"] = float(np.mean(cross_supports))
    fb["weighted_js_divergence"] = float(sum(HEAD_WEIGHTS[h] * js for h, js in zip(HEADS, js_values)))
    fb["strong_conflict_score"] = float(weighted_disagreement * min(mean_v_conf, mean_t_conf))

    return fa, fb

# ==============================================================================
# 5. DATASET VALIDATION & LOADING
# ==============================================================================
def load_and_validate_dataset():
    df = pd.read_csv(DATASET_CSV)
    df["sample_id"] = df["sample_id"].astype(str)
    df["scenario_id"] = df["scenario_id"].astype(str)
    
    required_master_columns = [
        "sample_id", "scenario_id", "transcript", "domain_label",
        "subdomain_label", "topic_label", "document_type_label", "split"
    ]
    missing = [c for c in required_master_columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"Master dataset is missing required columns: {missing}")

    assert len(df) == 6000, f"Expected 6000 samples, got {len(df)}"
    assert df["sample_id"].nunique() == 6000, "Sample IDs are not unique"
    assert df["scenario_id"].nunique() == 600, "Scenario IDs are not unique"
    assert set(df["split"].unique()) == {"train", "validation", "unseen"}, "Invalid splits"
    
    scenario_split_counts = df.groupby("scenario_id")["split"].nunique()
    assert (scenario_split_counts > 1).sum() == 0, "Scenario leakage detected"
    
    split_counts = df["split"].value_counts()
    assert split_counts["train"] == 4200
    assert split_counts["validation"] == 900
    assert split_counts["unseen"] == 900
    
    df_unseen = df[df["split"] == "unseen"].copy()
    
    # Audio availability validation
    expected_audio_ids = set(df_unseen["sample_id"].astype(str))
    available_audio_ids = {p.stem for p in AUDIO_DIR.glob("*.wav")}
    missing_audio_ids = expected_audio_ids - available_audio_ids
    
    if missing_audio_ids:
        raise RuntimeError(f"Missing {len(missing_audio_ids)} unseen WAV files. Examples: {sorted(missing_audio_ids)[:10]}")

    return df_unseen

# ==============================================================================
# 6. MAIN PIPELINE EXECUTION
# ==============================================================================
def run_pipeline():
    record_environment()
    validate_artifacts()
    
    df_unseen = load_and_validate_dataset()
    print(f"\nUnseen runtime evaluation samples: {len(df_unseen)}")
    
    # Load Foundation Models
    print("\nLoading Foundation Models...")
    whisper_model = whisper.load_model(WHISPER_PATH, device=DEVICE)
    minilm_model = SentenceTransformer(MINILM_PATH, device=DEVICE)
    gemma_tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, local_files_only=True)
    gemma_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    gemma_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_PATH, local_files_only=True, torch_dtype=gemma_dtype, device_map="auto"
    )
    
    # Load NLU & Detectors
    print("Loading NLU Projections, MLPs & Detectors...")
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
        if "threshold_A" not in thresholds or "threshold_B" not in thresholds:
            raise RuntimeError("Threshold JSON missing threshold_A or threshold_B")
        thresh_A = float(thresholds["threshold_A"])
        thresh_B = float(thresholds["threshold_B"])

    # Artifact Compatibility Validation
    assert v_scaler.n_features_in_ == 512, "Voice scaler expects 512"
    assert t_scaler.n_features_in_ == 384, "Text scaler expects 384"
    for h in HEADS:
        assert v_mlps[h].n_features_in_ == 128, f"Voice MLP {h} expects 128"
        assert t_mlps[h].n_features_in_ == 128, f"Text MLP {h} expects 128"

    def get_nlu_preds(emb_128, mlps):
        preds, probs_dict = {}, {}
        for h in HEADS:
            probs = mlps[h].predict_proba(emb_128)[0]
            probs_dict[h] = probs
            pred_class_int = mlps[h].classes_[np.argmax(probs)]
            preds[h] = str(shared_encoders[f"{h}_label"].inverse_transform([pred_class_int])[0])
        return preds, probs_dict

    def generate_ans(prompt):
        inp = gemma_tokenizer(prompt, return_tensors="pt").to(DEVICE)
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0_gen = time.perf_counter()
        with torch.no_grad():
            out = gemma_model.generate(**inp, max_new_tokens=30, do_sample=False, temperature=None)
        if DEVICE == "cuda": torch.cuda.synchronize()
        lat = (time.perf_counter() - t0_gen) * 1000
        gen = out[:, inp["input_ids"].shape[1]:]
        return gemma_tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip(), lat

    inference_results = []
    latencies = []
    warmup_count = 0
    measured_count = 0
    
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("\nStarting Full 900-Sample Runtime Loop...")
    for idx, row in tqdm(df_unseen.iterrows(), total=len(df_unseen), desc="Live Runtime Evaluation"):
        sample_id = str(row['sample_id'])
        scenario_id = str(row['scenario_id'])
        split = row['split']
        
        # 1. Authoritative ground truth transcript
        ground_truth_text = str(row["transcript"])
        if not ground_truth_text.strip():
            raise RuntimeError(f"Empty authoritative transcript for sample {sample_id}")
            
        timers = {}
        
        # 2. Live Audio & Whisper Encoder
        audio_file = AUDIO_DIR / f"{sample_id}.wav"
        t0 = time.perf_counter()
        audio = whisper.load_audio(str(audio_file))
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(DEVICE)
        if DEVICE == "cuda": torch.cuda.synchronize()
        timers['audio_load_ms'] = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        with torch.no_grad():
            enc_out = whisper_model.encoder(mel.unsqueeze(0))
            emb_512 = enc_out.mean(dim=1).cpu().numpy().astype(np.float32)
        if DEVICE == "cuda": torch.cuda.synchronize()
        timers['whisper_encoder_ms'] = (time.perf_counter() - t0) * 1000
        
        # 3. Voice NLU Path
        t0 = time.perf_counter()
        v_scaled = v_scaler.transform(emb_512).astype(np.float32)
        with torch.no_grad():
            v_128 = v_proj(torch.tensor(v_scaled, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        v_preds, v_probs = get_nlu_preds(v_128, v_mlps)
        if DEVICE == "cuda": torch.cuda.synchronize()
        timers['voice_nlu_ms'] = (time.perf_counter() - t0) * 1000
        
        # 4. Whisper Decoder (LIVE Transcript)
        t0 = time.perf_counter()
        decode_options = whisper.DecodingOptions(fp16=(DEVICE == "cuda"), temperature=0.0, language="en")
        with torch.no_grad():
            dec_res = whisper.decode(whisper_model, mel, decode_options)
            decoded_transcript = dec_res.text.strip()
        if DEVICE == "cuda": torch.cuda.synchronize()
        timers['whisper_decoder_ms'] = (time.perf_counter() - t0) * 1000
        
        # 5. Live Text NLU Path
        t0 = time.perf_counter()
        emb_384 = minilm_model.encode([decoded_transcript], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
        t_scaled = t_scaler.transform(emb_384).astype(np.float32)
        with torch.no_grad():
            t_128 = t_proj(torch.tensor(t_scaled, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        t_preds, t_probs = get_nlu_preds(t_128, t_mlps)
        if DEVICE == "cuda": torch.cuda.synchronize()
        timers['text_nlu_ms'] = (time.perf_counter() - t0) * 1000
        
        # 6. Detector B Extraction
        t0 = time.perf_counter()
        feat_a, feat_b = extract_detector_features(v_preds, t_preds, v_probs, t_probs, v_mlps, t_mlps, shared_encoders)
        
        df_a = pd.DataFrame([feat_a])
        df_b = pd.DataFrame([feat_b])
        
        # Bypass Scikit-Learn's broken feature_names_in_ attribute validation.
        # Dictionaries are built in the exact deterministic order required.
        prob_a = float(detector_A.predict_proba(df_a.values)[0, 1])
        prob_b = float(detector_B.predict_proba(df_b.values)[0, 1])

        pred_a = 1 if prob_a >= thresh_A else 0
        pred_b = 1 if prob_b >= thresh_B else 0
        if DEVICE == "cuda": torch.cuda.synchronize()
        timers['detector_ms'] = (time.perf_counter() - t0) * 1000
        
        # 7. Gemma Generation
        base_prompt = f"You are answering a user's request represented by the speech transcript below.\n\nTranscript:\n{decoded_transcript}\n\nAnswer briefly and only using information supported by the transcript.\nDo not invent facts."
        voice_prompt = f"You are answering a user's request represented by the speech transcript below.\n\nTranscript:\n{decoded_transcript}\n\nIndependent acoustic semantic evidence:\nDomain: {v_preds['domain']}\nTopic: {v_preds['topic']}\n\nUse the acoustic semantic evidence only as supporting evidence when interpreting ambiguity.\nDo not blindly override the transcript.\nDo not invent facts.\nAnswer briefly."
        gated_prompt = f"You are answering a user's request represented by the speech transcript below.\n\nThe transcript has been flagged as potentially containing an ASR-induced semantic error.\n\nTranscript:\n{decoded_transcript}\n\nIndependent acoustic semantic evidence:\nDomain: {v_preds['domain']}\nTopic: {v_preds['topic']}\n\nUse the acoustic semantic evidence to resolve ambiguity only when it provides supporting evidence.\nDo not invent facts.\nAnswer briefly."
        
        ans_base, lat_base = generate_ans(base_prompt)
        ans_voice, lat_voice = generate_ans(voice_prompt)
        
        if pred_b == 1:
            ans_gated, lat_gated = generate_ans(gated_prompt)
        else:
            ans_gated, lat_gated = ans_base, 0.0
            
        timers['gemma_baseline_ms'] = lat_base
        timers['gemma_voice_ms'] = lat_voice
        timers['gemma_gated_ms'] = lat_gated
        
        timers["semantic_pipeline_ms"] = (timers["voice_nlu_ms"] + timers["whisper_decoder_ms"] + timers["text_nlu_ms"] + timers["detector_ms"])
        
        # Exact architectural timing
        timers["total_baseline_ms"] = timers["audio_load_ms"] + timers["whisper_encoder_ms"] + timers["whisper_decoder_ms"] + timers["text_nlu_ms"] + timers["gemma_baseline_ms"]
        timers["total_voice_ms"] = timers["audio_load_ms"] + timers["whisper_encoder_ms"] + timers["whisper_decoder_ms"] + timers["voice_nlu_ms"] + timers["gemma_voice_ms"]
        timers["total_gated_ms"] = timers["audio_load_ms"] + timers["whisper_encoder_ms"] + timers["whisper_decoder_ms"] + timers["voice_nlu_ms"] + timers["text_nlu_ms"] + timers["detector_ms"] + timers["gemma_gated_ms"]
        
        # 8. Offline Clean-Text NLU Evaluation & Error Calcs
        clean_emb_384 = minilm_model.encode([ground_truth_text], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
        clean_scaled = t_scaler.transform(clean_emb_384).astype(np.float32)
        with torch.no_grad():
            clean_128 = t_proj(torch.tensor(clean_scaled, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        text_clean_preds, _ = get_nlu_preds(clean_128, t_mlps)
        
        gt_labels = {h: str(row[f"{h}_label"]) for h in HEADS}
        clean_text_semantically_correct = int(all(text_clean_preds[h] == gt_labels[h] for h in HEADS))
        live_text_semantically_correct = int(all(t_preds[h] == gt_labels[h] for h in HEADS))
        strict_live_asr_error = int(clean_text_semantically_correct == 1 and live_text_semantically_correct == 0)
        
        wer_value = jiwer.wer(ground_truth_text, decoded_transcript)
        cer_value = jiwer.cer(ground_truth_text, decoded_transcript)
        
        # 3-Sample Pre-flight Diagnostic
        if idx < 3:
            print(f"\n--- PRE-FLIGHT SAMPLE {idx+1} ---")
            print(f"Sample ID: {sample_id} | WER: {wer_value:.3f} | CER: {cer_value:.3f}")
            print(f"Authoritative Transcript: {ground_truth_text}")
            print(f"Live Whisper Transcript:  {decoded_transcript}")
            print(f"Clean Correct: {clean_text_semantically_correct} | Live Correct: {live_text_semantically_correct} | Strict Error Target: {strict_live_asr_error}")
            print(f"Detector B Prob: {prob_b:.4f} | Pred: {pred_b}")
            print(f"Gemma Baseline: {ans_base}")
            print(f"Gemma Gated: {ans_gated}")

        res_dict = {
            "sample_id": sample_id,
            "scenario_id": scenario_id,
            "split": split,
            "ground_truth": ground_truth_text,
            "whisper_transcript": decoded_transcript,
            "WER": wer_value,
            "CER": cer_value,
            "clean_text_semantically_correct": clean_text_semantically_correct,
            "live_text_semantically_correct": live_text_semantically_correct,
            "strict_live_asr_error": strict_live_asr_error,
            "detector_A_prob": prob_a,
            "detector_A_pred": pred_a,
            "detector_B_prob": prob_b,
            "detector_B_pred": pred_b,
            "gemma_baseline_ans": ans_base,
            "gemma_voice_ans": ans_voice,
            "gemma_gated_ans": ans_gated,
            "live_semantic_changed_heads": sum(1 for h in HEADS if text_clean_preds[h] != t_preds[h])
        }
        for h in HEADS:
            res_dict[f"gt_{h}"] = gt_labels[h]
            res_dict[f"voice_pred_{h}"] = v_preds[h]
            res_dict[f"text_pred_{h}"] = t_preds[h]
            res_dict[f"clean_text_pred_{h}"] = text_clean_preds[h]
            res_dict[f"clean_vs_live_{h}_changed"] = int(text_clean_preds[h] != t_preds[h])
            
        inference_results.append(res_dict)
        
        # Add latency tracking ignoring warmup
        if warmup_count < WARMUP_RUNS:
            warmup_count += 1
        elif measured_count < MEASURED_RUNS:
            latencies.append(timers)
            measured_count += 1

    # ==============================================================================
    # 7. METRICS & SAVING
    # ==============================================================================
    print("\nCalculating Final Metrics...")
    df_res = pd.DataFrame(inference_results)
    df_res.to_csv(RESULTS_DIR / "inference_results.csv", index=False)
    
    asr_metrics = df_res[["sample_id", "split", "WER", "CER"]]
    asr_metrics.to_csv(RESULTS_DIR / "ASR_RUNTIME_METRICS.csv", index=False)
    
    df_lat = pd.DataFrame(latencies)
    df_lat.to_csv(RESULTS_DIR / "LATENCY_BREAKDOWN.csv", index=False)
    lat_summary = df_lat.describe(percentiles=[.50, .90, .95, .99]).T
    lat_summary.to_csv(RESULTS_DIR / "LATENCY_SUMMARY.csv")
    
    # Detector Validation Check
    pos_count = df_res['strict_live_asr_error'].sum()
    print(f"\nLIVE STRICT ASR ERROR TARGET DISTRIBUTION")
    print(f"0: {len(df_res) - pos_count}")
    print(f"1: {pos_count}")
    print(f"Positive rate: {pos_count / len(df_res):.3f}")
    
    if pos_count == 0:
        raise RuntimeError(
            "No positive strict_live_asr_error samples were found. "
            "Do not compute detector ROC-AUC/F1 as a meaningful error-detection result."
        )
    
    # Detector Metrics
    y_true = df_res['strict_live_asr_error']
    det_metrics = []
    for name, pred, prob in [("Detector_A", df_res['detector_A_pred'], df_res['detector_A_prob']),
                             ("Detector_B", df_res['detector_B_pred'], df_res['detector_B_prob'])]:
        if len(y_true.unique()) > 1:
            roc = roc_auc_score(y_true, prob)
            pr_curve = precision_recall_curve(y_true, prob)
            pr = auc(pr_curve[1], pr_curve[0])
        else:
            roc, pr = np.nan, np.nan
            print(f"ROC-AUC unavailable for {name} because unseen runtime target contains only one class.")
            
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        det_metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, pred),
            "F1": f1_score(y_true, pred, zero_division=0),
            "ROC-AUC": roc,
            "PR-AUC": pr,
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred, zero_division=0),
            "Specificity": tn / max((tn + fp), 1),
            "FPR": fp / max((fp + tn), 1),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn
        })
    pd.DataFrame(det_metrics).to_csv(RESULTS_DIR / "DETECTOR_RUNTIME_RESULTS.csv", index=False)
    
    reasoning_metrics = [{"Condition": "NOT_OBJECTIVELY_EVALUATED", "Reason": "No authoritative reasoning answer exists in the dataset."}]
    pd.DataFrame(reasoning_metrics).to_csv(RESULTS_DIR / "REASONING_RESULTS.csv", index=False)
    
    nlu_metrics = []
    for h in HEADS:
        v_f1 = f1_score(df_res[f"gt_{h}"], df_res[f"voice_pred_{h}"], average='macro', zero_division=0)
        t_f1 = f1_score(df_res[f"gt_{h}"], df_res[f"text_pred_{h}"], average='macro', zero_division=0)
        nlu_metrics.append({"Head": h, "Voice-NLU_LIVE_AUDIO_RUNTIME_F1": v_f1, "Text-NLU_LIVE_WHISPER_TRANSCRIPT_RUNTIME_F1": t_f1})
    pd.DataFrame(nlu_metrics).to_csv(RESULTS_DIR / "NLU_RUNTIME_RESULTS.csv", index=False)
    
    # Final Output Report
    summary_text = f"""LIVE END-TO-END RUNTIME EVALUATION

Dataset:
6000 utterances
600 scenarios
900 unseen utterances
90 unseen scenarios

Voice-NLU live metrics:
Topic F1: {nlu_metrics[2]['Voice-NLU_LIVE_AUDIO_RUNTIME_F1']:.4f}

Text-NLU live metrics:
Topic F1: {nlu_metrics[2]['Text-NLU_LIVE_WHISPER_TRANSCRIPT_RUNTIME_F1']:.4f}

Live Whisper WER:
{df_res['WER'].mean():.4f}

Live Whisper CER:
{df_res['CER'].mean():.4f}

Strict live ASR-induced semantic error rate:
{pos_count / len(df_res):.4f}

Detector A:
F1: {det_metrics[0]['F1']:.4f}
ROC-AUC: {det_metrics[0]['ROC-AUC']:.4f}
PR-AUC: {det_metrics[0]['PR-AUC']:.4f}
FPR: {det_metrics[0]['FPR']:.4f}

Detector B:
F1: {det_metrics[1]['F1']:.4f}
ROC-AUC: {det_metrics[1]['ROC-AUC']:.4f}
PR-AUC: {det_metrics[1]['PR-AUC']:.4f}
FPR: {det_metrics[1]['FPR']:.4f}

Latency:
Baseline mean: {lat_summary.loc['total_baseline_ms', 'mean']:.1f} ms
Baseline P95: {lat_summary.loc['total_baseline_ms', '95%']:.1f} ms
Voice-grounded mean: {lat_summary.loc['total_voice_ms', 'mean']:.1f} ms
Voice-grounded P95: {lat_summary.loc['total_voice_ms', '95%']:.1f} ms
Detector-gated mean: {lat_summary.loc['total_gated_ms', 'mean']:.1f} ms
Detector-gated P95: {lat_summary.loc['total_gated_ms', '95%']:.1f} ms

Peak GPU memory:
{torch.cuda.max_memory_reserved(DEVICE)/(1024**3) if DEVICE == 'cuda' else 0:.2f} GB
"""
    with open(RESULTS_DIR / "FINAL_RUNTIME_SUMMARY.txt", "w") as f:
        f.write(summary_text)

    print("\n" + "="*50 + "\nFINAL RESEARCH SUMMARY\n" + "="*50)
    print(summary_text)

if __name__ == "__main__":
    run_pipeline()
