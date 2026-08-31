import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & LOCAL PATHS
# ==============================================================================
# Point this to your specific audio file to test
TEST_AUDIO_PATH = "/home/spark2/users/intern/Atreyee-Das/NLU_Robust_Experiment/audio/nlu_0001_01.wav"

PROJECT_ROOT = Path("/home/spark2/users/intern/Atreyee-Das/NLU_Robust_Experiment")
MODELS_DIR = PROJECT_ROOT / "models"
DETECTOR_DIR = PROJECT_ROOT / "detector"
EXP_MODELS_DIR = PROJECT_ROOT / "error_detector_experiments" / "artifacts" / "models"

WHISPER_PATH = "/home/spark2/Models/base.en.pt"
MINILM_PATH = "/home/spark2/Models/all-MiniLM-L6-v2"
GEMMA_PATH = "/home/spark2/Models/gemma_2_models/gemma-3-1b-it"

HEADS = ["domain", "subdomain", "topic", "document_type"]
HEAD_WEIGHTS = {"domain": 0.20, "subdomain": 0.25, "topic": 0.40, "document_type": 0.15}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-12

# ==============================================================================
# 2. ARCHITECTURE DEFINITIONS
# ==============================================================================
class VoiceHierarchicalProjection(nn.Module):
    def __init__(self, input_dim=512, projection_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.10), nn.Linear(256, projection_dim)
        )
    def forward(self, x): return F.normalize(self.projector(x), p=2, dim=1)

class TextHierarchicalProjection(nn.Module):
    def __init__(self, input_dim=384, projection_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.10), nn.Linear(256, projection_dim)
        )
    def forward(self, x): return F.normalize(self.projector(x), p=2, dim=1)

# ==============================================================================
# 3. FEATURE EXTRACTION & POSTERIOR METRICS
# ==============================================================================
def calc_entropy(probs):
    p = np.clip(probs, EPS, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))

def calc_margin(probs):
    s = np.sort(probs)[::-1]
    return float(s[0] - s[1]) if len(s) >= 2 else 1.0

def aligned_js(vp, vc, tp, tc):
    classes = sorted(set(vc) | set(tc))
    vmap = {c: p for c, p in zip(vc, vp)}
    tmap = {c: p for c, p in zip(tc, tp)}
    va = np.array([vmap.get(c, EPS) for c in classes])
    ta = np.array([tmap.get(c, EPS) for c in classes])
    va, ta = va / va.sum(), ta / ta.sum()
    d = jensenshannon(va, ta, base=2)
    return 0.0 if np.isnan(d) else float(d ** 2)

def extract_detector_features(v_preds, t_preds, v_probs, t_probs, v_mlps, t_mlps, encoders):
    fa, fb = {}, {}
    total_disagreement, weighted_disagreement = 0.0, 0.0
    voice_confs, text_confs, cross_supports, js_values = [], [], [], []

    for head in HEADS:
        disagreement = float(v_preds[head] != t_preds[head])
        short = head.replace("_label", "")
        fa[f"{short}_disagreement"] = disagreement
        total_disagreement += disagreement
        weighted_disagreement += HEAD_WEIGHTS[head] * disagreement

        vp, tp = v_probs[head], t_probs[head]
        v_classes = encoders[f"{head}_label"].inverse_transform(v_mlps[head].classes_)
        t_classes = encoders[f"{head}_label"].inverse_transform(t_mlps[head].classes_)
        v_idx, t_idx = int(np.argmax(vp)), int(np.argmax(tp))
        v_label, t_label = v_classes[v_idx], t_classes[t_idx]
        v_conf, t_conf = float(vp[v_idx]), float(tp[t_idx])

        v_class_map = {c: i for i, c in enumerate(v_classes)}
        t_class_map = {c: i for i, c in enumerate(t_classes)}
        text_prob_voice = float(tp[t_class_map[v_label]]) if v_label in t_class_map else 0.0
        voice_prob_text = float(vp[v_class_map[t_label]]) if t_label in v_class_map else 0.0
        js = aligned_js(vp, v_classes, tp, t_classes)

        fb[f"{short}_voice_top1_confidence"] = v_conf
        fb[f"{short}_text_top1_confidence"] = t_conf
        fb[f"{short}_confidence_gap"] = abs(v_conf - t_conf)
        fb[f"{short}_text_prob_of_voice_label"] = text_prob_voice
        fb[f"{short}_voice_prob_of_text_label"] = voice_prob_text
        fb[f"{short}_js_divergence"] = js
        fb[f"{short}_voice_entropy"] = calc_entropy(vp)
        fb[f"{short}_text_entropy"] = calc_entropy(tp)
        fb[f"{short}_voice_margin"] = calc_margin(vp)
        fb[f"{short}_text_margin"] = calc_margin(tp)

        voice_confs.append(v_conf)
        text_confs.append(t_conf)
        cross_supports.extend([text_prob_voice, voice_prob_text])
        js_values.append(js)

    fa["total_disagreements"] = total_disagreement
    fa["weighted_disagreement"] = weighted_disagreement
    fb.update(fa)
    
    mean_v_conf, mean_t_conf = float(np.mean(voice_confs)), float(np.mean(text_confs))
    fb["mean_voice_confidence"] = mean_v_conf
    fb["mean_text_confidence"] = mean_t_conf
    fb["mean_cross_model_support"] = float(np.mean(cross_supports))
    fb["weighted_js_divergence"] = float(sum(HEAD_WEIGHTS[h] * js for h, js in zip(HEADS, js_values)))
    fb["strong_conflict_score"] = float(weighted_disagreement * min(mean_v_conf, mean_t_conf))

    return fa, fb

def get_nlu_preds(emb_128, mlps, shared_encoders):
    preds, probs_dict = {}, {}
    for h in HEADS:
        probs = mlps[h].predict_proba(emb_128)[0]
        probs_dict[h] = probs
        pred_class_int = mlps[h].classes_[np.argmax(probs)]
        preds[h] = str(shared_encoders[f"{h}_label"].inverse_transform([pred_class_int])[0])
    return preds, probs_dict

# ==============================================================================
# 4. LOAD DETECTOR ARTIFACTS
# ==============================================================================
def load_detector():
    """Finds and loads the requested real_whisper_detector_B artifacts."""
    search_paths = [
        (EXP_MODELS_DIR / "real_whisper_detector_B.joblib",
         EXP_MODELS_DIR / "real_whisper_detector_B_features.json",
         EXP_MODELS_DIR / "real_whisper_detector_B_threshold.json"),
        (DETECTOR_DIR / "real_whisper_detector_B.joblib",
         DETECTOR_DIR / "real_whisper_detector_B_features.json",
         DETECTOR_DIR / "real_whisper_detector_B_threshold.json"),
        (DETECTOR_DIR / "Detector_B_STRICT_ASR_INDUCED.joblib",
         None,
         DETECTOR_DIR / "strict_detector_thresholds.json")
    ]
    
    detector_model, feature_names, threshold = None, None, 0.50

    for model_p, feat_p, thresh_p in search_paths:
        if model_p.exists():
            print(f"[+] Loading Detector Model from: {model_p}")
            detector_model = joblib.load(model_p)
            
            if feat_p and feat_p.exists():
                with open(feat_p, "r") as f:
                    feature_names = json.load(f)
                print(f"[+] Loaded Feature Schema ({len(feature_names)} features) from: {feat_p}")
            
            if thresh_p and thresh_p.exists():
                with open(thresh_p, "r") as f:
                    t_data = json.load(f)
                    threshold = float(t_data.get("threshold", t_data.get("threshold_B", 0.50)))
                print(f"[+] Loaded Threshold ({threshold:.4f}) from: {thresh_p}")
            break

    if detector_model is None:
        raise FileNotFoundError("Could not find a valid detector .joblib file.")
        
    return detector_model, feature_names, threshold

# ==============================================================================
# 5. SINGLE-SAMPLE PIPELINE EXECUTION
# ==============================================================================
def run_single_audio_diagnostic():
    if not os.path.exists(TEST_AUDIO_PATH):
        print(f"[!] Error: Test audio file not found at {TEST_AUDIO_PATH}")
        return

    print("\n[+] Loading Base Models (Whisper, MiniLM, Gemma)...")
    whisper_model = whisper.load_model(WHISPER_PATH, device=DEVICE)
    minilm_model = SentenceTransformer(MINILM_PATH, device=DEVICE)
    
    gemma_tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, local_files_only=True, use_fast=False)
    gemma_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    gemma_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_PATH, local_files_only=True, torch_dtype=gemma_dtype, device_map="auto"
    )

    # Load NLU Pipelines
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
    
    detector_b, feature_schema, threshold_b = load_detector()

    print(f"\n[+] Executing Inference on: {os.path.basename(TEST_AUDIO_PATH)}")
    
    # 1. Whisper Acoustic (Encoder)
    audio = whisper.load_audio(str(TEST_AUDIO_PATH))
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(DEVICE)
    
    with torch.no_grad():
        enc_out = whisper_model.encoder(mel.unsqueeze(0))
        emb_512 = enc_out.mean(dim=1).cpu().numpy().astype(np.float32)

    # 2. Voice NLU Path
    v_scaled = v_scaler.transform(emb_512).astype(np.float32)
    with torch.no_grad():
        v_128 = v_proj(torch.tensor(v_scaled, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    v_preds, v_probs = get_nlu_preds(v_128, v_mlps, shared_encoders)

    # 3. Whisper Decoder (Language explicitly set to English)
    decode_options = whisper.DecodingOptions(fp16=(DEVICE == "cuda"), temperature=0.0, language="en")
    with torch.no_grad():
        dec_res = whisper.decode(whisper_model, mel, decode_options)
        decoded_transcript = dec_res.text.strip()

    # 4. Text NLU Path
    emb_384 = minilm_model.encode([decoded_transcript], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
    t_scaled = t_scaler.transform(emb_384).astype(np.float32)
    with torch.no_grad():
        t_128 = t_proj(torch.tensor(t_scaled, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    t_preds, t_probs = get_nlu_preds(t_128, t_mlps, shared_encoders)

    # 5. Detector B Feature Alignment & Prediction
    feat_a, feat_b = extract_detector_features(v_preds, t_preds, v_probs, t_probs, v_mlps, t_mlps, shared_encoders)
    
    if feature_schema is not None:
        # Reorder columns to exactly match training order
        df_feat = pd.DataFrame([feat_b])
        missing = [f for f in feature_schema if f not in df_feat.columns]
        if missing:
            print(f"[!] Warning: Missing features for schema: {missing}")
            for m in missing: df_feat[m] = 0.0
        detector_input = df_feat[feature_schema].values
    else:
        detector_input = pd.DataFrame([feat_b]).values

    prob_b = float(detector_b.predict_proba(detector_input)[0, 1])
    is_suspicious = prob_b >= threshold_b

    # ==============================================================================
    # 6. DETAILED OUTPUT DISPLAY
    # ==============================================================================
    print("\n" + "="*70)
    print("TRANSCRIPT & NLU POSTERIOR PROBABILITY BREAKDOWN")
    print("="*70)
    print(f"Whisper Decoded Transcript:\n\"{decoded_transcript}\"")
    
    print("\n--- Semantic Head Posteriors & Disagreements ---")
    for h in HEADS:
        match_status = "MATCH" if v_preds[h] == t_preds[h] else "MISMATCH"
        v_p = v_probs[h]
        t_p = t_probs[h]
        v_cls = shared_encoders[f"{h}_label"].inverse_transform(v_mlps[h].classes_)
        t_cls = shared_encoders[f"{h}_label"].inverse_transform(t_mlps[h].classes_)
        
        v_top_idx = np.argsort(v_p)[::-1][:3]
        t_top_idx = np.argsort(t_p)[::-1][:3]
        
        print(f"\n[{match_status}] Head: {h.upper()}")
        print(f"  * Voice NLU Prediction : {v_preds[h]:<25} (Top-1 Conf: {np.max(v_p):.4f}, Entropy: {calc_entropy(v_p):.3f})")
        print(f"    Top-3 Distribution   : {', '.join([f'{v_cls[i]}: {v_p[i]:.3f}' for i in v_top_idx])}")
        print(f"  * Text NLU Prediction  : {t_preds[h]:<25} (Top-1 Conf: {np.max(t_p):.4f}, Entropy: {calc_entropy(t_p):.3f})")
        print(f"    Top-3 Distribution   : {', '.join([f'{t_cls[i]}: {t_p[i]:.3f}' for i in t_top_idx])}")
        print(f"  * Cross-modal Divergence: JS Divergence = {feat_b[f'{h}_js_divergence']:.4f}")

    print("\n" + "="*70)
    print("DETECTOR B EVALUATION")
    print("="*70)
    print(f"Detector B Output Probability : {prob_b:.4f}")
    print(f"Operational Decision Threshold: {threshold_b:.4f}")
    if is_suspicious:
        print("[!] VERDICT: ERROR / SUSPICIOUS (Cross-modal divergence exceeds threshold)")
    else:
        print("[✓] VERDICT: NORMAL / RELIABLE (Transcript aligns with acoustic evidence)")

    # ==============================================================================
    # 7. LLM REASONING & INTENT RECOVERY
    # ==============================================================================
    print("\n" + "="*70)
    print("GEMMA REASONING & USER INTENT DIAGNOSIS")
    print("="*70)
    
    if is_suspicious:
        prompt = f"""You are an expert AI assistant analyzing a spoken audio transcript that has been flagged as potentially containing an ASR (speech recognition) error.

Decoded Transcript:
"{decoded_transcript}"

Independent Acoustic Evidence (extracted directly from speech audio):
- Domain: {v_preds['domain']} (Confidence: {np.max(v_probs['domain']):.2f})
- Subdomain: {v_preds['subdomain']} (Confidence: {np.max(v_probs['subdomain']):.2f})
- Topic: {v_preds['topic']} (Confidence: {np.max(v_probs['topic']):.2f})
- Document Type: {v_preds['document_type']} (Confidence: {np.max(v_probs['document_type']):.2f})

Please answer the following three points concisely:
1. What does the raw transcript literally say?
2. What acoustic/semantic discrepancy was detected, and what was the user's likely intended request?
3. Provide the corrected response addressing the user's true intent."""
    else:
        prompt = f"""You are an expert AI assistant analyzing a user's spoken request.

Decoded Transcript:
"{decoded_transcript}"

Acoustic Verification Status: Verified reliable by speech semantic model.

Please provide a concise, direct response addressing the user's statement or question."""

    inputs = gemma_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = gemma_model.generate(
            **inputs, 
            max_new_tokens=150, 
            do_sample=False, 
            temperature=None
        )
    gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    answer = gemma_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()
    
    print(answer)
    print("="*70 + "\n")

if __name__ == "__main__":
    run_single_audio_diagnostic()
