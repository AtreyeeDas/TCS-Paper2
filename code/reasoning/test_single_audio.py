import os
import time
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
# CHANGE THIS TO YOUR SPECIFIC TEST FILE (.wav, .flac, or .mp3)
TEST_AUDIO_PATH = "/home/spark2/users/intern/Atreyee-Das/NLU_Robust_Experiment/audio/your_test_file.flac" 

PROJECT_ROOT = Path("/home/spark2/users/intern/Atreyee-Das/NLU_Robust_Experiment")
MODELS_DIR = PROJECT_ROOT / "models"
DETECTOR_DIR = PROJECT_ROOT / "detector"

WHISPER_PATH = "/home/spark2/Models/base.en.pt"
MINILM_PATH = "/home/spark2/Models/all-MiniLM-L6-v2"
GEMMA_PATH = "/home/spark2/Models/gemma_2_models/gemma-3-1b-it"

HEADS = ["domain", "subdomain", "topic", "document_type"]
HEAD_WEIGHTS = {"domain": 0.20, "subdomain": 0.25, "topic": 0.40, "document_type": 0.15}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-12

# ==============================================================================
# 2. ARCHITECTURE & DETECTOR FEATURE FUNCTIONS[span_6](start_span)[span_6](end_span)
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

def entropy(p):
    p = np.clip(p, EPS, 1.0)
    return float(-np.sum((p / p.sum()) * np.log(p / p.sum())))

def margin(p):
    s = np.sort(p)[::-1]
    return float(s[0] - s[1]) if len(p) >= 2 else 1.0

def aligned_js(vp, vc, tp, tc):
    classes = sorted(set(vc) | set(tc))
    vmap, tmap = {c: p for c, p in zip(vc, vp)}, {c: p for c, p in zip(tc, tp)}
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
        fb[f"{short}_voice_entropy"] = entropy(vp)
        fb[f"{short}_text_entropy"] = entropy(tp)
        fb[f"{short}_voice_margin"] = margin(vp)
        fb[f"{short}_text_margin"] = margin(tp)

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
# 3. SINGLE INFERENCE SCRIPT
# ==============================================================================
def run_single_inference():
    if not os.path.exists(TEST_AUDIO_PATH):
        print(f"[!] Error: Test audio file not found at {TEST_AUDIO_PATH}")
        return

    print("\n[+] Loading Models (this takes a moment)...")
    whisper_model = whisper.load_model(WHISPER_PATH, device=DEVICE)
    minilm_model = SentenceTransformer(MINILM_PATH, device=DEVICE)
    
    # Gemma loaded with use_fast=False to fix Rust parsing crash[span_7](start_span)[span_7](end_span)
    gemma_tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, local_files_only=True, use_fast=False) 
    gemma_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    gemma_model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, local_files_only=True, torch_dtype=gemma_dtype, device_map="auto")

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
    
    detector_A = joblib.load(DETECTOR_DIR / "Detector_A_STRICT_ASR_INDUCED.joblib")
    detector_B = joblib.load(DETECTOR_DIR / "Detector_B_STRICT_ASR_INDUCED.joblib")

    import json
    with open(DETECTOR_DIR / "strict_detector_thresholds.json", "r") as f:
        thresholds = json.load(f)
        thresh_B = float(thresholds["threshold_B"])

    print(f"\n[+] Processing Audio: {os.path.basename(TEST_AUDIO_PATH)}")
    
    # 1. Whisper Acoustic
    audio = whisper.load_audio(str(TEST_AUDIO_PATH))
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(DEVICE)
    
    with torch.no_grad():
        enc_out = whisper_model.encoder(mel.unsqueeze(0))
        emb_512 = enc_out.mean(dim=1).cpu().numpy().astype(np.float32)

    # 2. Voice NLU
    v_scaled = v_scaler.transform(emb_512).astype(np.float32)
    with torch.no_grad():
        v_128 = v_proj(torch.tensor(v_scaled, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    v_preds, v_probs = get_nlu_preds(v_128, v_mlps, shared_encoders)

    # 3. Whisper Decoder[span_8](start_span)[span_8](end_span)[span_9](start_span)[span_9](end_span)
    decode_options = whisper.DecodingOptions(fp16=(DEVICE == "cuda"), temperature=0.0, language="en")
    with torch.no_grad():
        dec_res = whisper.decode(whisper_model, mel, decode_options)
        decoded_transcript = dec_res.text.strip()

    # 4. Text NLU
    emb_384 = minilm_model.encode([decoded_transcript], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
    t_scaled = t_scaler.transform(emb_384).astype(np.float32)
    with torch.no_grad():
        t_128 = t_proj(torch.tensor(t_scaled, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    t_preds, t_probs = get_nlu_preds(t_128, t_mlps, shared_encoders)

    # 5. Extract Features & Detect Error[span_10](start_span)[span_10](end_span)
    feat_a, feat_b = extract_detector_features(v_preds, t_preds, v_probs, t_probs, v_mlps, t_mlps, shared_encoders)
    df_b = pd.DataFrame([feat_b])
    prob_b = float(detector_B.predict_proba(df_b.values)[0, 1])
    is_suspicious = prob_b >= thresh_B

    print("\n" + "="*50)
    print("PIPELINE RESULTS")
    print("="*50)
    print(f"Whisper Transcript:  \"{decoded_transcript}\"")
    print("\n--- NLU Disagreements ---")
    for h in HEADS:
        match_status = "MATCH" if v_preds[h] == t_preds[h] else "MISMATCH"
        print(f"[{match_status}] {h.upper()}:")
        print(f"     Voice NLU -> {v_preds[h]}")
        print(f"     Text NLU  -> {t_preds[h]}")

    print("\n--- Error Detection ---")
    print(f"Detector B Probability:  {prob_b:.4f} (Threshold: {thresh_B:.4f})")
    if is_suspicious:
        print("[!] DETECTOR FLAG: SUSPICIOUS (Routing to Gemma Correction)")
    else:
        print("[✓] DETECTOR FLAG: NORMAL (Routing to Baseline Gemma)")

    # 6. Gemma Reasoning
    print("\n--- Gemma Response ---")
    if is_suspicious:
        prompt = f"""You are answering a user's request represented by the speech transcript below.
The transcript has been flagged as potentially containing an ASR-induced semantic error.

Transcript:
{decoded_transcript}

Independent acoustic semantic evidence:
Domain: {v_preds['domain']}
Topic: {v_preds['topic']}

Use the acoustic semantic evidence to resolve ambiguity only when it provides supporting evidence.
Do not invent facts. Answer briefly."""
    else:
        prompt = f"""You are answering a user's request represented by the speech transcript below.

Transcript:
{decoded_transcript}

Answer briefly and only using information supported by the transcript.
Do not invent facts."""

    inputs = gemma_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = gemma_model.generate(**inputs, max_new_tokens=30, do_sample=False, temperature=None)
    gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    answer = gemma_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()
    
    print(f"Gemma Answer: {answer}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_single_inference()
