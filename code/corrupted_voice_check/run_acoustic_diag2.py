import os
import json
import warnings
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import whisper
import joblib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon, cosine
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.nn.functional as F
import re

warnings.filterwarnings("ignore")

# ==========================================
# STEP 0: CONFIGURATION & PATH AUDIT
# ==========================================
ROOT_DIR = Path("/home/spark2/users/intern/Atreyee-Das/NLU_Robust_Experiment")
DATASET_CSV = ROOT_DIR / "dataset" / "whisper_domain_multitarget_6000.csv"
AUDIO_DIR = ROOT_DIR / "audio"

# Artifacts
WHISPER_MODEL_PATH = "/home/spark2/Models/base.en.pt"
VOICE_MODELS_DIR = Path("/home/spark2/users/intern/Atreyee-Das/NLU_Robust_Experiment/audio_nlu_models")
TEXT_MODELS_DIR = Path("/home/spark2/users/intern/Atreyee-Das/NLU_Robust_Experiment/text_nlu_models")
TEXT_ENCODER_PATH = "/home/spark2/Models/all-MiniLM-L6-v2"

# Outputs
EXP_DIR = ROOT_DIR / "acoustic_final_diagnostic"
for sub in ["audio", "embeddings", "embeddings/clean", "results", "posterior_npz", "logs", "figures"]:
    (EXP_DIR / sub).mkdir(parents=True, exist_ok=True)

HEADS = ["domain", "subdomain", "topic", "document_type"]

# Perturbation Grid
SNRS = [30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0]
COVERAGES = [0.25, 0.40, 0.50, 0.70, 1.00]
EXTENSIONS_MS = [0, 50, 100, 200]

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
        z = self.projector(x)
        return F.normalize(z, p=2, dim=1)

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
        z = self.projector(x)
        return F.normalize(z, p=2, dim=1)

def audit_artifacts():
    print("="*50 + "\nSTEP 0: ARTIFACT AUDIT\n" + "="*50)
    assert DATASET_CSV.exists(), f"Missing {DATASET_CSV}"
    assert AUDIO_DIR.exists(), f"Missing {AUDIO_DIR}"
    assert os.path.exists(WHISPER_MODEL_PATH), f"Missing local Whisper model at {WHISPER_MODEL_PATH}"
    
    assert (VOICE_MODELS_DIR / "best_hierarchical_projection.pt").exists(), "Missing Voice Projection"
    assert (VOICE_MODELS_DIR / "whisper_scaler.joblib").exists(), "Missing Voice Scaler"
    for head in HEADS:
        assert (VOICE_MODELS_DIR / f"{head}_mlp.joblib").exists(), f"Missing Voice {head} MLP"
    assert (VOICE_MODELS_DIR / "label_encoders.joblib").exists(), "Missing Voice Encoders"
    
    assert (TEXT_MODELS_DIR / "best_text_hierarchical_projection.pt").exists(), "Missing Text Projection"
    assert (TEXT_MODELS_DIR / "text_scaler.joblib").exists(), "Missing Text Scaler"
    for head in HEADS:
        assert (TEXT_MODELS_DIR / f"text_{head}_mlp.joblib").exists(), f"Missing Text {head} MLP"
    print("[✓] All project files and NLU artifacts verified.")

# ==========================================
# STEP 3 & 5: LOCALIZED ACOUSTIC CORRUPTION
# ==========================================
def inject_localized_noise(audio, sr, start_sec, end_sec, snr_db, coverage, extension_ms):
    # Apply coverage first (center around the midpoint of the target)
    dur = end_sec - start_sec
    midpoint = start_sec + (dur / 2.0)
    new_dur = dur * coverage
    c_start = midpoint - (new_dur / 2.0)
    c_end = midpoint + (new_dur / 2.0)
    
    # Apply temporal extension
    ext_sec = extension_ms / 1000.0
    c_start = max(0, c_start - ext_sec)
    c_end = min(len(audio)/sr, c_end + ext_sec)
    
    start_samp = int(c_start * sr)
    end_samp = int(c_end * sr)
    
    if start_samp >= end_samp or start_samp >= len(audio):
        return audio, float('inf')
        
    target_segment = audio[start_samp:end_samp]
    sig_rms = np.sqrt(np.mean(target_segment**2))
    if sig_rms == 0:
        sig_rms = 1e-5
        
    # White Gaussian noise
    noise = np.random.randn(len(target_segment))
    noise_rms = np.sqrt(np.mean(noise**2))
    
    snr_linear = 10 ** (snr_db / 20)
    desired_noise_rms = sig_rms / snr_linear
    noise = noise * (desired_noise_rms / noise_rms)
    
    # Smooth fade-in/out (10ms)
    fade_len = min(int(0.01 * sr), len(noise)//2)
    if fade_len > 0:
        window = np.hanning(fade_len * 2)
        noise[:fade_len] *= window[:fade_len]
        noise[-fade_len:] *= window[-fade_len:]
        
    corrupted_audio = audio.copy()
    corrupted_audio[start_samp:end_samp] += noise
    
    # Calc actual SNR for verification
    actual_snr = 20 * np.log10(sig_rms / (np.sqrt(np.mean(noise**2)) + 1e-12))
    
    return corrupted_audio, actual_snr

# ==========================================
# METRICS & INSTABILITY
# ==========================================
def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[\'"]', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return " ".join(text.split())

def calc_entropy(probs):
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log2(p))

def calc_instability(probs):
    k = len(probs)
    h_norm = calc_entropy(probs) / np.log2(k) if k > 1 else 0
    sorted_p = np.sort(probs)[::-1]
    margin = sorted_p[0] - sorted_p[1] if k > 1 else 1.0
    return 0.5 * h_norm + 0.5 * (1 - margin)

def align_posteriors(v_probs, t_probs, v_enc, t_enc, head):
    v_classes = v_enc[f"{head}_label"].classes_
    t_classes = t_enc[f"{head}_label"].classes_
    union_classes = sorted(list(set(v_classes) | set(t_classes)))
    
    v_aligned = np.zeros(len(union_classes))
    t_aligned = np.zeros(len(union_classes))
    
    for i, cls in enumerate(union_classes):
        if cls in v_classes:
            v_idx = np.where(v_classes == cls)[0][0]
            v_aligned[i] = v_probs[v_idx]
        if cls in t_classes:
            t_idx = np.where(t_classes == cls)[0][0]
            t_aligned[i] = t_probs[t_idx]
            
    v_aligned = np.clip(v_aligned, 1e-12, 1.0)
    v_aligned /= v_aligned.sum()
    t_aligned = np.clip(t_aligned, 1e-12, 1.0)
    t_aligned /= t_aligned.sum()
    
    return v_aligned, t_aligned

# ==========================================
# MAIN EXPERIMENT PIPELINE
# ==========================================
def run_diagnostic():
    audit_artifacts()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load Data
    df = pd.read_csv(DATASET_CSV)
    df = df[df['split'] == 'unseen'].copy()
    
    print(f"\n[+] Loading Whisper from local checkpoint: {WHISPER_MODEL_PATH}...")
    whisper_model = whisper.load_model(WHISPER_MODEL_PATH, device=device)
    
    # Load Voice NLUs
    voice_encoders = joblib.load(VOICE_MODELS_DIR / "label_encoders.joblib")
    voice_scaler = joblib.load(VOICE_MODELS_DIR / "whisper_scaler.joblib")
    voice_proj = VoiceHierarchicalProjection(512, 128).to(device)
    voice_proj.load_state_dict(torch.load(VOICE_MODELS_DIR / "best_hierarchical_projection.pt", map_location=device, weights_only=True))
    voice_proj.eval()
    voice_mlps = {h: joblib.load(VOICE_MODELS_DIR / f"{h}_mlp.joblib") for h in HEADS}
    
    # Load Text NLUs
    text_encoder = SentenceTransformer(TEXT_ENCODER_PATH, device=device)
    text_scaler = joblib.load(TEXT_MODELS_DIR / "text_scaler.joblib")
    text_proj = TextHierarchicalProjection(384, 128).to(device)
    text_proj.load_state_dict(torch.load(TEXT_MODELS_DIR / "best_text_hierarchical_projection.pt", map_location=device, weights_only=True))
    text_proj.eval()
    text_encoders = joblib.load(TEXT_MODELS_DIR / "text_label_encoders.joblib")
    text_mlps = {h: joblib.load(TEXT_MODELS_DIR / f"text_{h}_mlp.joblib") for h in HEADS}
    
    if os.path.exists(EXP_DIR / "results" / "all_perturbations.csv"):
        print("[+] Checkpoint found. To restart, delete acoustic_final_diagnostic/.")
        return

    results, voice_posteriors_dict, text_posteriors_dict = [], {}, {}
    
    # Subsample grid for practicality in standard run:
    # Full grid is 11 x 5 x 4 = 220 conditions per sample. 
    # For testing, we subset if needed, but per prompt we evaluate all.
    active_snrs = SNRS
    active_covs = COVERAGES
    active_exts = EXTENSIONS_MS
    
    # Clean Reference Cache
    clean_cache = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Samples"):
        sample_id = str(row['sample_id'])
        audio_path = AUDIO_DIR / f"{sample_id}.wav"
        if not audio_path.exists(): continue
        
        target_term = str(row.get('target_terms', ''))
        if not target_term or str(target_term).lower() == 'nan': continue
        
        norm_target = normalize_text(target_term)
        target_tokens = norm_target.split()
        if len(target_tokens) == 0: continue
        
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Determine Timestamps (Fallback included)
        start_sec, end_sec = None, None
        try:
            clean_res = whisper_model.transcribe(audio_path.as_posix(), word_timestamps=True, language="en")
            for segment in clean_res.get('segments', []):
                for word in segment.get('words', []):
                    if target_tokens[0] in normalize_text(word.get('word', '')):
                        start_sec, end_sec = word['start'], word['end']
                        break
                if start_sec: break
        except TypeError:
            clean_res = whisper_model.transcribe(audio_path.as_posix(), language="en")
            for segment in clean_res.get('segments', []):
                seg_text = normalize_text(segment.get('text', ''))
                if norm_target in seg_text:
                    idx_term = seg_text.find(norm_target)
                    char_ratio_start = idx_term / max(1, len(seg_text))
                    char_ratio_end = (idx_term + len(norm_target)) / max(1, len(seg_text))
                    
                    seg_dur = segment['end'] - segment['start']
                    start_sec = segment['start'] + (seg_dur * char_ratio_start)
                    end_sec = segment['start'] + (seg_dur * char_ratio_end)
                    break
                    
        if not start_sec: continue
        
        # Clean Cache Computation (Rule 6)
        clean_audio_32 = audio.astype(np.float32)
        mel_clean = whisper.log_mel_spectrogram(whisper.pad_or_trim(clean_audio_32)).to(device)
        with torch.no_grad():
            clean_enc_out = whisper_model.encoder(mel_clean.unsqueeze(0))
            clean_512 = clean_enc_out.mean(dim=1).cpu().numpy().astype(np.float32)
            dec_clean_res = whisper.decode(whisper_model, mel_clean, whisper.DecodingOptions(language="en", fp16=False))
            clean_transcript = dec_clean_res.text.strip()
            
        clean_v_scaled = voice_scaler.transform(clean_512).astype(np.float32)
        with torch.no_grad():
            clean_128_v = voice_proj(torch.tensor(clean_v_scaled, device=device)).cpu().numpy()
            
        clean_cache[sample_id] = {
            '512': clean_512,
            '128_v': clean_128_v,
            'transcript': clean_transcript
        }
        
        np.save(EXP_DIR / "embeddings" / "clean" / f"{sample_id}.npy", clean_512)
        
        # Grid Perturbation
        for snr in active_snrs:
            for cov in active_covs:
                for ext in active_exts:
                    condition_str = f"SNR{snr}_C{cov}_E{ext}"
                    
                    test_audio, actual_snr = inject_localized_noise(audio, sr, start_sec, end_sec, snr, cov, ext)
                    test_audio_32 = test_audio.astype(np.float32)
                    
                    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(test_audio_32)).to(device)
                    with torch.no_grad():
                        enc_out = whisper_model.encoder(mel.unsqueeze(0))
                        emb_512 = enc_out.mean(dim=1).cpu().numpy().astype(np.float32)
                        dec_res = whisper.decode(whisper_model, mel, whisper.DecodingOptions(language="en", fp16=False))
                        corr_transcript = dec_res.text.strip()
                        
                    # Geometry distances
                    l2_512 = np.linalg.norm(clean_512 - emb_512)
                    cos_512 = cosine(clean_512.flatten(), emb_512.flatten())
                    rel_l2_512 = l2_512 / (np.linalg.norm(clean_512) + 1e-8)
                    
                    v_scaled = voice_scaler.transform(emb_512).astype(np.float32)
                    scaled_l2 = np.linalg.norm(clean_v_scaled - v_scaled)
                    
                    with torch.no_grad():
                        v_128 = voice_proj(torch.tensor(v_scaled, device=device)).cpu().numpy()
                        
                    l2_128 = np.linalg.norm(clean_128_v - v_128)
                    cos_128 = cosine(clean_128_v.flatten(), v_128.flatten())

                    # Inference Voice
                    v_preds = {}
                    for h in HEADS:
                        probs = voice_mlps[h].predict_proba(v_128)[0]
                        v_preds[h] = probs
                        
                    # Inference Text
                    emb_384 = text_encoder.encode([corr_transcript], convert_to_numpy=True)
                    t_scaled = text_scaler.transform(emb_384).astype(np.float32)
                    with torch.no_grad():
                        t_128 = text_proj(torch.tensor(t_scaled, device=device)).cpu().numpy()
                        
                    t_preds = {}
                    for h in HEADS:
                        probs = text_mlps[h].predict_proba(t_128)[0]
                        t_preds[h] = probs
                        
                    # Corruption Scoring
                    norm_corr = normalize_text(corr_transcript)
                    preserved = sum(1 for tok in target_tokens if tok in norm_corr)
                    target_was_corrupted = 1 if preserved < len(target_tokens) else 0
                    corruption_rate = 1.0 - (preserved / len(target_tokens))
                    
                    row_res = {
                        "sample_id": sample_id,
                        "scenario_id": row.get('scenario_id', ''),
                        "split": row.get('split', ''),
                        "target_term": target_term,
                        "snr_db": snr,
                        "coverage": cov,
                        "temporal_extension_ms": ext,
                        "actual_snr_db": actual_snr,
                        "clean_transcript": clean_transcript,
                        "corrupted_whisper_transcript": corr_transcript,
                        "target_token_count": len(target_tokens),
                        "target_tokens_preserved": preserved,
                        "target_corruption_rate": corruption_rate,
                        "exact_target_preserved": 1 if preserved == len(target_tokens) else 0,
                        "target_was_corrupted": target_was_corrupted,
                        "embedding_512_l2": l2_512,
                        "embedding_512_cosine": cos_512,
                        "embedding_512_relative_l2": rel_l2_512,
                        "scaled_embedding_l2": scaled_l2,
                        "embedding_128_l2": l2_128,
                        "embedding_128_cosine": cos_128
                    }
                    
                    asym_count = 0
                    for h in HEADS:
                        gt_label = str(row.get(f"{h}_label", row.get(h)))
                        row_res[f"{h}_label"] = gt_label
                        
                        v_p = v_preds[h]
                        v_top = np.argsort(v_p)[::-1]
                        v_pred_lbl = voice_encoders[f"{h}_label"].inverse_transform([v_top[0]])[0]
                        row_res[f"voice_{h}"] = v_pred_lbl
                        row_res[f"voice_{h}_prob"] = v_p[v_top[0]]
                        row_res[f"voice_{h}_margin"] = v_p[v_top[0]] - (v_p[v_top[1]] if len(v_p)>1 else 0)
                        row_res[f"voice_{h}_entropy"] = calc_entropy(v_p)
                        row_res[f"voice_{h}_normalized_entropy"] = calc_entropy(v_p) / (np.log2(len(v_p)) if len(v_p)>1 else 1)
                        row_res[f"voice_{h}_correct"] = int(v_pred_lbl == gt_label)
                        
                        t_p = t_preds[h]
                        t_top = np.argsort(t_p)[::-1]
                        t_pred_lbl = text_encoders[f"{h}_label"].inverse_transform([t_top[0]])[0]
                        row_res[f"text_{h}"] = t_pred_lbl
                        row_res[f"text_{h}_prob"] = t_p[t_top[0]]
                        row_res[f"text_{h}_margin"] = t_p[t_top[0]] - (t_p[t_top[1]] if len(t_p)>1 else 0)
                        row_res[f"text_{h}_entropy"] = calc_entropy(t_p)
                        row_res[f"text_{h}_normalized_entropy"] = calc_entropy(t_p) / (np.log2(len(t_p)) if len(t_p)>1 else 1)
                        row_res[f"text_{h}_correct"] = int(t_pred_lbl == gt_label)
                        
                        v_align, t_align = align_posteriors(v_p, t_p, voice_encoders, text_encoders, h)
                        row_res[f"js_{h}"] = jensenshannon(v_align, t_align)**2
                        row_res[f"aligned_l1_{h}"] = np.sum(np.abs(v_align - t_align))
                        row_res[f"aligned_l2_{h}"] = np.linalg.norm(v_align - t_align)
                        row_res[f"aligned_cosine_{h}"] = cosine(v_align, t_align)
                        
                        is_asym = int(row_res[f"voice_{h}_correct"] == 1 and (row_res[f"text_{h}_correct"] == 0 or row_res[f"text_{h}_prob"] < 0.60))
                        row_res[f"asymmetric_{h}"] = is_asym
                        asym_count += is_asym
                        
                    row_res["number_of_asymmetric_heads"] = asym_count
                    
                    # Category Logic
                    if target_was_corrupted:
                        if row_res["voice_topic_correct"] and (not row_res["text_topic_correct"] or row_res["text_topic_prob"] < 0.6):
                            category = "DESIRED_ASYMMETRIC"
                        elif row_res["voice_topic_correct"] and row_res["text_topic_correct"]:
                            category = "ASR_ERROR_BUT_NLU_ROBUST"
                        elif not row_res["voice_topic_correct"] and not row_res["text_topic_correct"]:
                            category = "BOTH_DEGRADED"
                        elif not row_res["voice_topic_correct"] and row_res["text_topic_correct"]:
                            category = "VOICE_ONLY_FAILURE"
                        else:
                            category = "OTHER"
                    else:
                        category = "NO_ASR_ERROR"
                        
                    row_res["corruption_category"] = category
                    row_res["most_unstable_voice_head"] = max(HEADS, key=lambda h: calc_instability(v_preds[h]))
                    row_res["most_unstable_text_head"] = max(HEADS, key=lambda h: calc_instability(t_preds[h]))
                    row_res["highest_voice_text_js_head"] = max(HEADS, key=lambda h: row_res[f"js_{h}"])
                    
                    results.append(row_res)
                    post_key = f"{sample_id}_{condition_str}"
                    voice_posteriors_dict[post_key] = v_preds
                    text_posteriors_dict[post_key] = t_preds
                    
                    if category == "DESIRED_ASYMMETRIC":
                        audio_out = EXP_DIR / "audio" / f"{post_key}.wav"
                        emb_out = EXP_DIR / "embeddings" / f"{post_key}.npy"
                        sf.write(audio_out, test_audio, sr)
                        np.save(emb_out, emb_512)

    df_res = pd.DataFrame(results)
    df_res.to_csv(EXP_DIR / "results" / "all_perturbations.csv", index=False)
    
    # Save partial posteriors safely to avoid giant arrays
    np.savez_compressed(EXP_DIR / "posterior_npz" / "voice_posteriors.npz", **{str(k): v for k, v in voice_posteriors_dict.items()})
    np.savez_compressed(EXP_DIR / "posterior_npz" / "text_posteriors.npz", **{str(k): v for k, v in text_posteriors_dict.items()})
    
    # Grid Aggregation
    grid_cols = ["snr_db", "coverage", "temporal_extension_ms"]
    grid_summary = df_res.groupby(grid_cols).agg(
        N=('sample_id', 'count'),
        target_ASR_error_rate=('target_was_corrupted', 'mean'),
        target_partial_corruption_rate=('target_corruption_rate', lambda x: np.mean((x > 0) & (x < 1))),
        target_complete_corruption_rate=('target_corruption_rate', lambda x: np.mean(x == 1)),
        voice_topic_accuracy=('voice_topic_correct', 'mean'),
        text_topic_accuracy=('text_topic_correct', 'mean'),
        desired_asymmetric_rate=('corruption_category', lambda x: (x == "DESIRED_ASYMMETRIC").mean()),
        both_degraded_rate=('corruption_category', lambda x: (x == "BOTH_DEGRADED").mean()),
        asr_error_nlu_robust_rate=('corruption_category', lambda x: (x == "ASR_ERROR_BUT_NLU_ROBUST").mean()),
        mean_js_topic=('js_topic', 'mean'),
        mean_512d_displacement=('embedding_512_l2', 'mean'),
        mean_128d_displacement=('embedding_128_l2', 'mean')
    ).reset_index()
    
    grid_summary.to_csv(EXP_DIR / "results" / "condition_summary.csv", index=False)
    
    # Best condition
    grid_summary['best_score'] = 0.4 * grid_summary['target_ASR_error_rate'] + 0.4 * grid_summary['voice_topic_accuracy'] + 0.2 * grid_summary['desired_asymmetric_rate']
    best_cond = grid_summary.sort_values(by='best_score', ascending=False).iloc[0]
    
    # Rep cases
    asym_df = df_res[df_res['corruption_category'] == 'DESIRED_ASYMMETRIC'].sort_values(by='js_topic', ascending=False).head(20)
    asym_df.to_csv(EXP_DIR / "results" / "representative_asymmetric_cases.csv", index=False)
    
    # Plots
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10,6))
    sns.lineplot(data=grid_summary, x='snr_db', y='target_ASR_error_rate', label='ASR Error')
    sns.lineplot(data=grid_summary, x='snr_db', y='voice_topic_accuracy', label='Voice-NLU Topic Acc')
    sns.lineplot(data=grid_summary, x='snr_db', y='desired_asymmetric_rate', label='Desired Asymmetric Rate')
    plt.gca().invert_xaxis()
    plt.title("Performance vs Perturbation SNR")
    plt.savefig(EXP_DIR / "figures" / "snr_trends.png")
    plt.close()
    
    # Decision Matrix
    go_no_go_path = EXP_DIR / "results" / "final_go_no_go.csv"
    
    opt_a_valid = (best_cond['target_ASR_error_rate'] >= 0.20) and (best_cond['voice_topic_accuracy'] >= 0.70) and (best_cond['desired_asymmetric_rate'] >= 0.10)
    opt_b_valid = (best_cond['target_ASR_error_rate'] >= 0.20) and (best_cond['voice_topic_accuracy'] < 0.70)
    
    if opt_a_valid:
        decision = "PROCEED_WITH_EXISTING_VOICE_NLU_AND_TRAIN_NEW_DETECTOR"
        exp = "Robust separation found: ASR decoder fails while Voice-NLU retains high semantic accuracy, producing reliable detection bounds."
    elif opt_b_valid:
        decision = "RETRAIN_ROBUST_VOICE_NLU_THEN_TRAIN_NEW_DETECTOR"
        exp = f"ASR failed robustly ({best_cond['target_ASR_error_rate']:.2f}) but Voice-NLU degraded ({best_cond['voice_topic_accuracy']:.2f}). Needs robust retraining."
    else:
        decision = "NO_SUPPORTED_ENCODER_DECODER_ASYMMETRY"
        exp = "The perturbation space does not yield a domain where the Voice-NLU is independently reliable when the ASR decoder fundamentally fails."

    pd.DataFrame({"Decision": [decision], "Rationale": [exp]}).to_csv(go_no_go_path, index=False)
    
    print("\n" + "="*60)
    print("FINAL ACOUSTIC DIAGNOSTIC")
    print("="*60)
    print(f"1. Unique utterances tested: {len(df)}")
    print(f"2. Perturbation conditions: {len(active_snrs)*len(active_covs)*len(active_exts)}")
    print(f"3. Target ASR error rate at best condition (SNR {best_cond['snr_db']}): {best_cond['target_ASR_error_rate']:.2f}")
    print(f"4. Desired Asymmetric Rate at best condition: {best_cond['desired_asymmetric_rate']:.2f}")
    print("\nFINAL SCIENTIFIC DECISION:")
    print(f">> {decision}")
    print(f">> {exp}")

if __name__ == "__main__":
    run_diagnostic()
