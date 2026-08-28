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
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.nn.functional as F

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
EXP_DIR = ROOT_DIR / "corrupted_audio_voice_text_diagnostic"
for sub in ["audio", "embeddings", "results", "posterior_npz", "logs", "figures"]:
    (EXP_DIR / sub).mkdir(parents=True, exist_ok=True)

HEADS = ["domain", "subdomain", "topic", "document_type"]
SNRS = [15, 12, 9, 6, 3, 0]

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
    
    # Voice-NLU
    assert (VOICE_MODELS_DIR / "best_hierarchical_projection.pt").exists(), "Missing Voice Projection"
    assert (VOICE_MODELS_DIR / "whisper_scaler.joblib").exists(), "Missing Voice Scaler"
    for head in HEADS:
        assert (VOICE_MODELS_DIR / f"{head}_mlp.joblib").exists(), f"Missing Voice {head} MLP"
    assert (VOICE_MODELS_DIR / "label_encoders.joblib").exists(), "Missing Voice Encoders"
    
    # Text-NLU
    assert (TEXT_MODELS_DIR / "best_text_hierarchical_projection.pt").exists(), "Missing Text Projection"
    assert (TEXT_MODELS_DIR / "text_scaler.joblib").exists(), "Missing Text Scaler"
    for head in HEADS:
        assert (TEXT_MODELS_DIR / f"text_{head}_mlp.joblib").exists(), f"Missing Text {head} MLP"
    print("[✓] All project files and NLU artifacts verified.")

# ==========================================
# STEP 3: LOCALIZED ACOUSTIC CORRUPTION
# ==========================================
def inject_localized_noise(audio, sr, start_sec, end_sec, snr_db):
    start_samp = int(start_sec * sr)
    end_samp = int(end_sec * sr)
    
    start_samp = max(0, start_samp)
    end_samp = min(len(audio), end_samp)
    
    if start_samp >= end_samp:
        return audio
        
    target_segment = audio[start_samp:end_samp]
    
    sig_rms = np.sqrt(np.mean(target_segment**2))
    if sig_rms == 0:
        sig_rms = 1e-5
        
    noise = np.random.randn(len(target_segment))
    noise_rms = np.sqrt(np.mean(noise**2))
    
    snr_linear = 10 ** (snr_db / 20)
    desired_noise_rms = sig_rms / snr_linear
    noise = noise * (desired_noise_rms / noise_rms)
    
    fade_len = min(int(0.01 * sr), len(noise)//2)
    if fade_len > 0:
        window = np.hanning(fade_len * 2)
        noise[:fade_len] *= window[:fade_len]
        noise[-fade_len:] *= window[-fade_len:]
        
    corrupted_audio = audio.copy()
    corrupted_audio[start_samp:end_samp] += noise
    return corrupted_audio

# ==========================================
# STEPS 9 & 10: METRICS & INSTABILITY
# ==========================================
def calc_entropy(probs):
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log2(p))

def calc_instability(probs):
    k = len(probs)
    h_norm = calc_entropy(probs) / np.log2(k) if k > 1 else 0
    sorted_p = np.sort(probs)[::-1]
    margin = sorted_p[0] - sorted_p[1] if k > 1 else 1.0
    return 0.5 * h_norm + 0.5 * (1 - margin)

# ==========================================
# MAIN EXPERIMENT PIPELINE
# ==========================================
def run_diagnostic():
    audit_artifacts()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    df = pd.read_csv(DATASET_CSV)
    df = df[(df['split'] == 'unseen') & (df['targets_corrupted'] > 0)].copy()
    if len(df) > 500: df = df.sample(500, random_state=42)
    
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
    
    results, voice_posteriors_list, text_posteriors_list = [], [], []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Samples"):
        sample_id = str(row['sample_id'])
        audio_path = AUDIO_DIR / f"{sample_id}.wav"
        if not audio_path.exists(): continue
        
        target_term = str(row.get('target_terms', '')).lower()
        if not target_term: continue
        
        audio, sr = librosa.load(audio_path, sr=16000)
        
        start_sec, end_sec = None, None
        try:
            clean_res = whisper_model.transcribe(audio_path.as_posix(), word_timestamps=True)
            for segment in clean_res.get('segments', []):
                for word in segment.get('words', []):
                    if target_term in word.get('word', '').lower():
                        start_sec, end_sec = word['start'], word['end']
                        break
                if start_sec: break
        except TypeError:
            clean_res = whisper_model.transcribe(audio_path.as_posix())
            for segment in clean_res.get('segments', []):
                seg_text = segment.get('text', '').lower()
                if target_term in seg_text:
                    idx_term = seg_text.find(target_term)
                    char_ratio_start = idx_term / max(1, len(seg_text))
                    char_ratio_end = (idx_term + len(target_term)) / max(1, len(seg_text))
                    
                    seg_dur = segment['end'] - segment['start']
                    start_sec = segment['start'] + (seg_dur * char_ratio_start)
                    end_sec = segment['start'] + (seg_dur * char_ratio_end)
                    
                    start_sec = max(0, start_sec - 0.2)
                    end_sec = end_sec + 0.2
                    break
            
        if not start_sec: continue
        
        selected_snr, corr_audio, corr_emb, corr_text, v_preds, t_preds, category = None, None, None, None, {}, {}, "OTHER"
        
        for snr in SNRS:
            test_audio = inject_localized_noise(audio, sr, start_sec, end_sec, snr)
            test_audio_32 = test_audio.astype(np.float32)
            
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(test_audio_32)).to(device)
            with torch.no_grad():
                # 512-D Corrupted Embedding
                enc_out = whisper_model.encoder(mel.unsqueeze(0))
                emb_512 = enc_out.mean(dim=1).cpu().numpy().astype(np.float32)
                
                # Decoder
                dec_res = whisper.decode(whisper_model, mel, whisper.DecodingOptions(fp16=False),language='en')
                corrupted_transcript = dec_res.text.strip()
            
            # STEP 6: Voice-NLU Inference (Fixed 128-D Projection)
            v_scaled = voice_scaler.transform(emb_512).astype(np.float32)
            with torch.no_grad():
                v_128 = voice_proj(torch.tensor(v_scaled, device=device)).cpu().numpy()
            
            v_probs, v_correct = {}, True
            for h in HEADS:
                probs = voice_mlps[h].predict_proba(v_128)[0]
                v_probs[h] = probs
                pred_label = voice_encoders[f"{h}_label"].inverse_transform([np.argmax(probs)])[0]
                if pred_label != str(row.get(f"{h}_label", row.get(h))): v_correct = False
                
            # STEP 8: Text-NLU Inference
            emb_384 = text_encoder.encode([corrupted_transcript], convert_to_numpy=True)
            t_scaled = text_scaler.transform(emb_384).astype(np.float32)
            with torch.no_grad():
                z_128 = text_proj(torch.tensor(t_scaled, device=device)).cpu().numpy()
            
            t_probs, t_correct = {}, True
            for h in HEADS:
                probs = text_mlps[h].predict_proba(z_128)[0]
                t_probs[h] = probs
                pred_label = text_encoders[f"{h}_label"].inverse_transform([np.argmax(probs)])[0]
                if pred_label != str(row.get(f"{h}_label", row.get(h))): t_correct = False
            
            asr_failed = target_term not in corrupted_transcript.lower()
            
            if asr_failed and v_correct and not t_correct:
                category = "DESIRED_ASYMMETRIC"
            elif asr_failed and v_correct and t_correct:
                category = "ASR_ERROR_BUT_NLU_ROBUST"
            elif asr_failed and not v_correct and not t_correct:
                category = "BOTH_DEGRADED"
            elif not v_correct and t_correct:
                category = "VOICE_ONLY_FAILURE"
            elif not asr_failed:
                category = "NO_MEANINGFUL_ASR_ERROR"
                
            if category == "DESIRED_ASYMMETRIC" or snr == 0:
                selected_snr = snr
                corr_audio = test_audio
                corr_emb = emb_512
                corr_text = corrupted_transcript
                v_preds, t_preds = v_probs, t_probs
                break

        audio_out = EXP_DIR / "audio" / f"{sample_id}_corrupted.wav"
        emb_out = EXP_DIR / "embeddings" / f"{sample_id}_corrupted_encoder.npy"
        sf.write(audio_out, corr_audio, sr)
        np.save(emb_out, corr_emb)
        
        row_res = {
            "sample_id": sample_id,
            "scenario_id": row.get('scenario_id', ''),
            "split": row.get('split', ''),
            "target_term": target_term,
            "selected_snr_db": selected_snr,
            "clean_transcript": row.get('reference_transcript', ''),
            "corrupted_whisper_transcript": corr_text,
            "corruption_category": category,
            "most_unstable_voice_head": max(HEADS, key=lambda h: calc_instability(v_preds[h])),
            "most_unstable_text_head": max(HEADS, key=lambda h: calc_instability(t_preds[h]))
        }
        
        for h in HEADS:
            row_res[f"{h}_label"] = str(row.get(f"{h}_label", row.get(h)))
            
            v_p = v_preds[h]
            v_top = np.argsort(v_p)[::-1]
            row_res[f"voice_{h}"] = voice_encoders[f"{h}_label"].inverse_transform([v_top[0]])[0]
            row_res[f"voice_{h}_prob"] = v_p[v_top[0]]
            row_res[f"voice_{h}_margin"] = v_p[v_top[0]] - (v_p[v_top[1]] if len(v_p)>1 else 0)
            row_res[f"voice_{h}_entropy"] = calc_entropy(v_p)
            row_res[f"voice_{h}_correct"] = row_res[f"voice_{h}"] == row_res[f"{h}_label"]
            
            t_p = t_preds[h]
            t_top = np.argsort(t_p)[::-1]
            row_res[f"text_{h}"] = text_encoders[f"{h}_label"].inverse_transform([t_top[0]])[0]
            row_res[f"text_{h}_prob"] = t_p[t_top[0]]
            row_res[f"text_{h}_margin"] = t_p[t_top[0]] - (t_p[t_top[1]] if len(t_p)>1 else 0)
            row_res[f"text_{h}_entropy"] = calc_entropy(t_p)
            row_res[f"text_{h}_correct"] = row_res[f"text_{h}"] == row_res[f"{h}_label"]
            
            v_align = np.pad(v_p, (0, max(0, len(t_p)-len(v_p))))[:len(t_p)]
            row_res[f"js_{h}"] = jensenshannon(v_align, t_p)**2
            row_res[f"l1_{h}"] = np.sum(np.abs(v_align - t_p))
            
        results.append(row_res)
        voice_posteriors_list.append({h: v_preds[h] for h in HEADS})
        text_posteriors_list.append({h: t_preds[h] for h in HEADS})

    df_res = pd.DataFrame(results)
    df_res.to_csv(EXP_DIR / "results" / "corrupted_audio_diagnostic_results.csv", index=False)
    np.savez_compressed(EXP_DIR / "posterior_npz" / "voice_posteriors.npz", **{str(i): v for i, v in enumerate(voice_posteriors_list)})
    np.savez_compressed(EXP_DIR / "posterior_npz" / "text_posteriors.npz", **{str(i): t for i, t in enumerate(text_posteriors_list)})
    
    cat_sum = df_res['corruption_category'].value_counts(normalize=True).mul(100).round(2).reset_index()
    cat_sum.columns = ['Category', 'Percentage']
    cat_sum.to_csv(EXP_DIR / "results" / "corruption_category_summary.csv", index=False)
    
    dom_sum = df_res.groupby('domain_label')['corruption_category'].apply(lambda x: (x == 'DESIRED_ASYMMETRIC').mean() * 100).reset_index()
    dom_sum.columns = ['Domain', 'Desired_Asymmetric_Rate']
    dom_sum.to_csv(EXP_DIR / "results" / "domain_summary.csv", index=False)
    
    sns.set_theme(style="whitegrid")
    
    v_acc = [df_res[f"voice_{h}_correct"].mean() for h in HEADS]
    t_acc = [df_res[f"text_{h}_correct"].mean() for h in HEADS]
    
    x = np.arange(len(HEADS))
    plt.figure(figsize=(10,6))
    plt.bar(x - 0.2, v_acc, 0.4, label='Voice-NLU')
    plt.bar(x + 0.2, t_acc, 0.4, label='Text-NLU')
    plt.xticks(x, HEADS)
    plt.ylabel('Accuracy')
    plt.title('Voice vs Text NLU Accuracy on Corrupted Audio')
    plt.legend()
    plt.savefig(EXP_DIR / "figures" / "voice_vs_text_accuracy.png")
    plt.close()
    
    plt.figure(figsize=(8,8))
    for cat, color in zip(["DESIRED_ASYMMETRIC", "BOTH_DEGRADED", "ASR_ERROR_BUT_NLU_ROBUST"], ['green', 'red', 'gray']):
        subset = df_res[df_res['corruption_category'] == cat]
        plt.scatter(subset['voice_topic_prob'], subset['text_topic_prob'], label=cat, alpha=0.6, c=color)
    plt.xlabel('Voice-NLU Topic Confidence')
    plt.ylabel('Text-NLU Topic Confidence')
    plt.title('Confidence Displacement Scatter')
    plt.legend()
    plt.savefig(EXP_DIR / "figures" / "confidence_scatter.png")
    plt.close()
    
    print("\n[✓] Diagnostic completed successfully. All artifacts saved in:", EXP_DIR)

if __name__ == "__main__":
    run_diagnostic()
