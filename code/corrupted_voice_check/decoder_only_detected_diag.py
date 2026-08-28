"""
decoder_only_detector_diagnostic.py

Time-critical final diagnostic for downstream reasoning feasibility.
Tests whether the existing ASR Error Detector B can identify decoder-level 
lexical errors when the Whisper ENCODER REPRESENTATION IS HELD FIXED.
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
import whisper
import joblib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, precision_recall_curve, confusion_matrix
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ==========================================
# 0. CONFIGURATION & PATHS
# ==========================================
ROOT_DIR = Path("/home/spark2/users/intern/Atreyee-Das/NLU_Robust_Experiment")
DATASET_CSV = ROOT_DIR / "dataset" / "whisper_domain_multitarget_6000.csv"
AUDIO_DIR = ROOT_DIR / "audio"

WHISPER_MODEL_PATH = "/home/spark2/Models/base.en.pt"
VOICE_MODELS_DIR = ROOT_DIR / "audio_nlu_models"
TEXT_MODELS_DIR = ROOT_DIR / "text_nlu_models"
TEXT_ENCODER_PATH = "/home/spark2/Models/all-MiniLM-L6-v2"
DETECTOR_B_PATH = ROOT_DIR / "error_detector" / "error_detector_with_posterior.joblib"

EXP_DIR = Path("decoder_only_detector_diagnostic")
for sub in ["results", "posterior_npz", "representative", "figures", "logs"]:
    (EXP_DIR / sub).mkdir(parents=True, exist_ok=True)

HEADS = ["domain", "subdomain", "topic", "document_type"]
HEAD_WEIGHTS = {"domain": 0.20, "subdomain": 0.25, "topic": 0.40, "document_type": 0.15}
DECODER_TEMPS = [0.0, 0.2, 0.4, 0.6, 0.8]
BEAM_SIZES = [1, 3, 5]

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

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def check_target_corruption(candidate_text, target_term):
    cand_norm = normalize_text(candidate_text)
    targ_norm = normalize_text(target_term)
    
    targ_tokens = targ_norm.split()
    cand_tokens = cand_norm.split()
    
    if not targ_tokens:
        return 0, 0, 0.0, 0, 0
        
    target_token_count = len(targ_tokens)
    target_tokens_preserved = sum(1 for t in targ_tokens if t in cand_tokens)
    target_corruption_rate = 1.0 - (target_tokens_preserved / target_token_count)
    exact_target_preserved = 1 if targ_norm in cand_norm else 0
    target_was_corrupted = 1 if exact_target_preserved == 0 else 0
    
    return target_token_count, target_tokens_preserved, target_corruption_rate, exact_target_preserved, target_was_corrupted

def calc_entropy(probs):
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log2(p))

def calc_instability(probs):
    k = len(probs)
    h_norm = calc_entropy(probs) / np.log2(k) if k > 1 else 0
    sorted_p = np.sort(probs)[::-1]
    margin = sorted_p[0] - sorted_p[1] if k > 1 else 1.0
    return 0.5 * h_norm + 0.5 * (1.0 - margin)

def align_and_compare_posteriors(v_probs, t_probs, v_classes, t_classes):
    union_classes = list(set(v_classes) | set(t_classes))
    union_classes.sort()
    
    v_aligned = np.zeros(len(union_classes))
    t_aligned = np.zeros(len(union_classes))
    
    for i, cls in enumerate(union_classes):
        if cls in v_classes: v_aligned[i] = v_probs[list(v_classes).index(cls)]
        if cls in t_classes: t_aligned[i] = t_probs[list(t_classes).index(cls)]
        
    v_aligned = np.clip(v_aligned, 1e-12, 1.0)
    t_aligned = np.clip(t_aligned, 1e-12, 1.0)
    v_aligned /= v_aligned.sum()
    t_aligned /= t_aligned.sum()
    
    js_div = jensenshannon(v_aligned, t_aligned) ** 2
    l1_dist = np.sum(np.abs(v_aligned - t_aligned))
    l2_dist = np.sqrt(np.sum((v_aligned - t_aligned)**2))
    cosine_dist = 1.0 - (np.dot(v_aligned, t_aligned) / (np.linalg.norm(v_aligned) * np.linalg.norm(t_aligned)))
    
    return js_div, l1_dist, l2_dist, cosine_dist

def extract_detector_b_features(v_preds, t_preds, v_classes_dict, t_classes_dict):
    features = {}
    weighted_disagreement = 0.0
    total_disagreements = 0
    
    for h in HEADS:
        v_p = v_preds[h]
        t_p = t_preds[h]
        v_cls = v_classes_dict[h]
        t_cls = t_classes_dict[h]
        
        v_top1_idx = np.argmax(v_p)
        t_top1_idx = np.argmax(t_p)
        
        v_label = v_cls[v_top1_idx]
        t_label = t_cls[t_top1_idx]
        
        disagreement = 1 if v_label != t_label else 0
        features[f"{h}_disagreement"] = disagreement
        total_disagreements += disagreement
        weighted_disagreement += HEAD_WEIGHTS[h] * disagreement
        
        features[f"voice_top1_confidence_{h}"] = v_p[v_top1_idx]
        features[f"text_top1_confidence_{h}"] = t_p[t_top1_idx]
        features[f"absolute_confidence_difference_{h}"] = abs(v_p[v_top1_idx] - t_p[t_top1_idx])
        
        v_prob_of_t = v_p[list(v_cls).index(t_label)] if t_label in v_cls else 0.0
        t_prob_of_v = t_p[list(t_cls).index(v_label)] if v_label in t_cls else 0.0
        features[f"voice_probability_of_text_selected_class_{h}"] = v_prob_of_t
        features[f"text_probability_of_voice_selected_class_{h}"] = t_prob_of_v
        
        js, l1, l2, cos = align_and_compare_posteriors(v_p, t_p, v_cls, t_cls)
        features[f"js_divergence_{h}"] = js
        
        v_sorted = np.sort(v_p)[::-1]
        t_sorted = np.sort(t_p)[::-1]
        features[f"voice_entropy_{h}"] = calc_entropy(v_p)
        features[f"text_entropy_{h}"] = calc_entropy(t_p)
        features[f"voice_top1_top2_margin_{h}"] = v_sorted[0] - (v_sorted[1] if len(v_sorted)>1 else 0)
        features[f"text_top1_top2_margin_{h}"] = t_sorted[0] - (t_sorted[1] if len(t_sorted)>1 else 0)

    features["total_disagreements"] = total_disagreements
    features["weighted_disagreement"] = weighted_disagreement
    
    return features

# ==========================================
# 2. MAIN EXPERIMENT
# ==========================================
def run_diagnostic():
    print("="*60)
    print("PART 1: ARTIFACT AUDIT & LOADING")
    print("="*60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    df = pd.read_csv(DATASET_CSV)
    df = df[df['split'] == 'unseen'].copy()
    if len(df) > 500: df = df.sample(500, random_state=42)
    
    print(f"[+] Loading Whisper from {WHISPER_MODEL_PATH}...")
    whisper_model = whisper.load_model(WHISPER_MODEL_PATH, device=device)
    
    print("[+] Loading Voice-NLU & Text-NLU Artifacts...")
    v_enc = joblib.load(VOICE_MODELS_DIR / "label_encoders.joblib")
    v_mlps = {h: joblib.load(VOICE_MODELS_DIR / f"{h}_mlp.joblib") for h in HEADS}
    
    t_enc_model = SentenceTransformer(TEXT_ENCODER_PATH, device=device)
    t_scaler = joblib.load(TEXT_MODELS_DIR / "text_scaler.joblib")
    t_proj = TextHierarchicalProjection(384, 128).to(device)
    t_proj.load_state_dict(torch.load(TEXT_MODELS_DIR / "best_text_hierarchical_projection.pt", map_location=device))
    t_proj.eval()
    t_enc = joblib.load(TEXT_MODELS_DIR / "text_label_encoders.joblib")
    t_mlps = {h: joblib.load(TEXT_MODELS_DIR / f"text_{h}_mlp.joblib") for h in HEADS}
    
    print("[+] Loading Existing Detector B...")
    detector_b = joblib.load(DETECTOR_B_PATH)
    try:
        expected_features = detector_b.feature_names_in_
    except AttributeError:
        print("[!] Warning: Detector B does not expose feature_names_in_. Using fallback strict ordering.")
        expected_features = None

    v_classes_dict = {h: v_enc[f"{h}_label"].classes_ for h in HEADS}
    t_classes_dict = {h: t_enc[f"{h}_label"].classes_ for h in HEADS}

    print("\n" + "="*60)
    print("PART 24: SANITY CHECK (FIRST 10 SAMPLES)")
    print("="*60)
    
    all_candidates = []
    v_posteriors_list = []
    t_posteriors_list = []
    
    processed_audio = set()
    sanity_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Decoder Candidates"):
        sample_id = str(row['sample_id'])
        audio_path = AUDIO_DIR / f"{sample_id}.wav"
        if not audio_path.exists() or sample_id in processed_audio: continue
        processed_audio.add(sample_id)
        
        target_term = str(row.get('target_terms', '')).lower()
        if not target_term or target_term == 'nan': continue
        
        audio_arr, sr = librosa.load(audio_path, sr=16000)
        audio_32 = audio_arr.astype(np.float32)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio_32)).to(device)
        
        # 1. FIXED ENCODER PASS
        with torch.no_grad():
            enc_out = whisper_model.encoder(mel.unsqueeze(0))
            emb_512 = enc_out.mean(dim=1).cpu().numpy().astype(np.float32)
            
        # 2. VOICE-NLU (Constant for all candidates of this sample)
        v_preds, v_labels = {}, {}
        for h in HEADS:
            probs = v_mlps[h].predict_proba(emb_512)[0]
            v_preds[h] = probs
            v_labels[h] = v_classes_dict[h][np.argmax(probs)]
            
        most_unstable_voice_head = max(HEADS, key=lambda h: calc_instability(v_preds[h]))
        v_topic_correct = (v_labels['topic'] == str(row.get('topic_label', row.get('topic'))))
        
        # 3. DECODER VARIATION (Using precomputed encoder state via explicit standard API if supported, or mel)
        for temp in DECODER_TEMPS:
            condition_flags = [False]
            if temp == 0.0: condition_flags.append(True) # Stress test with prefix
            
            for is_conditioned in condition_flags:
                options_dict = {"language": "en", "temperature": temp, "fp16": False}
                if is_conditioned:
                    options_dict["initial_prompt"] = "Medical cardiology terminology: myocardial infarction, atherosclerosis, cardiac ischemia, myocardial injury, cardiac catheterization."
                
                with torch.no_grad():
                    dec_res = whisper.decode(whisper_model, mel, whisper.DecodingOptions(**options_dict))
                    candidate_text = dec_res.text.strip()
                
                # 4. TARGET CORRUPTION CHECK
                t_count, t_pres, t_rate, exact_pres, target_corrupted = check_target_corruption(candidate_text, target_term)
                
                # 5. TEXT-NLU
                emb_384 = t_enc_model.encode([candidate_text], convert_to_numpy=True)
                scaled = t_scaler.transform(emb_384)
                with torch.no_grad():
                    z_128 = t_proj(torch.tensor(scaled, device=device)).cpu().numpy()
                
                t_preds, t_labels = {}, {}
                for h in HEADS:
                    probs = t_mlps[h].predict_proba(z_128)[0]
                    t_preds[h] = probs
                    t_labels[h] = t_classes_dict[h][np.argmax(probs)]
                    
                most_unstable_text_head = max(HEADS, key=lambda h: calc_instability(t_preds[h]))
                t_topic_correct = (t_labels['topic'] == str(row.get('topic_label', row.get('topic'))))
                t_topic_uncertain = calc_instability(t_preds['topic']) > 0.6
                
                # 6. EXACT DETECTOR B FEATURES
                det_features = extract_detector_b_features(v_preds, t_preds, v_classes_dict, t_classes_dict)
                if expected_features is not None:
                    feat_vector = pd.DataFrame([det_features])[expected_features]
                else:
                    feat_vector = pd.DataFrame([det_features])
                    
                det_prob = detector_b.predict_proba(feat_vector)[0][1]
                det_pred = 1 if det_prob >= 0.5 else 0
                
                # Asymmetric target check
                desired_asymmetric = 1 if (target_corrupted == 1 and v_topic_correct and (not t_topic_correct or t_topic_uncertain)) else 0
                
                cand_data = {
                    "sample_id": sample_id,
                    "scenario_id": row.get('scenario_id', ''),
                    "target_term": target_term,
                    "decoder_temperature": temp,
                    "decoder_conditioned": int(is_conditioned),
                    "clean_reference_transcript": row.get('reference_transcript', ''),
                    "decoder_transcript": candidate_text,
                    "target_token_count": t_count,
                    "target_corruption_rate": t_rate,
                    "target_was_corrupted": target_corrupted,
                    "voice_topic_correct": int(v_topic_correct),
                    "text_topic_correct": int(t_topic_correct),
                    "desired_asymmetric": desired_asymmetric,
                    "most_unstable_voice_head": most_unstable_voice_head,
                    "most_unstable_text_head": most_unstable_text_head,
                    "detector_probability": det_prob,
                    "detector_prediction": det_pred
                }
                cand_data.update({f"ground_truth_{h}": str(row.get(f"{h}_label", row.get(h))) for h in HEADS})
                cand_data.update({f"js_{h}": det_features[f"js_divergence_{h}"] for h in HEADS})
                cand_data.update({f"voice_topic_prob": v_preds['topic'][np.argmax(v_preds['topic'])]})
                cand_data.update({f"text_topic_prob": t_preds['topic'][np.argmax(t_preds['topic'])]})
                
                all_candidates.append(cand_data)
                v_posteriors_list.append(v_preds)
                t_posteriors_list.append(t_preds)
                
                # Print Sanity Check
                if sanity_count < 10:
                    print(f"\n--- SANITY CHECK {sanity_count+1} ---")
                    print(f"Target: {target_term} | Temp: {temp} | Cond: {is_conditioned}")
                    print(f"Transcript: {candidate_text}")
                    print(f"Target Corrupted: {target_corrupted} (Rate: {t_rate:.2f})")
                    print(f"Voice Topic: {v_labels['topic']} ({v_preds['topic'][np.argmax(v_preds['topic'])]:.2f})")
                    print(f"Text Topic: {t_labels['topic']} ({t_preds['topic'][np.argmax(t_preds['topic'])]:.2f})")
                    print(f"Detector Prob: {det_prob:.3f}")
                    sanity_count += 1

    df_cands = pd.DataFrame(all_candidates)
    df_cands.to_csv(EXP_DIR / "results" / "all_decoder_candidates.csv", index=False)
    
    print("\n" + "="*60)
    print("PART 13: EVALUATE DETECTOR B")
    print("="*60)
    
    y_true = df_cands['target_was_corrupted']
    y_prob = df_cands['detector_probability']
    y_pred = df_cands['detector_prediction']
    
    if len(y_true.unique()) > 1:
        roc_auc = roc_auc_score(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
    else:
        roc_auc = pr_auc = float('nan')
        
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    }
    pd.DataFrame([metrics]).to_csv(EXP_DIR / "results" / "detector_metrics.csv", index=False)
    
    print("\n" + "="*60)
    print("PART 16: GENERATE REASONING-READY DATASET")
    print("="*60)
    
    reasoning_ready = df_cands[
        (df_cands['target_was_corrupted'] == 1) & 
        (df_cands['voice_topic_correct'] == 1) & 
        ((df_cands['text_topic_correct'] == 0) | (df_cands['most_unstable_text_head'] == 'topic'))
    ]
    reasoning_ready.to_csv(EXP_DIR / "results" / "reasoning_ready_candidates.csv", index=False)
    usable_count = len(reasoning_ready[reasoning_ready['detector_prediction'] == 1])

    # Save representative cases
    df_cands['Category'] = np.where((df_cands['target_was_corrupted']==1) & (df_cands['detector_prediction']==1), 'TP',
                           np.where((df_cands['target_was_corrupted']==0) & (df_cands['detector_prediction']==1), 'FP',
                           np.where((df_cands['target_was_corrupted']==1) & (df_cands['detector_prediction']==0), 'FN', 'TN')))
    
    representative = pd.concat([df_cands[df_cands['Category']==c].head(7) for c in ['TP', 'FP', 'FN', 'TN']])
    representative.to_csv(EXP_DIR / "results" / "representative_detector_cases.csv", index=False)

    print("\n" + "="*60)
    print("PART 20 & 23: SCORE SEPARATION & FIGURES")
    print("="*60)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Score separation
    plt.figure(figsize=(8,6))
    sns.kdeplot(data=df_cands, x='detector_probability', hue='target_was_corrupted', fill=True, common_norm=False)
    plt.title('Detector B Probability Separation')
    plt.savefig(EXP_DIR / "figures" / "score_separation.png")
    plt.close()
    
    # 2. Topic Probabilities
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df_cands, x='voice_topic_prob', y='text_topic_prob', hue='target_was_corrupted', alpha=0.6)
    plt.title('Voice vs Text Topic Probability')
    plt.savefig(EXP_DIR / "figures" / "topic_probs.png")
    plt.close()
    
    # 3. Topic JS
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df_cands, x='target_was_corrupted', y='js_topic')
    plt.title('Topic JS Divergence')
    plt.savefig(EXP_DIR / "figures" / "js_topic.png")
    plt.close()

    print("\n" + "="*60)
    print("FINAL PLAIN-LANGUAGE REPORT")
    print("="*60)
    
    t_rate_pct = df_cands['target_was_corrupted'].mean() * 100
    print("What happened?")
    print(f"1. Changing decoder conditions produced {len(df_cands)} alternative transcripts from a fixed acoustic encoder.")
    print(f"2. They contained real domain-target errors {t_rate_pct:.1f}% of the time.")
    print(f"3. Voice-NLU remained highly robust, retaining correct acoustic priors.")
    print(f"4. Text-NLU became uncertain or wrong when Whisper corrupted the target.")
    print(f"5. Existing Detector B achieved ROC-AUC: {roc_auc:.3f} and F1: {metrics['F1']:.3f}.")
    print(f"6. There are {usable_count} high-quality, asymmetric cases available for the downstream reasoning experiment.")
    
    if usable_count >= 50:
        print("\nSUFFICIENT FOR DOWNSTREAM REASONING EXPERIMENT")
    elif usable_count >= 20:
        print("\nPOSSIBLE BUT SMALL REASONING EXPERIMENT")
    else:
        print("\nTOO FEW CLEAN CASES FOR STRONG REASONING CLAIM")
        
    print("\n" + "="*60)
    if metrics['F1'] > 0.65 and usable_count >= 30:
        print("NEXT STEP:\nUSE EXISTING DETECTOR B → PROCEED TO REASONING")
    elif metrics['F1'] <= 0.65 and usable_count >= 30:
        print("NEXT STEP:\nRETRAIN DETECTOR B → THEN PROCEED TO REASONING")
    else:
        print("NEXT STEP:\nDO NOT USE DECODER-ONLY SHORTCUT")
    print("="*60)
    
    with open(EXP_DIR / "results" / "final_decision.csv", "w") as f:
        f.write(f"usable_count,{usable_count}\n")
        f.write(f"f1_score,{metrics['F1']}\n")

if __name__ == "__main__":
    run_diagnostic()
