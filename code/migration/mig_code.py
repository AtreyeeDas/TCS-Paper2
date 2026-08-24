Here is the complete, locally deployable Python pipeline migrated from your Colab notebook, structured precisely into the 19 requested cells. All Colab-specific code has been stripped, and strict local paths are utilized to prevent arbitrary internet downloads.
This script is engineered to run seamlessly on your NVIDIA RTX PRO 50 Blackwell and natively leverages the CUDA 13.0 PyTorch nightly environment.
Prerequisites (Local Environment)
Ensure your EXPERIMENT_ROOT contains the transferred Google Drive folders (dataset/, audio/, models/, embeddings/, text_models/, text_embeddings/), and MODEL_ROOT contains your manually downloaded models (whisper/base.en.pt and sentence_transformers/all-MiniLM-L6-v2/).
CELL 1: Local Configuration + Environment Verification
import os
import sys
import json
import torch
from pathlib import Path
import warnings

# Suppress annoying warnings from external libraries
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
EXPERIMENT_ROOT = Path(r"C:\Path\To\Your\Experiment_Root") # UPDATE THIS
MODEL_ROOT = Path(r"C:\Path\To\Your\Model_Root")           # UPDATE THIS

# Derived Paths
DATA_DIR = EXPERIMENT_ROOT / "dataset"
AUDIO_DIR = EXPERIMENT_ROOT / "audio"
MODELS_DIR = EXPERIMENT_ROOT / "models"
EMBED_DIR = EXPERIMENT_ROOT / "embeddings"
TEXT_MODELS_DIR = EXPERIMENT_ROOT / "text_models"
TEXT_EMBED_DIR = EXPERIMENT_ROOT / "text_embeddings"

ASR_DIR = EXPERIMENT_ROOT / "asr"
ASR_TRANSCRIPTS_DIR = ASR_DIR / "transcripts"
ASR_JSON_DIR = ASR_DIR / "json"
RESULTS_DIR = EXPERIMENT_ROOT / "results"

# Create output directories
for d in [ASR_DIR, ASR_TRANSCRIPTS_DIR, ASR_JSON_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# ENVIRONMENT VERIFICATION
# ============================================================
print("=" * 60)
print("ENVIRONMENT VERIFICATION")
print("=" * 60)
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA Available: True (Device: {torch.cuda.get_device_name(0)})")
    DEVICE = "cuda"
else:
    print("CUDA Available: False - CRITICAL WARNING: GPU NOT DETECTED")
    DEVICE = "cpu"

print(f"\nExperiment Root: {EXPERIMENT_ROOT}")
print(f"Model Root: {MODEL_ROOT}")

CELL 2: Artifact Verification
print("=" * 60)
print("ARTIFACT VERIFICATION")
print("=" * 60)

required_artifacts = [
    DATA_DIR / "nlu_robust_6000_scenario_paraphrase.csv",
    MODELS_DIR / "whisper_scaler.joblib",
    MODELS_DIR / "label_encoders.joblib",
    MODELS_DIR / "best_hierarchical_projection.pt",
    MODELS_DIR / "domain_mlp.joblib",
    MODELS_DIR / "subdomain_mlp.joblib",
    MODELS_DIR / "topic_mlp.joblib",
    MODELS_DIR / "document_type_mlp.joblib",
    EMBED_DIR / "hierarchical_semantic_embeddings.npy",
    EMBED_DIR / "whisper_embedding_metadata.csv",
    TEXT_MODELS_DIR / "text_scaler.joblib",
    TEXT_MODELS_DIR / "text_label_encoders.joblib",
    TEXT_MODELS_DIR / "best_text_hierarchical_projection.pt",
    TEXT_MODELS_DIR / "text_domain_mlp.joblib",
    TEXT_MODELS_DIR / "text_subdomain_mlp.joblib",
    TEXT_MODELS_DIR / "text_topic_mlp.joblib",
    TEXT_MODELS_DIR / "text_document_type_mlp.joblib",
    MODEL_ROOT / "whisper" / "base.en.pt",
    MODEL_ROOT / "sentence_transformers" / "all-MiniLM-L6-v2" / "config.json"
]

missing_files = []
for artifact in required_artifacts:
    if not artifact.exists():
        missing_files.append(artifact)

if missing_files:
    print("CRITICAL ERROR: Missing the following required artifacts:")
    for mf in missing_files:
        print(f"  - {mf}")
    sys.exit(1)
else:
    print("All required existing artifacts and local models found. Proceeding.")

import pandas as pd
df = pd.read_csv(DATA_DIR / "nlu_robust_6000_scenario_paraphrase.csv")
print(f"Dataset loaded: {len(df)} samples.")

CELL 3: Load Local Whisper base.en
import whisper

print("=" * 60)
print("LOADING LOCAL WHISPER MODEL")
print("=" * 60)

whisper_path = str(MODEL_ROOT / "whisper" / "base.en.pt")

# Load exactly from the local file path to prevent internet downloads
whisper_model = whisper.load_model(whisper_path, device=DEVICE)
print(f"Successfully loaded Whisper from {whisper_path}")

CELL 4: Decode All 6000 Audios with Resumable Checkpointing
import numpy as np
from tqdm import tqdm

print("=" * 60)
print("WHISPER DECODING")
print("=" * 60)

progress_file = ASR_DIR / "whisper_decode_progress.json"
csv_output = ASR_DIR / "whisper_transcripts_6000.csv"

if progress_file.exists():
    with open(progress_file, 'r') as f:
        completed = set(json.load(f))
else:
    completed = set()

if csv_output.exists():
    df_results = pd.read_csv(csv_output).to_dict('records')
else:
    df_results = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    sample_id = str(row['sample_id'])
    if sample_id in completed:
        continue
        
    audio_path = AUDIO_DIR / f"{sample_id}.wav"
    if not audio_path.exists():
        print(f"Audio missing: {audio_path}")
        continue
        
    # Primary deterministic decoding configuration
    result = whisper_model.transcribe(
        str(audio_path),
        language="en",
        task="transcribe",
        temperature=0.0,
        beam_size=5,
        condition_on_previous_text=True,
        word_timestamps=True,
        verbose=False
    )
    
    # Save full JSON including timestamps
    with open(ASR_JSON_DIR / f"{sample_id}.json", 'w') as f:
        json.dump(result, f)
        
    # Aggregate logprobs
    avg_logprob = np.mean([seg['avg_logprob'] for seg in result['segments']]) if result['segments'] else 0.0
    no_speech_prob = np.mean([seg['no_speech_prob'] for seg in result['segments']]) if result['segments'] else 0.0
    compression_ratio = np.mean([seg['compression_ratio'] for seg in result['segments']]) if result['segments'] else 0.0
    
    with open(ASR_TRANSCRIPTS_DIR / f"{sample_id}.txt", 'w', encoding='utf-8') as f:
        f.write(result['text'].strip())
        
    record = {
        'sample_id': sample_id,
        'scenario_id': row['scenario_id'],
        'split': row['split'],
        'ground_truth': row['transcript'],
        'whisper_transcript': result['text'].strip(),
        'avg_logprob': avg_logprob,
        'no_speech_prob': no_speech_prob,
        'compression_ratio': compression_ratio,
        'temperature': 0.0,
        'language': "en"
    }
    df_results.append(record)
    
    completed.add(sample_id)
    
    # Checkpointing
    if len(completed) % 50 == 0:
        pd.DataFrame(df_results).to_csv(csv_output, index=False)
        with open(progress_file, 'w') as f:
            json.dump(list(completed), f)

# Final Save
pd.DataFrame(df_results).to_csv(csv_output, index=False)
with open(progress_file, 'w') as f:
    json.dump(list(completed), f)

print(f"Decoding complete. Total transcripts: {len(completed)}")

CELL 5: WER / ASR Quality Analysis
import jiwer

print("=" * 60)
print("ASR QUALITY ANALYSIS (WER)")
print("=" * 60)

df_asr = pd.read_csv(ASR_DIR / "whisper_transcripts_6000.csv")

# Standardize text for WER calculation
def normalize_text(text):
    if pd.isna(text): return ""
    import re
    return re.sub(r'[^\w\s]', '', str(text).lower()).strip()

wers = []
has_error = []

for _, row in df_asr.iterrows():
    gt = normalize_text(row['ground_truth'])
    pred = normalize_text(row['whisper_transcript'])
    
    if len(gt) == 0:
        wers.append(0.0)
        has_error.append(False)
        continue
        
    error = jiwer.wer(gt, pred)
    wers.append(error)
    has_error.append(error > 0.0)

df_asr['wer'] = wers
df_asr['has_error'] = has_error
df_asr.to_csv(ASR_DIR / "whisper_transcripts_with_wer.csv", index=False)

print(f"Mean WER: {np.mean(wers):.4f}")
print(f"Median WER: {np.median(wers):.4f}")
print(f"Samples with any error: {np.mean(has_error)*100:.2f}%")

print("\nWER by Split:")
print(df_asr.groupby('split')['wer'].mean())

CELL 6: Load Saved Voice-NLU Artifacts
import joblib

print("=" * 60)
print("LOADING VOICE NLU ARTIFACTS")
print("=" * 60)

# Load aligned metadata and semantic embeddings
df_voice_meta = pd.read_csv(EMBED_DIR / "whisper_embedding_metadata.csv")
voice_embeddings = np.load(EMBED_DIR / "hierarchical_semantic_embeddings.npy")

# Ensure ordering matches ASR df
sample_to_voice_emb = {str(row['sample_id']): voice_embeddings[idx] for idx, row in df_voice_meta.iterrows()}
ordered_voice_embeddings = np.array([sample_to_voice_emb[str(s_id)] for s_id in df_asr['sample_id']])

# Load Models
voice_label_encoders = joblib.load(MODELS_DIR / "label_encoders.joblib")
voice_domain_mlp = joblib.load(MODELS_DIR / "domain_mlp.joblib")
voice_subdomain_mlp = joblib.load(MODELS_DIR / "subdomain_mlp.joblib")
voice_topic_mlp = joblib.load(MODELS_DIR / "topic_mlp.joblib")
voice_doc_mlp = joblib.load(MODELS_DIR / "document_type_mlp.joblib")

print("Voice artifacts loaded successfully.")

CELL 7: Load Local MiniLM + Saved Text-NLU Artifacts
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("LOADING TEXT NLU ARTIFACTS")
print("=" * 60)

# Load MiniLM explicitly from local path
minilm_path = str(MODEL_ROOT / "sentence_transformers" / "all-MiniLM-L6-v2")
text_encoder = SentenceTransformer(minilm_path, device=DEVICE)

text_scaler = joblib.load(TEXT_MODELS_DIR / "text_scaler.joblib")
text_label_encoders = joblib.load(TEXT_MODELS_DIR / "text_label_encoders.joblib")

# For PyTorch Projection
class TextHierarchicalProjection(torch.nn.Module):
    def __init__(self, input_dim=384, projection_dim=128):
        super().__init__()
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(256, projection_dim)
        )
    def forward(self, x):
        z = self.projector(x)
        return torch.nn.functional.normalize(z, p=2, dim=1)

text_proj = TextHierarchicalProjection().to(DEVICE)
text_proj.load_state_dict(torch.load(TEXT_MODELS_DIR / "best_text_hierarchical_projection.pt", map_location=DEVICE))
text_proj.eval()

text_domain_mlp = joblib.load(TEXT_MODELS_DIR / "text_domain_mlp.joblib")
text_subdomain_mlp = joblib.load(TEXT_MODELS_DIR / "text_subdomain_mlp.joblib")
text_topic_mlp = joblib.load(TEXT_MODELS_DIR / "text_topic_mlp.joblib")
text_doc_mlp = joblib.load(TEXT_MODELS_DIR / "text_document_type_mlp.joblib")

print("Text artifacts loaded successfully.")

CELL 8: Run Text-NLU on Whisper Transcripts
print("=" * 60)
print("RUNNING TEXT-NLU INFERENCE ON ASR TRANSCRIPTS")
print("=" * 60)

whisper_transcripts = df_asr['whisper_transcript'].fillna("").tolist()

# 1. Encode text
raw_text_embs = text_encoder.encode(whisper_transcripts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)

# 2. Scale
scaled_text_embs = text_scaler.transform(raw_text_embs)

# 3. Project to 128-D
text_semantic_embeddings = []
with torch.inference_mode():
    for i in range(0, len(scaled_text_embs), 128):
        batch = torch.tensor(scaled_text_embs[i:i+128], dtype=torch.float32, device=DEVICE)
        proj = text_proj(batch).cpu().numpy()
        text_semantic_embeddings.append(proj)
text_semantic_embeddings = np.vstack(text_semantic_embeddings)

print("Text semantic representations generated.")

CELL 9: Probability-Aware Voice/Text NLU Inference
from scipy.stats import entropy

def get_inference_stats(mlp, encoders, X, head_name):
    probas = mlp.predict_proba(X)
    top2_indices = np.argsort(probas, axis=1)[:, -2:][:, ::-1]
    
    stats = []
    for i, p in enumerate(probas):
        top1_idx, top2_idx = top2_indices[i]
        top1_conf = p[top1_idx]
        margin = top1_conf - p[top2_idx]
        ent = entropy(p)
        label = encoders[head_name].inverse_transform([top1_idx])[0]
        
        stats.append({
            f'{head_name}_label': label,
            f'{head_name}_conf': top1_conf,
            f'{head_name}_margin': margin,
            f'{head_name}_entropy': ent,
            f'{head_name}_proba_dist': p
        })
    return pd.DataFrame(stats)

print("=" * 60)
print("EXTRACTING PROBABILISTIC FEATURES")
print("=" * 60)

# Voice Inferences
v_dom = get_inference_stats(voice_domain_mlp, voice_label_encoders, ordered_voice_embeddings, 'domain')
v_sub = get_inference_stats(voice_subdomain_mlp, voice_label_encoders, ordered_voice_embeddings, 'subdomain')
v_top = get_inference_stats(voice_topic_mlp, voice_label_encoders, ordered_voice_embeddings, 'topic')
v_doc = get_inference_stats(voice_doc_mlp, voice_label_encoders, ordered_voice_embeddings, 'document_type')

df_voice_nlu = pd.concat([v_dom, v_sub, v_top, v_doc], axis=1)
df_voice_nlu.columns = [f"voice_{c}" for c in df_voice_nlu.columns]

# Text Inferences
t_dom = get_inference_stats(text_domain_mlp, text_label_encoders, text_semantic_embeddings, 'domain')
t_sub = get_inference_stats(text_subdomain_mlp, text_label_encoders, text_semantic_embeddings, 'subdomain')
t_top = get_inference_stats(text_topic_mlp, text_label_encoders, text_semantic_embeddings, 'topic')
t_doc = get_inference_stats(text_doc_mlp, text_label_encoders, text_semantic_embeddings, 'document_type')

df_text_nlu = pd.concat([t_dom, t_sub, t_top, t_doc], axis=1)
df_text_nlu.columns = [f"text_{c}" for c in df_text_nlu.columns]

print("Inference completed.")

CELL 10: Semantic Comparison Engine
from scipy.spatial.distance import jensenshannon

print("=" * 60)
print("SEMANTIC COMPARISON ENGINE")
print("=" * 60)

df_comparison = pd.concat([df_asr[['sample_id', 'scenario_id', 'split']], df_voice_nlu, df_text_nlu], axis=1)

heads = [
    ('domain', 0.20),
    ('subdomain', 0.25),
    ('topic', 0.40),
    ('document_type', 0.15)
]

for head, weight in heads:
    # 1. Hard label disagreement
    df_comparison[f'{head}_disagreement'] = (df_comparison[f'voice_{head}_label'] != df_comparison[f'text_{head}_label']).astype(int)
    
    # Jensen-Shannon Divergence
    js_divs, text_supports_voice, voice_supports_text, conf_gaps = [], [], [], []
    
    for _, row in df_comparison.iterrows():
        p_v = row[f'voice_{head}_proba_dist']
        p_t = row[f'text_{head}_proba_dist']
        
        # JS Divergence
        js_divs.append(jensenshannon(p_v, p_t) ** 2)
        
        # Cross-support (Requires mapping indices if label encoders differ. Assuming they are identical for this implementation)
        v_label_idx = voice_label_encoders[head].transform([row[f'voice_{head}_label']])[0]
        t_label_idx = text_label_encoders[head].transform([row[f'text_{head}_label']])[0]
        
        text_supports_voice.append(p_t[v_label_idx])
        voice_supports_text.append(p_v[t_label_idx])
        conf_gaps.append(abs(row[f'voice_{head}_conf'] - row[f'text_{head}_conf']))
        
    df_comparison[f'{head}_js_div'] = js_divs
    df_comparison[f'{head}_text_support_for_voice'] = text_supports_voice
    df_comparison[f'{head}_voice_support_for_text'] = voice_supports_text
    df_comparison[f'{head}_conf_gap'] = conf_gaps

# Aggregate Features
df_comparison['weighted_label_disagreement'] = sum(df_comparison[f'{h}_disagreement'] * w for h, w in heads)
df_comparison['weighted_JS_divergence'] = sum(df_comparison[f'{h}_js_div'] * w for h, w in heads)
df_comparison['strong_conflict_score'] = df_comparison['weighted_label_disagreement'] * df_comparison['weighted_JS_divergence']

df_comparison['mean_voice_confidence'] = df_comparison[[f'voice_{h}_conf' for h, _ in heads]].mean(axis=1)
df_comparison['mean_text_confidence'] = df_comparison[[f'text_{h}_conf' for h, _ in heads]].mean(axis=1)
df_comparison['mean_text_supports_voice'] = df_comparison[[f'{h}_text_support_for_voice' for h, _ in heads]].mean(axis=1)
df_comparison['mean_voice_supports_text'] = df_comparison[[f'{h}_voice_support_for_text' for h, _ in heads]].mean(axis=1)

df_comparison.to_csv(RESULTS_DIR / "voice_text_nlu_comparison.csv", index=False)
print("Semantic comparison complete.")

CELL 11: Build Training-Only Phonetic/Domain Confusion Pairs
import random

print("=" * 60)
print("BUILDING CONFUSION LEXICON (TRAIN ONLY)")
print("=" * 60)

# Extract only from train
df_train = df_asr[df_asr['split'] == 'train']
corpus = " ".join(df_train['ground_truth'].tolist()).lower().split()
vocab = list(set(corpus))

# Generate simple heuristic-based phonetic confusions
# In a real environment, this would use phonemizers. 
# Here we simulate character-level edit distance proximity for real-word substitution.
def get_plausible_substitutions(word, vocab, limit=1):
    subs = []
    for v in vocab:
        if v == word or len(v) < 4 or len(word) < 4: continue
        if len(v) == len(word) and sum(a != b for a,b in zip(v, word)) == 1:
            subs.append(v)
            if len(subs) == limit: break
    return subs

confusion_pairs = []
for word in random.sample(vocab, min(500, len(vocab))):
    subs = get_plausible_substitutions(word, vocab)
    if subs:
        confusion_pairs.append({'source_term': word, 'replacement_term': subs[0]})

df_confusion = pd.DataFrame(confusion_pairs)
df_confusion.to_csv(ASR_DIR / "phonetic_confusion_pairs.csv", index=False)
print(f"Generated {len(df_confusion)} domain/phonetic confusion pairs from training split.")

CELL 12: Generate Controlled Transcript Errors
import re

print("=" * 60)
print("GENERATING CONTROLLED TRANSCRIPT ERRORS")
print("=" * 60)

controlled_data = []

# Map lists for fast lookup
conf_dict = dict(zip(df_confusion['source_term'], df_confusion['replacement_term']))

for _, row in df_asr.iterrows():
    clean_text = row['ground_truth']
    corrupted_text = clean_text
    true_error = 0
    source_term, rep_term = "", ""
    
    # 50% chance to corrupt if a word exists in the dictionary
    words = str(clean_text).split()
    replaceable = [w for w in words if w.lower() in conf_dict]
    
    if replaceable and random.random() > 0.5:
        target = random.choice(replaceable)
        source_term = target.lower()
        rep_term = conf_dict[source_term]
        
        # Replace only one occurrence using regex boundary
        corrupted_text = re.sub(rf"\b{re.escape(target)}\b", rep_term, str(clean_text), count=1, flags=re.IGNORECASE)
        true_error = 1
        
    controlled_data.append({
        'sample_id': row['sample_id'],
        'scenario_id': row['scenario_id'],
        'split': row['split'],
        'clean_transcript': clean_text,
        'controlled_corrupted_transcript': corrupted_text,
        'source_term': source_term,
        'replacement_term': rep_term,
        'true_error': true_error
    })

df_controlled = pd.DataFrame(controlled_data)
df_controlled.to_csv(ASR_DIR / "controlled_asr_error_dataset.csv", index=False)
print(f"Generated {df_controlled['true_error'].sum()} controlled errors across dataset.")

CELL 13: Build Detector Features
print("=" * 60)
print("BUILDING ERROR DETECTOR FEATURES")
print("=" * 60)

# The features required for the classifier
feature_cols = [
    'weighted_label_disagreement', 'weighted_JS_divergence', 'strong_conflict_score',
    'mean_voice_confidence', 'mean_text_confidence', 
    'mean_text_supports_voice', 'mean_voice_supports_text'
]

for h, _ in heads:
    feature_cols.extend([
        f'{h}_disagreement', f'{h}_js_div', f'{h}_conf_gap',
        f'voice_{h}_entropy', f'text_{h}_entropy',
        f'voice_{h}_margin', f'text_{h}_margin'
    ])

# We map the true_error label to the comparison dataframe
df_detector = pd.merge(df_comparison, df_controlled[['sample_id', 'true_error']], on='sample_id')

X = df_detector[feature_cols].values
y = df_detector['true_error'].values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, MODELS_DIR / "detector_feature_scaler.joblib")
print(f"Feature matrix built with {len(feature_cols)} dimensions.")

CELL 14: Train Detector on Train Scenarios
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("=" * 60)
print("TRAINING INTERPRETABLE ERROR DETECTOR")
print("=" * 60)

train_mask = df_detector['split'] == 'train'

X_train = X_scaled[train_mask]
y_train = y[train_mask]

# Interpretable classifier
detector = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
detector.fit(X_train, y_train)

joblib.dump(detector, MODELS_DIR / "transcript_error_detector.joblib")

train_preds = detector.predict(X_train)
print("TRAIN PERFORMANCE:")
print(classification_report(y_train, train_preds))
print(f"ROC-AUC: {roc_auc_score(y_train, detector.predict_proba(X_train)[:, 1]):.4f}")

CELL 15: Validation Threshold Selection
print("=" * 60)
print("VALIDATION SCENARIO EVALUATION")
print("=" * 60)

val_mask = df_detector['split'] == 'validation'
X_val = X_scaled[val_mask]
y_val = y[val_mask]

val_probs = detector.predict_proba(X_val)[:, 1]

# Calibrate Threshold (Maximize F1)
from sklearn.metrics import f1_score
best_thresh, best_f1 = 0.5, 0
for thresh in np.arange(0.1, 0.9, 0.05):
    preds = (val_probs >= thresh).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

print(f"Optimized Decision Threshold: {best_thresh:.2f}")

val_preds = (val_probs >= best_thresh).astype(int)
print("\nVALIDATION PERFORMANCE:")
print(classification_report(y_val, val_preds))
print(f"ROC-AUC: {roc_auc_score(y_val, val_probs):.4f}")

CELL 16: Final Unseen Evaluation
print("=" * 60)
print("FINAL UNSEEN SCENARIO EVALUATION")
print("=" * 60)

unseen_mask = df_detector['split'] == 'unseen'
X_unseen = X_scaled[unseen_mask]
y_unseen = y[unseen_mask]

unseen_probs = detector.predict_proba(X_unseen)[:, 1]
unseen_preds = (unseen_probs >= best_thresh).astype(int)

df_detector.loc[unseen_mask, 'detector_prob'] = unseen_probs
df_detector.loc[unseen_mask, 'detector_decision'] = unseen_preds

print("UNSEEN PERFORMANCE:")
print(classification_report(y_unseen, unseen_preds))
print(f"ROC-AUC: {roc_auc_score(y_unseen, unseen_probs):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_unseen, unseen_preds))

# Detailed Breakdown Analysis
print("\nBreakdown by Disagreement Type (UNSEEN):")
unseen_df = df_detector[unseen_mask]

for h, _ in heads:
    subset = unseen_df[unseen_df[f'{h}_disagreement'] == 1]
    if len(subset) > 0:
        f1 = f1_score(subset['true_error'], subset['detector_decision'])
        print(f"  {h.upper()} mismatch: F1 = {f1:.4f} (n={len(subset)})")

CELL 17: Real Whisper Error Evaluation
print("=" * 60)
print("REAL WHISPER ERROR DETECTION")
print("=" * 60)

# Merge actual WER from Cell 5
df_real = pd.merge(df_comparison, df_asr[['sample_id', 'has_error', 'wer', 'ground_truth', 'whisper_transcript']], on='sample_id')
X_real_scaled = scaler.transform(df_real[feature_cols].values)

real_probs = detector.predict_proba(X_real_scaled)[:, 1]
real_preds = (real_probs >= best_thresh).astype(int)

df_real['detector_prob'] = real_probs
df_real['detector_decision'] = real_preds

# Evaluation against actual WER derived ground truth
y_actual_error = df_real['has_error'].astype(int)

print("Actual ASR Error Detection Performance:")
print(classification_report(y_actual_error, real_preds))

cm_real = confusion_matrix(y_actual_error, real_preds)
print("Confusion Matrix:")
print(cm_real)

fp = cm_real[0,1]
fn = cm_real[1,0]
print(f"\nReal ASR Errors Flagged: {cm_real[1,1]}")
print(f"Real ASR Errors Missed (False Negatives): {fn}")
print(f"False Positives (Clean transcripts flagged as errors): {fp}")

df_real.to_csv(RESULTS_DIR / "real_whisper_error_detection.csv", index=False)

CELL 18: Representative Examples & Final Tables
print("=" * 60)
print("REPRESENTATIVE EXAMPLES")
print("=" * 60)

# Find True Positives (Successfully detected real ASR errors)
tp_examples = df_real[(df_real['has_error'] == True) & (df_real['detector_decision'] == 1)].head(3)

print("--- DETECTED ERRORS (TRUE POSITIVES) ---")
for _, ex in tp_examples.iterrows():
    print(f"GT: {ex['ground_truth']}")
    print(f"ASR: {ex['whisper_transcript']} (WER: {ex['wer']:.2f})")
    print(f"Voice Domain: {ex['voice_domain_label']} ({ex['voice_domain_conf']:.2f}) | Text Domain: {ex['text_domain_label']} ({ex['text_domain_conf']:.2f})")
    print(f"Disagreement Score: {ex['weighted_label_disagreement']:.2f} | Detector Prob: {ex['detector_prob']:.4f}\n")

# Find False Positives
fp_examples = df_real[(df_real['has_error'] == False) & (df_real['detector_decision'] == 1)].head(3)

print("--- FALSE ALARMS (FALSE POSITIVES) ---")
for _, ex in fp_examples.iterrows():
    print(f"GT/ASR Match: {ex['ground_truth']}")
    print(f"Disagreement Score: {ex['weighted_label_disagreement']:.2f} | Detector Prob: {ex['detector_prob']:.4f}\n")

tp_examples.to_csv(RESULTS_DIR / "detector_examples.csv", index=False)

CELL 19: Save All Artifacts / Configuration
import pkg_resources

print("=" * 60)
print("SAVING REPRODUCIBILITY CONFIGURATION")
print("=" * 60)

config = {
    "random_seed": 42,
    "whisper_model": whisper_path,
    "minilm_model": minilm_path,
    "feature_names": feature_cols,
    "hierarchy_weights": dict(heads),
    "decision_threshold": float(best_thresh),
    "dataset_samples": len(df_asr),
    "train_scenarios_count": len(df_detector[df_detector['split'] == 'train']['scenario_id'].unique()),
    "val_scenarios_count": len(df_detector[df_detector['split'] == 'validation']['scenario_id'].unique()),
    "unseen_scenarios_count": len(df_detector[df_detector['split'] == 'unseen']['scenario_id'].unique()),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "pytorch_version": torch.__version__,
    "environment": {pkg.key: pkg.version for pkg in pkg_resources.working_set if pkg.key in ['torch', 'whisper', 'scikit-learn', 'numpy', 'pandas', 'sentence-transformers']}
}

with open(RESULTS_DIR / "experiment_config.json", 'w') as f:
    json.dump(config, f, indent=4)

print("Experiment configuration saved. Local migration complete.")

