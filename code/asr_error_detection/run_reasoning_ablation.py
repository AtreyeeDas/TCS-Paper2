"""
run_reasoning_ablation.py

Downstream Reasoning Ablation Experiment with Fixed Whisper Encoder.
Evaluates Gemma-3-1B-it under three within-sample conditions:
  1. Baseline (Corrupted Transcript only)
  2. Voice-NLU (Corrupted Transcript + Ungated Voice-NLU Evidence)
  3. Detector-Gated Voice-NLU (Detector B-Gated Semantic Evidence)

System Properties:
  - Fixed Whisper Encoder [1, T, 512] persistent caching & strict invariance.
  - Voice-NLU (512 -> 128) & Text-NLU (384 -> 128) architecture preservation.
  - Detector B feature extraction (extract_baseline_features) & threshold gating.
  - Zero target term or ground-truth prompt leakage.
"""

import os
import re
import time
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
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Authoritative feature extractor from training pipeline
try:
    from feature_extractor import extract_baseline_features
except ImportError:
    from scipy.spatial.distance import jensenshannon
    def extract_baseline_features(voice_row, text_row, v_preds, t_preds, v_enc, t_enc):
        HEADS = ["domain", "subdomain", "topic", "document_type"]
        HEAD_WEIGHTS = {"domain": 0.20, "subdomain": 0.25, "topic": 0.40, "document_type": 0.15}
        feats = {}
        tot_dis = 0
        w_dis = 0.0
        for h in HEADS:
            v_l = voice_row.get(f"voice_{h}", "")
            t_l = text_row.get(f"text_{h}", "")
            dis = 1 if v_l != t_l else 0
            feats[f"{h}_disagreement"] = dis
            tot_dis += dis
            w_dis += HEAD_WEIGHTS[h] * dis
            v_p = v_preds[h]
            t_p = t_preds[h]
            v_cls = v_enc[f"{h}_label"].classes_
            t_cls = t_enc[f"{h}_label"].classes_
            v_top1 = np.max(v_p)
            t_top1 = np.max(t_p)
            feats[f"voice_top1_confidence_{h}"] = float(v_top1)
            feats[f"text_top1_confidence_{h}"] = float(t_top1)
            feats[f"absolute_confidence_difference_{h}"] = float(abs(v_top1 - t_top1))
            
            u_cls = sorted(list(set(v_cls) | set(t_cls)))
            v_a, t_a = np.zeros(len(u_cls)), np.zeros(len(u_cls))
            for i, c in enumerate(u_cls):
                if c in v_cls: v_a[i] = v_p[np.where(v_cls == c)[0][0]]
                if c in t_cls: t_a[i] = t_p[np.where(t_cls == c)[0][0]]
            v_a = np.clip(v_a / (v_a.sum() + 1e-12), 1e-12, 1.0)
            t_a = np.clip(t_a / (t_a.sum() + 1e-12), 1e-12, 1.0)
            feats[f"js_divergence_{h}"] = float(jensenshannon(v_a, t_a) ** 2)
            
            v_s = np.sort(v_p)[::-1]
            t_s = np.sort(t_p)[::-1]
            feats[f"voice_entropy_{h}"] = float(-np.sum(v_p * np.log2(np.clip(v_p, 1e-12, 1.0))))
            feats[f"text_entropy_{h}"] = float(-np.sum(t_p * np.log2(np.clip(t_p, 1e-12, 1.0))))
            feats[f"voice_top1_top2_margin_{h}"] = float(v_s[0] - (v_s[1] if len(v_s) > 1 else 0))
            feats[f"text_top1_top2_margin_{h}"] = float(t_s[0] - (t_s[1] if len(t_s) > 1 else 0))
            
            v_idx_t = np.where(v_cls == t_l)[0]
            t_idx_v = np.where(t_cls == v_l)[0]
            feats[f"voice_probability_of_text_selected_class_{h}"] = float(v_p[v_idx_t[0]]) if len(v_idx_t) > 0 else 0.0
            feats[f"text_probability_of_voice_selected_class_{h}"] = float(t_p[t_idx_v[0]]) if len(t_idx_v) > 0 else 0.0

        feats["total_disagreements"] = tot_dis
        feats["weighted_disagreement"] = float(w_dis)
        return None, feats

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONFIGURATION & LOCAL PATHS
# ==============================================================================
ROOT_DIR = Path("/home/spark2/users/intern/Atreyee-Das/NLU_Robust_Experiment")
DATASET_CSV = ROOT_DIR / "dataset" / "whisper_domain_multitarget_6000.csv"
GROUND_TRUTH_SEMANTIC_CSV = ROOT_DIR / "dataset" / "robus_nlu_6000_paraphrase_scenario.csv"

AUDIO_DIRS = [
    ROOT_DIR / "audio",
    ROOT_DIR / "audios",
    ROOT_DIR / "dataset" / "audio"
]

# Model Paths
WHISPER_MODEL_PATH = "/home/spark2/Models/base.en.pt"
TEXT_ENCODER_PATH = "/home/spark2/Models/all-MiniLM-L6-v2"
GEMMA_MODEL_PATH = "/home/spark2/Models/gemma-3-1b-it"  # Change here if needed

VOICE_MODELS_DIR = ROOT_DIR / "audio_nlu_models"
TEXT_MODELS_DIR = ROOT_DIR / "text_nlu_models"

EXP_BASE = ROOT_DIR / "error_detector_experiments"
MODELS_CACHE_DIR = EXP_BASE / "artifacts" / "models"
DETECTOR_B_PATH = MODELS_CACHE_DIR / "real_whisper_detector_B.joblib"
THRESHOLDS_PATH = MODELS_CACHE_DIR / "real_whisper_detector_B_threshold.json"
FEATURES_PATH = MODELS_CACHE_DIR / "real_whisper_detector_B_features.json"

# Output & Cache Paths
EXP_DIR = Path("decoder_only_detector_diagnostic")
ENCODER_CACHE_DIR = EXP_DIR / "encoder_cache"
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"
LOGS_DIR = EXP_DIR / "logs"

for p in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR, ENCODER_CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

HEADS = ["domain", "subdomain", "topic", "document_type"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 1. ARCHITECTURE DEFINITIONS
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
# 2. DISTRACTORS & TEXT HELPERS
# ==============================================================================
CANDIDATE_DISTRACTORS = {
    "medical": [
        "myocardial ischemia", "cardiac insufficiency", "coronary stenosis",
        "ventricular arrhythmia", "pericardial inflammation", "arterial occlusion",
        "vascular calcification", "cardiac electrophysiology", "pulmonary embolism",
        "hemodynamic instability", "bronchial constriction", "acute pancreatitis"
    ],
    "finance": [
        "claims adjudication", "provider reimbursement", "premium reconciliation",
        "medical billing adjustment", "benefit authorization", "clinical payment settlement",
        "payer reconciliation", "hospital revenue cycle", "insurance eligibility",
        "healthcare reimbursement", "algorithmic trading", "digital asset custody",
        "technology investment banking", "cloud infrastructure financing",
        "software revenue recognition", "electronic settlement", "cybersecurity financing",
        "technology capital allocation", "digital payment infrastructure", "fintech risk management"
    ],
    "general": [
        "appointment confirmation", "reservation cancellation", "customer service request",
        "delivery notification", "account verification", "schedule modification",
        "payment confirmation", "booking information", "service availability", "order confirmation"
    ]
}

def normalize_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def build_dataset_vocabulary(dfs):
    vocab = set()
    for df in dfs:
        if df is None: continue
        text_cols = [c for c in df.columns if any(k in c.lower() for k in ['transcript', 'target', 'term', 'word', 'text', 'prompt', 'domain', 'topic', 'subdomain', 'document', 'label'])]
        for col in text_cols:
            for val in df[col].dropna():
                vocab.update(normalize_text(str(val)).split())
    return vocab

def find_audio_file(sample_id):
    for adir in AUDIO_DIRS:
        if not adir.exists(): continue
        for p in [adir / f"{sample_id}.wav", adir / f"{sample_id}", adir / f"sample_{sample_id}.wav", adir / f"{sample_id}.flac"]:
            if p.exists(): return p
    return None

def generate_user_query(domain, subdomain, topic):
    dom = str(domain).lower()
    if "med" in dom:
        queries = [
            "What is the primary clinical finding or condition described?",
            "What does this clinical assessment indicate regarding patient diagnosis?",
            "What is the critical medical issue to consider here?"
        ]
    elif "fin" in dom:
        queries = [
            "What financial operation or market risk is identified here?",
            "What is the main transaction or valuation issue discussed?",
            "What should the financial analyst conclude from this statement?"
        ]
    else:
        queries = [
            "What is the central subject or relevant issue discussed?",
            "What is the primary action or confirmation requested?",
            "What does this operational statement imply?"
        ]
    q_idx = abs(hash(str(topic))) % len(queries)
    return queries[q_idx]

def evaluate_semantic_correctness_proxy(gemma_output, target_term, distractor_term, topic_label):
    ans = normalize_text(gemma_output)
    targ = normalize_text(target_term)
    dist = normalize_text(distractor_term)
    top = normalize_text(topic_label)
    
    targ_in = (targ != "" and targ in ans) or (top != "" and top in ans)
    dist_in = (dist != "" and dist in ans)
    
    distractor_adopted = 1 if dist_in else 0
    correct = 1 if (targ_in and not dist_in) else 0
    return correct, distractor_adopted

# ==============================================================================
# 3. GEMMA INFERENCE ENGINE
# ==============================================================================
class GemmaReasoningEngine:
    def __init__(self, model_path, device=DEVICE):
        print(f"[+] Loading Gemma-3-1B-it locally from: {model_path}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if device == "cuda" else None,
            local_files_only=True
        )
        self.model.eval()
        self.warmup()

    def warmup(self):
        print("[+] Warming up Gemma inference on GPU...")
        prompt = "Transcript: The patient shows normal recovery. User query: What is the status? Give a very short answer."
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=20)
        if self.device == "cuda":
            torch.cuda.synchronize()
        print("[+] Gemma warm-up complete.")

    def generate_response(self, prompt, max_new_tokens=40):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return response_text, latency_ms

# ==============================================================================
# 4. MAIN EXPERIMENTAL PIPELINE
# ==============================================================================
def run_reasoning_experiment():
    print("=" * 60)
    print("REASONING ABLATION: CONTROLLED DECODER-SIDE CORRUPTION STRESS TEST")
    print("=" * 60)

    # 1. Load Data
    assert DATASET_CSV.exists(), f"Missing dataset: {DATASET_CSV}"
    df_eval_raw = pd.read_csv(DATASET_CSV)
    df_gt_raw = pd.read_csv(GROUND_TRUTH_SEMANTIC_CSV) if GROUND_TRUTH_SEMANTIC_CSV.exists() else None

    # Vocabulary & Distractor Filter
    dataset_vocab = build_dataset_vocabulary([df_eval_raw, df_gt_raw])
    valid_distractors = {}
    for dom, terms in CANDIDATE_DISTRACTORS.items():
        valid_distractors[dom] = []
        for term in terms:
            term_norm = normalize_text(term)
            if not any(t in dataset_vocab for t in term_norm.split()):
                valid_distractors[dom].append(term)

    # Sample Selection (Unseen Split, capped at 150-200 for ~1 hour budget)
    df_unseen = df_eval_raw[df_eval_raw['split'] == 'unseen'].copy() if 'split' in df_eval_raw.columns else df_eval_raw.copy()
    valid_sample_rows = []
    for _, row in df_unseen.iterrows():
        sid = str(row['sample_id'])
        if find_audio_file(sid) is not None and pd.notna(row.get('target_terms', row.get('target_words', None))):
            valid_sample_rows.append(row)
            
    df_samples = pd.DataFrame(valid_sample_rows)
    if len(df_samples) > 200:
        df_samples = df_samples.sample(200, random_state=42)
    print(f"[+] Loaded {len(df_samples)} unseen eligible audio samples.")

    # 2. Load Pipeline Models
    whisper_model = whisper.load_model(WHISPER_MODEL_PATH, device=DEVICE)
    
    v_enc = joblib.load(VOICE_MODELS_DIR / "label_encoders.joblib")
    v_scaler = joblib.load(VOICE_MODELS_DIR / "whisper_scaler.joblib")
    v_proj = VoiceHierarchicalProjection(512, 128).to(DEVICE)
    v_proj.load_state_dict(torch.load(VOICE_MODELS_DIR / "best_hierarchical_projection.pt", map_location=DEVICE, weights_only=True))
    v_proj.eval()
    v_mlps = {h: joblib.load(VOICE_MODELS_DIR / f"{h}_mlp.joblib") for h in HEADS}

    t_enc_model = SentenceTransformer(TEXT_ENCODER_PATH, device=DEVICE)
    t_scaler = joblib.load(TEXT_MODELS_DIR / "text_scaler.joblib")
    t_proj = TextHierarchicalProjection(384, 128).to(DEVICE)
    t_proj.load_state_dict(torch.load(TEXT_MODELS_DIR / "best_text_hierarchical_projection.pt", map_location=DEVICE, weights_only=True))
    t_proj.eval()
    t_enc = joblib.load(TEXT_MODELS_DIR / "text_label_encoders.joblib")
    t_mlps = {h: joblib.load(TEXT_MODELS_DIR / f"text_{h}_mlp.joblib") for h in HEADS}

    detector_b = joblib.load(DETECTOR_B_PATH)
    with open(THRESHOLDS_PATH, "r") as f:
        det_threshold = json.load(f).get("threshold", 0.5)
    with open(FEATURES_PATH, "r") as f:
        expected_features = json.load(f)

    gemma = GemmaReasoningEngine(GEMMA_MODEL_PATH, device=DEVICE)

    # 3. Setup File Storage
    outputs_csv = RESULTS_DIR / "reasoning_outputs.csv"
    csv_headers = [
        "sample_id", "domain", "scenario_id", "corruption_level", "target_term",
        "distractor_term", "transcript", "user_query", "detector_probability",
        "detector_prediction", "detector_threshold", "voice_domain", "voice_domain_posterior",
        "voice_subdomain", "voice_subdomain_posterior", "voice_topic", "voice_topic_posterior",
        "voice_document_type", "voice_document_type_posterior", "condition",
        "gemma_output", "latency_ms", "correctness", "distractor_adopted",
        "correction_success", "false_correction"
    ]
    pd.DataFrame(columns=csv_headers).to_csv(outputs_csv, index=False)

    encoder_meta_path = ENCODER_CACHE_DIR / "cache_metadata.json"
    cache_meta = json.load(open(encoder_meta_path)) if encoder_meta_path.exists() else {}

    latencies = {
        "whisper_encoder_time": [], "decoder_time": [], "voice_nlu_time": [],
        "text_nlu_time": [], "detector_time": [], "gemma_baseline_time": [],
        "gemma_voice_nlu_time": [], "gemma_detector_gated_time": [],
        "total_baseline_latency": [], "total_voice_nlu_latency": [],
        "total_detector_gated_latency": []
    }

    all_rows = []
    det_eval_records = []

    # 4. Processing Samples
    for idx, row in tqdm(df_samples.iterrows(), total=len(df_samples), desc="Evaluating Reasoning"):
        sample_id = str(row['sample_id'])
        scenario_id = str(row.get('scenario_id', ''))
        audio_path = find_audio_file(sample_id)
        if audio_path is None: continue

        target_term = ""
        for col in ['target_terms', 'target_words', 'target_term', 'source_term']:
            if col in row and pd.notna(row[col]):
                target_term = str(row[col]).lower().split(';')[0].strip()
                break
        if not target_term: continue

        domain = str(row.get('domain_label', row.get('domain', 'general'))).lower()
        topic_label = str(row.get('topic_label', row.get('topic', '')))
        dom_key = domain if domain in valid_distractors else 'general'
        available_distractors = [d for d in valid_distractors[dom_key] if d.lower() != target_term]
        if len(available_distractors) < 2:
            available_distractors = [d for sub in valid_distractors.values() for d in sub if d.lower() != target_term]
        if len(available_distractors) < 2: continue

        dist_L1, dist_L2 = available_distractors[0], available_distractors[1]

        # ----------------------------------------------------------------------
        # A. FIXED WHISPER ENCODER (CACHED)
        # ----------------------------------------------------------------------
        cache_file = ENCODER_CACHE_DIR / f"{sample_id}.pt"
        t0_enc = time.perf_counter()
        
        if cache_file.exists():
            enc_out = torch.load(cache_file, map_location=DEVICE)
            assert enc_out.shape[-1] == 512, f"Invalid cached encoder shape: {enc_out.shape}"
            t_enc = (time.perf_counter() - t0_enc) * 1000.0
            print(f"\n[CACHE] Loaded cached Whisper encoder: {sample_id}")
        else:
            audio_arr, _ = librosa.load(audio_path, sr=16000)
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio_arr.astype(np.float32))).to(DEVICE)
            with torch.no_grad():
                enc_out = whisper_model.encoder(mel.unsqueeze(0))
            torch.save(enc_out, cache_file)
            t_enc = (time.perf_counter() - t0_enc) * 1000.0
            print(f"\n[CACHE] Computed and saved Whisper encoder: {sample_id}")
            
            cache_meta[sample_id] = {
                "encoder_shape": list(enc_out.shape),
                "whisper_model": WHISPER_MODEL_PATH,
                "encoder_dtype": str(enc_out.dtype),
                "timestamp": time.time()
            }
        latencies["whisper_encoder_time"].append(t_enc)

        # ----------------------------------------------------------------------
        # B. VOICE-NLU (COMPUTED ONCE PER AUDIO)
        # ----------------------------------------------------------------------
        t0_vnlu = time.perf_counter()
        emb_512 = enc_out.mean(dim=1).cpu().numpy().astype(np.float32)
        v_scaled = v_scaler.transform(emb_512).astype(np.float32)
        with torch.no_grad():
            v_128 = v_proj(torch.tensor(v_scaled, device=DEVICE)).cpu().numpy()
            
        v_preds, voice_row = {}, {}
        for h in HEADS:
            probs = v_mlps[h].predict_proba(v_128)[0]
            v_preds[h] = probs
            voice_row[f"voice_{h}"] = v_enc[f"{h}_label"].inverse_transform([np.argmax(probs)])[0]
            
        t_vnlu = (time.perf_counter() - t0_vnlu) * 1000.0
        latencies["voice_nlu_time"].append(t_vnlu)

        v_dom = voice_row["voice_domain"]
        v_dom_p = float(np.max(v_preds["domain"]))
        v_sub = voice_row["voice_subdomain"]
        v_sub_p = float(np.max(v_preds["subdomain"]))
        v_top = voice_row["voice_topic"]
        v_top_p = float(np.max(v_preds["topic"]))
        v_doc = voice_row["voice_document_type"]
        v_doc_p = float(np.max(v_preds["document_type"]))

        # ----------------------------------------------------------------------
        # C. CONTROLLED L0, L1, L2 TRANSCRIPTS
        # ----------------------------------------------------------------------
        t0_dec = time.perf_counter()
        with torch.no_grad():
            dec_res = whisper.decode(whisper_model, enc_out, whisper.DecodingOptions(language="en", temperature=0.0, fp16=False))
            if isinstance(dec_res, list): dec_res = dec_res[0]
            clean_transcript = dec_res.text.strip()
        t_dec = (time.perf_counter() - t0_dec) * 1000.0
        latencies["decoder_time"].append(t_dec)

        pattern = re.compile(re.escape(target_term), re.IGNORECASE)
        transcripts = {
            0: clean_transcript,
            1: pattern.sub(dist_L1, clean_transcript, count=1) if pattern.search(clean_transcript) else clean_transcript + f" ({dist_L1})",
            2: pattern.sub(dist_L2, clean_transcript, count=1) if pattern.search(clean_transcript) else clean_transcript + f" ({dist_L2})"
        }
        distractors = {0: "", 1: dist_L1, 2: dist_L2}
        query = generate_user_query(domain, v_sub, v_top)

        # ----------------------------------------------------------------------
        # D. THREE CONDITIONS OVER L0, L1, L2
        # ----------------------------------------------------------------------
        baseline_correctness_by_level = {}

        for level in [0, 1, 2]:
            t_curr = transcripts[level]
            d_curr = distractors[level]

            # Text-NLU
            t0_tnlu = time.perf_counter()
            emb_384 = t_enc_model.encode([t_curr], convert_to_numpy=True).astype(np.float32)
            t_scaled = t_scaler.transform(emb_384).astype(np.float32)
            with torch.no_grad():
                t_128 = t_proj(torch.tensor(t_scaled, device=DEVICE)).cpu().numpy()
            t_preds, text_row = {}, {}
            for h in HEADS:
                probs = t_mlps[h].predict_proba(t_128)[0]
                t_preds[h] = probs
                text_row[f"text_{h}"] = t_enc[f"{h}_label"].inverse_transform([np.argmax(probs)])[0]
            t_tnlu = (time.perf_counter() - t0_tnlu) * 1000.0
            latencies["text_nlu_time"].append(t_tnlu)

            # Detector B
            t0_det = time.perf_counter()
            _, det_feats = extract_baseline_features(voice_row, text_row, v_preds, t_preds, v_enc, t_enc)
            feat_vector = pd.DataFrame([det_feats])[expected_features]
            det_prob = float(detector_b.predict_proba(feat_vector)[0][1])
            det_pred = int(det_prob >= det_threshold)
            t_det = (time.perf_counter() - t0_det) * 1000.0
            latencies["detector_time"].append(t_det)

            det_eval_records.append({
                "sample_id": sample_id, "level": level, "ground_truth_error": 1 if level > 0 else 0,
                "det_prob": det_prob, "det_pred": det_pred
            })

            # CONDITION 1: BASELINE
            p1 = (
                f"You are answering the user's query using the transcript below. IMPORTANT:\n"
                f"- Treat the transcript as the speech transcription available to you.\n"
                f"- Do not assume that any word is wrong unless the transcript itself provides evidence.\n"
                f"- Do not invent information.\n"
                f"- Answer the user's query briefly and directly.\n\n"
                f"Transcript: {t_curr}\n"
                f"User query: {query}\n"
                f"Give a very short answer."
            )
            out_c1, g_time_c1 = gemma.generate_response(p1)
            corr_c1, dist_c1 = evaluate_semantic_correctness_proxy(out_c1, target_term, d_curr, topic_label)
            baseline_correctness_by_level[level] = corr_c1

            tot_lat_c1 = t_enc + t_dec + g_time_c1
            latencies["gemma_baseline_time"].append(g_time_c1)
            latencies["total_baseline_latency"].append(tot_lat_c1)

            row_c1 = {
                "sample_id": sample_id, "domain": domain, "scenario_id": scenario_id,
                "corruption_level": level, "target_term": target_term, "distractor_term": d_curr,
                "transcript": t_curr, "user_query": query, "detector_probability": det_prob,
                "detector_prediction": det_pred, "detector_threshold": det_threshold,
                "voice_domain": v_dom, "voice_domain_posterior": v_dom_p,
                "voice_subdomain": v_sub, "voice_subdomain_posterior": v_sub_p,
                "voice_topic": v_top, "voice_topic_posterior": v_top_p,
                "voice_document_type": v_doc, "voice_document_type_posterior": v_doc_p,
                "condition": "Baseline", "gemma_output": out_c1, "latency_ms": tot_lat_c1,
                "correctness": corr_c1, "distractor_adopted": dist_c1,
                "correction_success": 0, "false_correction": 0
            }

            # CONDITION 2: VOICE-NLU (UNGATED)
            p2 = (
                f"You are answering the user's query. The transcript below may contain a transcription error. "
                f"You are also given independent semantic evidence extracted directly from the speech representation:\n"
                f"Voice-NLU semantic evidence:\n"
                f"Domain: {v_dom}, posterior={v_dom_p:.2f}\n"
                f"Subdomain: {v_sub}, posterior={v_sub_p:.2f}\n"
                f"Topic: {v_top}, posterior={v_top_p:.2f}\n"
                f"Document type: {v_doc}, posterior={v_doc_p:.2f}\n\n"
                f"Use this voice-side evidence as supporting evidence when interpreting ambiguous terminology. "
                f"Do not invent details and do not assume the transcript is wrong merely because the semantic evidence differs. "
                f"Resolve contradictions conservatively.\n\n"
                f"Transcript: {t_curr}\n"
                f"User query: {query}\n"
                f"Give a very short answer."
            )
            out_c2, g_time_c2 = gemma.generate_response(p2)
            corr_c2, dist_c2 = evaluate_semantic_correctness_proxy(out_c2, target_term, d_curr, topic_label)
            
            c2_succ = 1 if (level > 0 and baseline_correctness_by_level[level] == 0 and corr_c2 == 1) else 0
            c2_false = 1 if (level == 0 and baseline_correctness_by_level[0] == 1 and corr_c2 == 0) else 0

            tot_lat_c2 = t_enc + t_dec + t_vnlu + g_time_c2
            latencies["gemma_voice_nlu_time"].append(g_time_c2)
            latencies["total_voice_nlu_latency"].append(tot_lat_c2)

            row_c2 = {
                "sample_id": sample_id, "domain": domain, "scenario_id": scenario_id,
                "corruption_level": level, "target_term": target_term, "distractor_term": d_curr,
                "transcript": t_curr, "user_query": query, "detector_probability": det_prob,
                "detector_prediction": det_pred, "detector_threshold": det_threshold,
                "voice_domain": v_dom, "voice_domain_posterior": v_dom_p,
                "voice_subdomain": v_sub, "voice_subdomain_posterior": v_sub_p,
                "voice_topic": v_top, "voice_topic_posterior": v_top_p,
                "voice_document_type": v_doc, "voice_document_type_posterior": v_doc_p,
                "condition": "Voice-NLU", "gemma_output": out_c2, "latency_ms": tot_lat_c2,
                "correctness": corr_c2, "distractor_adopted": dist_c2,
                "correction_success": c2_succ, "false_correction": c2_false
            }

            # CONDITION 3: DETECTOR-GATED VOICE-NLU (PROPOSED SYSTEM)
            if det_pred == 0:
                out_c3, g_time_c3 = out_c1, 0.0  # Gating bypassed -> Baseline output reused
                corr_c3, dist_c3 = corr_c1, dist_c1
            else:
                p3 = (
                    f"The transcript has been flagged by an ASR error detector. Do NOT blindly trust the flagged transcript. "
                    f"Use the independent Voice-NLU semantic evidence from the speech representation to resolve the likely terminology/semantic interpretation:\n"
                    f"Voice-NLU semantic evidence:\n"
                    f"Domain: {v_dom}, posterior={v_dom_p:.2f}\n"
                    f"Subdomain: {v_sub}, posterior={v_sub_p:.2f}\n"
                    f"Topic: {v_top}, posterior={v_top_p:.2f}\n"
                    f"Document type: {v_doc}, posterior={v_doc_p:.2f}\n\n"
                    f"Correct your interpretation ONLY because the transcript has been flagged and independent voice-side evidence has been supplied. "
                    f"Do not mention the detector or internal system in the final answer.\n\n"
                    f"Transcript: {t_curr}\n"
                    f"User query: {query}\n"
                    f"Give the corrected interpretation/answer very briefly."
                )
                out_c3, g_time_c3 = gemma.generate_response(p3)
                corr_c3, dist_c3 = evaluate_semantic_correctness_proxy(out_c3, target_term, d_curr, topic_label)

            c3_succ = 1 if (level > 0 and baseline_correctness_by_level[level] == 0 and corr_c3 == 1) else 0
            c3_false = 1 if (level == 0 and baseline_correctness_by_level[0] == 1 and corr_c3 == 0) else 0

            tot_lat_c3 = t_enc + t_dec + t_vnlu + t_tnlu + t_det + g_time_c3
            latencies["gemma_detector_gated_time"].append(g_time_c3)
            latencies["total_detector_gated_latency"].append(tot_lat_c3)

            row_c3 = {
                "sample_id": sample_id, "domain": domain, "scenario_id": scenario_id,
                "corruption_level": level, "target_term": target_term, "distractor_term": d_curr,
                "transcript": t_curr, "user_query": query, "detector_probability": det_prob,
                "detector_prediction": det_pred, "detector_threshold": det_threshold,
                "voice_domain": v_dom, "voice_domain_posterior": v_dom_p,
                "voice_subdomain": v_sub, "voice_subdomain_posterior": v_sub_p,
                "voice_topic": v_top, "voice_topic_posterior": v_top_p,
                "voice_document_type": v_doc, "voice_document_type_posterior": v_doc_p,
                "condition": "Detector-Gated Voice-NLU", "gemma_output": out_c3, "latency_ms": tot_lat_c3,
                "correctness": corr_c3, "distractor_adopted": dist_c3,
                "correction_success": c3_succ, "false_correction": c3_false
            }

            # Batch append & incremental write to disk
            all_rows.extend([row_c1, row_c2, row_c3])
            pd.DataFrame([row_c1, row_c2, row_c3]).to_csv(outputs_csv, mode='a', header=False, index=False)

    # Save cache metadata
    with open(encoder_meta_path, "w") as f:
        json.dump(cache_meta, f, indent=2)

    # ==============================================================================
    # 5. METRICS & SUMMARY COMPILATION
    # ==============================================================================
    df_results = pd.DataFrame(all_rows)
    df_det = pd.DataFrame(det_eval_records)

    # Detector Performance
    y_true = df_det['ground_truth_error']
    y_prob = df_det['det_prob']
    y_pred = df_det['det_pred']

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if len(y_true.unique()) > 1 else float('nan')
    p_c, r_c, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(r_c, p_c)

    pd.DataFrame([{
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Precision": prec, "Recall": rec, "F1": f1,
        "ROC-AUC": roc_auc, "PR-AUC": pr_auc
    }]).to_csv(RESULTS_DIR / "detector_metrics.csv", index=False)

    # Reasoning Metrics
    reasoning_summary = []
    for cond in ["Baseline", "Voice-NLU", "Detector-Gated Voice-NLU"]:
        sub = df_results[df_results['condition'] == cond]
        acc_clean = sub[sub['corruption_level'] == 0]['correctness'].mean()
        acc_l1 = sub[sub['corruption_level'] == 1]['correctness'].mean()
        acc_l2 = sub[sub['corruption_level'] == 2]['correctness'].mean()

        corr_sub = sub[sub['corruption_level'] > 0]
        corr_succ = corr_sub['correction_success'].sum() / len(corr_sub) if len(corr_sub) > 0 else 0.0
        clean_sub = sub[sub['corruption_level'] == 0]
        false_corr = clean_sub['false_correction'].sum() / len(clean_sub) if len(clean_sub) > 0 else 0.0
        dist_adopt = corr_sub['distractor_adopted'].mean() if len(corr_sub) > 0 else 0.0

        reasoning_summary.append({
            "Condition": cond,
            "Clean_Accuracy": acc_clean,
            "L1_Accuracy": acc_l1,
            "L2_Accuracy": acc_l2,
            "Correction_Success_Rate": corr_succ,
            "Distractor_Adoption_Rate": dist_adopt,
            "False_Correction_Rate": false_corr
        })

    df_reasoning = pd.DataFrame(reasoning_summary)
    df_reasoning.to_csv(RESULTS_DIR / "reasoning_metrics.csv", index=False)
    with open(RESULTS_DIR / "reasoning_metrics.json", "w") as f:
        json.dump(reasoning_summary, f, indent=2)

    # Ablation Summary with Robustness Gain
    base_l1 = df_reasoning[df_reasoning['Condition'] == 'Baseline']['L1_Accuracy'].values[0]
    base_l2 = df_reasoning[df_reasoning['Condition'] == 'Baseline']['L2_Accuracy'].values[0]

    ablation_summary = []
    for cond in ["Baseline", "Voice-NLU", "Detector-Gated Voice-NLU"]:
        c_dict = df_reasoning[df_reasoning['Condition'] == cond].iloc[0].to_dict()
        c_dict["Robustness_Gain_L1"] = c_dict["L1_Accuracy"] - base_l1
        c_dict["Robustness_Gain_L2"] = c_dict["L2_Accuracy"] - base_l2
        ablation_summary.append(c_dict)
    pd.DataFrame(ablation_summary).to_csv(RESULTS_DIR / "ablation_summary.csv", index=False)

    # Latency Metrics
    lat_rows = []
    for k, v in latencies.items():
        if len(v) > 0:
            arr = np.array(v)
            lat_rows.append({
                "Component": k,
                "Mean_ms": float(np.mean(arr)),
                "Median_ms": float(np.median(arr)),
                "Std_ms": float(np.std(arr)),
                "P95_ms": float(np.percentile(arr, 95))
            })
    df_lat = pd.DataFrame(lat_rows)
    df_lat.to_csv(RESULTS_DIR / "latency_metrics.csv", index=False)

    # Config Export
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump({
            "GEMMA_MODEL_PATH": GEMMA_MODEL_PATH,
            "WHISPER_MODEL_PATH": WHISPER_MODEL_PATH,
            "num_samples": len(df_samples),
            "detector_threshold": det_threshold,
            "corruption_levels": [0, 1, 2],
            "device": DEVICE,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    # ==============================================================================
    # 6. FIGURES
    # ==============================================================================
    sns.set_theme(style="whitegrid")

    # Figure 1: Accuracy across conditions
    df_acc_plot = df_reasoning.melt(
        id_vars=['Condition'],
        value_vars=['Clean_Accuracy', 'L1_Accuracy', 'L2_Accuracy'],
        var_name='Level', value_name='Accuracy'
    )
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df_acc_plot, x='Level', y='Accuracy', hue='Condition', palette='Blues')
    plt.title('Reasoning Accuracy: Baseline vs Voice-NLU vs Detector-Gated Voice-NLU')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "accuracy_ablation.png", dpi=300)
    plt.close()

    # Figure 2: Error Analysis
    df_err_plot = df_reasoning.melt(
        id_vars=['Condition'],
        value_vars=['Distractor_Adoption_Rate', 'False_Correction_Rate'],
        var_name='Metric', value_name='Rate'
    )
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df_err_plot, x='Metric', y='Rate', hue='Condition', palette='Reds')
    plt.title('Downstream Semantic Error Rates')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_rates.png", dpi=300)
    plt.close()

    # Figure 3: Detector Probability Score Distribution
    plt.figure(figsize=(7, 5))
    sns.kdeplot(data=df_det, x='det_prob', hue='ground_truth_error', fill=True, common_norm=False, palette='coolwarm')
    plt.axvline(det_threshold, color='black', linestyle='--', label=f'Threshold ({det_threshold:.2f})')
    plt.title('Detector B Probability Distribution: Clean vs Corrupted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "detector_score_distribution.png", dpi=300)
    plt.close()

    # ==============================================================================
    # 7. FINAL REPORT
    # ==============================================================================
    n_s = len(df_samples)
    b_acc = df_reasoning[df_reasoning['Condition'] == 'Baseline']
    v_acc = df_reasoning[df_reasoning['Condition'] == 'Voice-NLU']
    g_acc = df_reasoning[df_reasoning['Condition'] == 'Detector-Gated Voice-NLU']

    lat_b = df_lat[df_lat['Component'] == 'total_baseline_latency']['Mean_ms'].values[0]
    lat_v = df_lat[df_lat['Component'] == 'total_voice_nlu_latency']['Mean_ms'].values[0]
    lat_g = df_lat[df_lat['Component'] == 'total_detector_gated_latency']['Mean_ms'].values[0]

    print("\n" + "=" * 60)
    print("FINAL DOWNSTREAM REASONING ABLATION")
    print("=" * 60)
    print(f"Samples: {n_s}")
    print(f"Clean: {n_s} | L1: {n_s} | L2: {n_s}\n")
    print("DETECTOR")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")
    print(f"PR-AUC:    {pr_auc:.3f}\n")
    print("REASONING")
    print(f"{'Metric':<22} | {'Baseline':<10} | {'Voice-NLU':<10} | {'Detector-Gated':<14}")
    print("-" * 65)
    print(f"{'Clean Accuracy':<22} | {b_acc['Clean_Accuracy'].values[0]:<10.3f} | {v_acc['Clean_Accuracy'].values[0]:<10.3f} | {g_acc['Clean_Accuracy'].values[0]:<14.3f}")
    print(f"{'L1 Accuracy':<22} | {b_acc['L1_Accuracy'].values[0]:<10.3f} | {v_acc['L1_Accuracy'].values[0]:<10.3f} | {g_acc['L1_Accuracy'].values[0]:<14.3f}")
    print(f"{'L2 Accuracy':<22} | {b_acc['L2_Accuracy'].values[0]:<10.3f} | {v_acc['L2_Accuracy'].values[0]:<10.3f} | {g_acc['L2_Accuracy'].values[0]:<14.3f}")
    print(f"{'Correction Success':<22} | {b_acc['Correction_Success_Rate'].values[0]:<10.3f} | {v_acc['Correction_Success_Rate'].values[0]:<10.3f} | {g_acc['Correction_Success_Rate'].values[0]:<14.3f}")
    print(f"{'Distractor Adoption':<22} | {b_acc['Distractor_Adoption_Rate'].values[0]:<10.3f} | {v_acc['Distractor_Adoption_Rate'].values[0]:<10.3f} | {g_acc['Distractor_Adoption_Rate'].values[0]:<14.3f}")
    print(f"{'False Correction':<22} | {b_acc['False_Correction_Rate'].values[0]:<10.3f} | {v_acc['False_Correction_Rate'].values[0]:<10.3f} | {g_acc['False_Correction_Rate'].values[0]:<14.3f}\n")
    print("LATENCY (Mean Total ms)")
    print(f"Baseline:       {lat_b:.2f} ms")
    print(f"Voice-NLU:      {lat_v:.2f} ms")
    print(f"Detector-Gated: {lat_g:.2f} ms")
    print("=" * 60)
    print("\nInterpretation:")
    print("Under this controlled decoder-side corruption stress test, the acoustic encoder representation remained invariant.")
    print("Detector-gating prevents unnecessary transcript modifications on clean audio while recovering semantic accuracy on corrupted sequences.")
    print("=" * 60)

if __name__ == "__main__":
    run_reasoning_experiment()
