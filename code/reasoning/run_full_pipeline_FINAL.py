#!/usr/bin/env python3
"""
FINAL END-TO-END ASR -> NLU -> ERROR DETECTOR -> GEMMA PIPELINE

Purpose:
1. Run REAL Whisper audio inference for objective latency measurement.
2. Use the SAVED clean/error transcript pairing for the semantic-error experiment.
   This is necessary because spontaneous clean-audio decoding did not contain enough
   strict semantic-error positives for detector evaluation.
3. Run Voice-NLU from the REAL Whisper encoder representation.
4. Run Text-NLU on the SAVED clean and erroneous transcript pair.
5. Evaluate STRICT ASR-induced semantic errors:
       clean Text-NLU correct on all 4 heads
       AND erroneous Text-NLU wrong on >=1 head.
6. Evaluate Detector A vs Detector B.
7. Run three Gemma 3 1B reasoning conditions:
       A: erroneous transcript only
       B: detector-gated transcript + Voice-NLU evidence when flagged
       C: transcript + Voice-NLU evidence
   Ground-truth labels are NEVER supplied to Gemma.
8. Save complete results and latency statistics.

IMPORTANT:
- No retraining.
- No split modification.
- No ground-truth labels are used as runtime features.
- Ground-truth labels are used only after generation for offline evaluation.
- The saved erroneous transcript is used for the semantic detector/reasoning
  experiment; REAL Whisper decoding is still performed from audio so encoder,
  decoder and end-to-end latency are measured.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import jiwer
import whisper

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(
    "/home/spark2/users/intern/Atreyee-Das/NLU_Robust_Experiment"
)

MODELS_DIR = PROJECT_ROOT / "models"
DETECTOR_DIR = PROJECT_ROOT / "detector"
DATASET_DIR = PROJECT_ROOT / "dataset"
AUDIO_DIR = PROJECT_ROOT / "audio"
RESULTS_DIR = PROJECT_ROOT / "runtime_results"

WHISPER_PATH = "/home/spark2/Models/base.en.pt"
MINILM_PATH = "/home/spark2/Models/all-MiniLM-L6-v2"
GEMMA_PATH = "/home/spark2/Models/gemma_2_models/gemma-3-1b-it"

MASTER_CSV = (
    DATASET_DIR /
    "nlu_robust_6000_scenario_paraphrase_FINAL_70_10_20.csv"
)

# This MUST be the repaired paired ASR file.
ASR_PAIR_CANDIDATES = [
    DATASET_DIR / "whisper_domain_multitarget_FINAL_REPAIRED_6000.csv",
    DATASET_DIR / "whisper_domain_multitarget_final_70_10_20.csv",
]

HEADS = ["domain", "subdomain", "topic", "document_type"]

WARMUP_RUNS = 20
MEASURED_RUNS = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(
        "None of the ASR paired datasets exists:\n" +
        "\n".join(str(x) for x in paths)
    )


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


# ============================================================
# PROJECTION ARCHITECTURES
# ============================================================

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


# ============================================================
# DETECTOR FEATURE RECONSTRUCTION
# IMPORTANT: probabilities are aligned using the actual
# classifier classes, not by slicing encoder.classes_.
# ============================================================

EPS = 1e-12
HEAD_WEIGHTS = {
    "domain": 0.20,
    "subdomain": 0.25,
    "topic": 0.40,
    "document_type": 0.15,
}


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def margin(p):
    p = np.sort(np.asarray(p))[::-1]
    return float(p[0] - p[1]) if len(p) > 1 else 1.0


def aligned_js(vp, vc, tp, tc):
    vmap = {c: float(p) for c, p in zip(vc, vp)}
    tmap = {c: float(p) for c, p in zip(tc, tp)}
    classes = sorted(set(vmap) | set(tmap))

    va = np.asarray([vmap.get(c, EPS) for c in classes], dtype=float)
    ta = np.asarray([tmap.get(c, EPS) for c in classes], dtype=float)

    va = np.clip(va / va.sum(), EPS, 1.0)
    ta = np.clip(ta / ta.sum(), EPS, 1.0)

    d = jensenshannon(va, ta, base=2)
    return 0.0 if np.isnan(d) else float(d ** 2)


def actual_label_classes(mlp, label_encoder):
    """
    Convert the MLP's integer class indices back to the semantic labels.
    This preserves the exact class-to-probability mapping.
    """
    return np.asarray(
        label_encoder.inverse_transform(
            np.asarray(mlp.classes_, dtype=int)
        )
    )


def extract_detector_features(
    voice_preds,
    text_preds,
    voice_probs,
    text_probs,
    voice_mlps,
    text_mlps,
    label_encoders,
):
    feat_a = {}
    feat_b = {}

    total_disagreement = 0.0
    weighted_disagreement = 0.0

    v_confs = []
    t_confs = []
    cross_support = []

    for h in HEADS:

        dis = float(voice_preds[h] != text_preds[h])

        feat_a[f"{h}_disagreement"] = dis
        total_disagreement += dis
        weighted_disagreement += HEAD_WEIGHTS[h] * dis

        vp = voice_probs[h]
        tp = text_probs[h]

        vc = actual_label_classes(
            voice_mlps[h],
            label_encoders[f"{h}_label"]
        )
        tc = actual_label_classes(
            text_mlps[h],
            label_encoders[f"{h}_label"]
        )

        v_top = float(np.max(vp))
        t_top = float(np.max(tp))

        v_confs.append(v_top)
        t_confs.append(t_top)

        v_idx_text = np.where(vc == text_preds[h])[0]
        t_idx_voice = np.where(tc == voice_preds[h])[0]

        v_prob_text = (
            float(vp[v_idx_text[0]])
            if len(v_idx_text) else 0.0
        )
        t_prob_voice = (
            float(tp[t_idx_voice[0]])
            if len(t_idx_voice) else 0.0
        )

        cross_support.extend([
            v_prob_text,
            t_prob_voice
        ])

        feat_b[f"{h}_voice_top1_confidence"] = v_top
        feat_b[f"{h}_text_top1_confidence"] = t_top
        feat_b[f"{h}_confidence_gap"] = abs(v_top - t_top)
        feat_b[f"{h}_text_prob_of_voice_label"] = t_prob_voice
        feat_b[f"{h}_voice_prob_of_text_label"] = v_prob_text
        feat_b[f"{h}_js_divergence"] = aligned_js(
            vp, vc, tp, tc
        )
        feat_b[f"{h}_voice_entropy"] = entropy(vp)
        feat_b[f"{h}_text_entropy"] = entropy(tp)
        feat_b[f"{h}_voice_margin"] = margin(vp)
        feat_b[f"{h}_text_margin"] = margin(tp)

    feat_a["total_disagreements"] = total_disagreement
    feat_a["weighted_disagreement"] = weighted_disagreement

    feat_b.update(feat_a)

    feat_b["mean_voice_confidence"] = float(np.mean(v_confs))
    feat_b["mean_text_confidence"] = float(np.mean(t_confs))
    feat_b["mean_cross_model_support"] = float(
        np.mean(cross_support)
    )

    return feat_a, feat_b


# ============================================================
# NLU INFERENCE
# ============================================================

def run_nlu(
    embedding,
    scaler,
    projection,
    mlps,
    encoders,
):
    scaled = scaler.transform(
        np.asarray(embedding, dtype=np.float32)
    )

    with torch.no_grad():
        z = projection(
            torch.tensor(
                scaled,
                dtype=torch.float32,
                device=DEVICE
            )
        ).cpu().numpy()

    preds = {}
    probs = {}

    for h in HEADS:
        p = mlps[h].predict_proba(z)[0]
        probs[h] = p

        class_idx = mlps[h].classes_[np.argmax(p)]

        preds[h] = encoders[
            f"{h}_label"
        ].inverse_transform(
            [class_idx]
        )[0]

    return preds, probs, z


# ============================================================
# GEMMA
# ============================================================

def generate_gemma(
    prompt,
    tokenizer,
    model,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    sync()
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
        )

    sync()

    latency_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    generated = outputs[
        :,
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True
    )[0].strip()

    return answer, latency_ms


# ============================================================
# ENVIRONMENT / ARTIFACT VALIDATION
# ============================================================

def validate():
    required = [
        WHISPER_PATH,
        MINILM_PATH,
        GEMMA_PATH,
        MASTER_CSV,
        first_existing(ASR_PAIR_CANDIDATES),
        MODELS_DIR /
            "best_voice_projection_FINAL_70_10_20.pt",
        MODELS_DIR /
            "best_text_projection_FINAL_70_10_20.pt",
        MODELS_DIR /
            "voice_whisper_scaler_FINAL_70_10_20.joblib",
        MODELS_DIR /
            "text_scaler_FINAL_70_10_20.joblib",
        MODELS_DIR /
            "shared_label_encoders_FINAL_70_10_20.joblib",
        DETECTOR_DIR /
            "Detector_A_STRICT_ASR_INDUCED.joblib",
        DETECTOR_DIR /
            "Detector_B_STRICT_ASR_INDUCED.joblib",
        DETECTOR_DIR /
            "strict_detector_thresholds.json",
    ]

    for h in HEADS:
        required += [
            MODELS_DIR /
            f"voice_{h}_label_mlp_FINAL_70_10_20.joblib",
            MODELS_DIR /
            f"text_{h}_label_mlp_FINAL_70_10_20.joblib",
        ]

    missing = [
        str(x)
        for x in required
        if not Path(x).exists()
    ]

    if missing:
        print("MISSING ARTIFACTS:")
        print("\n".join(missing))
        raise SystemExit(1)

    print("ARTIFACT VALIDATION PASSED")


# ============================================================
# MAIN
# ============================================================

def main():

    validate()

    paired_csv = first_existing(
        ASR_PAIR_CANDIDATES
    )

    print("\n==================================================")
    print("FINAL END-TO-END PIPELINE")
    print("==================================================")
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    print(f"Master dataset: {MASTER_CSV}")
    print(f"Paired ASR dataset: {paired_csv}")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    master = pd.read_csv(MASTER_CSV)
    paired = pd.read_csv(paired_csv)

    master["sample_id"] = master["sample_id"].astype(str)
    paired["sample_id"] = paired["sample_id"].astype(str)

    master_unseen = master[
        master["split"].astype(str) == "unseen"
    ].copy()

    # Pair by sample_id.
    pair_cols = [
        "sample_id",
        "corrupted_whisper_transcript",
        "clean_whisper_transcript",
    ]

    available_pair_cols = [
        c for c in pair_cols
        if c in paired.columns
    ]

    if "corrupted_whisper_transcript" not in available_pair_cols:
        raise RuntimeError(
            "The paired ASR CSV does not contain "
            "'corrupted_whisper_transcript'."
        )

    paired_small = paired[
        available_pair_cols
    ].drop_duplicates("sample_id")

    df = master_unseen.merge(
        paired_small,
        on="sample_id",
        how="left"
    )

    missing_pair = df[
        df["corrupted_whisper_transcript"].isna()
    ]

    if len(missing_pair):
        raise RuntimeError(
            f"{len(missing_pair)} unseen samples have no "
            "paired erroneous transcript."
        )

    print(f"Unseen samples: {len(df)}")

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print("\nLoading Whisper...")
    whisper_model = whisper.load_model(
        WHISPER_PATH,
        device=DEVICE
    )
    whisper_model.eval()

    print("Loading MiniLM...")
    minilm = SentenceTransformer(
        MINILM_PATH,
        device=DEVICE
    )

    print("Loading Gemma 3 1B...")
    tokenizer = AutoTokenizer.from_pretrained(
        GEMMA_PATH,
        local_files_only=True
    )

    dtype = (
        torch.bfloat16
        if DEVICE == "cuda"
        else torch.float32
    )

    gemma = AutoModelForCausalLM.from_pretrained(
        GEMMA_PATH,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto"
    )
    gemma.eval()

    # --------------------------------------------------------
    # NLU ARTIFACTS
    # --------------------------------------------------------

    v_scaler = joblib.load(
        MODELS_DIR /
        "voice_whisper_scaler_FINAL_70_10_20.joblib"
    )

    t_scaler = joblib.load(
        MODELS_DIR /
        "text_scaler_FINAL_70_10_20.joblib"
    )

    encoders = joblib.load(
        MODELS_DIR /
        "shared_label_encoders_FINAL_70_10_20.joblib"
    )

    v_proj = VoiceHierarchicalProjection().to(DEVICE)
    v_proj.load_state_dict(
        torch.load(
            MODELS_DIR /
            "best_voice_projection_FINAL_70_10_20.pt",
            map_location=DEVICE
        )
    )
    v_proj.eval()

    t_proj = TextHierarchicalProjection().to(DEVICE)
    t_proj.load_state_dict(
        torch.load(
            MODELS_DIR /
            "best_text_projection_FINAL_70_10_20.pt",
            map_location=DEVICE
        )
    )
    t_proj.eval()

    v_mlps = {
        h: joblib.load(
            MODELS_DIR /
            f"voice_{h}_label_mlp_FINAL_70_10_20.joblib"
        )
        for h in HEADS
    }

    t_mlps = {
        h: joblib.load(
            MODELS_DIR /
            f"text_{h}_label_mlp_FINAL_70_10_20.joblib"
        )
        for h in HEADS
    }

    # --------------------------------------------------------
    # DETECTORS
    # --------------------------------------------------------

    detector_A = joblib.load(
        DETECTOR_DIR /
        "Detector_A_STRICT_ASR_INDUCED.joblib"
    )

    detector_B = joblib.load(
        DETECTOR_DIR /
        "Detector_B_STRICT_ASR_INDUCED.joblib"
    )

    with open(
        DETECTOR_DIR /
        "strict_detector_thresholds.json"
    ) as f:
        thresholds = json.load(f)

    thresh_A = float(
        thresholds["threshold_A"]
    )
    thresh_B = float(
        thresholds["threshold_B"]
    )

    print(
        f"\nDetector thresholds: "
        f"A={thresh_A:.4f}, B={thresh_B:.4f}"
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []
    latency_rows = []

    warmups_done = 0
    measured_done = 0

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Running final unseen pipeline"
    ):

        sample_id = str(row["sample_id"])
        audio_path = (
            AUDIO_DIR /
            f"{sample_id}.wav"
        )

        if not audio_path.exists():
            print(
                f"\nWARNING: missing audio "
                f"{audio_path}; skipping."
            )
            continue

        gt_text = clean_text(
            row.get("ground_truth", "")
        )

        # This is the saved erroneous transcript pair.
        # It is used for the semantic detector experiment.
        erroneous_text = clean_text(
            row["corrupted_whisper_transcript"]
        )

        query = clean_text(
            row.get("user_query", "")
        )

        if not query:
            query = (
                "Answer the user's request using "
                "the supplied speech transcript."
            )

        # ====================================================
        # REAL AUDIO -> WHISPER
        # ====================================================

        t0 = time.perf_counter()

        audio = whisper.load_audio(
            str(audio_path)
        )

        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(
            audio
        ).to(DEVICE)

        sync()

        audio_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        # ====================================================
        # REAL WHISPER ENCODER -> VOICE NLU
        # ====================================================

        t0 = time.perf_counter()

        with torch.no_grad():
            enc = whisper_model.encoder(
                mel.unsqueeze(0)
            )

            # EXACT pooling used by the current runtime:
            # final encoder sequence -> mean over time.
            emb512 = (
                enc.mean(dim=1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        sync()

        encoder_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        # Voice NLU
        t0 = time.perf_counter()

        voice_preds, voice_probs, voice_z = run_nlu(
            emb512,
            v_scaler,
            v_proj,
            v_mlps,
            encoders
        )

        sync()

        voice_nlu_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        # ====================================================
        # REAL WHISPER DECODER
        # USED FOR TRUE DECODER LATENCY
        # ====================================================

        t0 = time.perf_counter()

        decode_options = whisper.DecodingOptions(
            fp16=(DEVICE == "cuda"),
            temperature=0.0,
            language="en"
        )

        with torch.no_grad():
            decoded_real = whisper.decode(
                whisper_model,
                mel,
                decode_options
            ).text.strip()

        sync()

        decoder_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        # IMPORTANT:
        # decoded_real is retained for reporting/WER.
        # erroneous_text is used for the controlled semantic
        # detector/reasoning experiment.

        # ====================================================
        # CLEAN GROUND-TRUTH TEXT -> TEXT NLU
        # OFFLINE LABEL REFERENCE ONLY
        # ====================================================

        clean_emb = minilm.encode(
            [gt_text],
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype(np.float32)

        clean_text_preds, clean_text_probs, _ = run_nlu(
            clean_emb,
            t_scaler,
            t_proj,
            t_mlps,
            encoders
        )

        # ====================================================
        # ERRONEOUS TRANSCRIPT -> TEXT NLU
        # THIS IS THE ACTUAL DETECTOR INPUT
        # ====================================================

        t0 = time.perf_counter()

        err_emb = minilm.encode(
            [erroneous_text],
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype(np.float32)

        text_preds, text_probs, _ = run_nlu(
            err_emb,
            t_scaler,
            t_proj,
            t_mlps,
            encoders
        )

        sync()

        text_nlu_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        # ====================================================
        # STRICT OFFLINE GROUND TRUTH
        # ====================================================

        gt_labels = {
            h: clean_text(
                row.get(f"{h}_label", "")
            )
            for h in HEADS
        }

        clean_text_correct = int(
            all(
                str(clean_text_preds[h]) ==
                gt_labels[h]
                for h in HEADS
            )
        )

        erroneous_text_correct = int(
            all(
                str(text_preds[h]) ==
                gt_labels[h]
                for h in HEADS
            )
        )

        strict_error = int(
            clean_text_correct == 1
            and erroneous_text_correct == 0
        )

        # ====================================================
        # DETECTOR
        # ====================================================

        t0 = time.perf_counter()

        feat_a, feat_b = extract_detector_features(
            voice_preds,
            text_preds,
            voice_probs,
            text_probs,
            v_mlps,
            t_mlps,
            encoders
        )

        df_a = pd.DataFrame(
            [feat_a]
        )[
            detector_A.feature_names_in_
        ]

        df_b = pd.DataFrame(
            [feat_b]
        )[
            detector_B.feature_names_in_
        ]

        prob_a = float(
            detector_A.predict_proba(df_a)[0, 1]
        )

        prob_b = float(
            detector_B.predict_proba(df_b)[0, 1]
        )

        pred_a = int(
            prob_a >= thresh_A
        )

        pred_b = int(
            prob_b >= thresh_B
        )

        sync()

        detector_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        # ====================================================
        # GEMMA REASONING
        # NO GROUND-TRUTH LABELS IN PROMPTS
        # ====================================================

        base_prompt = f"""
You are a concise reasoning assistant.

Answer the user's request using the speech transcript.
Do not invent facts.
Do not assume a transcript word is wrong without evidence.

Transcript:
{erroneous_text}

User request:
{query}

Answer briefly and directly.
""".strip()

        gated_prompt = f"""
You are a concise reasoning assistant.

The speech transcript has been flagged as potentially unreliable.
Use the transcript, but use the independent acoustic semantic evidence
below to resolve ambiguity when the evidence supports it.

Acoustic semantic evidence:
Domain: {voice_preds["domain"]}
Subdomain: {voice_preds["subdomain"]}
Topic: {voice_preds["topic"]}
Document Type: {voice_preds["document_type"]}

Do not blindly override the transcript.
Do not invent facts.

Transcript:
{erroneous_text}

User request:
{query}

Answer briefly and directly.
""".strip()

        voice_prompt = f"""
You are a concise reasoning assistant.

Answer the user's request using the transcript together with an
independent semantic interpretation derived from the acoustic speech
representation.

Acoustic semantic evidence:
Domain: {voice_preds["domain"]}
Subdomain: {voice_preds["subdomain"]}
Topic: {voice_preds["topic"]}
Document Type: {voice_preds["document_type"]}

Use this evidence only when it supports resolving ambiguity.
Do not blindly override the transcript.
Do not invent facts.

Transcript:
{erroneous_text}

User request:
{query}

Answer briefly and directly.
""".strip()

        # Baseline
        ans_base, gemma_base_ms = generate_gemma(
            base_prompt,
            tokenizer,
            gemma
        )

        # Always voice-grounded
        ans_voice, gemma_voice_ms = generate_gemma(
            voice_prompt,
            tokenizer,
            gemma
        )

        # Detector-gated condition
        if pred_b == 1:
            ans_gated, gemma_gated_ms = generate_gemma(
                gated_prompt,
                tokenizer,
                gemma
            )
        else:
            ans_gated = ans_base
            gemma_gated_ms = 0.0

        # ====================================================
        # WER / CER OF REAL WHISPER DECODER
        # ====================================================

        real_wer = jiwer.wer(
            gt_text,
            decoded_real
        )

        real_cer = jiwer.cer(
            gt_text,
            decoded_real
        )

        # WER/CER of the saved erroneous semantic transcript
        paired_wer = jiwer.wer(
            gt_text,
            erroneous_text
        )

        paired_cer = jiwer.cer(
            gt_text,
            erroneous_text
        )

        # ====================================================
        # LATENCY
        # ====================================================

        speech_to_decision_ms = (
            audio_ms
            + encoder_ms
            + voice_nlu_ms
            + decoder_ms
            + text_nlu_ms
            + detector_ms
        )

        baseline_total_ms = (
            audio_ms
            + encoder_ms
            + decoder_ms
            + gemma_base_ms
        )

        voice_grounded_total_ms = (
            audio_ms
            + encoder_ms
            + decoder_ms
            + voice_nlu_ms
            + gemma_voice_ms
        )

        gated_total_ms = (
            speech_to_decision_ms
            + gemma_gated_ms
        )

        timers = {
            "sample_id": sample_id,
            "audio_load_ms": audio_ms,
            "whisper_encoder_ms": encoder_ms,
            "voice_nlu_ms": voice_nlu_ms,
            "whisper_decoder_ms": decoder_ms,
            "text_nlu_ms": text_nlu_ms,
            "detector_ms": detector_ms,
            "gemma_baseline_ms": gemma_base_ms,
            "gemma_gated_ms": gemma_gated_ms,
            "gemma_voice_ms": gemma_voice_ms,
            "speech_to_decision_ms": speech_to_decision_ms,
            "total_baseline_ms": baseline_total_ms,
            "total_gated_ms": gated_total_ms,
            "total_voice_grounded_ms": voice_grounded_total_ms,
        }

        # Warm-up samples are executed but not included in latency stats.
        if warmups_done < WARMUP_RUNS:
            warmups_done += 1
        elif measured_done < MEASURED_RUNS:
            latency_rows.append(timers)
            measured_done += 1

        # ====================================================
        # SAVE SAMPLE RESULT
        # ====================================================

        out = {
            "sample_id": sample_id,
            "scenario_id": row.get("scenario_id", ""),
            "split": "unseen",

            "ground_truth": gt_text,
            "real_whisper_transcript": decoded_real,
            "paired_erroneous_transcript": erroneous_text,

            "real_whisper_WER": real_wer,
            "real_whisper_CER": real_cer,
            "paired_transcript_WER": paired_wer,
            "paired_transcript_CER": paired_cer,

            "clean_text_semantically_correct": clean_text_correct,
            "erroneous_text_semantically_correct": erroneous_text_correct,
            "strict_asr_induced_error": strict_error,

            "detector_A_probability": prob_a,
            "detector_A_prediction": pred_a,
            "detector_B_probability": prob_b,
            "detector_B_prediction": pred_b,

            "gemma_baseline_answer": ans_base,
            "gemma_detector_gated_answer": ans_gated,
            "gemma_voice_grounded_answer": ans_voice,
        }

        for h in HEADS:
            out[f"gt_{h}"] = gt_labels[h]
            out[f"voice_pred_{h}"] = voice_preds[h]
            out[f"text_clean_pred_{h}"] = clean_text_preds[h]
            out[f"text_erroneous_pred_{h}"] = text_preds[h]
            out[f"voice_confidence_{h}"] = float(
                np.max(voice_probs[h])
            )
            out[f"text_confidence_{h}"] = float(
                np.max(text_probs[h])
            )

        results.append(out)

    # ========================================================
    # SAVE
    # ========================================================

    res = pd.DataFrame(results)

    res.to_csv(
        RESULTS_DIR / "inference_results_FINAL.csv",
        index=False
    )

    lat = pd.DataFrame(latency_rows)

    lat.to_csv(
        RESULTS_DIR / "LATENCY_BREAKDOWN_FINAL.csv",
        index=False
    )

    if len(lat):
        summary = lat.describe(
            percentiles=[0.50, 0.90, 0.95, 0.99]
        ).T
    else:
        summary = pd.DataFrame()

    summary.to_csv(
        RESULTS_DIR / "LATENCY_SUMMARY_FINAL.csv"
    )

    # ========================================================
    # FINAL OBJECTIVE METRICS
    # ========================================================

    u = res.copy()

    # ---- NLU ----
    nlu_rows = []

    for h in HEADS:
        nlu_rows.append({
            "head": h,
            "voice_accuracy":
                accuracy_score(
                    u[f"gt_{h}"],
                    u[f"voice_pred_{h}"]
                ),
            "voice_macro_f1":
                f1_score(
                    u[f"gt_{h}"],
                    u[f"voice_pred_{h}"],
                    average="macro",
                    zero_division=0
                ),
            "text_clean_accuracy":
                accuracy_score(
                    u[f"gt_{h}"],
                    u[f"text_clean_pred_{h}"]
                ),
            "text_clean_macro_f1":
                f1_score(
                    u[f"gt_{h}"],
                    u[f"text_clean_pred_{h}"],
                    average="macro",
                    zero_division=0
                ),
            "text_erroneous_accuracy":
                accuracy_score(
                    u[f"gt_{h}"],
                    u[f"text_erroneous_pred_{h}"]
                ),
            "text_erroneous_macro_f1":
                f1_score(
                    u[f"gt_{h}"],
                    u[f"text_erroneous_pred_{h}"],
                    average="macro",
                    zero_division=0
                ),
        })

    nlu_df = pd.DataFrame(nlu_rows)

    nlu_df.to_csv(
        RESULTS_DIR /
        "NLU_RUNTIME_RESULTS_FINAL.csv",
        index=False
    )

    # ---- ASR ----
    asr_df = pd.DataFrame({
        "real_decoder_WER":
            [u["real_whisper_WER"].mean()],
        "real_decoder_CER":
            [u["real_whisper_CER"].mean()],
        "paired_error_WER":
            [u["paired_transcript_WER"].mean()],
        "paired_error_CER":
            [u["paired_transcript_CER"].mean()],
    })

    asr_df.to_csv(
        RESULTS_DIR /
        "ASR_RUNTIME_METRICS_FINAL.csv",
        index=False
    )

    # ---- STRICT DETECTOR ----
    y = u[
        "strict_asr_induced_error"
    ].astype(int).values

    detector_metrics = []

    for name in ["A", "B"]:

        pred = u[
            f"detector_{name}_prediction"
        ].astype(int).values

        prob = u[
            f"detector_{name}_probability"
        ].astype(float).values

        tn, fp, fn, tp = confusion_matrix(
            y,
            pred,
            labels=[0, 1]
        ).ravel()

        if len(np.unique(y)) == 2:
            roc = roc_auc_score(
                y,
                prob
            )
            pr_auc = average_precision_score(
                y,
                prob
            )
        else:
            roc = np.nan
            pr_auc = np.nan

        detector_metrics.append({
            "detector": f"Detector_{name}",
            "positive_count": int(y.sum()),
            "negative_count": int((1-y).sum()),
            "accuracy":
                accuracy_score(y, pred),
            "precision":
                precision_score(
                    y, pred, zero_division=0
                ),
            "recall":
                recall_score(
                    y, pred, zero_division=0
                ),
            "f1":
                f1_score(
                    y, pred, zero_division=0
                ),
            "specificity":
                tn / (tn + fp)
                if (tn + fp) else 0.0,
            "false_positive_rate":
                fp / (tn + fp)
                if (tn + fp) else 0.0,
            "roc_auc": roc,
            "pr_auc": pr_auc,
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        })

    det_df = pd.DataFrame(
        detector_metrics
    )

    det_df.to_csv(
        RESULTS_DIR /
        "DETECTOR_RUNTIME_RESULTS_FINAL.csv",
        index=False
    )

    # ---- REASONING OUTPUTS ----
    # No authoritative reasoning-answer field exists in the supplied dataset.
    # Therefore DO NOT manufacture an answer-accuracy metric.
    reasoning_summary = {
        "evaluation_status":
            "GENERATED_BUT_NOT_AUTOMATICALLY_SCORED",
        "reason":
            "The supplied dataset contains semantic labels, not authoritative "
            "free-form reasoning answers. Gemma responses are saved for manual "
            "or separate judge-based evaluation.",
        "conditions": [
            "transcript_only",
            "detector_gated",
            "voice_grounded"
        ],
        "sample_count": int(len(u))
    }

    with open(
        RESULTS_DIR /
        "REASONING_RESULTS_FINAL.json",
        "w"
    ) as f:
        json.dump(
            reasoning_summary,
            f,
            indent=2
        )

    # Save just the reasoning table too.
    u[
        [
            "sample_id",
            "strict_asr_induced_error",
            "detector_B_prediction",
            "detector_B_probability",
            "gemma_baseline_answer",
            "gemma_detector_gated_answer",
            "gemma_voice_grounded_answer",
        ]
    ].to_csv(
        RESULTS_DIR /
        "REASONING_RESPONSES_FINAL.csv",
        index=False
    )

    # ========================================================
    # LATENCY SUMMARY
    # ========================================================

    if len(lat):

        latency_summary = pd.DataFrame({
            "metric": lat.columns,
            "mean_ms":
                [lat[c].mean() for c in lat.columns],
            "median_ms":
                [lat[c].median() for c in lat.columns],
            "std_ms":
                [lat[c].std() for c in lat.columns],
            "P50_ms":
                [lat[c].quantile(.50) for c in lat.columns],
            "P90_ms":
                [lat[c].quantile(.90) for c in lat.columns],
            "P95_ms":
                [lat[c].quantile(.95) for c in lat.columns],
            "P99_ms":
                [lat[c].quantile(.99) for c in lat.columns],
            "min_ms":
                [lat[c].min() for c in lat.columns],
            "max_ms":
                [lat[c].max() for c in lat.columns],
        })

        latency_summary.to_csv(
            RESULTS_DIR /
            "LATENCY_OBJECTIVE_SUMMARY_FINAL.csv",
            index=False
        )

    # ========================================================
    # RUNTIME CONFIG
    # ========================================================

    runtime_config = {
        "project_root": str(PROJECT_ROOT),
        "master_dataset": str(MASTER_CSV),
        "paired_dataset": str(paired_csv),
        "device": DEVICE,
        "whisper_path": WHISPER_PATH,
        "minilm_path": MINILM_PATH,
        "gemma_path": GEMMA_PATH,
        "split": "unseen",
        "warmup_runs": WARMUP_RUNS,
        "measured_runs_requested": MEASURED_RUNS,
        "actual_measured_runs": len(lat),
        "whisper_decode_temperature": 0.0,
        "whisper_language": "en",
        "strict_error_definition":
            "clean Text-NLU correct on all four heads AND "
            "paired erroneous Text-NLU wrong on at least one head",
        "ground_truth_used_as_runtime_feature": False,
        "detector_threshold_A": thresh_A,
        "detector_threshold_B": thresh_B,
    }

    with open(
        RESULTS_DIR /
        "runtime_config_FINAL.json",
        "w"
    ) as f:
        json.dump(
            runtime_config,
            f,
            indent=2
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    topic_voice = nlu_df[
        nlu_df.head == "topic"
    ].iloc[0]

    topic_text = topic_voice

    b = det_df[
        det_df.detector == "Detector_B"
    ].iloc[0]

    a = det_df[
        det_df.detector == "Detector_A"
    ].iloc[0]

    strict_rate = u[
        "strict_asr_induced_error"
    ].mean()

    print("\n")
    print("=" * 65)
    print("FINAL RESEARCH SUMMARY — UNSEEN 90 SCENARIOS")
    print("=" * 65)

    print(
        f"A. Voice-NLU Topic Macro-F1: "
        f"{topic_voice['voice_macro_f1']:.3f}"
    )

    print(
        f"B. Text-NLU Topic Macro-F1 "
        f"(clean text): "
        f"{topic_text['text_clean_macro_f1']:.3f}"
    )

    print(
        f"C. Text-NLU Topic Macro-F1 "
        f"(erroneous transcript): "
        f"{topic_text['text_erroneous_macro_f1']:.3f}"
    )

    print(
        f"D. Real Whisper decoder WER: "
        f"{u['real_whisper_WER'].mean():.3f}"
    )

    print(
        f"E. Real Whisper decoder CER: "
        f"{u['real_whisper_CER'].mean():.3f}"
    )

    print(
        f"F. STRICT ASR-induced semantic-error rate: "
        f"{strict_rate:.1%} "
        f"({int(u['strict_asr_induced_error'].sum())}/{len(u)})"
    )

    print(
        f"G. Detector A ROC-AUC: "
        f"{a['roc_auc']:.3f} | "
        f"F1: {a['f1']:.3f}"
    )

    print(
        f"H. Detector B ROC-AUC: "
        f"{b['roc_auc']:.3f} | "
        f"F1: {b['f1']:.3f}"
    )

    if not np.isnan(a["roc_auc"]) and not np.isnan(b["roc_auc"]):
        print(
            f"I. Detector B - A ROC-AUC: "
            f"{b['roc_auc'] - a['roc_auc']:+.3f}"
        )

    if len(lat):
        for c in [
            "total_baseline_ms",
            "total_gated_ms",
            "total_voice_grounded_ms",
        ]:
            print(
                f"{c}: "
                f"{lat[c].mean():.1f} ms "
                f"(P95={lat[c].quantile(.95):.1f} ms)"
            )

    if DEVICE == "cuda":
        peak_gb = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 3)
        )
        print(
            f"Peak GPU reserved: "
            f"{peak_gb:.2f} GB"
        )

    print("=" * 65)

    final_summary = {
        "unseen_samples": int(len(u)),
        "strict_asr_induced_error_rate":
            float(strict_rate),
        "strict_asr_induced_error_count":
            int(u["strict_asr_induced_error"].sum()),
        "real_whisper_mean_WER":
            float(u["real_whisper_WER"].mean()),
        "real_whisper_mean_CER":
            float(u["real_whisper_CER"].mean()),
        "detector_A_ROC_AUC":
            None if np.isnan(a["roc_auc"])
            else float(a["roc_auc"]),
        "detector_A_F1":
            float(a["f1"]),
        "detector_B_ROC_AUC":
            None if np.isnan(b["roc_auc"])
            else float(b["roc_auc"]),
        "detector_B_F1":
            float(b["f1"]),
        "reasoning_evaluation":
            "responses saved; no authoritative free-form reasoning reference",
        "measured_latency_samples":
            int(len(lat)),
    }

    if len(lat):
        final_summary.update({
            "baseline_mean_ms":
                float(lat["total_baseline_ms"].mean()),
            "baseline_P95_ms":
                float(lat["total_baseline_ms"].quantile(.95)),
            "gated_mean_ms":
                float(lat["total_gated_ms"].mean()),
            "gated_P95_ms":
                float(lat["total_gated_ms"].quantile(.95)),
            "voice_grounded_mean_ms":
                float(lat["total_voice_grounded_ms"].mean()),
            "voice_grounded_P95_ms":
                float(lat["total_voice_grounded_ms"].quantile(.95)),
        })

    if DEVICE == "cuda":
        final_summary["peak_gpu_reserved_GB"] = float(
            torch.cuda.max_memory_reserved()
            / (1024 ** 3)
        )

    with open(
        RESULTS_DIR /
        "FINAL_RUNTIME_SUMMARY.json",
        "w"
    ) as f:
        json.dump(
            final_summary,
            f,
            indent=2
        )

    with open(
        RESULTS_DIR /
        "FINAL_RUNTIME_SUMMARY.txt",
        "w"
    ) as f:
        f.write(
            json.dumps(
                final_summary,
                indent=2
            )
        )

    print(
        "\nDONE. Results saved to:\n"
        f"{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
