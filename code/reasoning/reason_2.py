# -*- coding: utf-8 -*-

"""
reason.py
FINAL FULL-TIME INFERENCE PIPELINE
NLU_Robust_Experiment

Purpose
-------
Run the complete inference-time architecture on the 900-sample UNSEEN split:

    Audio
      |
      +--> Whisper Encoder
      |       |
      |       +--> 512-D embedding
      |              |
      |              +--> Voice Projection 512 -> 128
      |                     |
      |                     +--> Voice-NLU
      |
      +--> Whisper Decoder
              |
              +--> Live ASR transcript
                     |
                     +--> Text MiniLM 384-D
                            |
                            +--> Text Projection 384 -> 128
                                   |
                                   +--> Text-NLU
                                          |
                                          +--> Detector A/B
                                                  |
                                                  +--> Normal / Suspicious
                                                           |
                                                           +--> Gemma reasoning

IMPORTANT
---------
The clean/reference transcript is used ONLY for OFFLINE EVALUATION.

It is NEVER supplied to:
    - Voice-NLU
    - Text-NLU live inference
    - Detector A
    - Detector B
    - Gemma reasoning

The live detector sees only:
    Voice-NLU output
    +
    Live Whisper transcript -> Text-NLU output

This file does NOT retrain any model.
It loads the frozen artifacts produced by the final Colab experiment.
"""

# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import os
import sys
import json
import time
import psutil
import warnings

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import jiwer

from tqdm import tqdm

import whisper

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc
)

from scipy.spatial.distance import jensenshannon


warnings.filterwarnings("ignore")


# ==============================================================================
# 2. PROJECT CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(
    "/home/spark2/users/intern/Atreyee-Das/NLU_Robust_Experiment"
)

MODELS_DIR = PROJECT_ROOT / "models"
DETECTOR_DIR = PROJECT_ROOT / "detector"
DATASET_DIR = PROJECT_ROOT / "dataset"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
AUDIO_DIR = PROJECT_ROOT / "audio"
RESULTS_DIR = PROJECT_ROOT / "runtime_results"


# ------------------------------------------------------------------------------
# Foundation model paths
# ------------------------------------------------------------------------------

WHISPER_PATH = (
    "/home/spark2/Models/base.en.pt"
)

MINILM_PATH = (
    "/home/spark2/Models/all-MiniLM-L6-v2"
)

GEMMA_PATH = (
    "/home/spark2/Models/gemma_2_models/gemma-3-1b-it"
)


# ------------------------------------------------------------------------------
# Authoritative master dataset
#
# IMPORTANT:
# This file contains:
#
# sample_id
# scenario_id
# transcript
# domain_label
# subdomain_label
# topic_label
# document_type_label
# split
#
# It does NOT contain "ground_truth".
# "transcript" is the authoritative reference text.
# ------------------------------------------------------------------------------

DATASET_CSV = (
    DATASET_DIR
    / "nlu_robust_6000_scenario_paraphrase_FINAL_70_10_20.csv"
)


# ------------------------------------------------------------------------------
# Existing Whisper embedding artifacts
#
# These are NOT used for live encoder inference.
# They are loaded/validated only to make sure the correct frozen
# experiment artifacts are present.
# ------------------------------------------------------------------------------

WHISPER_EMBEDDINGS_NPY = (
    EMBEDDINGS_DIR
    / "whisper_embeddings_FINAL_70_10_20.npy"
)

WHISPER_EMBEDDINGS_META = (
    EMBEDDINGS_DIR
    / "whisper_embedding_metadata_FINAL_70_10_20.csv"
)


# ------------------------------------------------------------------------------
# NLU hierarchy
# ------------------------------------------------------------------------------

HEADS = [
    "domain",
    "subdomain",
    "topic",
    "document_type"
]


# These are the same weights used by the final Colab detector.
HEAD_WEIGHTS = {
    "domain": 0.20,
    "subdomain": 0.25,
    "topic": 0.40,
    "document_type": 0.15
}


# ------------------------------------------------------------------------------
# Device
# ------------------------------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ------------------------------------------------------------------------------
# Runtime latency measurement
# ------------------------------------------------------------------------------

WARMUP_RUNS = 20
MEASURED_RUNS = 200


# ==============================================================================
# 3. ENVIRONMENT RECORDING
# ==============================================================================

def record_environment():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        RESULTS_DIR / "figures"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    env_info = {

        "python_version":
            sys.version,

        "pytorch_version":
            torch.__version__,

        "cuda_version":
            torch.version.cuda,

        "cuda_available":
            torch.cuda.is_available(),

        "gpu_name":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "None"
            ),

        "cpu_ram_gb":
            psutil.virtual_memory().total
            / (1024 ** 3)
    }

    print(
        "\n--- ENVIRONMENT SPECIFICATIONS ---"
    )

    for k, v in env_info.items():

        print(
            f"{k}: {v}"
        )

    with open(
        RESULTS_DIR
        / "environment_info.json",
        "w"
    ) as f:

        json.dump(
            env_info,
            f,
            indent=4
        )


# ==============================================================================
# 4. HARD ARTIFACT VALIDATION
# ==============================================================================

def validate_artifacts():

    required_files = [

        Path(WHISPER_PATH),

        Path(MINILM_PATH),

        Path(GEMMA_PATH),

        DATASET_CSV,

        WHISPER_EMBEDDINGS_NPY,

        WHISPER_EMBEDDINGS_META,

        MODELS_DIR
        / "best_voice_projection_FINAL_70_10_20.pt",

        MODELS_DIR
        / "best_text_projection_FINAL_70_10_20.pt",

        MODELS_DIR
        / "voice_whisper_scaler_FINAL_70_10_20.joblib",

        MODELS_DIR
        / "text_scaler_FINAL_70_10_20.joblib",

        MODELS_DIR
        / "shared_label_encoders_FINAL_70_10_20.joblib",

        DETECTOR_DIR
        / "Detector_A_STRICT_ASR_INDUCED.joblib",

        DETECTOR_DIR
        / "Detector_B_STRICT_ASR_INDUCED.joblib",

        DETECTOR_DIR
        / "strict_detector_thresholds.json"
    ]

    for head in HEADS:

        required_files.extend([

            MODELS_DIR
            / f"voice_{head}_label_mlp_FINAL_70_10_20.joblib",

            MODELS_DIR
            / f"text_{head}_label_mlp_FINAL_70_10_20.joblib"
        ])


    missing = [
        str(f)
        for f in required_files
        if not Path(f).exists()
    ]


    if missing:

        print(
            "\nCRITICAL ERROR: Missing artifacts:"
        )

        for m in missing:

            print(
                f" - {m}"
            )

        raise FileNotFoundError(
            "Required final experiment artifacts are missing."
        )


    print(
        "\n✓ ARTIFACT VALIDATION PASSED"
    )


# ==============================================================================
# 5. ARCHITECTURE DEFINITIONS
# ==============================================================================

class VoiceHierarchicalProjection(nn.Module):

    def __init__(
        self,
        input_dim=512,
        projection_dim=128
    ):

        super().__init__()

        self.projector = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                256,
                projection_dim
            )
        )


    def forward(self, x):

        return F.normalize(
            self.projector(x),
            p=2,
            dim=1
        )


class TextHierarchicalProjection(nn.Module):

    def __init__(
        self,
        input_dim=384,
        projection_dim=128
    ):

        super().__init__()

        self.projector = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                256,
                projection_dim
            )
        )


    def forward(self, x):

        return F.normalize(
            self.projector(x),
            p=2,
            dim=1
        )


# ==============================================================================
# 6. EXACT DETECTOR FEATURE RECONSTRUCTION
#
# This follows the final Colab detector feature construction.
# ==============================================================================

EPS = 1e-12


def entropy(p):

    p = np.asarray(
        p,
        dtype=np.float64
    )

    p = np.clip(
        p,
        EPS,
        1.0
    )

    p = p / p.sum()

    # IMPORTANT:
    # Colab final detector uses natural log.
    return float(
        -np.sum(
            p * np.log(p)
        )
    )


def margin(p):

    p = np.asarray(
        p,
        dtype=np.float64
    )

    if len(p) < 2:

        return 1.0

    s = np.sort(
        p
    )[::-1]

    return float(
        s[0] - s[1]
    )


def aligned_js(
    vp,
    vc,
    tp,
    tc
):

    vc = np.asarray(
        vc
    )

    tc = np.asarray(
        tc
    )

    classes = sorted(
        set(vc) | set(tc)
    )

    vmap = {
        c: float(p)
        for c, p in zip(
            vc,
            vp
        )
    }

    tmap = {
        c: float(p)
        for c, p in zip(
            tc,
            tp
        )
    }

    va = np.asarray([
        vmap.get(
            c,
            EPS
        )
        for c in classes
    ])

    ta = np.asarray([
        tmap.get(
            c,
            EPS
        )
        for c in classes
    ])

    va = np.clip(
        va,
        EPS,
        None
    )

    ta = np.clip(
        ta,
        EPS,
        None
    )

    va /= va.sum()
    ta /= ta.sum()

    d = jensenshannon(
        va,
        ta,
        base=2
    )

    if np.isnan(d):

        return 0.0

    return float(
        d ** 2
    )


def extract_detector_features(
    voice_preds,
    text_preds,
    voice_probs,
    text_probs,
    voice_class_maps,
    text_class_maps
):
    """
    Reconstruct Detector A and Detector B features.

    Detector A:
        hard semantic disagreement only.

    Detector B:
        hard disagreement
        +
        posterior/confidence/entropy/margin/JS features.

    NO ground-truth information enters this function.
    """

    feat_a = {}
    feat_b = {}

    total_disagreement = 0.0
    weighted_disagreement = 0.0

    voice_confidences = []
    text_confidences = []

    cross_supports = []
    js_values = []


    # --------------------------------------------------------------------------
    # Detector A
    # --------------------------------------------------------------------------

    for h in HEADS:

        v_label = voice_preds[h]
        t_label = text_preds[h]

        disagreement = float(
            v_label != t_label
        )

        feat_a[
            f"{h}_disagreement"
        ] = disagreement

        total_disagreement += (
            disagreement
        )

        weighted_disagreement += (
            HEAD_WEIGHTS[h]
            * disagreement
        )


    feat_a[
        "total_disagreements"
    ] = total_disagreement

    feat_a[
        "weighted_disagreement"
    ] = weighted_disagreement


    # --------------------------------------------------------------------------
    # Detector B starts with Detector A features
    # --------------------------------------------------------------------------

    feat_b = dict(
        feat_a
    )


    # --------------------------------------------------------------------------
    # Posterior features
    # --------------------------------------------------------------------------

    for h in HEADS:

        vp = np.asarray(
            voice_probs[h],
            dtype=np.float64
        )

        tp = np.asarray(
            text_probs[h],
            dtype=np.float64
        )

        vc = np.asarray(
            voice_class_maps[h]
        )

        tc = np.asarray(
            text_class_maps[h]
        )


        v_idx = int(
            np.argmax(vp)
        )

        t_idx = int(
            np.argmax(tp)
        )


        v_label = vc[v_idx]
        t_label = tc[t_idx]


        v_conf = float(
            vp[v_idx]
        )

        t_conf = float(
            tp[t_idx]
        )


        # Probability that Text-NLU assigns to Voice-NLU's label
        t_label_to_index = {
            c: i
            for i, c in enumerate(tc)
        }

        text_prob_voice = (

            float(
                tp[
                    t_label_to_index[
                        v_label
                    ]
                ]
            )

            if v_label
            in t_label_to_index

            else 0.0
        )


        # Probability that Voice-NLU assigns to Text-NLU's label
        v_label_to_index = {
            c: i
            for i, c in enumerate(vc)
        }

        voice_prob_text = (

            float(
                vp[
                    v_label_to_index[
                        t_label
                    ]
                ]
            )

            if t_label
            in v_label_to_index

            else 0.0
        )


        js = aligned_js(
            vp,
            vc,
            tp,
            tc
        )


        v_ent = entropy(
            vp
        )

        t_ent = entropy(
            tp
        )


        v_margin = margin(
            vp
        )

        t_margin = margin(
            tp
        )


        feat_b[
            f"{h}_voice_top1_confidence"
        ] = v_conf

        feat_b[
            f"{h}_text_top1_confidence"
        ] = t_conf

        feat_b[
            f"{h}_confidence_gap"
        ] = abs(
            v_conf
            -
            t_conf
        )

        feat_b[
            f"{h}_text_prob_of_voice_label"
        ] = text_prob_voice

        feat_b[
            f"{h}_voice_prob_of_text_label"
        ] = voice_prob_text

        feat_b[
            f"{h}_js_divergence"
        ] = js

        feat_b[
            f"{h}_voice_entropy"
        ] = v_ent

        feat_b[
            f"{h}_text_entropy"
        ] = t_ent

        feat_b[
            f"{h}_voice_margin"
        ] = v_margin

        feat_b[
            f"{h}_text_margin"
        ] = t_margin


        voice_confidences.append(
            v_conf
        )

        text_confidences.append(
            t_conf
        )

        cross_supports.extend([
            text_prob_voice,
            voice_prob_text
        ])

        js_values.append(
            js
        )


    mean_voice_confidence = float(
        np.mean(
            voice_confidences
        )
    )

    mean_text_confidence = float(
        np.mean(
            text_confidences
        )
    )


    feat_b[
        "mean_voice_confidence"
    ] = mean_voice_confidence

    feat_b[
        "mean_text_confidence"
    ] = mean_text_confidence

    feat_b[
        "mean_cross_model_support"
    ] = float(
        np.mean(
            cross_supports
        )
    )


    feat_b[
        "weighted_js_divergence"
    ] = float(
        sum(
            HEAD_WEIGHTS[h]
            * js
            for h, js in zip(
                HEADS,
                js_values
            )
        )
    )


    feat_b[
        "strong_conflict_score"
    ] = float(
        weighted_disagreement
        *
        min(
            mean_voice_confidence,
            mean_text_confidence
        )
    )


    return (
        feat_a,
        feat_b
    )


# ==============================================================================
# 7. MODEL PREDICTION HELPERS
# ==============================================================================

def predict_nlu_heads(
    projected_embedding,
    mlps,
    label_encoders
):
    """
    Run all four frozen NLU heads.

    Returns:
        predictions
        probabilities
        class-label arrays corresponding exactly to probability columns
    """

    predictions = {}
    probabilities = {}
    class_maps = {}


    for h in HEADS:

        model = mlps[h]

        probs = model.predict_proba(
            projected_embedding
        )[0]

        probabilities[h] = probs


        # sklearn probability columns correspond to model.classes_
        class_ids = np.asarray(
            model.classes_
        )

        encoder = label_encoders[
            f"{h}_label"
        ]

        labels = encoder.inverse_transform(
            class_ids
        )

        class_maps[h] = labels


        top_index = int(
            np.argmax(probs)
        )

        top_class_id = class_ids[
            top_index
        ]

        predictions[h] = (
            encoder.inverse_transform(
                [top_class_id]
            )[0]
        )


    return (
        predictions,
        probabilities,
        class_maps
    )


# ==============================================================================
# 8. MAIN PIPELINE
# ==============================================================================

def run_pipeline():

    # --------------------------------------------------------------------------
    # Environment / artifact validation
    # --------------------------------------------------------------------------

    record_environment()

    validate_artifacts()


    # --------------------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------------------

    print(
        "\nLoading authoritative master dataset..."
    )

    df = pd.read_csv(
        DATASET_CSV
    )


    # --------------------------------------------------------------------------
    # HARD schema validation
    #
    # This prevents the exact failure that occurred previously.
    # --------------------------------------------------------------------------

    required_master_columns = [

        "sample_id",
        "scenario_id",
        "transcript",

        "domain_label",
        "subdomain_label",
        "topic_label",
        "document_type_label",

        "split"
    ]


    missing_master_columns = [
        c
        for c in required_master_columns
        if c not in df.columns
    ]


    if missing_master_columns:

        raise RuntimeError(
            "Master dataset is missing required columns: "
            + str(missing_master_columns)
        )


    if "ground_truth" in df.columns:

        print(
            "NOTE: master dataset contains an extra "
            "'ground_truth' column; this script will "
            "still use authoritative 'transcript'."
        )


    # --------------------------------------------------------------------------
    # IDs
    # --------------------------------------------------------------------------

    df["sample_id"] = (
        df["sample_id"]
        .astype(str)
    )

    df["scenario_id"] = (
        df["scenario_id"]
        .astype(str)
    )


    # --------------------------------------------------------------------------
    # Dataset validation
    # --------------------------------------------------------------------------

    if len(df) != 6000:

        raise RuntimeError(
            f"Expected 6000 master samples, got {len(df)}."
        )


    if df["sample_id"].nunique() != 6000:

        raise RuntimeError(
            "sample_id is not unique in the master dataset."
        )


    if df["scenario_id"].nunique() != 600:

        raise RuntimeError(
            "Expected 600 semantic scenarios."
        )


    scenario_split_counts = (
        df.groupby(
            "scenario_id"
        )["split"]
        .nunique()
    )


    leakage_count = int(
        (
            scenario_split_counts > 1
        ).sum()
    )


    if leakage_count != 0:

        raise RuntimeError(
            f"Scenario leakage detected: {leakage_count} scenarios."
        )


    scenario_counts = (
        df[
            [
                "scenario_id",
                "split"
            ]
        ]
        .drop_duplicates()
        ["split"]
        .value_counts()
    )


    expected_scenario_counts = {

        "train": 420,

        "validation": 90,

        "unseen": 90
    }


    for split_name, expected in (
        expected_scenario_counts.items()
    ):

        actual = int(
            scenario_counts.get(
                split_name,
                0
            )
        )

        if actual != expected:

            raise RuntimeError(
                f"{split_name} scenario count = "
                f"{actual}; expected {expected}."
            )


    print(
        "\n✓ Master dataset validated."
    )

    print(
        "  Samples:",
        len(df)
    )

    print(
        "  Scenarios:",
        df["scenario_id"].nunique()
    )

    print(
        "  Scenario split:",
        scenario_counts.to_dict()
    )


    # --------------------------------------------------------------------------
    # Unseen runtime subset
    # --------------------------------------------------------------------------

    df_unseen_runtime = (
        df[
            df["split"] == "unseen"
        ]
        .copy()
        .reset_index(drop=True)
    )


    if len(df_unseen_runtime) != 900:

        raise RuntimeError(
            "Expected exactly 900 unseen utterances."
        )


    if (
        df_unseen_runtime[
            "scenario_id"
        ].nunique()
        != 90
    ):

        raise RuntimeError(
            "Expected exactly 90 unseen scenarios."
        )


    print(
        "\nUnseen runtime evaluation:"
    )

    print(
        "  Utterances:",
        len(df_unseen_runtime)
    )

    print(
        "  Scenarios:",
        df_unseen_runtime[
            "scenario_id"
        ].nunique()
    )


    # --------------------------------------------------------------------------
    # Load foundation models
    # --------------------------------------------------------------------------

    print(
        "\nLoading Whisper..."
    )

    whisper_model = whisper.load_model(
        WHISPER_PATH,
        device=DEVICE
    )


    print(
        "✓ Whisper loaded."
    )


    print(
        "\nLoading MiniLM..."
    )

    minilm_model = SentenceTransformer(
        MINILM_PATH,
        device=DEVICE
    )


    print(
        "✓ MiniLM loaded."
    )


    print(
        "\nLoading Gemma..."
    )

    gemma_tokenizer = (
        AutoTokenizer.from_pretrained(
            GEMMA_PATH,
            local_files_only=True
        )
    )


    if gemma_tokenizer.pad_token is None:

        gemma_tokenizer.pad_token = (
            gemma_tokenizer.eos_token
        )


    gemma_dtype = (

        torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32
    )


    gemma_model = (
        AutoModelForCausalLM.from_pretrained(
            GEMMA_PATH,
            local_files_only=True,
            torch_dtype=gemma_dtype,
            device_map="auto"
        )
    )


    gemma_model.eval()


    print(
        "✓ Gemma loaded."
    )


    # --------------------------------------------------------------------------
    # Load scalers
    # --------------------------------------------------------------------------

    print(
        "\nLoading frozen scalers..."
    )


    v_scaler = joblib.load(
        MODELS_DIR
        / "voice_whisper_scaler_FINAL_70_10_20.joblib"
    )


    t_scaler = joblib.load(
        MODELS_DIR
        / "text_scaler_FINAL_70_10_20.joblib"
    )


    # --------------------------------------------------------------------------
    # Load frozen projections
    # --------------------------------------------------------------------------

    print(
        "\nLoading frozen projections..."
    )


    v_proj = VoiceHierarchicalProjection(
        input_dim=512,
        projection_dim=128
    ).to(DEVICE)


    v_proj.load_state_dict(
        torch.load(
            MODELS_DIR
            / "best_voice_projection_FINAL_70_10_20.pt",
            map_location=DEVICE
        )
    )


    v_proj.eval()


    t_proj = TextHierarchicalProjection(
        input_dim=384,
        projection_dim=128
    ).to(DEVICE)


    t_proj.load_state_dict(
        torch.load(
            MODELS_DIR
            / "best_text_projection_FINAL_70_10_20.pt",
            map_location=DEVICE
        )
    )


    t_proj.eval()


    print(
        "✓ Projections loaded."
    )


    # --------------------------------------------------------------------------
    # Shared label encoders
    # --------------------------------------------------------------------------

    shared_encoders = joblib.load(
        MODELS_DIR
        / "shared_label_encoders_FINAL_70_10_20.joblib"
    )


    # --------------------------------------------------------------------------
    # Load NLU MLPs
    # --------------------------------------------------------------------------

    print(
        "\nLoading Voice-NLU / Text-NLU heads..."
    )


    v_mlps = {}

    t_mlps = {}


    for h in HEADS:

        v_mlps[h] = joblib.load(
            MODELS_DIR
            / f"voice_{h}_label_mlp_FINAL_70_10_20.joblib"
        )

        t_mlps[h] = joblib.load(
            MODELS_DIR
            / f"text_{h}_label_mlp_FINAL_70_10_20.joblib"
        )


    print(
        "✓ All eight NLU MLP heads loaded."
    )


    # --------------------------------------------------------------------------
    # Load strict detectors
    # --------------------------------------------------------------------------

    print(
        "\nLoading strict ASR-induced detectors..."
    )


    detector_A = joblib.load(
        DETECTOR_DIR
        / "Detector_A_STRICT_ASR_INDUCED.joblib"
    )


    detector_B = joblib.load(
        DETECTOR_DIR
        / "Detector_B_STRICT_ASR_INDUCED.joblib"
    )


    with open(
        DETECTOR_DIR
        / "strict_detector_thresholds.json",
        "r"
    ) as f:

        thresholds = json.load(f)


    if "threshold_A" not in thresholds:

        raise RuntimeError(
            "strict_detector_thresholds.json does not contain threshold_A."
        )


    if "threshold_B" not in thresholds:

        raise RuntimeError(
            "strict_detector_thresholds.json does not contain threshold_B."
        )


    thresh_A = float(
        thresholds["threshold_A"]
    )

    thresh_B = float(
        thresholds["threshold_B"]
    )


    print(
        "✓ Detector A threshold:",
        thresh_A
    )

    print(
        "✓ Detector B threshold:",
        thresh_B
    )


    # --------------------------------------------------------------------------
    # Verify detector feature schemas BEFORE processing any audio
    #
    # This is extremely important.
    # --------------------------------------------------------------------------

    detector_A_features = list(
        detector_A.feature_names_in_
    )

    detector_B_features = list(
        detector_B.feature_names_in_
    )


    print(
        "\nDetector A expects",
        len(detector_A_features),
        "features."
    )

    print(
        "Detector B expects",
        len(detector_B_features),
        "features."
    )


    # --------------------------------------------------------------------------
    # Verify that our reconstructed feature generator contains all features
    # expected by the loaded detector.
    #
    # A missing feature is a fatal configuration error.
    # --------------------------------------------------------------------------

    # Build a harmless synthetic feature example using model class structures.
    # This only checks the names and does NOT perform detector inference.
    dummy_voice_preds = {}
    dummy_text_preds = {}
    dummy_voice_probs = {}
    dummy_text_probs = {}
    dummy_voice_classes = {}
    dummy_text_classes = {}


    for h in HEADS:

        enc = shared_encoders[
            f"{h}_label"
        ]

        classes = np.asarray(
            enc.classes_
        )

        dummy_voice_classes[h] = classes
        dummy_text_classes[h] = classes

        dummy_voice_probs[h] = (
            np.ones(
                len(classes),
                dtype=np.float64
            )
            / len(classes)
        )

        dummy_text_probs[h] = (
            np.ones(
                len(classes),
                dtype=np.float64
            )
            / len(classes)
        )

        dummy_voice_preds[h] = classes[0]
        dummy_text_preds[h] = classes[0]


    dummy_A, dummy_B = (
        extract_detector_features(
            dummy_voice_preds,
            dummy_text_preds,
            dummy_voice_probs,
            dummy_text_probs,
            dummy_voice_classes,
            dummy_text_classes
        )
    )


    reconstructed_A_features = set(
        dummy_A.keys()
    )

    reconstructed_B_features = set(
        dummy_B.keys()
    )


    missing_A_features = [
        f
        for f in detector_A_features
        if f not in reconstructed_A_features
    ]

    missing_B_features = [
        f
        for f in detector_B_features
        if f not in reconstructed_B_features
    ]


    if missing_A_features:

        raise RuntimeError(
            "Detector A feature reconstruction mismatch. "
            f"Missing: {missing_A_features}"
        )


    if missing_B_features:

        raise RuntimeError(
            "Detector B feature reconstruction mismatch. "
            f"Missing: {missing_B_features}"
        )


    print(
        "✓ Detector feature schemas validated."
    )


    # --------------------------------------------------------------------------
    # Gemma generation helper
    # --------------------------------------------------------------------------

    def generate_ans(prompt):

        inputs = (
            gemma_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            )
        )


        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }


        if DEVICE == "cuda":

            torch.cuda.synchronize()


        t0_gen = time.perf_counter()


        with torch.no_grad():

            output_ids = (
                gemma_model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=(
                        gemma_tokenizer.pad_token_id
                    )
                )
            )


        if DEVICE == "cuda":

            torch.cuda.synchronize()


        latency_ms = (
            time.perf_counter()
            -
            t0_gen
        ) * 1000


        input_length = (
            inputs[
                "input_ids"
            ].shape[1]
        )


        generated_ids = (
            output_ids[
                :,
                input_length:
            ]
        )


        answer = (
            gemma_tokenizer
            .batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            .strip()
        )


        return (
            answer,
            latency_ms
        )


    # --------------------------------------------------------------------------
    # Runtime storage
    # --------------------------------------------------------------------------

    inference_results = []

    latencies = []


    warmup_count = 0
    measured_count = 0


    if DEVICE == "cuda":

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()


    # ==========================================================================
    # 9. FULL 900-SAMPLE LIVE INFERENCE
    # ==========================================================================

    for idx, row in tqdm(
        df_unseen_runtime.iterrows(),
        total=len(df_unseen_runtime),
        desc="FULL E2E INFERENCE — UNSEEN"
    ):

        sample_id = str(
            row["sample_id"]
        )

        scenario_id = str(
            row["scenario_id"]
        )

        split = str(
            row["split"]
        )


        # ----------------------------------------------------------------------
        # AUTHORITATIVE reference text
        #
        # THIS IS OFFLINE EVALUATION ONLY.
        # It never enters the live path.
        # ----------------------------------------------------------------------

        reference_transcript = (
            str(
                row["transcript"]
            )
            if pd.notna(
                row["transcript"]
            )
            else ""
        )


        if not reference_transcript.strip():

            raise RuntimeError(
                f"Empty authoritative transcript for sample {sample_id}."
            )


        # ----------------------------------------------------------------------
        # The dataset contains no separate user_query.
        #
        # The utterance itself is the user's request.
        # Therefore Gemma receives the live transcript as the user utterance.
        # ----------------------------------------------------------------------

        query = reference_transcript


        # ----------------------------------------------------------------------
        # Audio
        # ----------------------------------------------------------------------

        audio_file = (
            AUDIO_DIR
            / f"{sample_id}.wav"
        )


        if not audio_file.exists():

            raise FileNotFoundError(
                f"Missing audio for sample {sample_id}: "
                f"{audio_file}"
            )


        timers = {}


        # ======================================================================
        # AUDIO LOAD + MEL
        # ======================================================================

        t0 = time.perf_counter()


        audio = whisper.load_audio(
            str(audio_file)
        )


        audio = whisper.pad_or_trim(
            audio
        )


        mel = whisper.log_mel_spectrogram(
            audio
        ).to(DEVICE)


        if DEVICE == "cuda":

            torch.cuda.synchronize()


        timers[
            "audio_load_ms"
        ] = (
            time.perf_counter()
            -
            t0
        ) * 1000


        # ======================================================================
        # WHISPER ENCODER
        #
        # This is the SAME 512-D mean-pooled representation used by the
        # Voice-NLU experiment.
        # ======================================================================

        t0 = time.perf_counter()


        with torch.no_grad():

            enc_out = (
                whisper_model.encoder(
                    mel.unsqueeze(0)
                )
            )


            emb_512 = (
                enc_out
                .mean(dim=1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )


        if emb_512.shape != (
            1,
            512
        ):

            raise RuntimeError(
                f"Unexpected Whisper embedding shape "
                f"for {sample_id}: {emb_512.shape}"
            )


        if DEVICE == "cuda":

            torch.cuda.synchronize()


        timers[
            "whisper_encoder_ms"
        ] = (
            time.perf_counter()
            -
            t0
        ) * 1000


        # ======================================================================
        # VOICE-NLU
        # ======================================================================

        t0 = time.perf_counter()


        v_scaled = (
            v_scaler
            .transform(
                emb_512
            )
            .astype(np.float32)
        )


        with torch.no_grad():

            v_128 = (
                v_proj(
                    torch.tensor(
                        v_scaled,
                        dtype=torch.float32,
                        device=DEVICE
                    )
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )


        if v_128.shape != (
            1,
            128
        ):

            raise RuntimeError(
                f"Unexpected Voice semantic embedding shape "
                f"for {sample_id}: {v_128.shape}"
            )


        (
            v_preds,
            v_probs,
            v_class_maps
        ) = predict_nlu_heads(
            v_128,
            v_mlps,
            shared_encoders
        )


        if DEVICE == "cuda":

            torch.cuda.synchronize()


        timers[
            "voice_nlu_ms"
        ] = (
            time.perf_counter()
            -
            t0
        ) * 1000


        # ======================================================================
        # WHISPER DECODER
        #
        # IMPORTANT:
        # Use the same decoding family/configuration used by the Colab ASR
        # experiment:
        #
        # language = en
        # task = transcribe
        # temperature = 0
        # beam_size = 5
        # condition_on_previous_text = True
        # word_timestamps = True
        #
        # This replaces the old whisper.decode() call.
        # ======================================================================

        t0 = time.perf_counter()


        with torch.no_grad():

            decode_result = (
                whisper_model.transcribe(
                    str(audio_file),

                    language="en",

                    task="transcribe",

                    temperature=0,

                    beam_size=5,

                    condition_on_previous_text=True,

                    word_timestamps=True,

                    verbose=False,

                    fp16=(
                        DEVICE == "cuda"
                    )
                )
            )


        decoded_transcript = (
            decode_result
            .get(
                "text",
                ""
            )
            .strip()
        )


        if DEVICE == "cuda":

            torch.cuda.synchronize()


        timers[
            "whisper_decoder_ms"
        ] = (
            time.perf_counter()
            -
            t0
        ) * 1000


        # ======================================================================
        # HARD CHECK:
        # Never allow an empty live transcript to silently propagate.
        # ======================================================================

        if not decoded_transcript:

            raise RuntimeError(
                f"Whisper produced an empty transcript for {sample_id}."
            )


        # ======================================================================
        # OFFLINE CLEAN-TEXT REFERENCE
        #
        # This branch is NOT part of inference.
        #
        # It exists ONLY to determine:
        #
        # strict_asr_error =
        #     clean semantic interpretation correct
        #     AND
        #     live decoded semantic interpretation wrong
        #
        # This is evaluation bookkeeping.
        # ======================================================================

        clean_emb_384 = (
            minilm_model
            .encode(
                [reference_transcript],
                convert_to_numpy=True,
                normalize_embeddings=False
            )
            .astype(np.float32)
        )


        clean_scaled = (
            t_scaler
            .transform(
                clean_emb_384
            )
            .astype(np.float32)
        )


        with torch.no_grad():

            clean_128 = (
                t_proj(
                    torch.tensor(
                        clean_scaled,
                        dtype=torch.float32,
                        device=DEVICE
                    )
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )


        (
            text_clean_preds,
            text_clean_probs,
            text_clean_class_maps
        ) = predict_nlu_heads(
            clean_128,
            t_mlps,
            shared_encoders
        )


        # ======================================================================
        # LIVE TEXT-NLU
        #
        # This is the actual inference path.
        #
        # It uses ONLY decoded_transcript.
        # ======================================================================

        t0 = time.perf_counter()


        emb_384 = (
            minilm_model
            .encode(
                [decoded_transcript],
                convert_to_numpy=True,
                normalize_embeddings=False
            )
            .astype(np.float32)
        )


        if emb_384.shape != (
            1,
            384
        ):

            raise RuntimeError(
                f"Unexpected MiniLM embedding shape "
                f"for {sample_id}: {emb_384.shape}"
            )


        t_scaled = (
            t_scaler
            .transform(
                emb_384
            )
            .astype(np.float32)
        )


        with torch.no_grad():

            t_128 = (
                t_proj(
                    torch.tensor(
                        t_scaled,
                        dtype=torch.float32,
                        device=DEVICE
                    )
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )


        if t_128.shape != (
            1,
            128
        ):

            raise RuntimeError(
                f"Unexpected Text semantic embedding shape "
                f"for {sample_id}: {t_128.shape}"
            )


        (
            t_preds,
            t_probs,
            t_class_maps
        ) = predict_nlu_heads(
            t_128,
            t_mlps,
            shared_encoders
        )


        if DEVICE == "cuda":

            torch.cuda.synchronize()


        timers[
            "text_nlu_ms"
        ] = (
            time.perf_counter()
            -
            t0
        ) * 1000


        # ======================================================================
        # LIVE DETECTOR
        #
        # Only Voice-NLU + live Text-NLU are supplied.
        #
        # NO reference_transcript.
        # NO semantic ground truth.
        # NO targets_corrupted.
        # ======================================================================

        t0 = time.perf_counter()


        (
            feat_a,
            feat_b
        ) = extract_detector_features(
            v_preds,
            t_preds,
            v_probs,
            t_probs,
            v_class_maps,
            t_class_maps
        )


        df_a = (
            pd.DataFrame(
                [feat_a]
            )
            .reindex(
                columns=detector_A_features
            )
        )


        df_b = (
            pd.DataFrame(
                [feat_b]
            )
            .reindex(
                columns=detector_B_features
            )
        )


        if df_a.isnull().any().any():

            raise RuntimeError(
                f"NaN/missing Detector A feature for {sample_id}."
            )


        if df_b.isnull().any().any():

            raise RuntimeError(
                f"NaN/missing Detector B feature for {sample_id}."
            )


        prob_a = float(
            detector_A
            .predict_proba(
                df_a
            )[0][1]
        )


        prob_b = float(
            detector_B
            .predict_proba(
                df_b
            )[0][1]
        )


        pred_a = int(
            prob_a >= thresh_A
        )


        pred_b = int(
            prob_b >= thresh_B
        )


        if DEVICE == "cuda":

            torch.cuda.synchronize()


        timers[
            "detector_ms"
        ] = (
            time.perf_counter()
            -
            t0
        ) * 1000


        # ======================================================================
        # OFFLINE STRICT ASR-INDUCED ERROR LABEL
        #
        # This does NOT enter detector inference.
        #
        # Definition from final Colab experiment:
        #
        # clean text semantics correct on ALL four heads
        # AND
        # corrupted/live text semantics wrong on >= 1 head.
        #
        # The final Colab strict detector used this concept to create the
        # strict target. 4
        # ======================================================================

        gt_labels = {

            h:
                str(
                    row[
                        f"{h}_label"
                    ]
                )

            for h in HEADS
        }


        clean_text_correct = int(
            all(
                str(
                    text_clean_preds[h]
                )
                ==
                gt_labels[h]

                for h in HEADS
            )
        )


        decoded_text_correct = int(
            all(
                str(
                    t_preds[h]
                )
                ==
                gt_labels[h]

                for h in HEADS
            )
        )


        strict_asr_error = int(
            (
                clean_text_correct == 1
            )
            and
            (
                decoded_text_correct == 0
            )
        )


        # ======================================================================
        # WER / CER
        #
        # Authoritative reference = master["transcript"]
        # ======================================================================

        wer_value = float(
            jiwer.wer(
                reference_transcript,
                decoded_transcript
            )
        )


        cer_value = float(
            jiwer.cer(
                reference_transcript,
                decoded_transcript
            )
        )


        # ======================================================================
        # GEMMA REASONING
        #
        # There is no separate user_query column in the final master dataset.
        # The transcript is the user's utterance.
        #
        # Baseline:
        #   transcript only
        #
        # Voice:
        #   transcript + acoustic Voice-NLU evidence
        #
        # Gated:
        #   transcript + Voice-NLU evidence ONLY if Detector B fires
        # ======================================================================

        base_prompt = f"""
You are answering the user's request using the speech transcript below.

Treat the transcript as the primary evidence.
Do not assume that any word is wrong unless the supplied evidence supports that conclusion.
Do not invent facts, entities, numbers, or details.
Answer very briefly and directly.

Transcript:
{decoded_transcript}

User request:
{query}
""".strip()


        voice_prompt = f"""
You are answering the user's request using the speech transcript below.

The transcript is the primary evidence.
You also have independent semantic evidence derived from the acoustic speech representation.

Acoustic semantic evidence:
Domain: {v_preds["domain"]}
Subdomain: {v_preds["subdomain"]}
Topic: {v_preds["topic"]}
Document type: {v_preds["document_type"]}

Use the acoustic evidence only as supporting evidence when resolving ambiguity.
Do not blindly override the transcript.
Do not invent facts, entities, numbers, or details.
Answer very briefly and directly.

Transcript:
{decoded_transcript}

User request:
{query}
""".strip()


        gated_prompt = f"""
You are answering the user's request using the speech transcript below.

The transcript has been flagged by an ASR-error detector as potentially containing
a domain-specific transcription error.

The transcript remains the primary evidence.
You also have independent semantic evidence derived from the acoustic speech representation.

Acoustic semantic evidence:
Domain: {v_preds["domain"]}
Subdomain: {v_preds["subdomain"]}
Topic: {v_preds["topic"]}
Document type: {v_preds["document_type"]}

Use this acoustic evidence only when it provides support for resolving an ambiguity
in the transcript.
Correct the interpretation only when the supplied acoustic evidence supports doing so.
Do not blindly replace transcript content.
Do not invent facts, entities, numbers, or details.
Answer very briefly and directly.

Transcript:
{decoded_transcript}

User request:
{query}
""".strip()


        # ----------------------------------------------------------------------
        # Baseline Gemma
        # ----------------------------------------------------------------------

        ans_base, lat_base = (
            generate_ans(
                base_prompt
            )
        )


        # ----------------------------------------------------------------------
        # Voice-grounded Gemma
        # ----------------------------------------------------------------------

        ans_voice, lat_voice = (
            generate_ans(
                voice_prompt
            )
        )


        # ----------------------------------------------------------------------
        # Detector-gated Gemma
        #
        # IMPORTANT:
        # If Detector B does NOT fire, gated answer is exactly the baseline
        # result. No extra generation is performed.
        # ----------------------------------------------------------------------

        if pred_b == 1:

            ans_gated, lat_gated = (
                generate_ans(
                    gated_prompt
                )
            )

        else:

            ans_gated = ans_base

            lat_gated = 0.0


        timers[
            "gemma_baseline_ms"
        ] = lat_base

        timers[
            "gemma_voice_ms"
        ] = lat_voice

        timers[
            "gemma_gated_ms"
        ] = lat_gated


        # ======================================================================
        # LATENCY DEFINITIONS
        # ======================================================================

        timers[
            "semantic_pipeline_ms"
        ] = (

            timers[
                "voice_nlu_ms"
            ]

            +

            timers[
                "whisper_decoder_ms"
            ]

            +

            timers[
                "text_nlu_ms"
            ]

            +

            timers[
                "detector_ms"
            ]
        )


        timers[
            "speech_to_decision_ms"
        ] = (

            timers[
                "audio_load_ms"
            ]

            +

            timers[
                "whisper_encoder_ms"
            ]

            +

            timers[
                "whisper_decoder_ms"
            ]

            +

            timers[
                "voice_nlu_ms"
            ]

            +

            timers[
                "text_nlu_ms"
            ]

            +

            timers[
                "detector_ms"
            ]
        )


        timers[
            "total_baseline_ms"
        ] = (

            timers[
                "audio_load_ms"
            ]

            +

            timers[
                "whisper_encoder_ms"
            ]

            +

            timers[
                "whisper_decoder_ms"
            ]

            +

            timers[
                "gemma_baseline_ms"
            ]
        )


        timers[
            "total_gated_ms"
        ] = (

            timers[
                "speech_to_decision_ms"
            ]

            +

            timers[
                "gemma_gated_ms"
            ]
        )


        timers[
            "total_voice_ms"
        ] = (

            timers[
                "audio_load_ms"
            ]

            +

            timers[
                "whisper_encoder_ms"
            ]

            +

            timers[
                "whisper_decoder_ms"
            ]

            +

            timers[
                "voice_nlu_ms"
            ]

            +

            timers[
                "gemma_voice_ms"
            ]
        )


        # ======================================================================
        # LATENCY WARMUP / MEASUREMENT
        # ======================================================================

        if (
            warmup_count
            <
            WARMUP_RUNS
        ):

            warmup_count += 1

        elif (
            measured_count
            <
            MEASURED_RUNS
        ):

            latencies.append(
                timers.copy()
            )

            measured_count += 1


        # ======================================================================
        # SAVE PER-SAMPLE RESULT
        # ======================================================================

        res_dict = {

            "sample_id":
                sample_id,

            "scenario_id":
                scenario_id,

            "split":
                split,

            # --------------------------------------------------------------
            # OFFLINE reference
            # --------------------------------------------------------------

            "reference_transcript":
                reference_transcript,

            # --------------------------------------------------------------
            # LIVE ASR
            # --------------------------------------------------------------

            "whisper_transcript":
                decoded_transcript,

            "WER":
                wer_value,

            "CER":
                cer_value,

            # --------------------------------------------------------------
            # Offline semantic evaluation
            # --------------------------------------------------------------

            "clean_text_semantically_correct":
                clean_text_correct,

            "decoded_text_semantically_correct":
                decoded_text_correct,

            "strict_asr_induced_error":
                strict_asr_error,

            # --------------------------------------------------------------
            # Detector
            # --------------------------------------------------------------

            "detector_A_probability":
                prob_a,

            "detector_A_prediction":
                pred_a,

            "detector_B_probability":
                prob_b,

            "detector_B_prediction":
                pred_b,

            # --------------------------------------------------------------
            # Reasoning
            # --------------------------------------------------------------

            "gemma_baseline_ans":
                ans_base,

            "gemma_voice_ans":
                ans_voice,

            "gemma_gated_ans":
                ans_gated
        }


        # ----------------------------------------------------------------------
        # Save ground-truth labels + semantic predictions
        # ----------------------------------------------------------------------

        for h in HEADS:

            res_dict[
                f"gt_{h}"
            ] = gt_labels[h]


            res_dict[
                f"voice_pred_{h}"
            ] = v_preds[h]


            res_dict[
                f"text_clean_pred_{h}"
            ] = text_clean_preds[h]


            res_dict[
                f"text_pred_{h}"
            ] = t_preds[h]


        # ----------------------------------------------------------------------
        # Add selected detector features for auditability
        # ----------------------------------------------------------------------

        res_dict[
            "detector_weighted_disagreement"
        ] = feat_b[
            "weighted_disagreement"
        ]


        res_dict[
            "detector_total_disagreements"
        ] = feat_b[
            "total_disagreements"
        ]


        res_dict[
            "detector_weighted_js_divergence"
        ] = feat_b[
            "weighted_js_divergence"
        ]


        res_dict[
            "detector_strong_conflict_score"
        ] = feat_b[
            "strong_conflict_score"
        ]


        inference_results.append(
            res_dict
        )


    # ==========================================================================
    # 10. FINAL RESULTS
    # ==========================================================================

    print(
        "\nCalculating final runtime metrics..."
    )


    df_res = pd.DataFrame(
        inference_results
    )


    if len(df_res) != 900:

        raise RuntimeError(
            f"Expected 900 inference results; got {len(df_res)}."
        )


    # --------------------------------------------------------------------------
    # Save complete inference results
    # --------------------------------------------------------------------------

    df_res.to_csv(
        RESULTS_DIR
        / "inference_results.csv",
        index=False
    )


    # --------------------------------------------------------------------------
    # ASR metrics
    # --------------------------------------------------------------------------

    asr_metrics = df_res[
        [
            "sample_id",
            "scenario_id",
            "split",
            "reference_transcript",
            "whisper_transcript",
            "WER",
            "CER"
        ]
    ]


    asr_metrics.to_csv(
        RESULTS_DIR
        / "ASR_RUNTIME_METRICS.csv",
        index=False
    )


    # --------------------------------------------------------------------------
    # Latency
    # --------------------------------------------------------------------------

    df_lat = pd.DataFrame(
        latencies
    )


    if len(df_lat) > 0:

        df_lat.to_csv(
            RESULTS_DIR
            / "LATENCY_BREAKDOWN.csv",
            index=False
        )


        lat_summary = (
            df_lat
            .describe(
                percentiles=[
                    0.50,
                    0.90,
                    0.95,
                    0.99
                ]
            )
            .T
        )


        lat_summary.to_csv(
            RESULTS_DIR
            / "LATENCY_SUMMARY.csv"
        )

    else:

        lat_summary = pd.DataFrame()


    # --------------------------------------------------------------------------
    # Unseen results
    # --------------------------------------------------------------------------

    df_unseen = (
        df_res[
            df_res["split"] == "unseen"
        ]
        .copy()
    )


    if len(df_unseen) != 900:

        raise RuntimeError(
            "Final unseen result count is not 900."
        )


    # ==========================================================================
    # NLU METRICS
    # ==========================================================================

    nlu_metrics = []


    for h in HEADS:

        v_f1 = f1_score(

            df_unseen[
                f"gt_{h}"
            ],

            df_unseen[
                f"voice_pred_{h}"
            ],

            average="macro",

            zero_division=0
        )


        t_f1 = f1_score(

            df_unseen[
                f"gt_{h}"
            ],

            df_unseen[
                f"text_pred_{h}"
            ],

            average="macro",

            zero_division=0
        )


        nlu_metrics.append({

            "Head":
                h,

            "Voice_Macro_F1":
                v_f1,

            "Text_Macro_F1":
                t_f1
        })


    pd.DataFrame(
        nlu_metrics
    ).to_csv(
        RESULTS_DIR
        / "NLU_RUNTIME_RESULTS.csv",
        index=False
    )


    # ==========================================================================
    # ASR STRICT ERROR METRICS
    # ==========================================================================

    y_true = (
        df_unseen[
            "strict_asr_induced_error"
        ]
        .astype(int)
    )


    y_prob_a = (
        df_unseen[
            "detector_A_probability"
        ]
        .astype(float)
    )


    y_pred_a = (
        df_unseen[
            "detector_A_prediction"
        ]
        .astype(int)
    )


    y_prob_b = (
        df_unseen[
            "detector_B_probability"
        ]
        .astype(float)
    )


    y_pred_b = (
        df_unseen[
            "detector_B_prediction"
        ]
        .astype(int)
    )


    # --------------------------------------------------------------------------
    # Binary detector metric helper
    # --------------------------------------------------------------------------

    def detector_metrics(
        name,
        y,
        probabilities,
        predictions
    ):

        cm = confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        )


        tn, fp, fn, tp = (
            cm.ravel()
        )


        if len(
            np.unique(y)
        ) >= 2:

            roc = roc_auc_score(
                y,
                probabilities
            )

            pr = average_precision_score(
                y,
                probabilities
            )

        else:

            roc = np.nan
            pr = np.nan


        return {

            "Model":
                name,

            "Accuracy":
                accuracy_score(
                    y,
                    predictions
                ),

            "F1":
                f1_score(
                    y,
                    predictions,
                    zero_division=0
                ),

            "ROC-AUC":
                roc,

            "PR-AUC":
                pr,

            "Precision":
                precision_score(
                    y,
                    predictions,
                    zero_division=0
                ),

            "Recall":
                recall_score(
                    y,
                    predictions,
                    zero_division=0
                ),

            "Specificity":
                (
                    tn
                    /
                    (tn + fp)
                )
                if (
                    tn + fp
                ) > 0
                else 0.0,

            "FPR":
                (
                    fp
                    /
                    (fp + tn)
                )
                if (
                    fp + tn
                ) > 0
                else 0.0,

            "TP":
                tp,

            "TN":
                tn,

            "FP":
                fp,

            "FN":
                fn
        }


    det_metrics = [

        detector_metrics(
            "Detector_A",
            y_true,
            y_prob_a,
            y_pred_a
        ),

        detector_metrics(
            "Detector_B",
            y_true,
            y_prob_b,
            y_pred_b
        )
    ]


    det_metrics_df = pd.DataFrame(
        det_metrics
    )


    det_metrics_df.to_csv(
        RESULTS_DIR
        / "DETECTOR_RUNTIME_RESULTS.csv",
        index=False
    )


    # ==========================================================================
    # STRICT ERROR DISTRIBUTION
    # ==========================================================================

    strict_distribution = (
        df_unseen[
            "strict_asr_induced_error"
        ]
        .value_counts()
        .sort_index()
    )


    print(
        "\nStrict ASR-induced semantic target:"
    )

    print(
        strict_distribution
    )


    strict_rate = float(
        y_true.mean()
    )


    # ==========================================================================
    # REASONING OUTPUT SUMMARY
    #
    # No authoritative reasoning-answer labels exist in the dataset, so this
    # file records generation counts/availability rather than inventing a
    # reasoning accuracy score.
    # ==========================================================================

    reasoning_summary = pd.DataFrame([{

        "samples":
            len(df_unseen),

        "baseline_answers_generated":
            int(
                df_unseen[
                    "gemma_baseline_ans"
                ]
                .notna()
                .sum()
            ),

        "voice_answers_generated":
            int(
                df_unseen[
                    "gemma_voice_ans"
                ]
                .notna()
                .sum()
            ),

        "gated_answers_generated":
            int(
                df_unseen[
                    "gemma_gated_ans"
                ]
                .notna()
                .sum()
            ),

        "detector_B_positive_count":
            int(
                df_unseen[
                    "detector_B_prediction"
                ]
                .sum()
            ),

        "detector_B_positive_rate":
            float(
                df_unseen[
                    "detector_B_prediction"
                ]
                .mean()
            ),

        "reasoning_accuracy":
            np.nan,

        "reasoning_accuracy_note":
            (
                "No authoritative reasoning-answer labels "
                "are present in the 6000-sample master dataset."
            )
    }])


    reasoning_summary.to_csv(
        RESULTS_DIR
        / "REASONING_RESULTS.csv",
        index=False
    )


    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "FINAL FULL-TIME INFERENCE SUMMARY — UNSEEN"
    )

    print(
        "=" * 80
    )


    print(
        "\nSamples:",
        len(df_unseen)
    )


    print(
        "Scenarios:",
        df_unseen[
            "scenario_id"
        ].nunique()
    )


    print(
        "\nASR:"
    )


    print(
        f"Mean WER: {df_unseen['WER'].mean():.4f}"
    )


    print(
        f"Mean CER: {df_unseen['CER'].mean():.4f}"
    )


    print(
        "\nStrict ASR-induced semantic error:"
    )


    print(
        f"Rate: {strict_rate:.2%}"
    )


    for row_metrics in det_metrics:

        print(
            f"\n{row_metrics['Model']}:"
        )

        print(
            f"  F1       : {row_metrics['F1']:.4f}"
        )

        print(
            f"  ROC-AUC  : {row_metrics['ROC-AUC']:.4f}"
        )

        print(
            f"  PR-AUC   : {row_metrics['PR-AUC']:.4f}"
        )

        print(
            f"  Precision: {row_metrics['Precision']:.4f}"
        )

        print(
            f"  Recall   : {row_metrics['Recall']:.4f}"
        )

        print(
            f"  FPR      : {row_metrics['FPR']:.4f}"
        )


    print(
        "\nNLU:"
    )


    for item in nlu_metrics:

        print(
            f"  {item['Head']:15s} "
            f"Voice F1={item['Voice_Macro_F1']:.4f} "
            f"Text F1={item['Text_Macro_F1']:.4f}"
        )


    print(
        "\nDetector B fired:",
        int(
            df_unseen[
                "detector_B_prediction"
            ].sum()
        ),
        "/",
        len(df_unseen)
    )


    if len(df_lat) > 0:

        print(
            "\nLatency:"
        )

        if (
            "total_baseline_ms"
            in lat_summary.index
        ):

            print(
                "  Baseline mean:",
                f"{lat_summary.loc['total_baseline_ms', 'mean']:.1f} ms"
            )


        if (
            "total_gated_ms"
            in lat_summary.index
        ):

            print(
                "  Gated mean:",
                f"{lat_summary.loc['total_gated_ms', 'mean']:.1f} ms"
            )


        if (
            "total_voice_ms"
            in lat_summary.index
        ):

            print(
                "  Voice-grounded mean:",
                f"{lat_summary.loc['total_voice_ms', 'mean']:.1f} ms"
            )


        if (
            "total_gated_ms"
            in lat_summary.index
        ):

            print(
                "  Gated P95:",
                f"{lat_summary.loc['total_gated_ms', '95%']:.1f} ms"
            )


    if DEVICE == "cuda":

        peak_memory_gb = (
            torch.cuda.max_memory_reserved()
            /
            (1024 ** 3)
        )


        print(
            "\nPeak GPU memory reserved:",
            f"{peak_memory_gb:.2f} GB"
        )


    print(
        "\n"
        +
        "=" * 80
    )


    # ==========================================================================
    # SAVE FINAL SUMMARY TEXT
    # ==========================================================================

    with open(
        RESULTS_DIR
        / "FINAL_RUNTIME_SUMMARY.txt",
        "w"
    ) as f:

        f.write(
            "FINAL FULL-TIME INFERENCE SUMMARY — UNSEEN\n"
        )

        f.write(
            "=" * 60
            +
            "\n"
        )

        f.write(
            f"Samples: {len(df_unseen)}\n"
        )

        f.write(
            "Scenarios: "
            f"{df_unseen['scenario_id'].nunique()}\n"
        )

        f.write(
            f"Mean WER: {df_unseen['WER'].mean():.6f}\n"
        )

        f.write(
            f"Mean CER: {df_unseen['CER'].mean():.6f}\n"
        )

        f.write(
            f"Strict ASR-induced error rate: "
            f"{strict_rate:.6f}\n"
        )


        for row_metrics in det_metrics:

            f.write(
                f"\n{row_metrics['Model']}\n"
            )

            f.write(
                f"F1: {row_metrics['F1']:.6f}\n"
            )

            f.write(
                f"ROC-AUC: {row_metrics['ROC-AUC']}\n"
            )

            f.write(
                f"PR-AUC: {row_metrics['PR-AUC']}\n"
            )

            f.write(
                f"FPR: {row_metrics['FPR']:.6f}\n"
            )


        f.write(
            "\nDetector B positive count: "
            +
            str(
                int(
                    df_unseen[
                        "detector_B_prediction"
                    ].sum()
                )
            )
            +
            "\n"
        )


    print(
        "\n✓ Complete inference results saved to:"
    )

    print(
        RESULTS_DIR
    )


# ==============================================================================
# 11. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":

    run_pipeline()
