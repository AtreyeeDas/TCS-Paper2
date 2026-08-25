"""
Feature Extraction Module for Detector A (No Posteriors) and Detector B (With Posteriors).
Computes label disagreements, Jensen-Shannon divergence over aligned vocabularies,
margins, entropies, and cross-model posterior supports.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from config import EPS, HEAD_WEIGHTS, HEADS


def compute_entropy(probs: np.ndarray) -> float:
    """Computes Shannon entropy: H(p) = -sum(p * log(p + eps))."""
    p = np.clip(probs, EPS, 1.0)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log(p)))


def compute_margin(probs: np.ndarray) -> float:
    """Computes difference between highest and second-highest class probability."""
    if len(probs) < 2:
        return 1.0
    sorted_probs = np.sort(probs)[::-1]
    return float(sorted_probs[0] - sorted_probs[1])


def align_and_compute_js_divergence(
    voice_probs: np.ndarray,
    voice_classes: np.ndarray,
    text_probs: np.ndarray,
    text_classes: np.ndarray,
) -> float:
    """Aligns Voice and Text probability distributions to a unified vocabulary and computes JS divergence."""
    all_classes = sorted(list(set(voice_classes) | set(text_classes)))
    v_map = {c: p for c, p in zip(voice_classes, voice_probs)}
    t_map = {c: p for c, p in zip(text_classes, text_probs)}

    v_aligned = np.array([v_map.get(c, EPS) for c in all_classes], dtype=float)
    t_aligned = np.array([t_map.get(c, EPS) for c in all_classes], dtype=float)

    v_aligned = v_aligned / np.sum(v_aligned)
    t_aligned = t_aligned / np.sum(t_aligned)

    js_dist = jensenshannon(v_aligned, t_aligned, base=2)
    # JS divergence is the square of JS distance
    return float(js_dist**2) if not np.isnan(js_dist) else 0.0


def extract_sample_features(
    voice_row: pd.Series,
    text_row: pd.Series,
    voice_posteriors_sample: Dict[str, np.ndarray],
    text_posteriors_sample: Dict[str, np.ndarray],
    voice_encoders: Dict[str, Any],
    text_encoders: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Extracts features for Detector A and Detector B for a single paired inference instance."""
    feat_a = {}
    feat_b = {}

    total_disagreements = 0.0
    weighted_disagreements = 0.0

    # 1. Step 2: Extract Detector A Features (Hard Disagreements Only)
    for head in HEADS:
        v_label = str(voice_row[f"voice_{head}"])
        t_label = str(text_row[f"text_{head}"])
        disagree = 1.0 if v_label != t_label else 0.0

        feat_a[f"{head}_disagreement"] = disagree
        total_disagreements += disagree
        weighted_disagreements += HEAD_WEIGHTS[head] * disagree

    feat_a["total_disagreements"] = total_disagreements
    feat_a["weighted_disagreement"] = weighted_disagreements

    # Detector B begins with all Detector A features
    feat_b.update(feat_a)

    # 2. Step 3: Extract Posterior & Evidence Features for Detector B
    voice_confs = []
    text_confs = []
    cross_supports = []
    js_divs = []

    for head in HEADS:
        v_label = str(voice_row[f"voice_{head}"])
        t_label = str(text_row[f"text_{head}"])

        v_probs = voice_posteriors_sample[head]
        t_probs = text_posteriors_sample[head]

        v_classes = voice_encoders[f"{head}_label"].classes_
        t_classes = text_encoders[f"{head}_label"].classes_

        # Confidences
        v_top1_conf = float(np.max(v_probs))
        t_top1_conf = float(np.max(t_probs))
        conf_gap = abs(v_top1_conf - t_top1_conf)

        voice_confs.append(v_top1_conf)
        text_confs.append(t_top1_conf)

        # Cross-model probability evaluation
        v_class_to_idx = {c: idx for idx, c in enumerate(v_classes)}
        t_class_to_idx = {c: idx for idx, c in enumerate(t_classes)}

        # Text probability of Voice selected class
        t_prob_of_v_choice = (
            float(t_probs[t_class_to_idx[v_label]])
            if v_label in t_class_to_idx
            else 0.0
        )
        # Voice probability of Text selected class
        v_prob_of_t_choice = (
            float(v_probs[v_class_to_idx[t_label]])
            if t_label in v_class_to_idx
            else 0.0
        )

        cross_supports.extend([t_prob_of_v_choice, v_prob_of_t_choice])

        # JS Divergence over aligned vocabulary
        js_div = align_and_compute_js_divergence(
            v_probs, v_classes, t_probs, t_classes
        )
        js_divs.append(js_div)

        # Entropy & Top-1/Top-2 Margins
        v_entropy = compute_entropy(v_probs)
        t_entropy = compute_entropy(t_probs)
        v_margin = compute_margin(v_probs)
        t_margin = compute_margin(t_probs)

        # Assign per-head posterior features
        feat_b[f"{head}_voice_top1_confidence"] = v_top1_conf
        feat_b[f"{head}_text_top1_confidence"] = t_top1_conf
        feat_b[f"{head}_confidence_gap"] = conf_gap
        feat_b[f"{head}_text_prob_of_voice_label"] = t_prob_of_v_choice
        feat_b[f"{head}_voice_prob_of_text_label"] = v_prob_of_t_choice
        feat_b[f"{head}_js_divergence"] = js_div
        feat_b[f"{head}_voice_entropy"] = v_entropy
        feat_b[f"{head}_text_entropy"] = t_entropy
        feat_b[f"{head}_voice_margin"] = v_margin
        feat_b[f"{head}_text_margin"] = t_margin

    # Global summary features
    mean_v_conf = float(np.mean(voice_confs))
    mean_t_conf = float(np.mean(text_confs))
    feat_b["mean_voice_confidence"] = mean_v_conf
    feat_b["mean_text_confidence"] = mean_t_conf
    feat_b["mean_cross_model_support"] = float(np.mean(cross_supports))
    feat_b["weighted_js_divergence"] = float(
        sum(HEAD_WEIGHTS[h] * js for h, js in zip(HEADS, js_divs))
    )
    feat_b["strong_conflict_score"] = float(
        weighted_disagreements * min(mean_v_conf, mean_t_conf)
    )

    return feat_a, feat_b
