#!/usr/bin/env python3
"""
feature_extractor.py
Comprehensive Feature Extraction Engine for ASR -> NLU Error Detection.
Supports:
- Detector A (Hard Label Disagreements)
- Detector B (Posterior & Information-Theoretic Disagreement)
- Detector C (Text Clean -> Corrupted Posterior Displacement)
- Detector D (Excess Cross-Modal Disagreement Deltas)
- Hierarchical Transition Encoding
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, jensenshannon
from config import EPS, HEAD_WEIGHTS, HEADS


def compute_entropy(probs: np.ndarray) -> float:
    """Computes Shannon entropy: H(p) = -sum(p * log(p + eps))."""
    p = np.clip(probs, EPS, 1.0)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log(p)))


def compute_margin(probs: np.ndarray) -> float:
    """Computes difference between top-1 and top-2 class probabilities."""
    if len(probs) < 2:
        return 1.0
    sorted_probs = np.sort(probs)[::-1]
    return float(sorted_probs[0] - sorted_probs[1])


def align_distributions(
    p1: np.ndarray, classes1: np.ndarray, p2: np.ndarray, classes2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Aligns two discrete distributions over the union of their class vocabularies."""
    all_classes = sorted(list(set(classes1) | set(classes2)))
    m1 = {c: p for c, p in zip(classes1, p1)}
    m2 = {c: p for c, p in zip(classes2, p2)}

    a1 = np.array([m1.get(c, EPS) for c in all_classes], dtype=float)
    a2 = np.array([m2.get(c, EPS) for c in all_classes], dtype=float)

    a1 = a1 / np.sum(a1)
    a2 = a2 / np.sum(a2)
    return a1, a2, all_classes


def align_and_compute_js_divergence(
    voice_probs: np.ndarray,
    voice_classes: np.ndarray,
    text_probs: np.ndarray,
    text_classes: np.ndarray,
) -> float:
    """Computes JS divergence between Voice and Text probability distributions."""
    v_aligned, t_aligned, _ = align_distributions(
        voice_probs, voice_classes, text_probs, text_classes
    )
    js_dist = jensenshannon(v_aligned, t_aligned, base=2)
    return float(js_dist**2) if not np.isnan(js_dist) else 0.0


def extract_baseline_features(
    voice_row: pd.Series,
    text_row: pd.Series,
    voice_posteriors_sample: Dict[str, np.ndarray],
    text_posteriors_sample: Dict[str, np.ndarray],
    voice_encoders: Dict[str, Any],
    text_encoders: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Extracts Detector A and Detector B features for a Voice-Text pair."""
    feat_a = {}
    feat_b = {}

    total_disagreements = 0.0
    weighted_disagreements = 0.0

    for head in HEADS:
        v_label = str(voice_row[f"voice_{head}"])
        t_label = str(text_row[f"text_{head}"])
        disagree = 1.0 if v_label != t_label else 0.0

        feat_a[f"{head}_disagreement"] = disagree
        total_disagreements += disagree
        weighted_disagreements += HEAD_WEIGHTS[head] * disagree

    feat_a["total_disagreements"] = float(total_disagreements)
    feat_a["weighted_disagreement"] = float(weighted_disagreements)

    feat_b.update(feat_a)

    voice_confs, text_confs, cross_supports, js_divs = [], [], [], []

    for head in HEADS:
        v_label = str(voice_row[f"voice_{head}"])
        t_label = str(text_row[f"text_{head}"])

        v_probs = voice_posteriors_sample[head]
        t_probs = text_posteriors_sample[head]

        v_classes = voice_encoders[f"{head}_label"].classes_
        t_classes = text_encoders[f"{head}_label"].classes_

        v_top1_conf = float(np.max(v_probs))
        t_top1_conf = float(np.max(t_probs))
        conf_gap = abs(v_top1_conf - t_top1_conf)

        voice_confs.append(v_top1_conf)
        text_confs.append(t_top1_conf)

        v_class_to_idx = {c: idx for idx, c in enumerate(v_classes)}
        t_class_to_idx = {c: idx for idx, c in enumerate(t_classes)}

        t_prob_of_v_choice = (
            float(t_probs[t_class_to_idx[v_label]])
            if v_label in t_class_to_idx
            else 0.0
        )
        v_prob_of_t_choice = (
            float(v_probs[v_class_to_idx[t_label]])
            if t_label in v_class_to_idx
            else 0.0
        )

        cross_supports.extend([t_prob_of_v_choice, v_prob_of_t_choice])

        js_div = align_and_compute_js_divergence(
            v_probs, v_classes, t_probs, t_classes
        )
        js_divs.append(js_div)

        v_entropy = compute_entropy(v_probs)
        t_entropy = compute_entropy(t_probs)
        v_margin = compute_margin(v_probs)
        t_margin = compute_margin(t_probs)

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


def extract_posterior_displacement_features(
    clean_text_row: pd.Series,
    corrupted_text_row: pd.Series,
    clean_posteriors: Dict[str, np.ndarray],
    corrupted_posteriors: Dict[str, np.ndarray],
    text_encoders: Dict[str, Any],
) -> Dict[str, float]:
    """Extracts features capturing movement from Text(clean) to Text(corrupted)."""
    feats = {}
    js_list, l1_list, l2_list, cos_list = [], [], [], []

    for head in HEADS:
        c_label = str(clean_text_row[f"text_{head}"])
        e_label = str(corrupted_text_row[f"text_{head}"])
        label_changed = 1.0 if c_label != e_label else 0.0
        feats[f"disp_{head}_label_changed"] = label_changed

        c_probs = clean_posteriors[head]
        e_probs = corrupted_posteriors[head]
        classes = text_encoders[f"{head}_label"].classes_

        c_aligned, e_aligned, _ = align_distributions(
            c_probs, classes, e_probs, classes
        )

        js_div = float(jensenshannon(c_aligned, e_aligned, base=2) ** 2)
        if np.isnan(js_div):
            js_div = 0.0
        l1_dist = float(np.sum(np.abs(c_aligned - e_aligned)))
        l2_dist = float(np.linalg.norm(c_aligned - e_aligned))
        cos_dist = float(cosine(c_aligned, e_aligned))
        if np.isnan(cos_dist):
            cos_dist = 0.0

        feats[f"disp_{head}_js_divergence"] = js_div
        feats[f"disp_{head}_l1_distance"] = l1_dist
        feats[f"disp_{head}_l2_distance"] = l2_dist
        feats[f"disp_{head}_cosine_distance"] = cos_dist

        js_list.append(js_div)
        l1_list.append(l1_dist)
        l2_list.append(l2_dist)
        cos_list.append(cos_dist)

        c_top1_conf = float(np.max(c_probs))
        e_top1_conf = float(np.max(e_probs))
        feats[f"disp_{head}_clean_top1_conf"] = c_top1_conf
        feats[f"disp_{head}_error_top1_conf"] = e_top1_conf
        feats[f"disp_{head}_conf_delta"] = e_top1_conf - c_top1_conf
        feats[f"disp_{head}_abs_conf_delta"] = abs(e_top1_conf - c_top1_conf)

        c_margin = compute_margin(c_probs)
        e_margin = compute_margin(e_probs)
        feats[f"disp_{head}_clean_margin"] = c_margin
        feats[f"disp_{head}_error_margin"] = e_margin
        feats[f"disp_{head}_margin_delta"] = e_margin - c_margin

        c_entropy = compute_entropy(c_probs)
        e_entropy = compute_entropy(e_probs)
        feats[f"disp_{head}_clean_entropy"] = c_entropy
        feats[f"disp_{head}_error_entropy"] = e_entropy
        feats[f"disp_{head}_entropy_delta"] = e_entropy - c_entropy

        cls_to_idx = {c: idx for idx, c in enumerate(classes)}
        p_clean_of_clean = (
            float(c_probs[cls_to_idx[c_label]]) if c_label in cls_to_idx else 0.0
        )
        p_err_of_clean = (
            float(e_probs[cls_to_idx[c_label]]) if c_label in cls_to_idx else 0.0
        )
        p_clean_of_err = (
            float(c_probs[cls_to_idx[e_label]]) if e_label in cls_to_idx else 0.0
        )
        p_err_of_err = (
            float(e_probs[cls_to_idx[e_label]]) if e_label in cls_to_idx else 0.0
        )

        feats[f"disp_{head}_p_clean_of_clean"] = p_clean_of_clean
        feats[f"disp_{head}_p_error_of_clean"] = p_err_of_clean
        feats[f"disp_{head}_delta_p_clean_label"] = (
            p_err_of_clean - p_clean_of_clean
        )
        feats[f"disp_{head}_p_clean_of_error"] = p_clean_of_err
        feats[f"disp_{head}_p_error_of_error"] = p_err_of_err

    feats["disp_mean_js"] = float(np.mean(js_list))
    feats["disp_mean_l1"] = float(np.mean(l1_list))
    feats["disp_mean_l2"] = float(np.mean(l2_list))
    feats["disp_mean_cosine"] = float(np.mean(cos_list))
    feats["disp_weighted_js"] = float(
        sum(HEAD_WEIGHTS[h] * js for h, js in zip(HEADS, js_list))
    )
    return feats


def extract_excess_cross_modal_features(
    clean_feat_b: Dict[str, float], corrupted_feat_b: Dict[str, float]
) -> Dict[str, float]:
    """Computes excess cross-modal delta: (Voice <-> Corrupted Text) - (Voice <-> Clean Text)."""
    feats = {}
    delta_keys = [
        "total_disagreements",
        "weighted_disagreement",
        "mean_voice_confidence",
        "mean_text_confidence",
        "mean_cross_model_support",
        "weighted_js_divergence",
        "strong_conflict_score",
    ]

    for head in HEADS:
        delta_keys.extend(
            [
                f"{head}_disagreement",
                f"{head}_voice_top1_confidence",
                f"{head}_text_top1_confidence",
                f"{head}_confidence_gap",
                f"{head}_text_prob_of_voice_label",
                f"{head}_voice_prob_of_text_label",
                f"{head}_js_divergence",
                f"{head}_voice_entropy",
                f"{head}_text_entropy",
                f"{head}_voice_margin",
                f"{head}_text_margin",
            ]
        )

    for k in delta_keys:
        if k in clean_feat_b and k in corrupted_feat_b:
            feats[f"delta_{k}"] = float(corrupted_feat_b[k] - clean_feat_b[k])

    return feats


def extract_hierarchical_transition_features(
    clean_text_row: pd.Series, corrupted_text_row: pd.Series
) -> Dict[str, Any]:
    """Explicitly extracts hierarchical change transitions across the 4 levels."""
    feats = {}
    changed_bits = []

    for head in HEADS:
        c_val = str(clean_text_row[f"text_{head}"])
        e_val = str(corrupted_text_row[f"text_{head}"])
        chg = 1 if c_val != e_val else 0
        feats[f"hier_{head}_changed"] = float(chg)
        changed_bits.append(chg)

    pattern_str = "".join(str(b) for b in changed_bits)
    feats["hier_num_changed_levels"] = float(sum(changed_bits))

    first_changed = -1
    deepest_changed = -1
    for idx, bit in enumerate(changed_bits):
        if bit == 1:
            if first_changed == -1:
                first_changed = idx
            deepest_changed = idx

    feats["hier_first_changed_level"] = float(first_changed)
    feats["hier_deepest_changed_level"] = float(deepest_changed)
    feats["hier_domain_changed_bit"] = float(changed_bits[0])
    feats["hier_subdomain_changed_bit"] = float(changed_bits[1])
    feats["hier_topic_changed_bit"] = float(changed_bits[2])
    feats["hier_document_type_changed_bit"] = float(changed_bits[3])
    feats["hier_pattern_code"] = pattern_str

    return feats
