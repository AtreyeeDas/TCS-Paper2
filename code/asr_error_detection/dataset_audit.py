"""
Dataset Audit & Split Generation.
Resolves audio paths, separates synthetic audio (train-only), 
and generates Group-Disjoint train/validation/test splits based on transcript groups.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from config import Config


def build_audio_lookup_index(base_dir: str) -> dict:
    lookup = {}
    if not os.path.exists(base_dir):
        print(f"[!] Warning: Data directory does not exist: {base_dir}")
        return lookup
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".wav", ".flac", ".mp3")):
                lookup[f] = os.path.abspath(os.path.join(root, f))
    return lookup


def resolve_audio_path(rel_path: str, audio_lookup: dict) -> str:
    if pd.isna(rel_path) or str(rel_path).strip() == "":
        return None
    str_path = str(rel_path).strip()
    if os.path.isabs(str_path) and os.path.exists(str_path):
        return os.path.abspath(str_path)
    direct_candidate = os.path.abspath(os.path.join(Config.DATA_DIR, str_path))
    if os.path.exists(direct_candidate):
        return direct_candidate
    base_name = os.path.basename(str_path)
    return audio_lookup.get(base_name, None)


def audit_and_split():
    print(f"[+] Loading Master Dataset: {Config.MASTER_CSV}")
    df = pd.read_csv(Config.MASTER_CSV)
    audio_lookup = build_audio_lookup_index(Config.DATA_DIR)

    df["resolved_path"] = df["audio_path"].apply(
        lambda p: resolve_audio_path(p, audio_lookup)
    )
    missing_df = df[df["resolved_path"].isnull()].copy()
    valid_df = df[df["resolved_path"].notnull()].copy()

    results_dir = os.path.join(Config.ROOT_DIR, "results")
    splits_dir = os.path.join(results_dir, "splits")
    label_maps_dir = os.path.join(results_dir, "label_maps")
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(label_maps_dir, exist_ok=True)

    if not missing_df.empty:
        print(f"[!] Warning: {len(missing_df)} audio files could not be located.")
        missing_df.to_csv(os.path.join(results_dir, "missing_audio.csv"), index=False)

    # Identify synthetic samples to enforce train-only allocation
    valid_df["is_synthetic"] = (
        valid_df["resolved_path"].str.lower().str.contains("synthetic")
    )
    synthetic_count = valid_df["is_synthetic"].sum()
    print(f"[+] Found {synthetic_count} synthetic samples (Allocated to train split).")

    # Group by transcript text to prevent leakage between train and test
    valid_df["group_id"] = (
        valid_df["transcript"]
        .fillna(valid_df["audio_path"])
        .astype(str)
        .str.lower()
        .str.strip()
    )

    real_df = valid_df[~valid_df["is_synthetic"]].copy()
    synth_df = valid_df[valid_df["is_synthetic"]].copy()

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=Config.SEED)
    train_idx, temp_idx = next(gss1.split(real_df, groups=real_df["group_id"]))
    train_df = real_df.iloc[train_idx].copy()
    temp_df = real_df.iloc[temp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=Config.SEED)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["group_id"]))
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # Append synthetic audio to train split
    train_df = pd.concat([train_df, synth_df], ignore_index=True)

    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)

    # Generate label maps
    distribution_records = []
    for head in Config.HEADS:
        unique_labels = sorted(
            [l for l in train_df[head].unique() if str(l) != Config.MASK_TOKEN]
        )
        l2i = {label: idx for idx, label in enumerate(unique_labels)}
        l2i[Config.MASK_TOKEN] = Config.MASK_ID
        with open(os.path.join(label_maps_dir, f"{head}.json"), "w") as f:
            json.dump(l2i, f, indent=4)

        for cls in unique_labels:
            tr_c = (train_df[head] == cls).sum()
            vl_c = (val_df[head] == cls).sum()
            te_c = (test_df[head] == cls).sum()
            distribution_records.append({
                "head": head,
                "class": cls,
                "train": tr_c,
                "val": vl_c,
                "test": te_c,
                "total": tr_c + vl_c + te_c,
            })

    pd.DataFrame(distribution_records).to_csv(
        os.path.join(splits_dir, "class_distribution_by_split.csv"), index=False
    )
    print("[✓] Dataset audit and group splits completed successfully.")


if __name__ == "__main__":
    audit_and_split()
