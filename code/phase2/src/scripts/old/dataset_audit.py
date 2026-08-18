import json
import os
from glob import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from config import Config


def build_audio_lookup_index(base_dir: str) -> dict:
    """Recursively indexes all .wav, .flac, and .mp3 files inside ASIL_DATASETS."""
    lookup = {}
    if not os.path.exists(base_dir):
        print(f"[!] Warning: Data directory does not exist: {base_dir}")
        return lookup

    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".wav", ".flac", ".mp3")):
                full_path = os.path.abspath(os.path.join(root, f))
                lookup[f] = full_path
    return lookup


def resolve_audio_path(rel_path: str, audio_lookup: dict) -> str:
    """Resolves relative audio paths by checking direct prefix, then falling back to recursive lookup."""
    if pd.isna(rel_path) or str(rel_path).strip() == "":
        return None

    str_path = str(rel_path).strip()

    # 1. Check if already absolute and exists
    if os.path.isabs(str_path) and os.path.exists(str_path):
        return os.path.abspath(str_path)

    # 2. Check direct relative path: ASIL_DATASETS/<audio_path>
    direct_candidate = os.path.abspath(os.path.join(Config.DATA_DIR, str_path))
    if os.path.exists(direct_candidate):
        return direct_candidate

    # 3. Fallback: Lookup by base filename in the recursive index
    base_name = os.path.basename(str_path)
    if base_name in audio_lookup:
        return audio_lookup[base_name]

    return None


def audit_and_split():
    print(f"[+] Loading Master Dataset: {Config.MASTER_CSV}")
    if not os.path.exists(Config.MASTER_CSV):
        raise FileNotFoundError(f"Cannot find master CSV at {Config.MASTER_CSV}")

    df = pd.read_csv(Config.MASTER_CSV)
    print(f"[+] Indexing audio files inside: {Config.DATA_DIR}")
    audio_lookup = build_audio_lookup_index(Config.DATA_DIR)
    print(f"[+] Found {len(audio_lookup)} audio files on disk.")

    # Resolve paths
    df["resolved_path"] = df["audio_path"].apply(
        lambda p: resolve_audio_path(p, audio_lookup)
    )

    missing_df = df[df["resolved_path"].isnull()].copy()
    valid_df = df[df["resolved_path"].notnull()].copy()

    # Log missing files
    results_dir = os.path.join(Config.ROOT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    if not missing_df.empty:
        missing_out = os.path.join(results_dir, "missing_audio.csv")
        missing_df.to_csv(missing_out, index=False)
        print(f"[!] Warning: {len(missing_df)} files not found. Saved to {missing_out}")
    else:
        print("[+] 100% of audio files resolved successfully.")

    print(f"[+] Valid Audio Records: {len(valid_df)} / {len(df)}")

    # Audit Task MASK Distribution
    audit_data = {
        "total_rows": int(len(df)),
        "valid_audio_rows": int(len(valid_df)),
        "missing_audio_rows": int(len(missing_df)),
        "duplicate_audio_paths": int(valid_df["resolved_path"].duplicated().sum()),
        "tasks": {},
    }

    for head in Config.HEADS:
        mask_count = int((valid_df[head] == Config.MASK_TOKEN).sum())
        audit_data["tasks"][head] = {
            "valid_samples": int(len(valid_df) - mask_count),
            "mask_samples": mask_count,
            "mask_percentage": round(float(mask_count / len(valid_df) * 100), 2),
        }

    with open(os.path.join(results_dir, "dataset_audit.json"), "w") as f:
        json.dump(audit_data, f, indent=4)

    # Transcript-Grouped Splitting (Avoid Data Leakage)
    valid_df["group_id"] = (
        valid_df["transcript"]
        .fillna(valid_df["audio_path"])
        .astype(str)
        .str.lower()
        .str.strip()
    )

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=Config.SEED)
    train_idx, temp_idx = next(
        gss1.split(valid_df, groups=valid_df["group_id"])
    )
    train_df = valid_df.iloc[train_idx].copy()
    temp_df = valid_df.iloc[temp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=Config.SEED)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["group_id"]))
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    splits_dir = os.path.join(results_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)
    print(
        f"[+] Splits Generated: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}"
    )

    # Label Encoders (Fitted strictly on TRAIN)
    label_maps_dir = os.path.join(results_dir, "label_maps")
    os.makedirs(label_maps_dir, exist_ok=True)

    for head in Config.HEADS:
        unique_labels = sorted(
            [l for l in train_df[head].unique() if str(l) != Config.MASK_TOKEN]
        )
        l2i = {label: idx for idx, label in enumerate(unique_labels)}
        l2i[Config.MASK_TOKEN] = Config.MASK_ID

        with open(os.path.join(label_maps_dir, f"{head}.json"), "w") as f:
            json.dump(l2i, f, indent=4)

    print("[+] Dataset audit and label mappings completed successfully.")


if __name__ == "__main__":
    audit_and_split()