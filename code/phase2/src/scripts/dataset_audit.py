import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from config import Config

def resolve_audio_path(rel_path):
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path
    resolved = os.path.join(Config.DATA_DIR, str(rel_path))
    return resolved if os.path.exists(resolved) else None

def audit_and_split():
    df = pd.read_csv(Config.MASTER_CSV)
    df['resolved_path'] = df['audio_path'].apply(resolve_audio_path)
    
    missing_df = df[df['resolved_path'].isnull()]
    if not missing_df.empty:
        missing_df.to_csv(os.path.join(Config.ROOT_DIR, "results", "missing_audio.csv"), index=False)
        print(f"[!] Found {len(missing_df)} missing audio files. Saved to missing_audio.csv")
    
    df = df[df['resolved_path'].notnull()].copy()
    
    # Generate Audit
    audit_data = {
        "total_rows": len(df) + len(missing_df),
        "valid_audio_rows": len(df),
        "missing_audio_rows": len(missing_df),
        "duplicate_audio_paths": int(df['resolved_path'].duplicated().sum()),
        "tasks": {}
    }
    
    for head in Config.HEADS:
        mask_count = (df[head] == Config.MASK_TOKEN).sum()
        audit_data["tasks"][head] = {
            "valid_samples": int(len(df) - mask_count),
            "mask_samples": int(mask_count),
            "mask_percentage": float(mask_count / len(df) * 100)
        }
        
    with open(os.path.join(Config.ROOT_DIR, "results", "dataset_audit.json"), "w") as f:
        json.dump(audit_data, f, indent=4)
        
    # Group by transcript to avoid leakage
    df['group_id'] = df['transcript'].fillna(df['audio_path']).str.lower().str.strip()
    
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=Config.SEED)
    train_idx, temp_idx = next(gss1.split(df, groups=df['group_id']))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=Config.SEED)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df['group_id']))
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]
    
    splits_dir = os.path.join(Config.ROOT_DIR, "results", "splits")
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)
    
    # Label Encoders (Fit strictly on TRAIN)
    label_maps = {}
    for head in Config.HEADS:
        unique_labels = [l for l in train_df[head].unique() if l != Config.MASK_TOKEN]
        l2i = {l: i for i, l in enumerate(unique_labels)}
        l2i[Config.MASK_TOKEN] = Config.MASK_ID
        label_maps[head] = l2i
        
        with open(os.path.join(Config.ROOT_DIR, "results", "label_maps", f"{head}.json"), "w") as f:
            json.dump(l2i, f, indent=4)

if __name__ == "__main__":
    audit_and_split()
