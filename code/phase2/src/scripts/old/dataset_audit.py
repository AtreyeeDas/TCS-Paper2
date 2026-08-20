import json
import os
from glob import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from config import Config

def run_audit_and_split():
    df = pd.read_csv(Config.MASTER_CSV)
    
    # Generate mock group_ids if missing to allow GroupShuffleSplit to work
    if "group_id" not in df.columns:
        df["group_id"] = df.index
        
    valid_df = df.copy()

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=Config.SEED)
    train_idx, temp_idx = next(gss1.split(valid_df, groups=valid_df["group_id"]))
    
    train_df = valid_df.iloc[train_idx].copy()
    temp_df = valid_df.iloc[temp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=Config.SEED)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["group_id"]))
    
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    splits_dir = os.path.join(Config.ROOT_DIR, "results", "splits")
    os.makedirs(splits_dir, exist_ok=True)
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)
    
    print(f"[+] Splits Generated: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    label_maps_dir = os.path.join(Config.ROOT_DIR, "results", "label_maps")
    os.makedirs(label_maps_dir, exist_ok=True)

    for head in Config.HEADS:
        le = LabelEncoder()
        
        # EXCLUDE MASK AND OUTLIERS FROM BECOMING CLASSES
        valid_mask = ~train_df[head].astype(str).isin(Config.INVALID_TOKENS)
        le.fit(train_df.loc[valid_mask, head].astype(str))
        
        mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
        
        with open(os.path.join(label_maps_dir, f"{head}.json"), "w") as f:
            json.dump(mapping, f, indent=4)
        print(f"  {head}: {len(mapping)} valid classes mapped.")

if __name__ == "__main__":
    run_audit_and_split()
