"""
ASIL NLU Diagnostic Suite - Part I: Dataset, Split, Ontology & Confounding Audit
Executes Parts 1, 2, 3, 4, 10, 13, 14 without modifying original data or re-extracting audio.
"""

import os
import json
import numpy as np
import pandas as pd
from config import Config

DIAG_DIR = os.path.join(Config.ROOT_DIR, "results", "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)

def run_dataset_and_split_diagnostics():
    print(f"\n{'='*80}\n[+] STARTING DATASET & SPLIT DIAGNOSTIC AUDIT\n{'='*80}")
    
    if not os.path.exists(Config.MASTER_CSV):
        raise FileNotFoundError(f"Master CSV not found at {Config.MASTER_CSV}")
        
    df = pd.read_csv(Config.MASTER_CSV)
    splits_dir = os.path.join(Config.ROOT_DIR, "results", "splits")
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "validation.csv")
    test_path = os.path.join(splits_dir, "test.csv")
    
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Split CSVs not found in results/splits/. Run dataset_audit.py first.")
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # -------------------------------------------------------------
    # 1. Dataset Overview & Label Coverage
    # -------------------------------------------------------------
    print("[1/6] Auditing Label Coverage and Out-Of-Vocabulary Classes...")
    coverage_records = []
    
    for head in Config.HEADS:
        valid_all = df[df[head] != Config.MASK_TOKEN][head]
        valid_train = train_df[train_df[head] != Config.MASK_TOKEN][head]
        valid_val = val_df[val_df[head] != Config.MASK_TOKEN][head]
        valid_test = test_df[test_df[head] != Config.MASK_TOKEN][head]
        
        train_classes = set(valid_train.unique())
        val_classes = set(valid_val.unique())
        test_classes = set(valid_test.unique())
        all_classes = set(valid_all.unique())
        
        val_oov = val_classes - train_classes
        test_oov = test_classes - train_classes
        
        counts = valid_all.value_counts()
        min_size = int(counts.min()) if not counts.empty else 0
        max_size = int(counts.max()) if not counts.empty else 0
        median_size = float(counts.median()) if not counts.empty else 0.0
        imbalance_ratio = round(max_size / max(min_size, 1), 2)
        
        coverage_records.append({
            "head": head,
            "total_valid_samples": len(valid_all),
            "mask_samples": (df[head] == Config.MASK_TOKEN).sum(),
            "total_unique_classes": len(all_classes),
            "train_classes_count": len(train_classes),
            "val_classes_count": len(val_classes),
            "test_classes_count": len(test_classes),
            "val_oov_classes": list(val_oov),
            "test_oov_classes": list(test_oov),
            "min_class_size": min_size,
            "max_class_size": max_size,
            "median_class_size": median_size,
            "imbalance_ratio": imbalance_ratio
        })
        
    cov_df = pd.DataFrame(coverage_records)
    cov_df.to_csv(os.path.join(DIAG_DIR, "label_coverage.csv"), index=False)
    print(f"    -> Saved label coverage to {os.path.join(DIAG_DIR, 'label_coverage.csv')}")

    # -------------------------------------------------------------
    # 2. Transcript Leakage & Grouping Verification
    # -------------------------------------------------------------
    print("[2/6] Verifying Transcript-Grouped Splitting & Checking Data Leakage...")
    
    def norm_text(t):
        return str(t).lower().strip() if pd.notna(t) else ""
        
    train_transcripts = set(train_df['transcript'].apply(norm_text))
    val_transcripts = set(val_df['transcript'].apply(norm_text))
    test_transcripts = set(test_df['transcript'].apply(norm_text))
    
    # Remove empty strings from intersection check
    train_transcripts.discard("")
    val_transcripts.discard("")
    test_transcripts.discard("")
    
    leak_train_val = train_transcripts.intersection(val_transcripts)
    leak_train_test = train_transcripts.intersection(test_transcripts)
    leak_val_test = val_transcripts.intersection(test_transcripts)
    
    leak_report = {
        "unique_transcripts_total": int(df['transcript'].apply(norm_text).nunique()),
        "train_val_transcript_leakage_count": len(leak_train_val),
        "train_test_transcript_leakage_count": len(leak_train_test),
        "val_test_transcript_leakage_count": len(leak_val_test),
        "train_val_leaked_transcripts": list(leak_train_val)[:10],
        "train_test_leaked_transcripts": list(leak_train_test)[:10]
    }
    
    with open(os.path.join(DIAG_DIR, "transcript_group_analysis.json"), "w") as f:
        json.dump(leak_report, f, indent=4)
        
    # -------------------------------------------------------------
    # 3. Source / Dataset Confounding Analysis
    # -------------------------------------------------------------
    print("[3/6] Computing Source-Dataset x Class Crosstabs...")
    # Infer source from audio_path or explicit column
    if "source_dataset" in df.columns:
        df["source"] = df["source_dataset"]
    else:
        df["source"] = df["audio_path"].apply(
            lambda p: str(p).split("/")[0] if "/" in str(p) else "unknown"
        )
        
    crosstab_records = []
    for head in Config.HEADS:
        valid_head_df = df[df[head] != Config.MASK_TOKEN]
        ct = pd.crosstab(valid_head_df["source"], valid_head_df[head], normalize="columns")
        
        for cls in ct.columns:
            dominant_source = ct[cls].idxmax()
            dominant_pct = round(ct[cls].max() * 100, 2)
            total_count = int((valid_head_df[head] == cls).sum())
            is_confounded = bool(dominant_pct >= 90.0)
            
            crosstab_records.append({
                "head": head,
                "class": cls,
                "total_count": total_count,
                "dominant_source": dominant_source,
                "dominant_source_percentage": dominant_pct,
                "is_source_confounded": is_confounded
            })
            
    crosstab_df = pd.DataFrame(crosstab_records)
    crosstab_df.to_csv(os.path.join(DIAG_DIR, "source_label_crosstabs.csv"), index=False)
    print(f"    -> Found {crosstab_df['is_source_confounded'].sum()} / {len(crosstab_df)} source-confounded classes.")

    # -------------------------------------------------------------
    # 4. Multi-Task MASK Distribution Topology
    # -------------------------------------------------------------
    print("[4/6] Analyzing Multi-Task MASK Distribution per Sample...")
    valid_head_counts = (df[Config.HEADS] != Config.MASK_TOKEN).sum(axis=1)
    mask_topology = valid_head_counts.value_counts().sort_index()
    
    topology_df = pd.DataFrame({
        "valid_heads_count": mask_topology.index,
        "sample_count": mask_topology.values,
        "percentage": (mask_topology.values / len(df) * 100).round(2)
    })
    topology_df.to_csv(os.path.join(DIAG_DIR, "mask_distribution_per_sample.csv"), index=False)

    # -------------------------------------------------------------
    # 5. Semantic Consistency & Representative Examples
    # -------------------------------------------------------------
    print("[5/6] Auditing Text Semantics for Intent & Urgency...")
    for head in ["intent", "urgency", "emotion"]:
        head_df = df[df[head] != Config.MASK_TOKEN]
        audit_rows = []
        for cls, group in head_df.groupby(head):
            samples = group["transcript"].dropna().astype(str).sample(
                min(15, len(group)), random_state=Config.SEED
            ).tolist()
            audit_rows.append({
                "class": cls,
                "frequency": len(group),
                "representative_transcripts": " | ".join(samples)
            })
        pd.DataFrame(audit_rows).to_csv(os.path.join(DIAG_DIR, f"label_semantic_audit_{head}.csv"), index=False)

    # -------------------------------------------------------------
    # 6. Generate Text Audit Report
    # -------------------------------------------------------------
    print("[6/6] Generating Diagnostic Overview Report...")
    with open(os.path.join(DIAG_DIR, "diagnostic_report.txt"), "w") as f:
        f.write("ASIL NLU - DATASET & SPLIT DIAGNOSTIC REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total Dataset Samples : {len(df)}\n")
        f.write(f"Train Split Samples   : {len(train_df)}\n")
        f.write(f"Val Split Samples     : {len(val_df)}\n")
        f.write(f"Test Split Samples    : {len(test_df)}\n\n")
        f.write("MASK PER HEAD:\n")
        for head in Config.HEADS:
            mask_c = (df[head] == Config.MASK_TOKEN).sum()
            f.write(f"  - {head:<15}: {mask_c:>5} MASK ({(mask_c/len(df)*100):.2f}%)\n")
        f.write("\nLEAKAGE VERIFICATION:\n")
        f.write(f"  - Train/Val Leaked Transcripts  : {len(leak_train_val)}\n")
        f.write(f"  - Train/Test Leaked Transcripts : {len(leak_train_test)}\n")
        f.write(f"  - Val/Test Leaked Transcripts   : {len(leak_val_test)}\n\n")
        f.write("OUT-OF-VOCABULARY TEST CLASSES (Missing from Train):\n")
        for _, r in cov_df.iterrows():
            if len(r['test_oov_classes']) > 0:
                f.write(f"  - {r['head']}: {r['test_oov_classes']}\n")
                
    print(f"[+] Dataset & Split Audit complete. Artifacts written to: {DIAG_DIR}\n")

if __name__ == "__main__":
    run_dataset_and_split_diagnostics()
