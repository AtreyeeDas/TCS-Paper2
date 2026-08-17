"""
Extracts raw frequencies, source datasets, and representative transcripts for every head.
Performs deterministic string normalization without semantic merging.
"""
import os
import re
import json
import pandas as pd
from ontology_config import OntologyConfig

def deterministic_normalize(text: str) -> str:
    """Performs mechanical string normalization (casing, whitespace, punctuation)."""
    if pd.isna(text) or str(text).strip() == "" or str(text).strip() == OntologyConfig.MASK_TOKEN:
        return OntologyConfig.MASK_TOKEN
    s = str(text).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)      # Replace spaces/hyphens with single underscore
    s = re.sub(r"[^\w_]", "", s)        # Remove punctuation
    s = re.sub(r"_+", "_", s).strip("_") # Collapse multiple underscores
    return s.upper()

def infer_source_dataset(audio_path: str) -> str:
    """Infers source dataset from folder hierarchy."""
    path_str = str(audio_path).lower()
    if "meld" in path_str:
        return "MELD"
    elif "med" in path_str:
        return "MedDialog"
    elif "slurp" in path_str:
        return "SLURP"
    elif "earnings" in path_str:
        return "Earnings-21"
    return "UNKNOWN"

def run_extraction():
    OntologyConfig.make_dirs()
    print(f"[+] Loading Master Dataset: {OntologyConfig.MASTER_CSV}")
    df = pd.read_csv(OntologyConfig.MASTER_CSV)
    df["source_dataset"] = df["audio_path"].apply(infer_source_dataset)
    df["transcript"] = df["transcript"].fillna("").astype(str)

    summary_stats = {}

    for head in OntologyConfig.HEADS:
        print(f"[+] Extracting raw & normalized statistics for head: '{head}'...")
        non_mask_df = df[df[head].astype(str) != OntologyConfig.MASK_TOKEN].copy()
        
        # 1. Raw statistics
        raw_rows = []
        for raw_val, group in non_mask_df.groupby(head):
            freq = len(group)
            pct = (freq / len(df)) * 100.0
            sources = sorted(group["source_dataset"].unique().tolist())
            examples = [t for t in group["transcript"].unique().tolist() if t.strip()][:5]
            
            norm_val = deterministic_normalize(raw_val)
            raw_rows.append({
                "raw_label": str(raw_val),
                "normalized_label": norm_val,
                "frequency": freq,
                "percentage": round(pct, 3),
                "source_datasets": "|".join(sources),
                "num_examples": len(examples),
                "example_transcripts": " || ".join(examples)
            })

        raw_df = pd.DataFrame(raw_rows).sort_values(by="frequency", ascending=False)
        raw_df.to_csv(os.path.join(OntologyConfig.RAW_STATS_DIR, f"{head}.csv"), index=False)

        # 2. Normalized consolidated statistics
        norm_rows = []
        non_mask_df["norm_val"] = non_mask_df[head].apply(deterministic_normalize)
        
        for norm_val, group in non_mask_df.groupby("norm_val"):
            orig_members = group[head].astype(str).unique().tolist()
            member_freqs = group[head].astype(str).value_counts().to_dict()
            total_freq = len(group)
            pct = (total_freq / len(df)) * 100.0
            sources = sorted(group["source_dataset"].unique().tolist())
            examples = [t for t in group["transcript"].unique().tolist() if t.strip()][:5]

            norm_rows.append({
                "normalized_label": norm_val,
                "original_labels": json.dumps(orig_members),
                "member_frequencies": json.dumps(member_freqs),
                "total_frequency": total_freq,
                "percentage": round(pct, 3),
                "source_datasets": json.dumps(sources),
                "example_transcripts": json.dumps(examples)
            })

        norm_df = pd.DataFrame(norm_rows).sort_values(by="total_frequency", ascending=False)
        norm_df.to_csv(os.path.join(OntologyConfig.NORM_STATS_DIR, f"{head}.csv"), index=False)

        summary_stats[head] = {
            "total_valid_rows": int(len(non_mask_df)),
            "mask_rows": int(len(df) - len(non_mask_df)),
            "unique_raw_labels": int(len(raw_df)),
            "unique_normalized_labels": int(len(norm_df))
        }

    with open(os.path.join(OntologyConfig.RESULTS_DIR, "extraction_summary.json"), "w") as f:
        json.dump(summary_stats, f, indent=4)

    print("[+] Raw and Normalized statistics successfully generated.")

if __name__ == "__main__":
    run_extraction()
