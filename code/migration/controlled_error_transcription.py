import os
import pandas as pd
import random
import jiwer
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION & PHONETIC DICTIONARY
# ---------------------------------------------------------
DATASET_CSV = "dataset/nlu_robust_6000_scenario_paraphrase.csv"
OUTPUT_CSV = "asr/controlled_error_benchmark.csv"

# Realistic phonetic/ASR confusions mapped to specific domains
# Expand this dictionary using extracted terms from your training split
SUBSTITUTIONS = {
    # Medical
    "troponin": "treponema",
    "ischemia": "schema",
    "tachycardia": "tacky cardia",
    "myocardial": "myocarditis",
    "arrhythmia": "erythema",
    # Finance
    "amortization": "amortization",
    "liquidity": "liquid tea",
    "EBITDA": "ebit da",
    "fiduciary": "fiduciary",
    # Legal
    "subpoena": "sabina",
    "affidavit": "after david",
    "litigation": "mitigation"
}

def inject_phonetic_errors(transcript, substitutions):
    """Replaces domain terms with phonetic hallucinations."""
    words = transcript.split()
    corrupted = False
    
    for i, word in enumerate(words):
        clean_word = word.strip('.,!?()').lower()
        if clean_word in substitutions:
            # Inject the error
            words[i] = word.lower().replace(clean_word, substitutions[clean_word])
            corrupted = True
            
    return " ".join(words), corrupted

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.read_csv(DATASET_CSV)
    
    # Isolate training scenarios to prevent leakage
    train_df = df[df['split'] == 'train'].copy()
    
    results = []
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Injecting Controlled Errors"):
        clean_transcript = str(row['transcript'])
        
        # Inject controlled errors
        corrupted_transcript, is_corrupted = inject_phonetic_errors(clean_transcript, SUBSTITUTIONS)
        
        # Calculate degradation metrics
        wer = jiwer.wer(clean_transcript, corrupted_transcript)
        
        results.append({
            "sample_id": row['sample_id'],
            "scenario_id": row['scenario_id'],
            "domain_label": row['domain_label'],
            "clean_transcript": clean_transcript,
            "corrupted_transcript": corrupted_transcript,
            "is_corrupted": int(is_corrupted), # 0 = correct, 1 = intentionally corrupted
            "injected_wer": wer
        })
        
    benchmark_df = pd.DataFrame(results)
    benchmark_df.to_csv(OUTPUT_CSV, index=False)
    
    total_corrupted = benchmark_df['is_corrupted'].sum()
    print(f"\n[+] Generated controlled benchmark: {OUTPUT_CSV}")
    print(f"[+] Total training samples processed: {len(benchmark_df)}")
    print(f"[+] Successfully corrupted samples: {total_corrupted}")

if __name__ == "__main__":
    # Ensure reproducible randomness if extending with random probability checks
    random.seed(42)
    main()
