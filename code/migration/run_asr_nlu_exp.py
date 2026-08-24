import os
import json
import pandas as pd
import numpy as np
import whisper
import torch
from tqdm import tqdm
import jiwer

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
WHISPER_MODEL_PATH = "/home/spark2/Models/base.en.pt"
DATASET_CSV = "dataset/nlu_robust_6000_scenario_paraphrase.csv"
AUDIO_DIR = "audio"
ASR_DIR = "asr"
JSON_DIR = os.path.join(ASR_DIR, "json")
PROGRESS_FILE = os.path.join(ASR_DIR, "whisper_decode_progress.json")
OUTPUT_CSV = os.path.join(ASR_DIR, "whisper_transcripts_6000.csv")

def main():
    os.makedirs(JSON_DIR, exist_ok=True)
    
    print(f"Loading dataset from {DATASET_CSV}...")
    df = pd.read_csv(DATASET_CSV)
    
    completed_samples = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            completed_samples = set(json.load(f))
            
    results = []
    if os.path.exists(OUTPUT_CSV):
        results = pd.read_csv(OUTPUT_CSV).to_dict('records')
    
    print(f"Loading Whisper from {WHISPER_MODEL_PATH}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(WHISPER_MODEL_PATH, device=device)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Decoding Audio"):
        sample_id = str(row['sample_id'])
        if sample_id in completed_samples:
            continue
            
        audio_path = os.path.join(AUDIO_DIR, f"{sample_id}.wav")
        if not os.path.exists(audio_path):
            continue
            
        ground_truth = str(row['transcript'])
        
        # Deterministic transcription baseline
        result = model.transcribe(audio_path, temperature=0.0)
        transcript = result['text'].strip()
        
        # Metric calculation
        wer = jiwer.wer(ground_truth, transcript)
        is_error = int(wer > 0)
        
        with open(os.path.join(JSON_DIR, f"{sample_id}.json"), 'w') as f:
            json.dump(result, f)
        
        avg_logprob = np.mean([s['avg_logprob'] for s in result['segments']]) if result['segments'] else 0.0
        no_speech_prob = np.mean([s['no_speech_prob'] for s in result['segments']]) if result['segments'] else 0.0
        compression_ratio = np.mean([s['compression_ratio'] for s in result['segments']]) if result['segments'] else 0.0
        
        record = {
            "sample_id": sample_id,
            "scenario_id": row['scenario_id'],
            "split": row['split'],
            "ground_truth": ground_truth,
            "whisper_transcript": transcript,
            "wer": wer,
            "is_error": is_error,
            "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech_prob,
            "compression_ratio": compression_ratio,
            "temperature": 0.0,
            "language": result.get('language', 'en')
        }
        
        results.append(record)
        completed_samples.add(sample_id)
        
        # Resumable checkpointing
        if len(completed_samples) % 50 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(list(completed_samples), f)
                
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(list(completed_samples), f)
        
    print(f"Decoding complete. Baseline results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
