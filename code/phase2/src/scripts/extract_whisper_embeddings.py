import os
import time
import json
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WhisperModel
from config import Config

def extract_embeddings():
    mean_pool_dir = os.path.join(Config.ROOT_DIR, "embeddings", "mean_pool")
    seq_dir = os.path.join(Config.ROOT_DIR, "embeddings", "sequence_embedding")
    os.makedirs(mean_pool_dir, exist_ok=True)
    os.makedirs(seq_dir, exist_ok=True)

    print(f"[+] Loading frozen Whisper Encoder from {Config.WHISPER_PATH}...")
    processor = AutoFeatureExtractor.from_pretrained(Config.WHISPER_PATH)
    model = WhisperModel.from_pretrained(Config.WHISPER_PATH, torch_dtype=Config.DTYPE).to(Config.DEVICE)
    model.eval()
    model.requires_grad_(False)
    
    full_df = pd.read_csv(os.path.join(Config.ROOT_DIR, "results", "acoustic_feature_manifest.csv"))
    
    manifest_records = []
    failure_records = []
    
    print("[+] Extracting Whisper Embeddings...")
    extraction_start = time.time()
    
    for idx, row in tqdm(full_df.iterrows(), total=len(full_df)):
        sample_id = row['sample_id']
        audio_path = row['resolved_path']
        mean_path = os.path.join(mean_pool_dir, f"{sample_id}.npz")
        seq_path = os.path.join(seq_dir, f"{sample_id}.npz")
        
        try:
            if not os.path.exists(mean_path) or not os.path.exists(seq_path):
                audio, sr = librosa.load(audio_path, sr=16000)
                inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(Config.DEVICE, dtype=Config.DTYPE)
                
                with torch.inference_mode():
                    out = model.encoder(inputs.input_features).last_hidden_state.squeeze(0).cpu().numpy()
                    
                mean_pool = np.mean(out, axis=0)
                
                np.savez_compressed(mean_path, embedding=mean_pool)
                np.savez_compressed(seq_path, embedding=out)
                
            manifest_records.append(row.to_dict())
        except Exception as e:
            failure_records.append({"sample_id": sample_id, "audio_path": audio_path, "error": str(e)})

    total_time = time.time() - extraction_start
    
    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(os.path.join(Config.ROOT_DIR, "results", "embedding_manifest.csv"), index=False)
    
    if failure_records:
        pd.DataFrame(failure_records).to_csv(os.path.join(Config.ROOT_DIR, "results", "embedding_failures.csv"), index=False)
        
    print(f"[+] Total Extraction Time: {total_time:.2f}s | Failures: {len(failure_records)}")

if __name__ == "__main__":
    extract_embeddings()
