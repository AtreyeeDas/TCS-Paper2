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
    print(f"[+] Loading frozen Whisper Encoder from {Config.WHISPER_PATH}...")
    processor = AutoFeatureExtractor.from_pretrained(Config.WHISPER_PATH)
    # Load base model to bypass decoder entirely
    model = WhisperModel.from_pretrained(Config.WHISPER_PATH, torch_dtype=Config.DTYPE).to(Config.DEVICE)
    model.eval()
    model.requires_grad_(False)
    
    train_df = pd.read_csv(os.path.join(Config.ROOT_DIR, "results", "splits", "train.csv"))
    val_df = pd.read_csv(os.path.join(Config.ROOT_DIR, "results", "splits", "validation.csv"))
    test_df = pd.read_csv(os.path.join(Config.ROOT_DIR, "results", "splits", "test.csv"))
    
    train_df['split'] = 'train'
    val_df['split'] = 'validation'
    test_df['split'] = 'test'
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    manifest_records = []
    
    # 1-Sample Verification
    sample_audio = full_df.iloc[0]['resolved_path']
    audio, sr = librosa.load(sample_audio, sr=16000)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(Config.DEVICE, dtype=Config.DTYPE)
    
    with torch.inference_mode():
        encoder_outputs = model.encoder(inputs.input_features)
        hidden_state = encoder_outputs.last_hidden_state # [B, T, 1280]
        
    verification = {
        "audio_path": sample_audio,
        "audio_duration_sec": len(audio) / sr,
        "encoder_shape": list(hidden_state.shape),
        "dtype": str(hidden_state.dtype),
        "minimum": float(hidden_state.min()),
        "maximum": float(hidden_state.max()),
        "mean": float(hidden_state.mean()),
        "std": float(hidden_state.std())
    }
    with open(os.path.join(Config.ROOT_DIR, "results", "whisper_encoder_verification.json"), "w") as f:
        json.dump(verification, f, indent=4)
        
    print("[+] Extracting Embeddings...")
    extraction_start = time.time()
    for idx, row in tqdm(full_df.iterrows(), total=len(full_df)):
        sample_id = f"sample_{idx}"
        audio_path = row['resolved_path']
        
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(Config.DEVICE, dtype=Config.DTYPE)
            
            with torch.inference_mode():
                out = model.encoder(inputs.input_features).last_hidden_state.squeeze(0).cpu().numpy() # [T, 1280]
                
            mean_pool = np.mean(out, axis=0) # [1280]
            
            np.savez_compressed(os.path.join(Config.ROOT_DIR, "embeddings", "mean_pool", f"{sample_id}.npz"), embedding=mean_pool)
            np.savez_compressed(os.path.join(Config.ROOT_DIR, "embeddings", "attention_pool", f"{sample_id}.npz"), embedding=out)
            
            record = row.to_dict()
            record['sample_id'] = sample_id
            manifest_records.append(record)
            
        except Exception as e:
            print(f"[!] Failed to extract {audio_path}: {e}")

    total_time = time.time() - extraction_start
    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(os.path.join(Config.ROOT_DIR, "results", "embedding_manifest.csv"), index=False)
    print(f"[+] Total Extraction Time: {total_time:.2f}s | Avg: {total_time/len(full_df):.2f}s/sample")

if __name__ == "__main__":
    extract_embeddings()
