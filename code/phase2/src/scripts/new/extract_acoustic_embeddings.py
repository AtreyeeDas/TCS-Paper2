import os
import time
import json
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from config import Config

os.makedirs(os.path.join(Config.ROOT_DIR, "embeddings/acoustic"), exist_ok=True)

def extract_acoustic():
    print(f"[+] Loading frozen HuBERT from {Config.HUBERT_PATH}...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(Config.HUBERT_PATH)
    # Load base HubertModel to bypass emotion classification head
    model = HubertModel.from_pretrained(Config.HUBERT_PATH).to(Config.DEVICE)
    model.to(Config.DTYPE)
    model.eval()
    model.requires_grad_(False)
    
    manifest_path = os.path.join(Config.ROOT_DIR, "results", "embedding_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("Run extract_whisper_embeddings.py first.")
    
    manifest_df = pd.read_csv(manifest_path)
    
    # 1-Sample Verification
    sample_audio = manifest_df.iloc[0]['resolved_path']
    audio, sr = librosa.load(sample_audio, sr=16000)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(Config.DEVICE, dtype=Config.DTYPE)
    
    with torch.inference_mode():
        hidden_state = model(**inputs).last_hidden_state # [1, T, 768]
        
    verification = {
        "audio_path": sample_audio,
        "audio_duration_sec": len(audio) / sr,
        "hidden_shape": list(hidden_state.shape),
        "dtype": str(hidden_state.dtype),
        "minimum": float(hidden_state.min()),
        "maximum": float(hidden_state.max()),
        "mean": float(hidden_state.mean()),
        "std": float(hidden_state.std())
    }
    with open(os.path.join(Config.ROOT_DIR, "results", "hubert_verification.json"), "w") as f:
        json.dump(verification, f, indent=4)
        
    extraction_start = time.time()
    acoustic_paths = []
    
    for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Extracting HuBERT"):
        audio_path = row['resolved_path']
        sample_id = row['sample_id']
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(Config.DEVICE, dtype=Config.DTYPE)
            
            with torch.inference_mode():
                out = model(**inputs).last_hidden_state.squeeze(0).cpu().numpy() # [T, 768]
                
            mean_pool = np.mean(out, axis=0) # [768]
            out_path = f"embeddings/acoustic/{sample_id}.npz"
            np.savez_compressed(os.path.join(Config.ROOT_DIR, out_path), embedding=mean_pool)
            acoustic_paths.append(out_path)
            
        except Exception as e:
            print(f"[!] Failed to extract HuBERT for {audio_path}: {e}")
            acoustic_paths.append(None)

    manifest_df['acoustic_path'] = acoustic_paths
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[+] Total HuBERT Extraction Time: {time.time() - extraction_start:.2f}s")

if __name__ == "__main__":
    extract_acoustic()
