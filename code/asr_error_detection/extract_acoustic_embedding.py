"""
Extracts frozen HuBERT acoustic representations using Hugging Face Transformers.
Caches mean-pooled acoustic vectors (768-D).
"""
import json
import os
import time
import librosa
import numpy as np
import pandas as pd
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm
from config import Config

os.makedirs(os.path.join(Config.ROOT_DIR, "embeddings", "acoustic"), exist_ok=True)


def extract_acoustic():
    print(f"[+] Loading frozen HuBERT from {Config.HUBERT_PATH}...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(Config.HUBERT_PATH)
    model = HubertModel.from_pretrained(
        Config.HUBERT_PATH, torch_dtype=Config.DTYPE
    ).to(Config.DEVICE)
    model.eval()
    model.requires_grad_(False)

    manifest_path = os.path.join(Config.ROOT_DIR, "results", "embedding_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            "Embedding manifest not found. Run extract_whisper_embeddings.py first."
        )

    manifest_df = pd.read_csv(manifest_path)

    # Verification run on single sample
    sample_audio = manifest_df.iloc[0]["resolved_path"]
    audio, sr = librosa.load(sample_audio, sr=16000)
    inputs = feature_extractor(
        audio, sampling_rate=sr, return_tensors="pt"
    ).to(Config.DEVICE, dtype=Config.DTYPE)

    with torch.inference_mode():
        hidden_state = model(**inputs).last_hidden_state

    verification = {
        "audio_path": sample_audio,
        "duration_sec": len(audio) / sr,
        "hidden_shape": list(hidden_state.shape),
        "dtype": str(hidden_state.dtype),
        "mean": float(hidden_state.float().mean()),
        "std": float(hidden_state.float().std()),
    }
    with open(
        os.path.join(Config.ROOT_DIR, "results", "hubert_verification.json"), "w"
    ) as f:
        json.dump(verification, f, indent=4)

    start_time = time.time()
    acoustic_paths = []

    for idx, row in tqdm(
        manifest_df.iterrows(), total=len(manifest_df), desc="Extracting HuBERT"
    ):
        audio_path = row["resolved_path"]
        sample_id = row["sample_id"]
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            inputs = feature_extractor(
                audio, sampling_rate=sr, return_tensors="pt"
            ).to(Config.DEVICE, dtype=Config.DTYPE)

            with torch.inference_mode():
                out = (
                    model(**inputs)
                    .last_hidden_state.squeeze(0)
                    .cpu()
                    .float()
                    .numpy()
                )  # Shape: [T, 768]

            mean_pool = np.mean(out, axis=0)  # Shape: [768]
            out_path = f"embeddings/acoustic/{sample_id}.npz"
            np.savez_compressed(
                os.path.join(Config.ROOT_DIR, out_path), embedding=mean_pool
            )
            acoustic_paths.append(out_path)

        except Exception as e:
            print(f"[!] Failed to extract HuBERT for {audio_path}: {e}")
            acoustic_paths.append(None)

    manifest_df["acoustic_path"] = acoustic_paths
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[✓] HuBERT extraction complete. Elapsed: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    extract_acoustic()
