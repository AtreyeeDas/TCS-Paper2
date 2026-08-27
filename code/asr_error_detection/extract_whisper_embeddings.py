"""
Extracts frozen Whisper-Large-v3-Turbo representations using native Hugging Face Transformers.
Caches both mean-pooled vectors (1280-D) and full temporal sequences ([T, 1280]).
"""
import os
import time
import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, WhisperModel
from tqdm import tqdm
from config import Config

os.makedirs(os.path.join(Config.ROOT_DIR, "embeddings", "mean_pool"), exist_ok=True)
os.makedirs(os.path.join(Config.ROOT_DIR, "embeddings", "attention_pool"), exist_ok=True)


def extract_embeddings():
    print(f"[+] Loading frozen Whisper Encoder from {Config.WHISPER_PATH}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(Config.WHISPER_PATH)
    model = WhisperModel.from_pretrained(
        Config.WHISPER_PATH, torch_dtype=Config.DTYPE
    ).to(Config.DEVICE)
    model.eval()
    model.requires_grad_(False)

    splits = ["train", "validation", "test"]
    dfs = []
    for split in splits:
        df = pd.read_csv(
            os.path.join(Config.ROOT_DIR, "results", "splits", f"{split}.csv")
        )
        df["split"] = split
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)

    manifest_records = []
    start_time = time.time()

    for idx, row in tqdm(
        full_df.iterrows(), total=len(full_df), desc="Extracting Whisper"
    ):
        sample_id = f"sample_{idx:05d}"
        audio_path = row["resolved_path"]

        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            inputs = feature_extractor(
                audio, sampling_rate=sr, return_tensors="pt"
            ).to(Config.DEVICE, dtype=Config.DTYPE)

            with torch.inference_mode():
                out = (
                    model.encoder(inputs.input_features)
                    .last_hidden_state.squeeze(0)
                    .cpu()
                    .float()
                    .numpy()
                )  # Shape: [T, 1280]

            mean_pool = np.mean(out, axis=0)  # Shape: [1280]

            np.savez_compressed(
                os.path.join(
                    Config.ROOT_DIR, "embeddings", "mean_pool", f"{sample_id}.npz"
                ),
                embedding=mean_pool,
            )
            np.savez_compressed(
                os.path.join(
                    Config.ROOT_DIR, "embeddings", "attention_pool", f"{sample_id}.npz"
                ),
                embedding=out,
            )

            record = row.to_dict()
            record["sample_id"] = sample_id
            record["whisper_mean_path"] = f"embeddings/mean_pool/{sample_id}.npz"
            record["whisper_attn_path"] = f"embeddings/attention_pool/{sample_id}.npz"
            manifest_records.append(record)

        except Exception as e:
            print(f"[!] Error processing {audio_path}: {e}")

    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(
        os.path.join(Config.ROOT_DIR, "results", "embedding_manifest.csv"), index=False
    )
    print(f"[✓] Whisper extraction complete. Elapsed: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    extract_embeddings()
