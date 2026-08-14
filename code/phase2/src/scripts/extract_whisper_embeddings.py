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




"""
 Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia428_utt4.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_13.npz'
  0%|▍                                                                                                                 | 14/3852 [00:03<16:24,  3.90it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia195_utt14.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_14.npz'
  0%|▍                                                                                                                 | 15/3852 [00:03<16:35,  3.85it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia759_utt8.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_15.npz'
  0%|▍                                                                                                                 | 16/3852 [00:04<16:14,  3.94it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia835_utt2.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_16.npz'
  0%|▌                                                                                                                 | 17/3852 [00:04<16:23,  3.90it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia78_utt2.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_17.npz'
  0%|▌                                                                                                                 | 18/3852 [00:04<16:08,  3.96it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia844_utt11.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_18.npz'
  0%|▌                                                                                                                 | 19/3852 [00:04<16:16,  3.92it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia768_utt3.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_19.npz'
  1%|▌                                                                                                                 | 20/3852 [00:05<16:03,  3.98it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia950_utt2.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_20.npz'
  1%|▌                                                                                                                 | 21/3852 [00:05<16:19,  3.91it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia707_utt14.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_21.npz'
  1%|▋                                                                                                                 | 22/3852 [00:05<16:13,  3.93it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia771_utt14.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_22.npz'
  1%|▋                                                                                                                 | 23/3852 [00:05<16:23,  3.89it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia39_utt3.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_23.npz'
  1%|▋                                                                                                                 | 24/3852 [00:06<16:08,  3.95it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia264_utt7.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_24.npz'
  1%|▋                                                                                                                 | 25/3852 [00:06<16:29,  3.87it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia478_utt10.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_25.npz'
  1%|▊                                                                                                                 | 26/3852 [00:06<16:13,  3.93it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia1026_utt15.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_26.npz'
  1%|▊                                                                                                                 | 27/3852 [00:06<16:27,  3.88it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia513_utt4.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_27.npz'
  1%|▊                                                                                                                 | 28/3852 [00:07<16:09,  3.94it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia1012_utt9.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_28.npz'
  1%|▊                                                                                                                 | 29/3852 [00:07<16:25,  3.88it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia1022_utt12.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_29.npz'
  1%|▉                                                                                                                 | 30/3852 [00:07<16:08,  3.95it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia195_utt8.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_30.npz'
  1%|▉                                                                                                                 | 31/3852 [00:07<16:20,  3.90it/s][!] Failed to extract /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/ASIL_Datasets/meld_balanced_1050/anger_dia232_utt7.flac: [Errno 2] No such file or directory: '/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/imp_newpipeline/ASIL_NLU/embeddings/mean_pool/sample_31.npz'
  1%|▉                                                               
  """
"""
