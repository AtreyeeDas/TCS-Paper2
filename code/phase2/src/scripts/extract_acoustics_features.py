import os
import json
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config

def extract_features(audio, sr):
    features = []
    # 1-4: Energy RMS
    rms = librosa.feature.rms(y=audio)[0]
    features.extend([np.mean(rms), np.std(rms), np.max(rms), np.max(rms)-np.min(rms) if len(rms)>0 else 0])
    
    # 5-9: Pitch
    f0, voiced_flag, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    f0_voiced = f0[voiced_flag] if f0 is not None and len(f0[voiced_flag]) > 0 else np.array([0])
    features.extend([np.mean(f0_voiced), np.std(f0_voiced), np.median(f0_voiced), np.max(f0_voiced)-np.min(f0_voiced), np.mean(voiced_flag)])
    
    # 10-15: Spectral
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    features.extend([np.mean(cent), np.std(cent), np.mean(bw), np.mean(rolloff), np.mean(zcr), np.std(zcr)])
    
    # 16-20: Temporal
    duration = len(audio) / sr
    features.extend([duration, np.mean(rms > np.mean(rms)*0.5), 1-np.mean(rms > np.mean(rms)*0.5), len(f0_voiced)/max(1, len(f0)), len(f0_voiced)])
    
    # 21-46: MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    
    # Clean NaN/Inf
    feats = np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    # Ensure exactly 46 dimensions
    if len(feats) < 46:
        feats = np.pad(feats, (0, 46 - len(feats)))
    return feats[:46]

def run_extraction():
    splits_dir = os.path.join(Config.ROOT_DIR, "results", "splits")
    ac_dir = os.path.join(Config.ROOT_DIR, "embeddings", "acoustic")
    res_dir = os.path.join(Config.ROOT_DIR, "results")
    os.makedirs(ac_dir, exist_ok=True)

    dfs = []
    for split in ["train", "validation", "test"]:
        df = pd.read_csv(os.path.join(splits_dir, f"{split}.csv"))
        df['split'] = split
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)

    print("[+] Extracting Acoustic Features...")
    train_features = []
    manifest = []
    failures = 0
    
    for idx, row in tqdm(full_df.iterrows(), total=len(full_df)):
        sample_id = f"sample_{idx}"
        npz_path = os.path.join(ac_dir, f"{sample_id}.npz")
        
        try:
            if not os.path.exists(npz_path):
                audio, sr = librosa.load(row['resolved_path'], sr=16000)
                feats = extract_features(audio, sr)
                np.savez_compressed(npz_path, features=feats)
            else:
                feats = np.load(npz_path)['features']
            
            if row['split'] == 'train':
                train_features.append(feats)
            
            rec = row.to_dict()
            rec['sample_id'] = sample_id
            manifest.append(rec)
        except Exception as e:
            failures += 1

    train_feats = np.array(train_features)
    scaler = {
        "mean": np.mean(train_feats, axis=0).tolist(),
        "std": (np.std(train_feats, axis=0) + 1e-8).tolist()
    }
    
    with open(os.path.join(res_dir, "acoustic_scaler.json"), "w") as f:
        json.dump(scaler, f, indent=4)
        
    pd.DataFrame(manifest).to_csv(os.path.join(res_dir, "acoustic_feature_manifest.csv"), index=False)
    print(f"[+] Extracted acoustic features. Failures: {failures}")

if __name__ == "__main__":
    run_extraction()
