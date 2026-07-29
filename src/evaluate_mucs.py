import os
import time
import torch
import whisper
import librosa
import numpy as np
from jiwer import wer
from scipy.spatial.distance import cosine
from speechbrain.inference.speaker import EncoderClassifier

# --- PATH CONFIGURATION ---
WHISPER_MODEL_PATH = "/home/spark2/Models/whisper_large_v3_turbo"
REF_AUDIO_PATH = "/home/spark2/ICASSP_Project/Monika_lively.wav"
GEN_AUDIO_DIR = "/home/spark2/ICASSP_Project/outputs"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("[+] Loading Offline Whisper Large-v3...")
whisper_model = whisper.load_model(WHISPER_MODEL_PATH, device=device)

print("[+] Loading ECAPA-TDNN for Speaker Similarity (SIM-R)...")
spk_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    run_opts={"device": device}
)

def compute_sim_r(ref_path, gen_path):
    signal_ref, sr1 = librosa.load(ref_path, sr=16000)
    signal_gen, sr2 = librosa.load(gen_path, sr=16000)
    
    emb1 = spk_encoder.encode_batch(torch.tensor(signal_ref).unsqueeze(0).to(device))
    emb2 = spk_encoder.encode_batch(torch.tensor(signal_gen).unsqueeze(0).to(device))
    
    vec1 = emb1.squeeze().cpu().numpy()
    vec2 = emb2.squeeze().cpu().numpy()
    
    return 1.0 - cosine(vec1, vec2)

def compute_thd_and_wer(gen_path, ground_truth_text):
    # Load audio duration
    audio_dur = librosa.get_duration(filename=gen_path)
    
    # Whisper Transcription with Word Timestamps
    result = whisper_model.transcribe(gen_path, word_timestamps=True, language="en")
    transcription = result["text"].strip().lower()
    
    # Calculate Normalized WER
    gt_clean = ground_truth_text.lower().replace(".", "").replace(",", "")
    hyp_clean = transcription.replace(".", "").replace(",", "")
    calculated_wer = wer(gt_clean, hyp_clean) * 100.0
    
    # Calculate THD (Trailing Hallucination Duration)
    last_word_end = 0.0
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            last_word_end = max(last_word_end, word["end"])
            
    thd = max(0.0, audio_dur - last_word_end)
    return calculated_wer, thd

def evaluate_ablation_arm(arm_name, test_dataset):
    print(f"\n=================== Evaluating {arm_name} ===================")
    wers, sims, thds, rtfs = [], [], [], []
    
    for idx, (gt_text, wav_id) in enumerate(test_dataset):
        gen_wav = os.path.join(GEN_AUDIO_DIR, f"{arm_name}_{wav_id}.wav")
        
        if not os.path.exists(gen_wav):
            continue
            
        start_time = time.time()
        
        # Metrics Calculation
        sim = compute_sim_r(REF_AUDIO_PATH, gen_wav)
        wer_val, thd_val = compute_thd_and_wer(gen_wav, gt_text)
        
        gen_time = time.time() - start_time
        audio_len = librosa.get_duration(filename=gen_wav)
        rtf = gen_time / audio_len if audio_len > 0 else 0
        
        wers.append(wer_val)
        sims.append(sim)
        thds.append(thd_val)
        rtfs.append(rtf)
        
    print(f"Results for {arm_name}:")
    print(f"WER (%): {np.mean(wers):.2f}")
    print(f"SIM-R:   {np.mean(sims):.2f}")
    print(f"THD (s): {np.mean(thds):.2f}")
    print(f"RTF:     {np.mean(rtfs):.3f}")
