import os
import re
import json
import torch
import librosa
import pandas as pd
from glob import glob
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM
)

# ==========================================
# 1. HARDWARE & LOCAL PATH CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16  # Optimized for Blackwell GPU

GEMMA_PATH = "/home/spark2/Models/gemma4-e4b-it"
WHISPER_PATH = "/home/spark2/Models/whisper_large_v3_turbo"

# Dataset Directories
MELD_DIR = "./meld_balanced_1050"
MED_DIR = "./medical_dialogue_1500"
SLURP_DIR = "./slurp_general_1500"
EARNINGS_DIR = "./earnings_21_clean_segments"
OUTPUT_DIR = "./processed_nlu_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[+] Initializing Blackwell CUDA Runtime (Device: {torch.cuda.get_device_name(0)} | Dtype: {DTYPE})")

# ==========================================
# 2. LOAD LOCAL MODELS DIRECTLY ON GPU
# ==========================================
print(f"[+] Loading local Whisper model from {WHISPER_PATH}...")
whisper_processor = AutoProcessor.from_pretrained(WHISPER_PATH)
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    WHISPER_PATH,
    torch_dtype=DTYPE,
    device_map="cuda",
    attn_implementation="sdpa"  # Bypasses the aarch64 flash-attn crash
).eval()

print(f"[+] Loading local Gemma model from {GEMMA_PATH}...")
gemma_tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
gemma_model = AutoModelForCausalLM.from_pretrained(
    GEMMA_PATH,
    torch_dtype=DTYPE,
    device_map="cuda",
    attn_implementation="sdpa"
).eval()


def transcribe_audio_gpu(audio_path: str) -> str:
    """Fast GPU-accelerated transcription using local Whisper via transformers."""
    try:
        if not os.path.exists(audio_path):
            return "MASK"
        
        # Load and resample audio to 16kHz mono natively
        audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)
        inputs = whisper_processor(audio_array, sampling_rate=16000, return_tensors="pt").to(DEVICE, dtype=DTYPE)
        
        forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="en", task="transcribe")
        
        with torch.inference_mode():
            predicted_ids = whisper_model.generate(
                **inputs,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=150
            )
            
        transcript = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        return transcript
    except Exception as e:
        print(f"[!] ASR Error on {audio_path}: {e}")
        return "MASK"


def extract_nlu_gemma(transcript: str, domain_context: str, instruction_prompt: str) -> dict:
    """Executes structured JSON inference on local Gemma entirely in VRAM."""
    if not transcript or transcript == "MASK":
        return {}

    messages = [
        {
            "role": "user",
            "content": f"""You are a precise NLU classification model for the {domain_context} domain.
Analyze the following transcript:
"{transcript}"

{instruction_prompt}

Respond ONLY with a valid, raw JSON object. Do not include markdown codeblocks or extra text.
"""
        }
    ]

    prompt_text = gemma_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = gemma_tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        outputs = gemma_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=gemma_tokenizer.eos_token_id
        )

    # Decode only the newly generated response tokens
    response_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    response_text = gemma_tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    # Robust JSON extraction to handle rogue markdown backticks
    try:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(response_text)
    except Exception:
        return {}


# ==========================================
# 3. DATASET PROCESSING FUNCTIONS
# ==========================================
def process_meld():
    print("\n[1/4] Processing MELD Dataset...")
    csv_path = os.path.join(MELD_DIR, "meld_unified_annotations.csv")
    if not os.path.exists(csv_path):
        print("[-] MELD annotations CSV not found. Skipping.")
        return

    df = pd.read_csv(csv_path)
    processed_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="MELD Labels"):
        audio_filename = str(row["audio_path"])
        emotion_match = re.match(r"^([a-zA-Z]+)_", audio_filename)
        emotion_label = emotion_match.group(1).lower() if emotion_match else str(row.get("emotion", "MASK"))

        processed_rows.append({
            "audio_path": f"meld_balanced_1050/{audio_filename}",
            "domain": "MASK",
            "subdomain": "MASK",
            "intent": "MASK",
            "entity_type": "MASK",
            "urgency": "MASK",
            "emotion": emotion_label,
            "transcript": str(row.get("transcript", "MASK"))
        })

    out_df = pd.DataFrame(processed_rows)
    out_path = os.path.join(OUTPUT_DIR, "meld_processed.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[+] Saved {len(out_df)} MELD records to {out_path}")


def process_medical():
    print("\n[2/4] Processing Medical Dialogue Dataset...")
    wav_files = sorted(glob(os.path.join(MED_DIR, "*.wav")) + glob(os.path.join(MED_DIR, "*.flac")))
    processed_rows = []
    
    instruction = """Extract the following fields into JSON:
1. "intent": snake_case action (e.g., patient_report_symptoms, doctor_prescribe_medication).
2. "entity_type": core entity discussed (e.g., symptom, medication, test_result, MASK).
3. "urgency": strictly one of ["Low", "Medium", "High", "Critical"].
4. "subdomain": clinical specialty (e.g., cardiology, neurology, general) if evident, otherwise "MASK"."""

    for audio_path in tqdm(wav_files, desc="Medical GPU Pipeline"):
        audio_name = os.path.basename(audio_path)
        transcript = transcribe_audio_gpu(audio_path)
        labels = extract_nlu_gemma(transcript, "Medical and Clinical Healthcare", instruction)

        processed_rows.append({
            "audio_path": f"medical_dialogue_1500/{audio_name}",
            "domain": "medical",
            "subdomain": labels.get("subdomain", "MASK"),
            "intent": labels.get("intent", "MASK"),
            "entity_type": labels.get("entity_type", "MASK"),
            "urgency": labels.get("urgency", "MASK"),
            "emotion": "MASK",
            "transcript": transcript
        })

    out_df = pd.DataFrame(processed_rows)
    out_path = os.path.join(OUTPUT_DIR, "medical_processed.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[+] Saved {len(out_df)} Medical records to {out_path}")


def process_slurp():
    print("\n[3/4] Processing SLURP General Dataset...")
    csv_path = os.path.join(SLURP_DIR, "slurp_unified_annotations.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    processed_rows = []
    instruction = """Extract the core entity_type discussed in this voice command.
Return JSON: {"entity_type": "<extracted_entity_or_MASK>"}"""

    for _, row in tqdm(df.iterrows(), total=len(df), desc="SLURP Processing"):
        audio_filename = str(row["audio_path"])
        full_audio_path = os.path.join(SLURP_DIR, audio_filename)

        transcript = str(row.get("transcript", ""))
        if not transcript or transcript in ["nan", "MASK", ""]:
            transcript = transcribe_audio_gpu(full_audio_path)

        intent = str(row.get("intent", "MASK"))
        labels = extract_nlu_gemma(transcript, "Voice Assistant Commands", instruction)

        processed_rows.append({
            "audio_path": f"slurp_general_1500/{audio_filename}",
            "domain": "general",
            "subdomain": "voice_command",
            "intent": intent,
            "entity_type": labels.get("entity_type", "MASK"),
            "urgency": "MASK",
            "emotion": "MASK",
            "transcript": transcript
        })

    out_df = pd.DataFrame(processed_rows)
    out_path = os.path.join(OUTPUT_DIR, "slurp_processed.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[+] Saved {len(out_df)} SLURP records to {out_path}")


def process_earnings21():
    print("\n[4/4] Processing Earnings-21 Dataset...")
    sectors = ["healthcare", "industrial_goods", "technology"]
    processed_rows = []

    for sector in sectors:
        sector_dir = os.path.join(EARNINGS_DIR, sector)
        wav_files = sorted(glob(os.path.join(sector_dir, "*.wav")))

        instruction = f"""Analyze this financial earning call excerpt from the {sector} sector.
Extract the following into JSON:
1. "intent": snake_case scenario and action (e.g., financial_report_revenue).
2. "entity_type": main financial metric or subject discussed (e.g., revenue, operating_margin, MASK).
3. "urgency": strictly one of ["Low", "Medium", "High", "Critical"]."""

        for wav_path in tqdm(wav_files, desc=f"Earnings ({sector})"):
            audio_name = os.path.basename(wav_path)
            transcript = transcribe_audio_gpu(wav_path)
            labels = extract_nlu_gemma(transcript, f"Corporate Finance ({sector} sector)", instruction)

            processed_rows.append({
                "audio_path": f"earnings_21_clean_segments/{sector}/{audio_name}",
                "domain": "finance",
                "subdomain": sector,
                "intent": labels.get("intent", "MASK"),
                "entity_type": labels.get("entity_type", "MASK"),
                "urgency": labels.get("urgency", "MASK"),
                "emotion": "MASK",
                "transcript": transcript
            })

    if processed_rows:
        out_df = pd.DataFrame(processed_rows)
        out_path = os.path.join(OUTPUT_DIR, "earnings_processed.csv")
        out_df.to_csv(out_path, index=False)
        print(f"[+] Saved {len(out_df)} Earnings-21 records to {out_path}")


def merge_master_dataset():
    print("\n[+] Merging processed CSVs into master_nlu_dataset.csv...")
    csv_files = glob(os.path.join(OUTPUT_DIR, "*_processed.csv"))
    if csv_files:
        master_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        master_path = os.path.join(OUTPUT_DIR, "master_nlu_dataset.csv")
        master_df.to_csv(master_path, index=False)
        print(f"Master Dataset generated successfully at '{master_path}'.")

if __name__ == "__main__":
    process_meld()
    process_medical()
    process_slurp()
    process_earnings21()
    merge_master_dataset()
