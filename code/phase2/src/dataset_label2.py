import os
import re
import json
import torch
import soundfile as sf
import pandas as pd
from glob import glob
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    WhisperForConditionalGeneration,
    pipeline
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
    attn_implementation="sdpa"
)
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    torch_dtype=DTYPE,
    device=0
)

print(f"[+] Loading local Gemma model from {GEMMA_PATH}...")
gemma_tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)
gemma_model = AutoModelForCausalLM.from_pretrained(
    GEMMA_PATH,
    torch_dtype=DTYPE,
    device_map="cuda",
    attn_implementation="sdpa",
    trust_remote_code=True
)


def transcribe_audio_gpu(audio_path: str) -> str:
    """Fast GPU-accelerated transcription using local Whisper."""
    try:
        if not os.path.exists(audio_path):
            return "MASK"
        result = asr_pipeline(audio_path, return_timestamps=False)
        return result["text"].strip()
    except Exception as e:
        print(f"[!] ASR Error on {audio_path}: {e}")
        return "MASK"


def extract_nlu_gemma(transcript: str, domain_context: str, instruction_prompt: str) -> dict:
    """Executes structured JSON inference on local Gemma without Ollama."""
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
            do_sample=False,  # Deterministic greedy decoding for labeling
            pad_token_id=gemma_tokenizer.eos_token_id
        )

    # Decode only the newly generated response tokens
    response_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    response_text = gemma_tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    # Robust JSON extraction
    try:
        # Extract content between first { and last }
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
    """Dataset 1: MELD (Emotion from filename, rest masked)."""
    print("\n" + "="*50)
    print("[1/4] Processing MELD Dataset...")
    csv_path = os.path.join(MELD_DIR, "meld_unified_annotations.csv")
    if not os.path.exists(csv_path):
        print("[-] MELD annotations CSV not found. Skipping.")
        return

    df = pd.read_csv(csv_path)
    processed_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="MELD Labels"):
        audio_filename = str(row["audio_path"])

        # Extract emotion directly from filename prefix (e.g. 'anger_dia10_utt0.flac')
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
    """Dataset 2: Medical Dialogue 1500 (Whisper ASR -> Gemma Intent, Entity, Urgency, Subdomain)."""
    print("\n" + "="*50)
    print("[2/4] Processing Medical Dialogue Dataset...")
    wav_files = sorted(glob(os.path.join(MED_DIR, "*.wav")) + glob(os.path.join(MED_DIR, "*.flac")))
    if not wav_files:
        print("[-] No Medical audio files found. Skipping.")
        return

    processed_rows = []
    instruction = """Extract the following fields into JSON:
1. "intent": snake_case action (e.g., patient_report_symptoms, doctor_prescribe_medication).
2. "entity_type": core entity discussed (e.g., symptom, medication, test_result, MASK).
3. "urgency": strictly one of ["Low", "Medium", "High", "Critical"].
4. "subdomain": clinical specialty (e.g., cardiology, neurology, respiratory, general) if evident, otherwise "MASK"."""

    for audio_path in tqdm(wav_files, desc="Medical GPU Pipeline"):
        audio_name = os.path.basename(audio_path)
        
        # Step A: GPU Transcription
        transcript = transcribe_audio_gpu(audio_path)

        # Step B: Gemma NLU Inference
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
    """Dataset 3: SLURP General 1500 (Preserve Intent, extract Entity via Gemma, Urgency=MASK)."""
    print("\n" + "="*50)
    print("[3/4] Processing SLURP General Dataset...")
    csv_path = os.path.join(SLURP_DIR, "slurp_unified_annotations.csv")
    if not os.path.exists(csv_path):
        print("[-] SLURP CSV not found. Skipping.")
        return

    df = pd.read_csv(csv_path)
    processed_rows = []
    instruction = """Extract the core entity_type discussed in this smart home / assistant voice command.
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
    """Dataset 4: Earnings-21 (Domain=finance, Subdomain=sector, extract Intent, Entity, Urgency)."""
    print("\n" + "="*50)
    print("[4/4] Processing Earnings-21 Dataset...")
    sectors = ["healthcare", "industrial_goods", "technology"]
    processed_rows = []

    for sector in sectors:
        sector_dir = os.path.join(EARNINGS_DIR, sector)
        wav_files = sorted(glob(os.path.join(sector_dir, "*.wav")))
        if not wav_files:
            continue

        instruction = f"""Analyze this financial earning call excerpt from the {sector} sector.
Extract the following into JSON:
1. "intent": snake_case scenario and action (e.g., financial_report_revenue, executive_forward_guidance, earnings_qna).
2. "entity_type": main financial metric or subject discussed (e.g., revenue, operating_margin, capital_expenditure, MASK).
3. "urgency": strictly one of ["Low", "Medium", "High", "Critical"]."""

        for wav_path in tqdm(wav_files, desc=f"Earnings ({sector})"):
            audio_name = os.path.basename(wav_path)
            
            # Step A: GPU Transcription
            transcript = transcribe_audio_gpu(wav_path)

            # Step B: Gemma NLU Inference
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
    """Merges all processed subsets into master_nlu_dataset.csv."""
    print("\n" + "="*50)
    print("[+] Merging processed CSVs into master_nlu_dataset.csv...")
    csv_files = glob(os.path.join(OUTPUT_DIR, "*_processed.csv"))
    if not csv_files:
        print("[-] No intermediate processed CSVs found.")
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    master_df = pd.concat(dfs, ignore_index=True)
    master_path = os.path.join(OUTPUT_DIR, "master_nlu_dataset.csv")
    master_df.to_csv(master_path, index=False)
    print(f"🎉 Master Dataset generated successfully at '{master_path}' with {len(master_df)} total records.")


# ==========================================
# 4. MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    process_meld()
    process_medical()
    process_slurp()
    process_earnings21()
    merge_master_dataset()
