import os
import json
import librosa
import pandas as pd
from glob import glob
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "/home/spark2/Models/gemma-4-e4b-it"

MED_DIR = "./medical_dialogue_1500"
EARNINGS_DIR = "./earnings21_clean_segments"

print("[+] Loading Gemma 4 E4B and Processor to RTX PRO 50 (Blackwell)...")
# device_map="cuda" automatically utilizes your GPU
# torch_dtype=torch.bfloat16 leverages Blackwell's ultra-fast mixed precision
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    device_map="cuda", 
    torch_dtype=torch.bfloat16
)
# Optional: Enable Flash Attention if your environment supports it
# model = model.to(memory_format=torch.bfloat16)

def ask_gemma4_local(audio_path, domain_context):
    """
    Loads raw audio, processes it through the local Gemma 4 model, 
    and forces a JSON schema output.
    """
    try:
        # Load audio at the exact sample rate expected by the model (usually 16000 or 24000)
        target_sr = processor.feature_extractor.sampling_rate
        audio_array, _ = librosa.load(audio_path, sr=target_sr)
        
        prompt = f"""
        You are an expert NLU classifier. Listen to the provided audio clip from the {domain_context} domain.
        
        Based on the speech, output exactly three fields:
        1. "transcript": The text transcription of what was said.
        2. "intent": A snake_case scenario and action (e.g., "patient_report_symptoms", "financial_update").
        3. "urgency": Rate the urgency of the speech. Must be exactly one of: "Low", "Medium", "High", "Critical".
        
        Respond ONLY in valid JSON format. Example:
        {{"transcript": "My chest hurts.", "intent": "patient_report_symptoms", "urgency": "High"}}
        """

        # Format for native multimodal instruction-tuning
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # Apply chat template and process tensors
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt_text, audios=audio_array, return_tensors="pt", sampling_rate=target_sr)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate output natively on GPU
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.1, do_sample=False)
        
        # Decode only the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        result_text = processor.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Clean markdown formatting if model wraps JSON in code blocks
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        return json.loads(result_text)

    except Exception as e:
        print(f"[ERROR] Failed to process {audio_path}: {e}")
        return {"transcript": "MASK", "intent": "MASK", "urgency": "MASK"}

# ==========================================
# 1. PROCESS MEDICAL DIALOGUE
# ==========================================
def label_medical_audio():
    print(f"\n--- Labeling Med-Dialogue ---")
    med_csv_path = os.path.join(MED_DIR, "med_unified_annotations.csv")
    if not os.path.exists(med_csv_path):
        print("Medical CSV not found. Skipping.")
        return
        
    df = pd.read_csv(med_csv_path)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling Medical"):
        if row["intent"] == "MASK" or pd.isna(row["intent"]):
            audio_path = os.path.join(MED_DIR, row["audio_path"])
            labels = ask_gemma4_local(audio_path, "Medical and Healthcare")
            
            df.at[idx, "transcript"] = labels.get("transcript", "MASK")
            df.at[idx, "intent"] = labels.get("intent", "MASK")
            df.at[idx, "urgency"] = labels.get("urgency", "MASK")
            
    df.to_csv(med_csv_path, index=False)

# ==========================================
# 2. PROCESS EARNINGS-21
# ==========================================
def label_earnings_audio():
    print(f"\n--- Labeling Earnings-21 ---")
    sectors = ["healthcare", "industrial_goods", "technology"]
    earnings_rows = []
    
    for sector in sectors:
        sector_dir = os.path.join(EARNINGS_DIR, sector)
        wav_files = glob(os.path.join(sector_dir, "*.wav"))
        
        for wav_path in tqdm(wav_files, desc=f"Labeling Finance: {sector}"):
            audio_name = os.path.basename(wav_path)
            labels = ask_gemma4_local(wav_path, f"Corporate Finance ({sector} sector)")
            
            earnings_rows.append({
                "audio_path": f"{sector}/{audio_name}",
                "domain": "finance",
                "intent": labels.get("intent", "MASK"),
                "entity_type": "MASK",
                "emotion": "MASK",
                "urgency": labels.get("urgency", "MASK"),
                "transcript": labels.get("transcript", "MASK")
            })
            
    if earnings_rows:
        df = pd.DataFrame(earnings_rows)
        output_csv = os.path.join(EARNINGS_DIR, "earnings_final_nlu.csv")
        df.to_csv(output_csv, index=False)

# ==========================================
# EXECUTE
# ==========================================
if __name__ == "__main__":
    label_medical_audio()
    label_earnings_audio()
    print("\n🎉 Local GPU Audio Labeling Complete!")
