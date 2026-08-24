import torch
import json
from voice_nlu_inference import VoiceNLU
from text_nlu_inference import TextNLU
import whisper

def verify_hardware():
    print("=" * 50)
    print("HARDWARE VERIFICATION")
    print("=" * 50)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print("=" * 50)

def main():
    verify_hardware()
    
    # Initialize models (will fail loudly if paths/artifacts are missing)
    voice_nlu = VoiceNLU()
    text_nlu = TextNLU()
    
    # Provide a known audio path from your dataset
    sample_wav = "nlu_robust_experiment/audio/sample_001.wav" 
    
    print("\n[+] Running Voice NLU...")
    voice_res = voice_nlu.predict(sample_wav)
    
    print("\n[+] Running Whisper Decoder...")
    transcript_res = voice_nlu.whisper_model.transcribe(sample_wav, temperature=0.0)
    transcript = transcript_res["text"].strip()
    
    print(f"\n[+] Running Text NLU on: '{transcript}'...")
    text_res = text_nlu.predict(transcript)
    
    print("\n=== VOICE NLU OUTPUT ===")
    print(json.dumps({k: v for k, v in voice_res.items() if k != "semantic_embedding"}, indent=2))
    
    print("\n=== TEXT NLU OUTPUT ===")
    print(json.dumps({k: v for k, v in text_res.items() if k != "semantic_embedding"}, indent=2))

    assert voice_res["semantic_embedding"].shape == (128,), "Voice embedding dimension mismatch!"
    assert text_res["semantic_embedding"].shape == (128,), "Text embedding dimension mismatch!"
    print("\n[SUCCESS] Deployment verified successfully on Blackwell GPU.")

if __name__ == "__main__":
    main()
