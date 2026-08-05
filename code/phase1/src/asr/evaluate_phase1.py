import os
import torch
from src.asr.config import WhisperConfig
from src.asr.whisper_engine import WhisperEngine

def run_evaluation():
    # 1. Update this path to your local model!
    LOCAL_MODEL_PATH = "/path/to/your/local/whisper-large-v3-turbo"
    
    # 2. Directory containing 10 REAL test .wav files
    TEST_AUDIO_DIR = "./test_audio" 
    
    print("Initializing Phase 1 Evaluation...")
    config = WhisperConfig(model_path=LOCAL_MODEL_PATH)
    engine = WhisperEngine(config)
    
    test_files = [f for f in os.listdir(TEST_AUDIO_DIR) if f.endswith(".wav")][:10]
    
    if not test_files:
        print(f"ERROR: No .wav files found in {TEST_AUDIO_DIR}. You MUST use real data.")
        return

    print(f"Evaluating {len(test_files)} files...\n")
    
    for idx, file_name in enumerate(test_files):
        audio_path = os.path.join(TEST_AUDIO_DIR, file_name)
        print(f"--- Processing File {idx+1}: {file_name} ---")
        
        # Execute the pipeline
        output = engine.process_audio(audio_path)
        
        # Log exact requested evaluation metrics
        print(f"Transcript: {output.transcript}")
        print(f"Top-3 Beam Candidates: {output.beam_candidates[:3]}")
        print(f"Average Token Confidence (Log Prob): {sum(output.token_log_probs) / len(output.token_log_probs):.4f}")
        print(f"Encoder Hidden State Size: {output.encoder_hidden_states.shape}")
        print(f"Latency: {output.latency:.3f} seconds")
        print(f"Audio Duration: {output.duration:.3f} seconds")
        print(f"Real-Time Factor (RTF): {output.latency / output.duration:.3f}\n")

if __name__ == "__main__":
    run_evaluation()
