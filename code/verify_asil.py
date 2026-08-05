import torch
import vllm
import faiss
import transformers
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

def verify_environment():
    print("--- ASIL Environment Verification ---")
    
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        # Verify Blackwell SM120 
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 12:
            print("[SUCCESS] Blackwell / SM120+ architecture detected.")
        else:
            print("[WARNING] Compute capability is lower than expected for Blackwell.")
            
        # ---------------------------------------------------------
        # NEW: Verify Local Whisper Model Loading via Transformers
        # ---------------------------------------------------------
        # REPLACE THIS PATH with your actual local directory path
        local_whisper_path = "/path/to/your/local/whisper-large-v3-turbo" 
        
        try:
            print(f"\nAttempting to load local Whisper model from: {local_whisper_path}")
            
            # Load processor
            processor = AutoProcessor.from_pretrained(local_whisper_path, local_files_only=True)
            
            # Load model directly to CUDA with FlashAttention-2/3 enabled
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                local_whisper_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                local_files_only=True,
                attn_implementation="flash_attention_2" # Works for FA3 backend in transformers
            ).to("cuda")
            
            print("[SUCCESS] Local Whisper model successfully loaded onto Blackwell GPU using HF Transformers.")
            
        except Exception as e:
            print(f"[ERROR] Failed to load local Whisper model: {e}")
            print("Ensure the path is correct and contains the proper HuggingFace config/safetensor files.")
            
    else:
        print("[ERROR] PyTorch cannot see the GPU.")

if __name__ == "__main__":
    verify_environment()
