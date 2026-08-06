import os
import torch
from dataclasses import dataclass

@dataclass
class WhisperConfig:
    # Use bfloat16 for Blackwell architecture stability and VRAM optimization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 
    
    # Absolute path to your local model
    model_path: str = "/home/spark2/Models/whisper_large_v3_turbo"
    
    # Generation parameters required for ASIL introspection
    num_beams: int = 5
    num_return_sequences: int = 5
    max_new_tokens: int = 256
    return_timestamps: bool = True
    language: str = "en"
    task: str = "transcribe"
    
    # I/O Config
    sample_rate: int = 16000
    save_all_encoder_states: bool = False # Set True only if deeply debugging acoustics
    save_decoder_states: bool = False     # Strongly recommended False to save I/O

@dataclass
class Paths:
    raw_audio_dir: str = "../datasets/raw_audio"
    results_dir: str = "../results"
