from dataclasses import dataclass
import torch

@dataclass
class WhisperConfig:
    # Use absolute local path as requested
    model_path: str = "/path/to/your/local/whisper-large-v3-turbo"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Blackwell optimally supports bfloat16 or float16; float16 is standard for Whisper
    dtype: torch.dtype = torch.float16 
    beam_size: int = 5
    max_tokens: int = 448
    language: str = "hi" # Hindi/Hinglish default, can be "en"
    task: str = "transcribe"
    return_timestamps: bool = True
