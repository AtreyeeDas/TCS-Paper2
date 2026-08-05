from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch

@dataclass
class ASRExtractionOutput:
    transcript: str
    segments: List[Dict[str, Any]]
    timestamps: List[tuple]
    encoder_hidden_states: torch.Tensor  # Shape: (1, acoustic_frames, 1280)
    decoder_hidden_states: tuple         # Tuple of tensors per generation step
    token_ids: List[int]
    token_strings: List[str]
    token_log_probs: List[float]
    beam_candidates: List[str]
    beam_scores: List[float]
    language: str
    duration: float
    latency: float
