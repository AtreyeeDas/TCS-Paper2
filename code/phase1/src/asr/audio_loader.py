import torch
import torchaudio
from loguru import logger

class AudioLoader:
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate

    def load(self, file_path: str) -> torch.Tensor:
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Whisper expects mono channel
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Resample strictly to 16kHz for the Whisper feature extractor
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=self.target_sample_rate
                )
                waveform = resampler(waveform)
                
            # Squeeze to 1D array as expected by the HF Processor
            return waveform.squeeze(0).numpy()
            
        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {str(e)}")
            raise e
