import librosa
import numpy as np

class AudioLoader:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def load(self, file_path: str) -> np.ndarray:
        """
        Loads wav, mp3, flac and resamples to 16kHz mono.
        """
        audio_array, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        return audio_array
