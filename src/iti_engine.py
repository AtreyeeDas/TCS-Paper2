import os
import torch
import librosa
import soundfile as sf
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

class ITIPipeline:
    def __init__(self, model_path, reference_audio_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[+] Loading Offline XTTS-v2 from {model_path}...")
        
        self.config = XttsConfig()
        self.config.load_json(os.path.join(model_path, "config.json"))
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=model_path, eval=True)
        self.model.to(self.device)
        
        print(f"[+] Extracting Native Speaker Latents from {reference_audio_path}...")
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=reference_audio_path
        )

    def get_exact_bpe_boundary(self, raw_text):
        """
        Transliterates Devanagari to Roman and calculates exact BPE token index (\beta).
        """
        words = raw_text.split()
        switch_word_idx = -1
        for i in range(1, len(words)):
            if words[i-1].isascii() != words[i].isascii():
                switch_word_idx = i
                break
                
        # Stage 1: Transliteration
        transliterated_text = transliterate(raw_text, sanscript.DEVANAGARI, sanscript.ITRANS)
        
        if switch_word_idx == -1:
            return transliterated_text, -1
            
        # Get exact token count up to switch boundary using model's BPE tokenizer
        prefix_words = " ".join(words[:switch_word_idx])
        prefix_transliterated = transliterate(prefix_words, sanscript.DEVANAGARI, sanscript.ITRANS)
        
        # Exact tokenization
        prefix_tokens = self.model.tokenizer.tokenize(prefix_transliterated)
        beta_token_index = len(prefix_tokens)
        
        return transliterated_text, beta_token_index

    def generate_with_iti(self, text, beta_idx, temp_penalty=0.4, enable_iti=True):
        """
        Runs AR Generation with in-place Logit Sharpness Hook at boundary index \beta.
        """
        step_counter = {"count": 0}

        def iti_hook(module, input, output):
            step_counter["count"] += 1
            # Check if current AR step matches code-switch boundary \beta
            if enable_iti and beta_idx > 0 and step_counter["count"] == beta_idx:
                # In-place temperature scaling (sharpening logits)
                output.div_(temp_penalty)

        # Attach hook to final projection head
        hook_handle = self.model.gpt.text_head.register_forward_hook(iti_hook)
        
        try:
            out = self.model.inference(
                text=text,
                language="en",
                gpt_cond_latent=self.gpt_cond_latent,
                speaker_embedding=self.speaker_embedding,
                temperature=0.75
            )
            wav = out["wav"]
        finally:
            hook_handle.remove()
            
        return wav

    def apply_rms_guardrail(self, wav, sr=24000, threshold=0.008, window_ms=30):
        """
        Stage 5 Energy-Gated Guardrail using Librosa RMS.
        """
        hop_length = int(sr * (window_ms / 1000.0))
        rms = librosa.feature.rms(y=wav, frame_length=hop_length*2, hop_length=hop_length)[0]
        active_frames = np.where(rms > threshold)[0]
        
        if len(active_frames) > 0:
            last_sample = (active_frames[-1] + 1) * hop_length
            return wav[:last_sample]
        return wav

    def process_sentence(self, raw_text, output_wav_path, arm="Arm1"):
        norm_text, beta_idx = self.get_exact_bpe_boundary(raw_text)
        
        enable_iti = arm in ["Arm1", "Arm2", "Arm3"]
        enable_guardrail = arm in ["Arm1", "Arm3", "Arm4"]
        
        # Generation
        wav = self.generate_with_iti(
            text=norm_text if arm != "Arm4" else raw_text, # Arm 4 tests minus normalization
            beta_idx=beta_idx,
            enable_iti=enable_iti
        )
        
        # Guardrail
        if enable_guardrail:
            wav = self.apply_rms_guardrail(wav)
            
        sf.write(output_wav_path, wav, 24000)
        return output_wav_path
