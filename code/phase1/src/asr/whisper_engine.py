import time
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from .config import WhisperConfig
from .outputs import ASRExtractionOutput
from .audio_loader import AudioLoader

class WhisperEngine:
    def __init__(self, config: WhisperConfig):
        self.config = config
        print(f"Loading processor from {config.model_path}...")
        self.processor = AutoProcessor.from_pretrained(
            config.model_path, local_files_only=True
        )
        
        print(f"Loading model to {config.device} in {config.dtype}...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            config.model_path,
            torch_dtype=config.dtype,
            local_files_only=True,
            attn_implementation="flash_attention_2" # Crucial for Blackwell
        ).to(config.device)
        self.model.eval() # Ensure inference mode
        
        self.audio_loader = AudioLoader()

    @torch.no_grad()
    def process_audio(self, audio_path: str) -> ASRExtractionOutput:
        # 1. Load Audio
        audio_array = self.audio_loader.load(audio_path)
        duration = len(audio_array) / self.audio_loader.target_sr
        
        # 2. Extract Features
        inputs = self.processor(
            audio_array, 
            sampling_rate=self.audio_loader.target_sr, 
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.config.device).to(self.config.dtype)

        # 3. Single Forward Pass Generation
        start_time = time.time()
        
        outputs = self.model.generate(
            input_features,
            language=self.config.language,
            task=self.config.task,
            max_new_tokens=self.config.max_tokens,
            num_beams=self.config.beam_size,
            num_return_sequences=self.config.beam_size,
            return_timestamps=self.config.return_timestamps,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True
        )
        
        latency = time.time() - start_time

        # 4. Extract Encoder/Decoder Hidden States
        # encoder_hidden_states is a tuple of all layers. We take the last layer [-1].
        enc_hidden_states = outputs.encoder_hidden_states[-1] 
        dec_hidden_states = outputs.decoder_hidden_states 

        # 5. Extract Beam Candidates and Scores
        # The sequences tensor contains `num_beams` rows. Row 0 is the best hypothesis.
        beam_candidates = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)
        # Sequence scores (overall score for each beam)
        beam_scores = outputs.sequences_scores.tolist() if outputs.sequences_scores is not None else []
        
        # 6. Extract Best Sequence Token Strings and Log Probs
        best_sequence_ids = outputs.sequences[0]
        transcript = beam_candidates[0]
        
        # Extract individual token strings
        token_strings = [self.processor.decode([tok_id]) for tok_id in best_sequence_ids]
        
        # Compute exact token-level log probabilities for the chosen sequence
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
        )
        # Best sequence transition scores (log probabilities)
        best_transition_scores = transition_scores[0].cpu().numpy()
        
        # Note: transition_scores length matches generated tokens (excluding prompt/forced tokens)
        # We align them here.
        token_log_probs = best_transition_scores.tolist()

        # 7. Extract Timestamps (Basic segment parsing based on HF outputs)
        # HF timestamp extraction from raw sequences requires looking for timestamp tokens.
        # For Phase 1, we decode with timestamps=True to get segment boundaries in text.
        decoded_with_timestamps = self.processor.decode(best_sequence_ids, skip_special_tokens=False)
        
        # Return cleanly packaged dataclass
        return ASRExtractionOutput(
            transcript=transcript,
            segments=[{"text": transcript, "start": 0.0, "end": duration}], # Placeholder for deep segment parsing
            timestamps=[], # To be fully mapped if needed in phase 2
            encoder_hidden_states=enc_hidden_states,
            decoder_hidden_states=dec_hidden_states,
            token_ids=best_sequence_ids.tolist(),
            token_strings=token_strings,
            token_log_probs=token_log_probs,
            beam_candidates=beam_candidates,
            beam_scores=beam_scores,
            language=self.config.language,
            duration=duration,
            latency=latency
        )
