import torch
import numpy as np
from transformers import AutoProcessor, WhisperForConditionalGeneration
from loguru import logger
from .config import WhisperConfig

class IntrospectionEngine:
    def __init__(self, config: WhisperConfig):
        self.config = config
        logger.info(f"Loading local Whisper model from {config.model_path} onto {config.device} ({config.dtype})")
        
        # Load strictly from local path, forcing native PyTorch SDPA
        self.processor = AutoProcessor.from_pretrained(config.model_path, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            config.model_path,
            torch_dtype=config.dtype,
            local_files_only=True,
            attn_implementation="sdpa"  # Native SDPA for Blackwell compatibility
        ).to(config.device)
        self.model.eval()

    @torch.no_grad()
    def process(self, audio_array: np.ndarray):
        inputs = self.processor(
            audio_array, 
            sampling_rate=self.config.sample_rate, 
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.config.device, dtype=self.config.dtype)

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.config.language, 
            task=self.config.task
        )

        logger.info("Executing forward pass with full tensor introspection...")
        
        outputs = self.model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
            num_return_sequences=self.config.num_return_sequences,
            return_timestamps=self.config.return_timestamps,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True
        )

        # -----------------------------------------------------------------
        # ROBUST CONFIDENCE ALIGNMENT LOGIC
        # -----------------------------------------------------------------
        # compute_transition_scores computes log probs for each generated step.
        # Length of transition_scores[0] = len(top_sequence) - num_prompt_tokens
        # -----------------------------------------------------------------
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        
        top_sequence_tensor = outputs.sequences[0]
        top_sequence_ids = top_sequence_tensor.tolist()
        top_trans_scores = transition_scores[0].cpu().float().numpy()
        
        num_generated = len(top_trans_scores)
        total_tokens = len(top_sequence_ids)
        prompt_len = total_tokens - num_generated

        raw_tokens_str = self.processor.tokenizer.convert_ids_to_tokens(top_sequence_ids)
        
        raw_tokens_data = []
        clean_tokens_data = []

        for idx, (token_id, raw_tok) in enumerate(zip(top_sequence_ids, raw_tokens_str)):
            # Determine if token was part of initial decoder prompt or newly generated
            if idx < prompt_len:
                log_prob = 0.0
                confidence = 1.0
            else:
                gen_idx = idx - prompt_len
                log_prob = float(top_trans_scores[gen_idx])
                confidence = float(np.exp(log_prob))

            is_timestamp = bool(raw_tok.startswith("<|") and raw_tok.endswith("|>") and raw_tok[2:-2].replace(".", "", 1).isdigit())

            raw_tokens_data.append({
                "token_id": int(token_id),
                "raw_token": raw_tok,
                "is_timestamp": is_timestamp,
                "decoder_log_prob": log_prob,
                "confidence": confidence
            })

            # Clean tokens filtering for clean_tokens.csv
            if not (raw_tok.startswith("<|") and raw_tok.endswith("|>")):
                cleaned_word = raw_tok.replace("Ġ", "").replace(" ", "").strip()
                if cleaned_word:
                    clean_tokens_data.append({
                        "cleaned_display_token": cleaned_word,
                        "confidence": confidence
                    })

        # -----------------------------------------------------------------
        # TRANSCRIPT AND BEAM CANDIDATES EXTRACTION
        # -----------------------------------------------------------------
        transcript = self.processor.decode(top_sequence_tensor, skip_special_tokens=True)

        beam_hypotheses = []
        for i in range(1, self.config.num_return_sequences):
            beam_text = self.processor.decode(outputs.sequences[i], skip_special_tokens=True)
            beam_hypotheses.append(beam_text)

        # -----------------------------------------------------------------
        # ENCODER HIDDEN STATES EXTRACTION
        # -----------------------------------------------------------------
        final_encoder_state = outputs.encoder_hidden_states[-1].cpu().float().numpy()
        
        all_encoder_states = None
        if self.config.save_all_encoder_states:
            all_encoder_states = [layer.cpu().float().numpy() for layer in outputs.encoder_hidden_states]

        return {
            "transcript": transcript.strip(),
            "raw_tokens_data": raw_tokens_data,
            "clean_tokens_data": clean_tokens_data,
            "beam_hypotheses": beam_hypotheses,
            "final_encoder_state": final_encoder_state,
            "all_encoder_states": all_encoder_states,
            "sequence_score": outputs.sequences_scores[0].item() if outputs.sequences_scores is not None else 1.0
        }
        logger.info("Executing forward pass with full tensor introspection...")
        
        # The monolithic generate call mapping to all your research questions
        outputs = self.model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
            num_return_sequences=self.config.num_return_sequences,
            return_timestamps=self.config.return_timestamps,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True
        )

        # 1. Compute rigorous Log Probabilities
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        
        # Confidence calculation: Confidence = exp(Log_Probability)
        confidences = torch.exp(transition_scores)

        # 2. Extract Top-1 Transcript and Timestamps
        top_sequence = outputs.sequences[0]
        top_confidence = confidences[0]
        
        # Decode without special tokens to get clean text
        transcript = self.processor.decode(top_sequence, skip_special_tokens=True)
        tokens_with_special = self.processor.convert_ids_to_tokens(top_sequence)
        
        # 3. Parse Token-Level Data
        parsed_tokens = []
        parsed_confidences = []
        for i, token in enumerate(tokens_with_special):
            if token.startswith("<|") and token.endswith("|>"):
                continue # Skip pure timestamp/control tokens in the CSV
            
            # Align token with its confidence score
            score_idx = i - (len(tokens_with_special) - len(top_confidence))
            conf = top_confidence[score_idx].item() if score_idx >= 0 else 1.0
            
            parsed_tokens.append(token.replace("Ġ", "")) 
            parsed_confidences.append(conf)

        # 4. Extract Beam Hypotheses (excluding the top sequence)
        beam_hypotheses = []
        for i in range(1, self.config.num_return_sequences):
            beam_text = self.processor.decode(outputs.sequences[i], skip_special_tokens=True)
            beam_hypotheses.append(beam_text)

        # 5. Extract Hidden States
        final_encoder_state = outputs.encoder_hidden_states[-1].cpu().float().numpy()
        
        all_encoder_states = None
        if self.config.save_all_encoder_states:
            all_encoder_states = [layer.cpu().float().numpy() for layer in outputs.encoder_hidden_states]

        return {
            "transcript": transcript.strip(),
            "tokens": parsed_tokens,
            "confidences": parsed_confidences,
            "beam_hypotheses": beam_hypotheses,
            "final_encoder_state": final_encoder_state,
            "all_encoder_states": all_encoder_states,
            "sequence_score": outputs.sequences_scores[0].item() if outputs.sequences_scores is not None else 1.0
        }
