import os
import json
import h5py
import torch
import pandas as pd
import numpy as np

class DataPacker:
    @staticmethod
    def save_results(output_dir: str, result_dict: dict, metrics: dict):
        os.makedirs(output_dir, exist_ok=True)

        # 1. Transcript
        with open(os.path.join(output_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(result_dict["transcript"])

        # 2 & 6 & 7. Token List, Log Probs, Confidences
        # Confidence = exp(Log_Probability)
        df_tokens = pd.DataFrame({
            "Token": result_dict["tokens"],
            "Confidence": result_dict["confidences"],
            "Log_Probability": np.log(np.clip(result_dict["confidences"], 1e-10, 1.0))
        })
        df_tokens.to_csv(os.path.join(output_dir, "token_metrics.csv"), index=False)

        # 8. Beam Candidates
        with open(os.path.join(output_dir, "beam_candidates.txt"), "w", encoding="utf-8") as f:
            for i, hyp in enumerate(result_dict["beam_hypotheses"]):
                f.write(f"Beam {i+1}: {hyp}\n")

        # 9. Encoder Hidden State (NumPy, PyTorch, and HDF5)
        enc_state = result_dict["final_encoder_state"]
        np.save(os.path.join(output_dir, "final_encoder_state.npy"), enc_state)
        torch.save(torch.tensor(enc_state), os.path.join(output_dir, "final_encoder_state.pt"))
        
        with h5py.File(os.path.join(output_dir, "encoder_states.h5"), "w") as hf:
            hf.create_dataset("final_layer", data=enc_state)
            if result_dict.get("all_encoder_states") is not None:
                for i, layer_state in enumerate(result_dict["all_encoder_states"]):
                    hf.create_dataset(f"layer_{i}", data=layer_state)

        # 12-17. Comprehensive JSON Report
        report = {
            "metadata": metrics,
            "generation_score": result_dict["sequence_score"],
            "status": "completed"
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(report, f, indent=4)
