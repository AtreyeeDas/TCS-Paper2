import os
import json
import h5py
import torch
import pandas as pd
import numpy as np

class DataPacker:
    @staticmethod
    def save_results(
        output_dir: str, 
        result_dict: dict, 
        metrics: dict, 
        pooling_results: dict,
        config
    ):
        # -----------------------------------------------------------------
        # SUBDIRECTORY INITIALIZATION
        # -----------------------------------------------------------------
        dirs = {
            "transcript": os.path.join(output_dir, "transcript"),
            "embeddings": os.path.join(output_dir, "embeddings"),
            "decoder": os.path.join(output_dir, "decoder"),
            "metadata": os.path.join(output_dir, "metadata"),
            "visualizations": os.path.join(output_dir, "visualizations"),
            "pooling": os.path.join(output_dir, "pooling"),
            "frame_mapping": os.path.join(output_dir, "frame_mapping")
        }
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)

        generated_files = []

        # 1. TRANSCRIPT
        t_path = os.path.join(dirs["transcript"], "transcript.txt")
        with open(t_path, "w", encoding="utf-8") as f:
            f.write(result_dict["transcript"])
        generated_files.append(t_path)

        # 2. DECODER OUTPUTS
        # a) raw_tokens.csv
        df_raw = pd.DataFrame(result_dict["raw_tokens_data"])
        raw_csv_path = os.path.join(dirs["decoder"], "raw_tokens.csv")
        df_raw.to_csv(raw_csv_path, index=False)
        generated_files.append(raw_csv_path)

        # b) clean_tokens.csv
        df_clean = pd.DataFrame(result_dict["clean_tokens_data"])
        clean_csv_path = os.path.join(dirs["decoder"], "clean_tokens.csv")
        df_clean.to_csv(clean_csv_path, index=False)
        generated_files.append(clean_csv_path)

        # c) beam_candidates.txt
        beam_path = os.path.join(dirs["decoder"], "beam_candidates.txt")
        with open(beam_path, "w", encoding="utf-8") as f:
            for i, hyp in enumerate(result_dict["beam_hypotheses"]):
                f.write(f"Beam {i+1}: {hyp}\n")
        generated_files.append(beam_path)

        # 3. EMBEDDINGS
        enc_state = result_dict["final_encoder_state"]
        npy_path = os.path.join(dirs["embeddings"], "final_encoder_state.npy")
        pt_path = os.path.join(dirs["embeddings"], "final_encoder_state.pt")
        h5_path = os.path.join(dirs["embeddings"], "encoder_states.h5")

        np.save(npy_path, enc_state)
        torch.save(torch.tensor(enc_state), pt_path)
        generated_files.extend([npy_path, pt_path])

        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("final_layer", data=enc_state)
            if result_dict.get("all_encoder_states") is not None:
                for i, layer_state in enumerate(result_dict["all_encoder_states"]):
                    hf.create_dataset(f"layer_{i}", data=layer_state)
        generated_files.append(h5_path)

        # 4. POOLING OUTPUTS
        mean_npy = os.path.join(dirs["pooling"], "mean_pool.npy")
        max_npy = os.path.join(dirs["pooling"], "max_pool.npy")
        att_npy = os.path.join(dirs["pooling"], "attention_pool.npy")
        pool_meta_path = os.path.join(dirs["pooling"], "pooling_metadata.json")

        np.save(mean_npy, pooling_results["mean_pool"])
        np.save(max_npy, pooling_results["max_pool"])
        np.save(att_npy, pooling_results["attention_pool"])
        
        with open(pool_meta_path, "w", encoding="utf-8") as f:
            json.dump(pooling_results["metadata"], f, indent=4)
        generated_files.extend([mean_npy, max_npy, att_npy, pool_meta_path])

        # 5. FRAME MAPPING
        seq_len = enc_state.shape[1] if enc_state.ndim == 3 else enc_state.shape[0]
        frame_indices = np.arange(seq_len)
        approx_sec = frame_indices * config.encoder_stride_sec
        approx_ms = approx_sec * 1000.0

        df_frame = pd.DataFrame({
            "Frame Index": frame_indices,
            "Approximate Time (seconds)": approx_sec,
            "Approximate Time (milliseconds)": approx_ms
        })
        frame_csv_path = os.path.join(dirs["frame_mapping"], "frame_metadata.csv")
        df_frame.to_csv(frame_csv_path, index=False)
        generated_files.append(frame_csv_path)

        # 6. ENCODER STATISTICS
        flat_state = enc_state.squeeze(0) if enc_state.ndim == 3 else enc_state  # (T, D)
        dim_means = np.mean(flat_state, axis=0).tolist()
        dim_stds = np.std(flat_state, axis=0).tolist()
        frame_l2_norms = np.linalg.norm(flat_state, axis=1)

        mean_p_norm = float(np.linalg.norm(pooling_results["mean_pool"]))
        max_p_norm = float(np.linalg.norm(pooling_results["max_pool"]))
        att_p_norm = float(np.linalg.norm(pooling_results["attention_pool"]))

        near_zero_pct = float(np.mean(np.abs(flat_state) < 1e-5) * 100.0)

        encoder_stats = {
            "mean_activation_per_dimension": dim_means,
            "std_activation_per_dimension": dim_stds,
            "frame_l2_norms": {
                "mean": float(np.mean(frame_l2_norms)),
                "min": float(np.min(frame_l2_norms)),
                "max": float(np.max(frame_l2_norms)),
                "std": float(np.std(frame_l2_norms))
            },
            "pooled_embedding_l2_norms": {
                "mean_pool": mean_p_norm,
                "max_pool": max_p_norm,
                "attention_pool": att_p_norm
            },
            "near_zero_activation_percentage": near_zero_pct,
            "min_activation_value": float(np.min(flat_state)),
            "max_activation_value": float(np.max(flat_state))
        }
        enc_stats_path = os.path.join(dirs["metadata"], "encoder_statistics.json")
        with open(enc_stats_path, "w", encoding="utf-8") as f:
            json.dump(encoder_stats, f, indent=4)
        generated_files.append(enc_stats_path)

        # 7. PHASE 1 SUMMARY & METADATA
        confidences = [item["confidence"] for item in result_dict["raw_tokens_data"]]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        min_conf = float(np.min(confidences)) if confidences else 0.0
        max_conf = float(np.max(confidences)) if confidences else 0.0

        summary_report = {
            "model": config.model_path,
            "audio_length_seconds": metrics["audio_duration_seconds"],
            "inference_time_seconds": metrics["inference_latency_seconds"],
            "real_time_factor": metrics["real_time_factor"],
            "encoder_shape": list(enc_state.shape),
            "pooling_shapes": pooling_results["metadata"]["tensor_shapes"],
            "transcript_word_count": len(result_dict["transcript"].split()),
            "transcript_character_count": len(result_dict["transcript"]),
            "beam_count": len(result_dict["beam_hypotheses"]) + 1,
            "confidence_statistics": {
                "average_token_confidence": avg_conf,
                "minimum_token_confidence": min_conf,
                "maximum_token_confidence": max_conf
            },
            "gpu_memory_peak_vram_mb": metrics["peak_vram_mb"],
            "files_generated": [os.path.relpath(p, output_dir) for p in generated_files]
        }
        
        summary_path = os.path.join(dirs["metadata"], "phase1_summary.json")
        meta_path = os.path.join(dirs["metadata"], "metadata.json")
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=4)
        
        # Legacy compatibility metadata.json at root level of audio output
        root_meta_path = os.path.join(output_dir, "metadata.json")
        meta_content = {
            "metadata": metrics,
            "generation_score": result_dict["sequence_score"],
            "status": "completed"
        }
        with open(root_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, indent=4)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, indent=4)
            
        return dirs
