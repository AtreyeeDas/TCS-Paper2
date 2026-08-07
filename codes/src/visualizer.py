import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    @staticmethod
    def generate_plots(
        vis_dir: str, 
        clean_tokens_data: list, 
        encoder_state: np.ndarray,
        pooling_results: dict
    ):
        os.makedirs(vis_dir, exist_ok=True)

        tokens = [item["cleaned_display_token"] for item in clean_tokens_data]
        confidences = [item["confidence"] for item in clean_tokens_data]

        # -----------------------------------------------------------------
        # Plot 1: Token Confidence Timeline
        # -----------------------------------------------------------------
        if tokens and confidences:
            plt.figure(figsize=(15, 5))
            plt.plot(confidences, marker='o', linestyle='-', color='b')
            plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=90, fontsize=8)
            plt.axhline(y=0.85, color='r', linestyle='--', label='Uncertainty Threshold (0.85)')
            plt.title("Clean Token-Level Confidence Timeline ($U_{ASR}$ Indicator)")
            plt.ylabel("Confidence Score $p(y_t)$")
            plt.ylim(-0.05, 1.05)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "token_confidence_plot.png"), dpi=300)
            plt.close()

        # -----------------------------------------------------------------
        # Plot 2: Encoder Hidden State Heatmap (Frames 0-100)
        # -----------------------------------------------------------------
        plt.figure(figsize=(12, 8))
        flat_state = encoder_state.squeeze(0) if encoder_state.ndim == 3 else encoder_state
        subset_state = flat_state[:100, :].T  # (1280, 100)
        sns.heatmap(subset_state, cmap="viridis", cbar=True, yticklabels=False)
        plt.title("Encoder Hidden State Heatmap (Dimension 1280 x Acoustic Frames 0-100)")
        plt.xlabel("Acoustic Frames (20ms steps)")
        plt.ylabel("Hidden Dimensions (1280)")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "encoder_heatmap.png"), dpi=300)
        plt.close()

        # -----------------------------------------------------------------
        # Plot 3: Pooling Cosine Similarity Comparison
        # -----------------------------------------------------------------
        mean_p = pooling_results["mean_pool"]
        max_p = pooling_results["max_pool"]
        att_p = pooling_results["attention_pool"]

        def cos_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

        sim_mean_att = cos_sim(mean_p, att_p)
        sim_mean_max = cos_sim(mean_p, max_p)
        sim_max_att = cos_sim(max_p, att_p)

        plt.figure(figsize=(8, 5))
        pairs = ['Mean vs Attn', 'Mean vs Max', 'Max vs Attn']
        sims = [sim_mean_att, sim_mean_max, sim_max_att]
        bars = plt.bar(pairs, sims, color=['#2b5c8f', '#d95f02', '#7570b3'])
        plt.ylim(0, 1.1)
        plt.ylabel("Cosine Similarity")
        plt.title("Cosine Similarity Between Pooling Strategies")
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom')
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "pool_cosine_similarity.png"), dpi=300)
        plt.close()

        # -----------------------------------------------------------------
        # Plot 4: Frame & Pooled Embedding Norms
        # -----------------------------------------------------------------
        frame_norms = np.linalg.norm(flat_state, axis=1)
        pooled_norms = [np.linalg.norm(mean_p), np.linalg.norm(max_p), np.linalg.norm(att_p)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(frame_norms, color='green', alpha=0.8)
        ax1.set_title("Frame-Level L2 Embedding Norms Across Time")
        ax1.set_xlabel("Frame Index (20ms steps)")
        ax1.set_ylabel("L2 Norm")
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2.bar(['Mean Pool', 'Max Pool', 'Attn Pool'], pooled_norms, color=['#1b9e77', '#d95f02', '#7570b3'])
        ax2.set_title("L2 Norm Comparison of Pooled Vectors")
        ax2.set_ylabel("L2 Norm")
        ax2.grid(axis='y', linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "embedding_norms.png"), dpi=300)
        plt.close()
