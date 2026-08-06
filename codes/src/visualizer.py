import os
import numpy as np
import matplotlib.pyplot import plt
import seaborn as sns

class Visualizer:
    @staticmethod
    def generate_plots(output_dir: str, tokens: list, confidences: list, encoder_state: np.ndarray):
        # Plot 1: Token Confidence Timeline
        plt.figure(figsize=(15, 5))
        plt.plot(confidences, marker='o', linestyle='-', color='b')
        plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=90, fontsize=8)
        plt.axhline(y=0.85, color='r', linestyle='--', label='Uncertainty Threshold')
        plt.title("Token-Level Confidence ($U_{ASR}$ indicator)")
        plt.ylabel("Confidence Score $p(y_t)$")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "token_confidence_plot.png"), dpi=300)
        plt.close()

        # Plot 2: Encoder Hidden State Heatmap (First 100 frames to prevent massive images)
        plt.figure(figsize=(10, 8))
        # Shape is (1, frames, 1280). We take the first batch, first 100 frames.
        subset_state = encoder_state[0, :100, :].T 
        sns.heatmap(subset_state, cmap="viridis", cbar=True, yticklabels=False)
        plt.title("Encoder Hidden State Heatmap (Dim 1280 x Frames 0-100)")
        plt.xlabel("Acoustic Frames (20ms steps)")
        plt.ylabel("Hidden Dimensions (1280)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "encoder_heatmap.png"), dpi=300)
        plt.close()
