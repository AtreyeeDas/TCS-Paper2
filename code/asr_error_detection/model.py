"""
ASIL NLU Multi-Task Neural Network Architecture.
Supports:
- Whisper Semantic Branch (1280-D with Mean or Attention Pooling)
- HuBERT Acoustic Branch (768-D)
- Task-Specific Gated Fusion: F_h = (1 - g_h) * S + g_h * A
- 6 Output Classification Heads
"""
import json
import os
import torch
import torch.nn as nn
from config import Config


class ASILNLU(nn.Module):

    def __init__(self, mode: str = "mean", label_counts: dict = None):
        super().__init__()
        self.mode = mode
        self.use_attention = "attention" in mode
        self.use_acoustic = "acoustic" in mode

        # Semantic Projection Branch
        if self.use_attention:
            self.attention_pool = nn.Sequential(
                nn.Linear(Config.WHISPER_DIM, Config.FUSION_DIM),
                nn.Tanh(),
                nn.Linear(Config.FUSION_DIM, 1, bias=False),
            )

        self.whisper_proj = nn.Sequential(
            nn.Linear(Config.WHISPER_DIM, Config.FUSION_DIM),
            nn.LayerNorm(Config.FUSION_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
        )

        # Acoustic Projection Branch & Task-Specific Gates
        if self.use_acoustic:
            self.acoustic_proj = nn.Sequential(
                nn.Linear(Config.ACOUSTIC_DIM, Config.FUSION_DIM),
                nn.LayerNorm(Config.FUSION_DIM),
                nn.GELU(),
                nn.Dropout(Config.DROPOUT),
            )

            self.gates = nn.ModuleDict({
                head: nn.Sequential(
                    nn.Linear(Config.FUSION_DIM * 2, 1),
                    nn.Sigmoid(),
                )
                for head in label_counts.keys()
            })

        # Classification Heads
        self.heads = nn.ModuleDict()
        for head, num_classes in label_counts.items():
            self.heads[head] = nn.Sequential(
                nn.Linear(Config.FUSION_DIM, Config.HEAD_HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(Config.DROPOUT),
                nn.Linear(Config.HEAD_HIDDEN_DIM, num_classes),
            )

    def forward(self, whisper_x: torch.Tensor, acoustic_x: torch.Tensor = None):
        # 1. Semantic Pooling and Projection
        if self.use_attention:
            attn_weights = torch.softmax(self.attention_pool(whisper_x), dim=1)
            whisper_x = torch.sum(whisper_x * attn_weights, dim=1)

        S = self.whisper_proj(whisper_x)

        # 2. Acoustic Fusion (if enabled)
        if self.use_acoustic and acoustic_x is not None:
            A = self.acoustic_proj(acoustic_x)
            concat_SA = torch.cat([S, A], dim=-1)

            outputs = {}
            gate_values = {}
            for head_name, classifier in self.heads.items():
                g_h = self.gates[head_name](concat_SA)
                F_h = (1 - g_h) * S + g_h * A
                outputs[head_name] = classifier(F_h)
                gate_values[head_name] = g_h
            return outputs, gate_values

        # 3. Semantic-Only Pathway
        outputs = {
            head_name: classifier(S) for head_name, classifier in self.heads.items()
        }
        return outputs, None


def get_label_counts() -> dict:
    counts = {}
    for head in Config.HEADS:
        map_path = os.path.join(Config.ROOT_DIR, "results", "label_maps", f"{head}.json")
        with open(map_path, "r") as f:
            counts[head] = len(json.load(f)) - 1  # Exclude MASK
    return counts
