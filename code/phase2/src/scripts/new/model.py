import torch
import torch.nn as nn
import json
import os
from config import Config

class ASILNLU(nn.Module):
    def __init__(self, mode="mean", label_counts=None):
        super().__init__()
        self.mode = mode
        self.use_attention = "attention" in mode
        self.use_acoustic = "acoustic" in mode
        
        # Whisper Semantic Projector
        if self.use_attention:
            self.attention_pool = nn.Sequential(
                nn.Linear(Config.WHISPER_DIM, Config.FUSION_DIM),
                nn.Tanh(),
                nn.Linear(Config.FUSION_DIM, 1, bias=False)
            )
        
        self.whisper_proj = nn.Sequential(
            nn.Linear(Config.WHISPER_DIM, Config.FUSION_DIM),
            nn.LayerNorm(Config.FUSION_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT)
        )
        
        # HuBERT Acoustic Projector
        if self.use_acoustic:
            self.acoustic_proj = nn.Sequential(
                nn.Linear(Config.ACOUSTIC_DIM, Config.FUSION_DIM),
                nn.LayerNorm(Config.FUSION_DIM),
                nn.GELU(),
                nn.Dropout(Config.DROPOUT)
            )
            
            # Task-Specific Gates: F_h = (1-g_h)S + g_h A
            self.gates = nn.ModuleDict({
                head: nn.Sequential(
                    nn.Linear(Config.FUSION_DIM * 2, 1),
                    nn.Sigmoid()
                ) for head in label_counts.keys()
            })
            
        self.heads = nn.ModuleDict()
        for head, num_classes in label_counts.items():
            self.heads[head] = nn.Sequential(
                nn.Linear(Config.FUSION_DIM, Config.HEAD_HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(Config.DROPOUT),
                nn.Linear(Config.HEAD_HIDDEN_DIM, num_classes)
            )
            
    def forward(self, whisper_x, acoustic_x=None):
        # 1. Process Semantic Branch
        if self.use_attention:
            attn_weights = torch.softmax(self.attention_pool(whisper_x), dim=1) # [B, T, 1]
            whisper_x = torch.sum(whisper_x * attn_weights, dim=1) # [B, 1280]
            
        S = self.whisper_proj(whisper_x) # [B, 256]
        
        # 2. Process Acoustic Branch & Gated Fusion
        if self.use_acoustic and acoustic_x is not None:
            A = self.acoustic_proj(acoustic_x) # [B, 256]
            concat_SA = torch.cat([S, A], dim=-1) # [B, 512]
            
            outputs = {}
            gate_values = {}
            for head_name, classifier in self.heads.items():
                g_h = self.gates[head_name](concat_SA) # [B, 1]
                F_h = (1 - g_h) * S + g_h * A
                outputs[head_name] = classifier(F_h)
                gate_values[head_name] = g_h
            return outputs, gate_values
        else:
            outputs = {}
            for head_name, classifier in self.heads.items():
                outputs[head_name] = classifier(S)
            return outputs, None

def get_label_counts():
    counts = {}
    for head in Config.HEADS:
        with open(os.path.join(Config.ROOT_DIR, "results", "label_maps", f"{head}.json"), "r") as f:
            counts[head] = len(json.load(f)) - 1 # Exclude MASK
    return counts

"""

self.whisper_proj = nn.Sequential(
    nn.Linear(Config.WHISPER_DIM, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(512, Config.FUSION_DIM),
    nn.GELU(),
    nn.Dropout(0.2)
)
self.heads[head] = nn.Linear(Config.FUSION_DIM, num_classes)
"""
