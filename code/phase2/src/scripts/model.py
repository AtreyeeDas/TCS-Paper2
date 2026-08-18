import torch
import torch.nn as nn
from config import Config

class ASILNLU(nn.Module):
    def __init__(self, label_counts):
        super().__init__()
        
        # Whisper Projection
        self.whisper_proj = nn.Sequential(
            nn.Linear(1280, Config.FUSION_DIM),
            nn.LayerNorm(Config.FUSION_DIM),
            nn.GELU()
        )
        
        # Acoustic Projection (Lightweight)
        if Config.USE_ACOUSTIC:
            self.acoustic_proj = nn.Sequential(
                nn.Linear(Config.ACOUSTIC_FEATURE_DIM, Config.ACOUSTIC_HIDDEN_DIM),
                nn.LayerNorm(Config.ACOUSTIC_HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(Config.DROPOUT),
                nn.Linear(Config.ACOUSTIC_HIDDEN_DIM, Config.FUSION_DIM)
            )
            
            if Config.USE_GATED_FUSION:
                self.gate = nn.Sequential(
                    nn.Linear(Config.FUSION_DIM * 2, Config.FUSION_DIM),
                    nn.Sigmoid()
                )
        
        # Head-Specific lightweight gating/projection
        self.heads = nn.ModuleDict()
        for head, num_classes in label_counts.items():
            self.heads[head] = nn.Sequential(
                nn.Linear(Config.FUSION_DIM, 256),
                nn.GELU(),
                nn.Dropout(Config.DROPOUT),
                nn.Linear(256, num_classes)
            )
            
    def forward(self, whisper_emb, acoustic_emb=None):
        w_proj = self.whisper_proj(whisper_emb)
        gate_vals = None
        
        if Config.USE_ACOUSTIC and acoustic_emb is not None:
            a_proj = self.acoustic_proj(acoustic_emb)
            if Config.USE_GATED_FUSION:
                gate_vals = self.gate(torch.cat([w_proj, a_proj], dim=-1))
                fused = w_proj + (gate_vals * a_proj)
            else:
                fused = w_proj + a_proj # Simple addition baseline
        else:
            fused = w_proj
            
        outputs = {}
        for head_name, layer in self.heads.items():
            outputs[head_name] = layer(fused)
            
        return outputs, gate_vals
