import torch
import torch.nn as nn
import json
import os
from config import Config

class ASILNLU(nn.Module):
    def __init__(self, mode="mean", label_counts=None):
        super().__init__()
        self.mode = mode
        self.input_dim = 1280
        
        if mode == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(self.input_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 1, bias=False)
            )
            
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.heads = nn.ModuleDict()
        for head, num_classes in label_counts.items():
            self.heads[head] = nn.Linear(256, num_classes)
            
    def forward(self, x):
        # x shape: [B, 1280] for mean, [B, T, 1280] for attention
        if self.mode == "attention":
            attn_weights = torch.softmax(self.attention_pool(x), dim=1) # [B, T, 1]
            x = torch.sum(x * attn_weights, dim=1) # [B, 1280]
            
        shared_rep = self.shared_mlp(x)
        
        outputs = {}
        for head_name, layer in self.heads.items():
            outputs[head_name] = layer(shared_rep)
            
        return outputs

def get_label_counts():
    counts = {}
    for head in Config.HEADS:
        with open(os.path.join(Config.ROOT_DIR, "results", "label_maps", f"{head}.json"), "r") as f:
            counts[head] = len(json.load(f)) - 1 # Exclude MASK
    return counts
