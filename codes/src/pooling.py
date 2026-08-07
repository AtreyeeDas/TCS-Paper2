import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightAttentionPooling(nn.Module):
    """
    Lightweight Linear Attention Pooling over sequence dimension T.
    w = softmax(H * W_a)
    e_att = sum(w_t * H_t)
    """
    def __init__(self, embed_dim: int = 1280):
        super().__init__()
        # Deterministic initialization for reproducible research extraction
        torch.manual_seed(42)
        self.attn_projection = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (1, T, D) or (T, D)
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        attn_logits = self.attn_projection(x)  # (1, T, 1)
        attn_weights = F.softmax(attn_logits, dim=1)  # (1, T, 1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (1, D)
        return pooled.squeeze(0)

class PoolingEngine:
    def __init__(self, embed_dim: int = 1280):
        self.embed_dim = embed_dim
        self.attention_pooling_module = LightweightAttentionPooling(embed_dim=embed_dim)
        self.attention_pooling_module.eval()

    def compute_all_pools(self, encoder_hidden_state: np.ndarray) -> dict:
        """
        Input:
            encoder_hidden_state: numpy array of shape (1, T, D) or (T, D)
        Returns:
            dict containing mean_pool, max_pool, attention_pool (all 1D arrays of size D)
        """
        tensor_state = torch.tensor(encoder_hidden_state, dtype=torch.float32)
        if tensor_state.dim() == 3:
            tensor_state = tensor_state.squeeze(0)  # (T, D)

        # 1. Mean Pooling
        mean_pooled = torch.mean(tensor_state, dim=0).numpy()

        # 2. Max Pooling
        max_pooled = torch.max(tensor_state, dim=0).values.numpy()

        # 3. Lightweight Attention Pooling
        with torch.no_grad():
            att_pooled = self.attention_pooling_module(tensor_state).numpy()

        return {
            "mean_pool": mean_pooled,
            "max_pool": max_pooled,
            "attention_pool": att_pooled,
            "metadata": {
                "embedding_dimension": self.embed_dim,
                "sequence_length": int(tensor_state.shape[0]),
                "tensor_shapes": {
                    "encoder_input": list(encoder_hidden_state.shape),
                    "mean_pool": list(mean_pooled.shape),
                    "max_pool": list(max_pooled.shape),
                    "attention_pool": list(att_pooled.shape)
                },
                "pooling_methods": [
                    "Mean Pooling (Dimension 0 average)",
                    "Max Pooling (Dimension 0 max values)",
                    "Lightweight Linear Self-Attention Pooling"
                ]
            }
        }
