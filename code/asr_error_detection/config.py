"""
Central configuration for ASIL NLU Pipeline.
Defines model paths, dimensions, training hyperparameters, and head schemas.
"""
import os
import torch


class Config:
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "ASIL_Datasets")
    MASTER_CSV = os.path.join(ROOT_DIR, "master_nlu_dataset_canonical_augmented.csv")

    # Local Model Paths (Hugging Face Transformers)
    WHISPER_PATH = "/home/spark2/Models/whisper_large_v3_turbo"
    HUBERT_PATH = "/home/spark2/Models/hubert_base_superb_er"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    # Classification Heads & Masking Definitions
    HEADS = ["domain", "subdomain", "intent", "entity_type", "urgency", "emotion"]
    MASK_TOKEN = "MASK"
    MASK_ID = -1

    # Training Hyperparameters
    BATCH_SIZE = 64
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 50
    PATIENCE = 10
    MAX_CLASS_WEIGHT = 5.0

    # Model Dimensions
    WHISPER_DIM = 1280
    ACOUSTIC_DIM = 768
    FUSION_DIM = 256
    HEAD_HIDDEN_DIM = 128
    DROPOUT = 0.2

    SEED = 42
