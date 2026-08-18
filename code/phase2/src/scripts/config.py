import os
import torch

class Config:
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "ASIL_DATASETS")
    MASTER_CSV = os.path.join(ROOT_DIR, "master_nlu_dataset.csv")
    WHISPER_PATH = "/home/spark2/Models/whisper_large_v3_turbo"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Safely fallback if bfloat16 is not supported
    DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    HEADS = ["domain", "subdomain", "intent", "entity_type", "urgency", "emotion"]
    MASK_TOKEN = "MASK"
    MASK_ID = -1
    
    # Training Hyperparameters
    BATCH_SIZE = 64
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 50
    PATIENCE = 10
    
    # Acoustic & Architecture Dimensions
    ACOUSTIC_FEATURE_DIM = 46
    ACOUSTIC_HIDDEN_DIM = 64
    FUSION_DIM = 512
    DROPOUT = 0.2
    
    # Ablation Matrix Configuration
    USE_ACOUSTIC = True
    USE_GATED_FUSION = True
    USE_CLASS_WEIGHTS = True
    USE_WEIGHTED_SAMPLER = True
    
    MAX_CLASS_WEIGHT = 5.0
    SEED = 42
