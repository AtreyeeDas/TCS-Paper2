import os
import torch

class Config:
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(ROOT_DIR, "ASIL_Datasets")
    MASTER_CSV = os.path.join(ROOT_DIR, "master_nlu_dataset_canonical_augmented.csv")
    WHISPER_PATH = "/home/spark2/Models/whisper_large_v3_turbo"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float32
    
    HEADS = ["domain", "subdomain", "intent", "entity_type", "urgency", "emotion"]
    MASK_TOKEN = "MASK"
    MASK_ID = -1
    
    BATCH_SIZE = 64
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 50
    PATIENCE = 10
    
    SEED = 42