import os
import torch

class Config:
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(ROOT_DIR, "ASIL_Datasets")
    
    # 1. POINT TO THE NEW RELABELLED DATASET
    MASTER_CSV = os.path.join(ROOT_DIR, "results", "ontology_relabel", "master_nlu_dataset_relabelled.csv")
    WHISPER_PATH = "/home/spark2/Models/whisper_large_v3_turbo"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float32
    
    # 2. READ THE NEW CANONICAL LABELS
    HEADS = [
        "canonical_domain", "canonical_subdomain", "canonical_intent", 
        "canonical_entity_type", "canonical_urgency", "canonical_emotion"
    ]
    
    # 3. HANDLE OUTLIERS AS MASKS
    INVALID_TOKENS = ["MASK", "OUTLIER_REVIEW", "INVALID_LLM_FALLBACK"]
    MASK_ID = -1
    
    BATCH_SIZE = 64
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 50
    PATIENCE = 10
    
    SEED = 42
