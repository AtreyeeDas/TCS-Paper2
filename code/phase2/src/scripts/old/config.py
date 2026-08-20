import os
import torch

class Config:
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__)) # Assumes running from Scripts-old
    PROJECT_ROOT = os.path.abspath(os.path.join(ROOT_DIR, ".."))
    
    # New Paths
    MASTER_CSV = os.path.join(PROJECT_ROOT, "results", "ontology_relabel", "master_nlu_dataset_relabelled.csv")
    FINAL_TRAIN_DIR = os.path.join(PROJECT_ROOT, "results", "final_training")
    FINAL_LABEL_MAPS_DIR = os.path.join(PROJECT_ROOT, "results", "final_label_maps")
    SPLITS_DIR = os.path.join(PROJECT_ROOT, "results", "splits")
    
    WHISPER_PATH = "/home/spark2/Models/whisper_large_v3_turbo"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float32
    
    # NEW CANONICAL TARGET LABELS
    HEADS = [
        "canonical_domain", "canonical_subdomain", "canonical_intent", 
        "canonical_entity_type", "canonical_urgency", "canonical_emotion"
    ]
    
    # Ignore targets safely
    INVALID_TOKENS = ["MASK", "OUTLIER_REVIEW", "INVALID_LLM_FALLBACK", "nan", ""]
    MASK_ID = -1
    
    # Training Hyperparameters
    BATCH_SIZE = 64
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 50
    PATIENCE = 10
