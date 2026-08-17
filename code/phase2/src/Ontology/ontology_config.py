"""
Global configuration for ASIL Label-Ontology Construction and Remapping Tool.
Locks paths, hardware settings, and domain rules.
"""
import os
import torch

class OntologyConfig:
    # Project Paths
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    MASTER_CSV = os.path.join(ROOT_DIR, "master_nlu_dataset.csv")
    OUTPUT_CANONICAL_CSV = os.path.join(ROOT_DIR, "master_nlu_dataset_canonical.csv")
    
    # Results Directories
    RESULTS_DIR = os.path.join(ROOT_DIR, "results", "ontology")
    RAW_STATS_DIR = os.path.join(RESULTS_DIR, "raw_label_statistics")
    NORM_STATS_DIR = os.path.join(RESULTS_DIR, "normalized_label_statistics")
    PROPOSED_DIR = os.path.join(RESULTS_DIR, "proposed")
    REVIEW_DIR = os.path.join(RESULTS_DIR, "final_class_review")
    APPROVED_DIR = os.path.join(RESULTS_DIR, "approved")
    VALIDATION_DIR = os.path.join(RESULTS_DIR, "validation")
    
    # Local Gemma Model Path
    GEMMA_PATH = "/home/spark2/Models/gemma4-e4b-it"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16
    
    # Heads and Special Values
    HEADS = ["domain", "subdomain", "intent", "entity_type", "urgency", "emotion"]
    MASK_TOKEN = "MASK"
    
    # Locked Reference Ontologies
    LOCKED_EMOTIONS = ["ANGER", "DISGUST", "FEAR", "JOY", "NEUTRAL", "SADNESS", "SURPRISE"]
    LOCKED_URGENCIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    @classmethod
    def make_dirs(cls):
        for d in [
            cls.RAW_STATS_DIR, cls.NORM_STATS_DIR, cls.PROPOSED_DIR,
            cls.REVIEW_DIR, cls.APPROVED_DIR, cls.VALIDATION_DIR
        ]:
            os.makedirs(d, exist_ok=True)
