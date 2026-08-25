"""
Central configuration for the Semantic Inconsistency ASR Error Detection Pipeline.
Update ROOT_DIR and model paths to match your workstation environment.
"""

from pathlib import Path

# =====================================================================
# EXPERIMENT & MODEL PATHS
# =====================================================================
ROOT_DIR = Path("/home/spark2/users/intern/Atreyee-Das/NLU_Robust_Experiment")

# Dataset
CONTROLLED_ERROR_CSV = ROOT_DIR / "dataset" / "controlled_asr_error_6000.csv"
DATASET_METADATA_CSV = ROOT_DIR / "dataset" / "nlu_robust_6000_scenario_paraphrase.csv"
WHISPER_EMBEDDING_METADATA_CSV = ROOT_DIR / "embeddings" / "whisper_embedding_metadata.csv"

# Voice-NLU Artifacts
VOICE_SEMANTIC_EMBEDDINGS_NPY = ROOT_DIR / "embeddings" / "hierarchical_semantic_embeddings.npy"
VOICE_MODELS_DIR = ROOT_DIR / "models"

VOICE_DOMAIN_MLP = VOICE_MODELS_DIR / "domain_mlp.joblib"
VOICE_SUBDOMAIN_MLP = VOICE_MODELS_DIR / "subdomain_mlp.joblib"
VOICE_TOPIC_MLP = VOICE_MODELS_DIR / "topic_mlp.joblib"
VOICE_DOCUMENT_TYPE_MLP = VOICE_MODELS_DIR / "document_type_mlp.joblib"
VOICE_LABEL_ENCODERS = VOICE_MODELS_DIR / "label_encoders.joblib"

# Text-NLU Artifacts
TEXT_MODELS_DIR = ROOT_DIR / "text_models"
TEXT_ENCODER_LOCAL_PATH = "/home/spark2/Models/all-MiniLM-L6-v2"

TEXT_SCALER = TEXT_MODELS_DIR / "text_scaler.joblib"
TEXT_PROJECTION_PT = TEXT_MODELS_DIR / "best_text_hierarchical_projection.pt"
TEXT_LABEL_ENCODERS = TEXT_MODELS_DIR / "text_label_encoders.joblib"
TEXT_DOMAIN_MLP = TEXT_MODELS_DIR / "text_domain_mlp.joblib"
TEXT_SUBDOMAIN_MLP = TEXT_MODELS_DIR / "text_subdomain_mlp.joblib"
TEXT_TOPIC_MLP = TEXT_MODELS_DIR / "text_topic_mlp.joblib"
TEXT_DOCUMENT_TYPE_MLP = TEXT_MODELS_DIR / "text_document_type_mlp.joblib"

# Output Directory
OUTPUT_DIR = ROOT_DIR / "error_detector"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# HIERARCHICAL WEIGHTS & HEADS
# =====================================================================
HEADS = ["domain", "subdomain", "topic", "document_type"]

HEAD_WEIGHTS = {
    "domain": 0.20,
    "subdomain": 0.25,
    "topic": 0.40,
    "document_type": 0.15,
}

# =====================================================================
# HYPERPARAMETERS & REPRODUCIBILITY
# =====================================================================
RANDOM_SEED = 42
LOGISTIC_REGRESSION_MAX_ITER = 3000
TEXT_BATCH_SIZE = 128
EPS = 1e-12
