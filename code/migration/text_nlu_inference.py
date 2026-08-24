import os
import torch
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from voice_nlu_inference import HierarchicalProjection

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
TEXT_MODEL_PATH = "/home/spark2/Models/all-MiniLM-L6-v2"
ARTIFACTS_DIR = "nlu_robust_experiment/text_models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TextNLU:
    def __init__(self):
        print(f"Loading Text Encoder from {TEXT_MODEL_PATH}...")
        self.encoder = SentenceTransformer(TEXT_MODEL_PATH, device=DEVICE)
        
        self.scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "text_scaler.joblib"))
        self.encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "text_label_encoders.joblib"))
        
        self.projection = HierarchicalProjection(384, 128).to(DEVICE)
        self.projection.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "best_text_hierarchical_projection.pt"), map_location=DEVICE))
        self.projection.eval()
        
        self.mlps = {
            "domain": joblib.load(os.path.join(ARTIFACTS_DIR, "text_domain_mlp.joblib")),
            "subdomain": joblib.load(os.path.join(ARTIFACTS_DIR, "text_subdomain_mlp.joblib")),
            "topic": joblib.load(os.path.join(ARTIFACTS_DIR, "text_topic_mlp.joblib")),
            "document_type": joblib.load(os.path.join(ARTIFACTS_DIR, "text_document_type_mlp.joblib"))
        }

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        embedding = self.encoder.encode(text, convert_to_numpy=True).astype(np.float32)
        scaled_emb = self.scaler.transform([embedding]).astype(np.float32)
        xb = torch.tensor(scaled_emb, device=DEVICE)
        z = self.projection(xb).cpu().numpy()
        
        result = {"semantic_embedding": z[0]}
        
        for head in ["domain", "subdomain", "topic", "document_type"]:
            probs = self.mlps[head].predict_proba(z)[0]
            pred_idx = np.argmax(probs)
            label = self.encoders[f"{head}_label"].inverse_transform([pred_idx])[0]
            
            result[head] = label
            result[f"{head}_confidence"] = float(probs[pred_idx])
            result[f"{head}_probs"] = {self.encoders[f"{head}_label"].inverse_transform([i])[0]: float(p) for i, p in enumerate(probs)}
            
        if result["domain"] == "general":
            for head in ["subdomain", "topic", "document_type"]:
                result[head] = "NONE"
                
        return result
