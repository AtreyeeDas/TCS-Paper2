import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import whisper
import joblib

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
WHISPER_MODEL_PATH = "/home/spark2/Models/whisper_large_v3_turbo" # Update to local base.en path
ARTIFACTS_DIR = "nlu_robust_experiment/models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HierarchicalProjection(nn.Module):
    def __init__(self, input_dim=512, projection_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        z = self.projector(x)
        return F.normalize(z, p=2, dim=1)

class VoiceNLU:
    def __init__(self):
        print(f"Loading Whisper from {WHISPER_MODEL_PATH}...")
        self.whisper_model = whisper.load_model(WHISPER_MODEL_PATH, device=DEVICE)
        
        self.scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "whisper_scaler.joblib"))
        self.encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoders.joblib"))
        
        self.projection = HierarchicalProjection(512, 128).to(DEVICE)
        self.projection.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "best_hierarchical_projection.pt"), map_location=DEVICE))
        self.projection.eval()
        
        self.mlps = {
            "domain": joblib.load(os.path.join(ARTIFACTS_DIR, "domain_mlp.joblib")),
            "subdomain": joblib.load(os.path.join(ARTIFACTS_DIR, "subdomain_mlp.joblib")),
            "topic": joblib.load(os.path.join(ARTIFACTS_DIR, "topic_mlp.joblib")),
            "document_type": joblib.load(os.path.join(ARTIFACTS_DIR, "document_type_mlp.joblib"))
        }

    @torch.inference_mode()
    def predict(self, audio_path: str) -> dict:
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(DEVICE)
        
        features = self.whisper_model.encoder(mel.unsqueeze(0))
        embedding = features.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        
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
