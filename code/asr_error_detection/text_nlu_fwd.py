"""
Text-NLU Forward Inference Module.
Performs forward inference on arbitrary text strings (reference or controlled erroneous)
using the local MiniLM encoder, saved PyTorch projection, and 4 Text-NLU MLPs.
"""

from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import (
    TEXT_BATCH_SIZE,
    TEXT_DOCUMENT_TYPE_MLP,
    TEXT_DOMAIN_MLP,
    TEXT_ENCODER_LOCAL_PATH,
    TEXT_LABEL_ENCODERS,
    TEXT_PROJECTION_PT,
    TEXT_SCALER,
    TEXT_SUBDOMAIN_MLP,
    TEXT_TOPIC_MLP,
)


class TextHierarchicalProjection(nn.Module):

    def __init__(self, input_dim: int = 384, projection_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projector(x)
        return F.normalize(z, p=2, dim=1)


class TextNLUForwardInference:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        print(
            f"[Text-NLU] Loading local text encoder from: {TEXT_ENCODER_LOCAL_PATH} on {self.device}..."
        )
        self.text_encoder = SentenceTransformer(
            TEXT_ENCODER_LOCAL_PATH, device=self.device
        )
        self.scaler = joblib.load(TEXT_SCALER)
        self.encoders = joblib.load(TEXT_LABEL_ENCODERS)

        self.projection = TextHierarchicalProjection(384, 128).to(self.device)
        self.projection.load_state_dict(
            torch.load(TEXT_PROJECTION_PT, map_location=self.device)
        )
        self.projection.eval()

        self.mlps = {
            "domain": joblib.load(TEXT_DOMAIN_MLP),
            "subdomain": joblib.load(TEXT_SUBDOMAIN_MLP),
            "topic": joblib.load(TEXT_TOPIC_MLP),
            "document_type": joblib.load(TEXT_DOCUMENT_TYPE_MLP),
        }

    @torch.inference_mode()
    def run_inference_on_transcripts(
        self, transcripts: List[str], desc: str = "Text-NLU Inference"
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Encodes text, projects to 128-D normalized space, and computes 4-head predictions."""
        # 1. 384-D sentence embeddings
        embeddings_384 = self.text_encoder.encode(
            transcripts,
            batch_size=TEXT_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

        # 2. Standardization
        scaled_384 = self.scaler.transform(embeddings_384).astype(np.float32)

        # 3. 128-D Hierarchical Projection
        z_list = []
        for start in range(0, len(scaled_384), TEXT_BATCH_SIZE):
            batch = torch.tensor(
                scaled_384[start : start + TEXT_BATCH_SIZE],
                dtype=torch.float32,
                device=self.device,
            )
            z_batch = self.projection(batch)
            z_list.append(z_batch.cpu().numpy())
        z_128 = np.vstack(z_list).astype(np.float32)

        # 4. Multi-head MLP classification
        raw_posteriors = {}
        for head in ["domain", "subdomain", "topic", "document_type"]:
            raw_posteriors[head] = self.mlps[head].predict_proba(z_128)

        results = []
        for i in range(len(transcripts)):
            row_data = {}

            # Domain
            dom_probs = raw_posteriors["domain"][i]
            dom_idx = int(np.argmax(dom_probs))
            dom_label = self.encoders["domain_label"].inverse_transform(
                [dom_idx]
            )[0]
            dom_conf = float(dom_probs[dom_idx])
            row_data["text_domain"] = dom_label
            row_data["text_domain_confidence"] = dom_conf

            # Subdomain
            sub_probs = raw_posteriors["subdomain"][i]
            sub_idx = int(np.argmax(sub_probs))
            sub_label = self.encoders["subdomain_label"].inverse_transform(
                [sub_idx]
            )[0]
            sub_conf = float(sub_probs[sub_idx])

            # Topic
            top_probs = raw_posteriors["topic"][i]
            top_idx = int(np.argmax(top_probs))
            top_label = self.encoders["topic_label"].inverse_transform(
                [top_idx]
            )[0]
            top_conf = float(top_probs[top_idx])

            # Document Type
            doc_probs = raw_posteriors["document_type"][i]
            doc_idx = int(np.argmax(doc_probs))
            doc_label = self.encoders["document_type_label"].inverse_transform(
                [doc_idx]
            )[0]
            doc_conf = float(doc_probs[doc_idx])

            # Enforce hierarchical general rule
            if dom_label == "general":
                sub_label = "NONE"
                top_label = "NONE"
                doc_label = "NONE"

            row_data["text_subdomain"] = sub_label
            row_data["text_subdomain_confidence"] = sub_conf
            row_data["text_topic"] = top_label
            row_data["text_topic_confidence"] = top_conf
            row_data["text_document_type"] = doc_label
            row_data["text_document_type_confidence"] = doc_conf

            results.append(row_data)

        df_results = pd.DataFrame(results)
        return df_results, raw_posteriors
