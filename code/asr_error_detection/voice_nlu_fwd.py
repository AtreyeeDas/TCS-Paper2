"""
Voice-NLU Forward Inference Module.
Performs forward inference on the pre-computed 128-D hierarchical semantic embeddings
using the 4 trained Voice-NLU MLPs, strictly enforcing alignment by sample_id.
"""

from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from config import (
    VOICE_DOCUMENT_TYPE_MLP,
    VOICE_DOMAIN_MLP,
    VOICE_LABEL_ENCODERS,
    VOICE_SEMANTIC_EMBEDDINGS_NPY,
    VOICE_SUBDOMAIN_MLP,
    VOICE_TOPIC_MLP,
    WHISPER_EMBEDDING_METADATA_CSV,
)


class VoiceNLUForwardInference:

    def __init__(self):
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        print("[Voice-NLU] Loading saved models and label encoders...")
        self.encoders = joblib.load(VOICE_LABEL_ENCODERS)
        self.mlps = {
            "domain": joblib.load(VOICE_DOMAIN_MLP),
            "subdomain": joblib.load(VOICE_SUBDOMAIN_MLP),
            "topic": joblib.load(VOICE_TOPIC_MLP),
            "document_type": joblib.load(VOICE_DOCUMENT_TYPE_MLP),
        }
        self.semantic_embeddings = np.load(
            VOICE_SEMANTIC_EMBEDDINGS_NPY
        ).astype(np.float32)
        self.metadata_df = pd.read_csv(WHISPER_EMBEDDING_METADATA_CSV)

        assert len(self.semantic_embeddings) == len(self.metadata_df), (
            f"Embedding count ({len(self.semantic_embeddings)}) does not match "
            f"metadata count ({len(self.metadata_df)})!"
        )

    def run_inference(self) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Runs forward inference across all samples and returns predictions and raw posteriors."""
        print(
            f"[Voice-NLU] Running forward inference over {len(self.semantic_embeddings)} samples..."
        )
        z = self.semantic_embeddings
        n_samples = len(z)

        raw_posteriors = {}
        for head in ["domain", "subdomain", "topic", "document_type"]:
            raw_posteriors[head] = self.mlps[head].predict_proba(z)

        results = []
        for i in range(n_samples):
            sample_id = str(self.metadata_df.iloc[i]["sample_id"])
            scenario_id = self.metadata_df.iloc[i]["scenario_id"]
            split = self.metadata_df.iloc[i]["split"]

            row_data = {
                "sample_id": sample_id,
                "scenario_id": scenario_id,
                "split": split,
            }

            # Domain Head
            dom_probs = raw_posteriors["domain"][i]
            dom_idx = int(np.argmax(dom_probs))
            dom_label = self.encoders["domain_label"].inverse_transform(
                [dom_idx]
            )[0]
            dom_conf = float(dom_probs[dom_idx])

            row_data["voice_domain"] = dom_label
            row_data["voice_domain_confidence"] = dom_conf

            # Subdomain Head
            sub_probs = raw_posteriors["subdomain"][i]
            sub_idx = int(np.argmax(sub_probs))
            sub_label = self.encoders["subdomain_label"].inverse_transform(
                [sub_idx]
            )[0]
            sub_conf = float(sub_probs[sub_idx])

            # Topic Head
            top_probs = raw_posteriors["topic"][i]
            top_idx = int(np.argmax(top_probs))
            top_label = self.encoders["topic_label"].inverse_transform(
                [top_idx]
            )[0]
            top_conf = float(top_probs[top_idx])

            # Document Type Head
            doc_probs = raw_posteriors["document_type"][i]
            doc_idx = int(np.argmax(doc_probs))
            doc_label = self.encoders["document_type_label"].inverse_transform(
                [doc_idx]
            )[0]
            doc_conf = float(doc_probs[doc_idx])

            # Preserve exact hierarchical rule: if domain == "general", subordinates are NONE
            if dom_label == "general":
                sub_label = "NONE"
                top_label = "NONE"
                doc_label = "NONE"

            row_data["voice_subdomain"] = sub_label
            row_data["voice_subdomain_confidence"] = sub_conf
            row_data["voice_topic"] = top_label
            row_data["voice_topic_confidence"] = top_conf
            row_data["voice_document_type"] = doc_label
            row_data["voice_document_type_confidence"] = doc_conf

            results.append(row_data)

        df_results = pd.DataFrame(results)
        return df_results, raw_posteriors
