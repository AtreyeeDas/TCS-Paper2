#!/bin/bash
echo "Phase A/B: Auditing Dataset & Splitting"
python scripts/dataset_audit.py

echo "Phase C/D/E: Extracting Whisper Embeddings"
python scripts/extract_whisper_embeddings.py

echo "Phase F/G: Training & Evaluating Mean-Pooling Model"
python scripts/train_evaluate.py
