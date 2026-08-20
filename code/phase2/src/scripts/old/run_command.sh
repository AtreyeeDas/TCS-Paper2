#!/bin/bash
# Exit immediately if any command fails
set -e

# ============================================================
# 1. ACTIVATE CONDA ENVIRONMENT
# ============================================================
# Adjust 'asil_nlu' to your actual environment name if different
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate asil_nlu

echo "============================================================"
echo "          RESUMING STAGE 5 (FAST ASSIGNMENT)                "
echo "============================================================"
# It will automatically find the .jsonl checkpoint and resume!
python relabel_pipeline.py --stage 5

echo "============================================================"
echo "          RUNNING STAGE 6 (LEARNABILITY GATE)               "
echo "============================================================"
python relabel_pipeline.py --stage 6

echo "============================================================"
echo "          STARTING FINAL NLU TRAINING PIPELINE              "
echo "============================================================"
cd Scripts-old/

# Run the final training script for both mean and attention pooling
python train_final.py --mode both --seed 42

echo "============================================================"
echo "   OVERNIGHT PIPELINE COMPLETE! CHECK RESULTS DIRECTORY.    "
echo "============================================================"
