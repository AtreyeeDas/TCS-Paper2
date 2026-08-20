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

python -c "
import pandas as pd, json
df = pd.read_pickle('results/ontology_relabel/01_normalized.pkl')
rows = [json.loads(line) for line in open('results/ontology_relabel/05_assignment_checkpoint.jsonl')]
df.join(pd.DataFrame(rows).set_index('_idx')).to_csv('results/ontology_relabel/master_nlu_dataset_relabelled.csv', index=False)
print('Partial CSV compiled!')
"

