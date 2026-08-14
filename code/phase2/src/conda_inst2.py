# Activate your existing Conda environment
conda activate asil_env

# Install/verify HuggingFace ecosystem, audio loaders, and acceleration packages
pip install transformers accelerate sentencepiece soundfile librosa pandas tqdm
conda activate asil_env
python workstation_nlu_labeler.py
