# Phase 2 Dependencies

Ensure these packages are installed in your `asil_nlu` Conda environment.

| Package | Required Version | Reason | Installation Command |
| :--- | :--- | :--- | :--- |
| `transformers` | `>=4.40.0` | For Whisper and HuBERT extraction | `pip install transformers` |
| `torch` | `>=2.3.0` | Core DL framework | Already available |
| `torchaudio` | `>=2.3.0` | Audio loading | Already available |
| `librosa` | `>=0.10.0` | Resampling to 16kHz | `pip install librosa` |
| `pandas` | `>=2.0.0` | Dataset manipulation | `pip install pandas` |
| `scikit-learn` | `>=1.3.0` | Metrics and Splitting | `pip install scikit-learn` |
| `matplotlib` | `>=3.7.0` | Confusion matrices | `pip install matplotlib` |
| `seaborn` | `>=0.12.0` | Heatmap plotting | `pip install seaborn` |
| `tqdm` | `>=4.65.0` | Progress bars | `pip install tqdm` |
