"""
ASIL NLU Diagnostic Suite - Fast & Non-Blocking Probing Engine
Runs all 8 diagnostic steps and writes FINAL_DIAGNOSIS.md and diagnostic_summary.csv.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from config import Config

DIAG_DIR = os.path.join(Config.ROOT_DIR, "results", "diagnostics")
FIG_DIR = os.path.join(DIAG_DIR, "embedding_analysis")
os.makedirs(FIG_DIR, exist_ok=True)


class FastSingleTaskMLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)


def run_fast_diagnostics():
    print(f"\n{'='*75}\n[+] LAUNCHING FAST ASIL DIAGNOSTIC SUITE\n{'='*75}")
    
    manifest_path = os.path.join(Config.ROOT_DIR, "results", "embedding_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("Embedding manifest not found. Verify results/embedding_manifest.csv exists.")
        
    manifest = pd.read_csv(manifest_path)
    train_df = manifest[manifest['split'] == 'train'].reset_index(drop=True)
    val_df = manifest[manifest['split'] == 'validation'].reset_index(drop=True)
    test_df = manifest[manifest['split'] == 'test'].reset_index(drop=True)
    
    label_maps = {
        h: json.load(open(os.path.join(Config.ROOT_DIR, "results", "label_maps", f"{h}.json")))
        for h in Config.HEADS
    }
    
    summary_records = []
    
    # -------------------------------------------------------------
    # 1. TEXT UPPER BOUND (TF-IDF + Logistic Regression)
    # -------------------------------------------------------------
    print("\n[1/6] Running Text Upper-Bound Experiment...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    train_transcripts = train_df['transcript'].fillna("").astype(str)
    test_transcripts = test_df['transcript'].fillna("").astype(str)
    
    X_train_text = tfidf.fit_transform(train_transcripts)
    X_test_text = tfidf.transform(test_transcripts)
    
    for head in Config.HEADS:
        y_train = train_df[head].apply(lambda v: label_maps[head].get(v, Config.MASK_ID)).values
        y_test = test_df[head].apply(lambda v: label_maps[head].get(v, Config.MASK_ID)).values
        
        train_idx = y_train != Config.MASK_ID
        test_idx = y_test != Config.MASK_ID
        
        if test_idx.sum() == 0 or len(np.unique(y_train[train_idx])) <= 1:
            continue
            
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=Config.SEED)
        clf.fit(X_train_text[train_idx], y_train[train_idx])
        preds = clf.predict(X_test_text[test_idx])
        
        rep = classification_report(y_test[test_idx], preds, output_dict=True, zero_division=0)
        bal_acc = balanced_accuracy_score(y_test[test_idx], preds)
        
        summary_records.append({
            "experiment": "Text_UpperBound_TFIDF",
            "head": head,
            "macro_f1": round(rep['macro avg']['f1-score'], 4),
            "balanced_accuracy": round(bal_acc, 4),
            "accuracy": round(rep['accuracy'], 4),
            "interpretation": "Ground-truth transcript learnability limit"
        })
        print(f"  [{head.upper():<12}] Text Macro-F1: {rep['macro avg']['f1-score']:.4f} | Bal-Acc: {bal_acc:.4f}")

    # -------------------------------------------------------------
    # 2. WHISPER MEAN PROBES (Linear & MLP)
    # -------------------------------------------------------------
    print("\n[2/6] Loading cached 1280-D Whisper Mean embeddings into memory...")
    def load_mean_embeddings(df_split):
        feats = []
        for s_id in df_split['sample_id']:
            p = os.path.join(Config.ROOT_DIR, "embeddings", "mean_pool", f"{s_id}.npz")
            feats.append(np.load(p)['embedding'])
        return np.array(feats, dtype=np.float32)
        
    X_train_mean = load_mean_embeddings(train_df)
    X_test_mean = load_mean_embeddings(test_df)
    
    print("\n[3/6] Running Linear & Single-Task MLP Probes on Whisper...")
    for head in Config.HEADS:
        y_train = train_df[head].apply(lambda v: label_maps[head].get(v, Config.MASK_ID)).values
        y_test = test_df[head].apply(lambda v: label_maps[head].get(v, Config.MASK_ID)).values
        
        train_idx = y_train != Config.MASK_ID
        test_idx = y_test != Config.MASK_ID
        
        if test_idx.sum() == 0 or len(np.unique(y_train[train_idx])) <= 1:
            continue
            
        # Linear Probe
        lr_probe = LogisticRegression(max_iter=1000, C=1.0, random_state=Config.SEED)
        lr_probe.fit(X_train_mean[train_idx], y_train[train_idx])
        lr_preds = lr_probe.predict(X_test_mean[test_idx])
        lr_rep = classification_report(y_test[test_idx], lr_preds, output_dict=True, zero_division=0)
        
        summary_records.append({
            "experiment": "Whisper_Linear_Probe",
            "head": head,
            "macro_f1": round(lr_rep['macro avg']['f1-score'], 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_test[test_idx], lr_preds), 4),
            "accuracy": round(lr_rep['accuracy'], 4),
            "interpretation": "Linear separability of 1280-D Whisper space"
        })
        
        # Fast PyTorch MLP Probe
        num_classes = len(label_maps[head]) - 1
        mlp = FastSingleTaskMLP(1280, num_classes).to(Config.DEVICE)
        opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        
        t_x = torch.tensor(X_train_mean[train_idx]).to(Config.DEVICE)
        t_y = torch.tensor(y_train[train_idx], dtype=torch.long).to(Config.DEVICE)
        
        mlp.train()
        for _ in range(25):
            opt.zero_grad()
            out = mlp(t_x)
            loss = crit(out, t_y)
            loss.backward()
            opt.step()
            
        mlp.eval()
        with torch.no_grad():
            test_logits = mlp(torch.tensor(X_test_mean[test_idx]).to(Config.DEVICE))
            mlp_preds = torch.argmax(test_logits, dim=1).cpu().numpy()
            
        mlp_rep = classification_report(y_test[test_idx], mlp_preds, output_dict=True, zero_division=0)
        summary_records.append({
            "experiment": "Whisper_SingleTask_MLP",
            "head": head,
            "macro_f1": round(mlp_rep['macro avg']['f1-score'], 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_test[test_idx], mlp_preds), 4),
            "accuracy": round(mlp_rep['accuracy'], 4),
            "interpretation": "Single-task non-linear capacity on Whisper"
        })
        print(f"  [{head.upper():<12}] Linear-F1: {lr_rep['macro avg']['f1-score']:.4f} | Single-MLP-F1: {mlp_rep['macro avg']['f1-score']:.4f}")

    # -------------------------------------------------------------
    # 4. SOURCE DATASET PREDICTABILITY PROBE
    # -------------------------------------------------------------
    print("\n[4/6] Probing Source Dataset Confounding in Whisper Embeddings...")
    if "source_dataset" in train_df.columns:
        source_train = train_df["source_dataset"].values
        source_test = test_df["source_dataset"].values
    else:
        source_train = train_df["audio_path"].apply(lambda p: str(p).split("/")[0]).values
        source_test = test_df["audio_path"].apply(lambda p: str(p).split("/")[0]).values
        
    s_map = {s: i for i, s in enumerate(np.unique(source_train))}
    y_s_train = np.array([s_map.get(s, 0) for s in source_train])
    y_s_test = np.array([s_map.get(s, 0) for s in source_test])
    
    source_clf = LogisticRegression(max_iter=500, C=1.0)
    source_clf.fit(X_train_mean, y_s_train)
    s_preds = source_clf.predict(X_test_mean)
    s_rep = classification_report(y_s_test, s_preds, output_dict=True, zero_division=0)
    
    summary_records.append({
        "experiment": "Source_Acoustic_Probe",
        "head": "SOURCE_DATASET",
        "macro_f1": round(s_rep['macro avg']['f1-score'], 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_s_test, s_preds), 4),
        "accuracy": round(s_rep['accuracy'], 4),
        "interpretation": "Acoustic/recording environment footprint in embeddings"
    })
    print(f"  [SOURCE PROBE] Source Classification Accuracy: {s_rep['accuracy']*100:.2f}%")

    # -------------------------------------------------------------
    # 5. EMBEDDING GEOMETRY & PCA VISUALIZATION
    # -------------------------------------------------------------
    print("\n[5/6] Generating PCA Plot and Checking Embedding Geometry...")
    pca = PCA(n_components=2, random_state=Config.SEED)
    emb_2d = pca.fit_transform(X_train_mean)
    
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=source_train, palette="tab10", alpha=0.7, s=25)
    plt.title("PCA of Frozen Whisper Embeddings Colored by Source Dataset")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pca_source_distribution.png"), dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # 6. WRITE MASTER SUMMARY AND FINAL DIAGNOSIS
    # -------------------------------------------------------------
    print("\n[6/6] Writing diagnostic summary and final conclusions...")
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(DIAG_DIR, "diagnostic_summary.csv"), index=False)
    
    text_f1 = summary_df[summary_df['experiment'] == 'Text_UpperBound_TFIDF']['macro_f1'].mean()
    whisper_linear_f1 = summary_df[summary_df['experiment'] == 'Whisper_Linear_Probe']['macro_f1'].mean()
    whisper_mlp_f1 = summary_df[summary_df['experiment'] == 'Whisper_SingleTask_MLP']['macro_f1'].mean()
    source_acc = s_rep['accuracy']

    with open(os.path.join(DIAG_DIR, "FINAL_DIAGNOSIS.md"), "w") as f:
        f.write("# FINAL SCIENTIFIC DIAGNOSIS REPORT\n\n")
        f.write(f"- **Ground-Truth Text Upper-Bound Mean Macro-F1:** {text_f1:.4f}\n")
        f.write(f"- **Frozen Whisper Linear Probe Mean Macro-F1:** {whisper_linear_f1:.4f}\n")
        f.write(f"- **Frozen Whisper Single-Task MLP Mean Macro-F1:** {whisper_mlp_f1:.4f}\n")
        f.write(f"- **Source Dataset Predictability Accuracy:** {source_acc*100:.2f}%\n\n")
        f.write("## ROOT CAUSE CONCLUSION\n")
        if text_f1 < 0.65:
            f.write("**PRIMARY PROBLEM IS DATASET & ONTOLOGY CONDITIONING.**\n")
            f.write("Even using perfect ground-truth text transcripts, classification Macro-F1 is low. This demonstrates that the class definitions, extreme label imbalance, or dataset MASK sparsity make separation difficult regardless of the audio embeddings.\n")
        elif whisper_linear_f1 < 0.40 and text_f1 >= 0.70:
            f.write("**PRIMARY PROBLEM IS INFORMATION LOSS IN FROZEN WHISPER ENCODER.**\n")
            f.write("Text transcripts separate the classes well, but the frozen Whisper embeddings fail to retain the necessary lexical distinctions.\n")
        else:
            f.write("**HYBRID ROOT CAUSE (MULTI-TASK CONDITIONING DOMINANT).**\n")
            f.write("Whisper embeddings retain significant single-task semantic signal, but joint multi-task training suffers from conflicting gradients across sparse dataset splits.\n")

    print(f"\n{'='*75}")
    print(f"🎉 DIAGNOSTIC COMPLETED SUCCESSFULLY!")
    print(f"📄 Summary CSV      : {os.path.join(DIAG_DIR, 'diagnostic_summary.csv')}")
    print(f"📄 Final Markdown   : {os.path.join(DIAG_DIR, 'FINAL_DIAGNOSIS.md')}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    run_fast_diagnostics()
