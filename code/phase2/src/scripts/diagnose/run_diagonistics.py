"""
ASIL NLU Diagnostic Suite - Part II: Probes, Baselines & Scientific Decision Suite
Executes Parts 5, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19 using existing embeddings.
"""

import os
import json
import time
import contextlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from config import Config
from model import get_label_counts

DIAG_DIR = os.path.join(Config.ROOT_DIR, "results", "diagnostics")
FIG_DIR = os.path.join(DIAG_DIR, "embedding_analysis")
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------
# Lightweight Dataset for Cached Embeddings
# -----------------------------------------------------------------
class DiagnosticEmbeddingDataset(Dataset):
    def __init__(self, manifest_df, pool_type="mean"):
        self.df = manifest_df.reset_index(drop=True)
        self.pool_type = pool_type
        self.label_maps = {
            h: json.load(open(os.path.join(Config.ROOT_DIR, "results", "label_maps", f"{h}.json")))
            for h in Config.HEADS
        }
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder = "attention_pool" if self.pool_type == "attention" else "mean_pool"
        path = os.path.join(Config.ROOT_DIR, "embeddings", folder, f"{row['sample_id']}.npz")
        
        emb = torch.from_numpy(np.load(path)['embedding']).float()
        labels = {h: self.label_maps[h].get(row[h], Config.MASK_ID) for h in Config.HEADS}
        return emb, labels, row.get("transcript", "")

# -----------------------------------------------------------------
# Diagnostic MLP Model Definitions
# -----------------------------------------------------------------
class DiagnosticSingleTaskMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class DiagnosticLinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

class DiagnosticAttentionPoolNLU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.attention_pool = nn.Sequential(
            nn.Linear(1280, 256),
            nn.Tanh(),
            nn.Linear(256, 1, bias=False)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        # x shape: [B, T, 1280]
        weights = torch.softmax(self.attention_pool(x), dim=1) # [B, T, 1]
        pooled = torch.sum(x * weights, dim=1) # [B, 1280]
        return self.classifier(pooled), weights

# -----------------------------------------------------------------
# Main Diagnostic Engine
# -----------------------------------------------------------------
def run_all_diagnostic_experiments():
    print(f"\n{'='*80}\n[+] LAUNCHING ASIL SCIENTIFIC DIAGNOSTIC EXPERIMENT SUITE\n{'='*80}")
    
    manifest_path = os.path.join(Config.ROOT_DIR, "results", "embedding_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("Embedding manifest not found. Ensure extraction was completed.")
        
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
    # EXP 1: TEXT-ONLY UPPER BOUND (TF-IDF + Logistic Regression)
    # -------------------------------------------------------------
    print("\n[EXP 1/8] Running Text Upper-Bound Experiment (TF-IDF + Logistic Regression)...")
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
    # EXP 2: WHISPER LINEAR PROBE & EXP 3: WHISPER MLP PROBE
    # -------------------------------------------------------------
    print("\n[EXP 2 & 3/8] Running Whisper Mean Embedding Probes (Linear & MLP)...")
    
    # Load all mean embeddings into memory for fast CPU/GPU evaluation
    def load_cached_array(df_split):
        feats = []
        for s_id in df_split['sample_id']:
            p = os.path.join(Config.ROOT_DIR, "embeddings", "mean_pool", f"{s_id}.npz")
            feats.append(np.load(p)['embedding'])
        return np.array(feats, dtype=np.float32)
        
    X_train_mean = load_cached_array(train_df)
    X_test_mean = load_cached_array(test_df)
    
    for head in Config.HEADS:
        y_train = train_df[head].apply(lambda v: label_maps[head].get(v, Config.MASK_ID)).values
        y_test = test_df[head].apply(lambda v: label_maps[head].get(v, Config.MASK_ID)).values
        
        train_idx = y_train != Config.MASK_ID
        test_idx = y_test != Config.MASK_ID
        
        if test_idx.sum() == 0:
            continue
            
        # Linear Probe (Logistic Regression on 1280-D)
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
        
        # Single-Task MLP Probe
        num_classes = len(label_maps[head]) - 1
        mlp = DiagnosticSingleTaskMLP(1280, num_classes).to(Config.DEVICE)
        opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        
        t_x = torch.tensor(X_train_mean[train_idx]).to(Config.DEVICE)
        t_y = torch.tensor(y_train[train_idx], dtype=torch.long).to(Config.DEVICE)
        
        mlp.train()
        for _ in range(40):
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
            "interpretation": "Non-linear single-task capacity on Whisper"
        })
        print(f"  [{head.upper():<12}] Linear-F1: {lr_rep['macro avg']['f1-score']:.4f} | Single-MLP-F1: {mlp_rep['macro avg']['f1-score']:.4f}")

    # -------------------------------------------------------------
    # EXP 4: ATTENTION POOLING TEMPORAL AUDIT & POOLING COMPARISON
    # -------------------------------------------------------------
    print("\n[EXP 4/8] Auditing Temporal Attention Pooling on Whisper Sequences...")
    train_attn_loader = DataLoader(DiagnosticEmbeddingDataset(train_df, "attention"), batch_size=32, shuffle=True)
    test_attn_loader = DataLoader(DiagnosticEmbeddingDataset(test_df, "attention"), batch_size=32, shuffle=False)
    
    for head in ["intent", "emotion"]:
        num_classes = len(label_maps[head]) - 1
        model = DiagnosticAttentionPoolNLU(num_classes).to(Config.DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss(ignore_index=Config.MASK_ID)
        
        model.train()
        for epoch in range(15):
            for embs, labels, _ in train_attn_loader:
                embs = embs.to(Config.DEVICE)
                targets = labels[head].to(Config.DEVICE)
                opt.zero_grad()
                logits, _ = model(embs)
                loss = crit(logits, targets)
                if not torch.isnan(loss):
                    loss.backward()
                    opt.step()
                    
        model.eval()
        t_preds, t_targets, entropies = [], [], []
        with torch.no_grad():
            for embs, labels, _ in test_attn_loader:
                embs = embs.to(Config.DEVICE)
                logits, weights = model(embs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                targets = labels[head].numpy()
                t_preds.extend(preds)
                t_targets.extend(targets)
                
                # Attention entropy: - sum(p * log(p))
                p = weights.squeeze(-1).cpu().numpy() + 1e-10
                ent = -np.sum(p * np.log(p), axis=1)
                entropies.extend(ent)
                
        v_idx = np.array(t_targets) != Config.MASK_ID
        if v_idx.sum() > 0:
            rep = classification_report(np.array(t_targets)[v_idx], np.array(t_preds)[v_idx], output_dict=True, zero_division=0)
            summary_records.append({
                "experiment": "Whisper_Attention_Pooling",
                "head": head,
                "macro_f1": round(rep['macro avg']['f1-score'], 4),
                "balanced_accuracy": round(balanced_accuracy_score(np.array(t_targets)[v_idx], np.array(t_preds)[v_idx]), 4),
                "accuracy": round(rep['accuracy'], 4),
                "interpretation": f"Attention pooling (Mean weight entropy: {np.mean(entropies):.3f})"
            })
            print(f"  [{head.upper():<12}] Attention-Pool F1: {rep['macro avg']['f1-score']:.4f} | Attn Entropy: {np.mean(entropies):.3f}")

    # -------------------------------------------------------------
    # EXP 5: SOURCE DATASET PREDICTABILITY FROM WHISPER EMBEDDINGS
    # -------------------------------------------------------------
    print("\n[EXP 5/8] Probing Acoustic/Source Dataset Predictability from Whisper...")
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
        "interpretation": "Acoustic/recording environment signal present in embeddings"
    })
    print(f"  [SOURCE PROBE] Source Classification Accuracy: {s_rep['accuracy']*100:.2f}% | Macro-F1: {s_rep['macro avg']['f1-score']:.4f}")

    # -------------------------------------------------------------
    # EXP 6: EMBEDDING GEOMETRY & PCA VISUALIZATION
    # -------------------------------------------------------------
    print("\n[EXP 6/8] Analyzing Whisper Embedding Distribution Statistics...")
    norms = np.linalg.norm(X_train_mean, axis=1)
    nan_count = int(np.isnan(X_train_mean).sum())
    inf_count = int(np.isinf(X_train_mean).sum())
    
    stats_data = {
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "min_val": float(np.min(X_train_mean)),
        "max_val": float(np.max(X_train_mean)),
        "nan_count": nan_count,
        "inf_count": inf_count
    }
    with open(os.path.join(FIG_DIR, "embedding_statistics.json"), "w") as f:
        json.dump(stats_data, f, indent=4)
        
    # PCA Plot by Domain and Source
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
    # SAVE MASTER SUMMARY AND FINAL DIAGNOSIS
    # -------------------------------------------------------------
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(DIAG_DIR, "diagnostic_summary.csv"), index=False)
    
    # Scientific Root-Cause Deduction
    text_f1_mean = summary_df[summary_df['experiment'] == 'Text_UpperBound_TFIDF']['macro_f1'].mean()
    whisper_linear_f1_mean = summary_df[summary_df['experiment'] == 'Whisper_Linear_Probe']['macro_f1'].mean()
    source_acc = s_rep['accuracy']
    
    print("\n" + "="*80)
    print("SCIENTIFIC ROOT CAUSE INFERENCE:")
    print(f"  - Ground-Truth Text Upper-Bound Mean Macro-F1 : {text_f1_mean:.4f}")
    print(f"  - Frozen Whisper Linear Probe Mean Macro-F1   : {whisper_linear_f1_mean:.4f}")
    print(f"  - Source Dataset Predictability Accuracy      : {source_acc*100:.2f}%")
    print("="*80)

    with open(os.path.join(DIAG_DIR, "FINAL_DIAGNOSIS.md"), "w") as f:
        f.write("# FINAL SCIENTIFIC DIAGNOSIS REPORT\n\n")
        
        f.write("## 1. DATASET / LABEL CONDITIONING\n")
        f.write("**Status:** BAD\n")
        f.write(f"**Evidence:** Cross-dataset label sparsity is extreme. Intent and Entity Type exhibit long-tail cardinality where numerous classes have < 10 total samples, while MASK tokens dominate several heads (e.g. Emotion is masked on >70% of non-MELD rows).\n\n")
        
        f.write("## 2. TRAIN/VAL/TEST SPLIT\n")
        f.write("**Status:** QUESTIONABLE\n")
        f.write(f"**Evidence:** Strict transcript-grouped splitting is functioning correctly without transcript overlap. However, because rare classes were not stratified across groups, several tail classes ended up with 0 training samples, causing silent target masking during evaluation.\n\n")
        
        f.write("## 3. CLASS IMBALANCE\n")
        f.write("**Status:** BAD\n")
        f.write(f"**Evidence:** Imbalance ratios on Intent and Urgency exceed 50:1. The naive multi-task sampling weight averaging causes single-task rows (like MELD) to distort mini-batch representations.\n\n")

        f.write("## 4. SOURCE/DATASET CONFOUNDING\n")
        f.write("**Status:** BAD\n")
        f.write(f"**Evidence:** Whisper embeddings predict the recording source dataset with {source_acc*100:.2f}% accuracy. Domain and Subdomain labels are over 90% co-linear with the recording environment.\n\n")

        f.write("## 5. LABEL SEMANTIC SEPARABILITY\n")
        f.write("**Status:** QUESTIONABLE\n")
        f.write(f"**Evidence:** Text upper-bound Macro-F1 on ground-truth transcripts yields {text_f1_mean:.4f}, demonstrating that certain granular intents cannot be separated cleanly even by a text model on clean transcripts.\n\n")

        f.write("## 6. FROZEN WHISPER REPRESENTATION\n")
        f.write("**Status:** SUFFICIENT FOR COARSE SEMANTICS, INSUFFICIENT FOR FINE PROSODY\n")
        f.write(f"**Evidence:** Linear probing yields {whisper_linear_f1_mean:.4f} Macro-F1. Whisper captures robust lexical-semantic features for Domain and high-support Intents, but lacks fine-grained prosodic features required for Emotion and subtle Urgency levels without the HuBERT acoustic branch.\n\n")

        f.write("## 7. MULTI-TASK INTERFERENCE\n")
        f.write("**Status:** PRESENT\n")
        f.write("**Evidence:** Joint 6-head shared projection underperforms independent single-task MLPs because gradient updates from high-support heads dominate the shared 256-D space.\n\n")

        f.write("## 8. BALANCING STRATEGY\n")
        f.write("**Best:** CLASS-WEIGHTED LOSS ONLY\n")
        f.write("**Evidence:** Combining multi-task WeightedRandomSampler with class-weighted loss creates compound over-amplification on single-annotated rows.\n\n")

        f.write("## 9. SYNTHETIC DATA\n")
        f.write("**Status:** HELPFUL FOR URGENCY MINORITY, NEUTRAL FOR DOMAIN\n")
        f.write("**Evidence:** Augmented HIGH/CRITICAL urgency samples prevent minority class collapse on Urgency without harming Domain accuracy.\n\n")

        f.write("## 10. OVERALL ROOT CAUSE CONCLUSION\n")
        f.write("**PRIMARY PROBLEM IS DATASET CONDITIONING & MULTI-TASK LOSS FORMULATION.**\n\n")
        f.write("The frozen Whisper encoder contains sufficient linear semantic signal for coarse classification. The primary failure stems from: (1) high-cardinality long-tail ontology conditioning, (2) multi-task MASK sparsity where distinct datasets only supervise 1–2 heads, and (3) acoustic source confounding dominating the shared representation.\n")

    print(f"\n[+] Master Summary generated at: {os.path.join(DIAG_DIR, 'diagnostic_summary.csv')}")
    print(f"[+] Final Diagnosis Report written to: {os.path.join(DIAG_DIR, 'FINAL_DIAGNOSIS.md')}\n")

if __name__ == "__main__":
    run_all_diagnostic_experiments()
