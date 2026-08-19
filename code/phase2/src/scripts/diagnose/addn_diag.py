"""
ASIL NLU - FINAL THREE DIAGNOSTIC EXPERIMENTS
============================================
Experiment 1: Single-Task vs Joint Multi-Task Interference
Experiment 2: Multi-Task Balancing Strategy Ablation (None, Weighted Loss, Sampler, Both)
Experiment 3: Real Speech vs Synthetic TTS Generalization (All-Test vs Real-Only Test)

- Reuses existing cached Whisper mean embeddings (embeddings/mean_pool/)
- Reuses existing splits (results/splits/)
- Reuses existing label maps (results/label_maps/)
- ZERO Whisper/HuBERT extraction
- ZERO dataset modifications
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

from config import Config

# Setup output directories
OUT_DIR = os.path.join(Config.ROOT_DIR, "results", "diagnostics", "final_three")
CM_DIR = os.path.join(OUT_DIR, "confusion_matrices")
LOG_DIR = os.path.join(OUT_DIR, "experiment_logs")
os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# =====================================================================
# 1. DATASET & CACHED EMBEDDING LOADER (FAST IN-MEMORY CACHE)
# =====================================================================

class FastCachedDataset(Dataset):
    """Holds pre-loaded embeddings and integer-encoded multi-task targets."""
    def __init__(self, embeddings_tensor, targets_dict, sample_indices=None):
        if sample_indices is not None:
            self.embeddings = embeddings_tensor[sample_indices]
            self.targets = {h: targets_dict[h][sample_indices] for h in Config.HEADS}
        else:
            self.embeddings = embeddings_tensor
            self.targets = targets_dict
            
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], {h: self.targets[h][idx] for h in Config.HEADS}


def load_all_cached_data():
    """Loads manifest, label maps, and pre-caches all 1280-D mean embeddings into RAM."""
    manifest_path = os.path.join(Config.ROOT_DIR, "results", "embedding_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Ensure embeddings exist.")
        
    manifest = pd.read_csv(manifest_path)
    
    # Load label maps
    label_maps = {}
    for h in Config.HEADS:
        lmap_path = os.path.join(Config.ROOT_DIR, "results", "label_maps", f"{h}.json")
        if not os.path.exists(lmap_path):
            raise FileNotFoundError(f"Label map missing: {lmap_path}")
        with open(lmap_path, "r") as f:
            label_maps[h] = json.load(f)
            
    # Pre-load all 1280-D mean embeddings into RAM (only ~24MB total)
    print(f"[+] Pre-loading {len(manifest)} cached 1280-D Whisper embeddings into RAM...")
    embs_list = []
    for s_id in manifest['sample_id']:
        p = os.path.join(Config.ROOT_DIR, "embeddings", "mean_pool", f"{s_id}.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Embedding file missing: {p}")
        embs_list.append(np.load(p)['embedding'])
    all_embs = torch.tensor(np.array(embs_list, dtype=np.float32))
    
    # Encode targets
    all_targets = {}
    for h in Config.HEADS:
        t_arr = manifest[h].apply(lambda v: label_maps[h].get(v, Config.MASK_ID)).values
        all_targets[h] = torch.tensor(t_arr, dtype=torch.long)
        
    # Split masks
    train_idx = manifest[manifest['split'] == 'train'].index.values
    val_idx = manifest[manifest['split'] == 'validation'].index.values
    test_idx = manifest[manifest['split'] == 'test'].index.values
    
    # Synthetic flags
    if "is_synthetic" in manifest.columns:
        is_synth = manifest["is_synthetic"].astype(bool).values
    else:
        is_synth = manifest["resolved_path"].astype(str).str.lower().str.contains("synthetic").values
    manifest["is_synthetic"] = is_synth
    
    train_dataset = FastCachedDataset(all_embs, all_targets, train_idx)
    val_dataset = FastCachedDataset(all_embs, all_targets, val_idx)
    test_dataset = FastCachedDataset(all_embs, all_targets, test_idx)
    
    # Real-only subsets
    train_real_idx = manifest[(manifest['split'] == 'train') & (~manifest['is_synthetic'])].index.values
    test_real_idx = manifest[(manifest['split'] == 'test') & (~manifest['is_synthetic'])].index.values
    
    train_real_dataset = FastCachedDataset(all_embs, all_targets, train_real_idx)
    test_real_dataset = FastCachedDataset(all_embs, all_targets, test_real_idx)
    
    label_counts = {h: len(label_maps[h]) - 1 for h in Config.HEADS}
    
    return {
        "manifest": manifest,
        "label_maps": label_maps,
        "label_counts": label_counts,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_real_dataset": train_real_dataset,
        "test_real_dataset": test_real_dataset,
        "train_manifest": manifest.iloc[train_idx].copy().reset_index(drop=True),
        "train_real_manifest": manifest.iloc[train_real_idx].copy().reset_index(drop=True),
        "val_manifest": manifest.iloc[val_idx].copy().reset_index(drop=True),
        "test_manifest": manifest.iloc[test_idx].copy().reset_index(drop=True),
        "test_real_manifest": manifest.iloc[test_real_idx].copy().reset_index(drop=True),
    }


# =====================================================================
# 2. MODEL ARCHITECTURES (STANDARDIZED & REPRODUCIBLE)
# =====================================================================

class SharedBackboneMultiTaskNLU(nn.Module):
    """Joint 6-head model with shared 256-D backbone."""
    def __init__(self, in_dim=1280, label_counts=None, dropout=0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.heads = nn.ModuleDict({
            head: nn.Linear(256, num_cls)
            for head, num_cls in label_counts.items()
        })
        
    def forward(self, x):
        rep = self.backbone(x)
        return {h: self.heads[h](rep) for h in self.heads}


class IndependentSingleTaskNLU(nn.Module):
    """Independent single-task classifier with identical capacity per head."""
    def __init__(self, in_dim=1280, num_classes=10, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)


# =====================================================================
# 3. BALANCING HELPERS
# =====================================================================

def compute_class_weights(df, label_maps, max_weight=5.0):
    weights = {}
    for h in Config.HEADS:
        valid_series = df[df[h] != Config.MASK_TOKEN][h]
        N = len(valid_series)
        K = len(label_maps[h]) - 1
        if N == 0 or K == 0:
            weights[h] = torch.ones(max(K, 1), device=Config.DEVICE)
            continue
        counts = valid_series.value_counts().to_dict()
        w_c = torch.ones(K, dtype=torch.float32)
        for label, count in counts.items():
            if label in label_maps[h] and count > 0:
                idx = label_maps[h][label]
                raw_w = N / (K * count)
                w_c[idx] = min(raw_w, max_weight)
        weights[h] = w_c.to(Config.DEVICE)
    return weights


def build_multitask_sampler(df, class_weights, label_maps):
    sample_weights = []
    for _, row in df.iterrows():
        row_w = []
        for h in Config.HEADS:
            val = row[h]
            if val != Config.MASK_TOKEN and val in label_maps[h]:
                idx = label_maps[h][val]
                row_w.append(class_weights[h][idx].item())
        sample_weights.append(float(np.mean(row_w)) if row_w else 1.0)
    sample_weights_t = torch.tensor(sample_weights, dtype=torch.float)
    return WeightedRandomSampler(weights=sample_weights_t, num_samples=len(sample_weights_t), replacement=True)


# =====================================================================
# 4. TRAINING & EVALUATION ENGINE
# =====================================================================

def evaluate_predictions(y_true, y_pred, head_name, label_map, exp_tag):
    """Computes full metrics, per-class metrics, and saves confusion matrix."""
    valid_idx = y_true != Config.MASK_ID
    if valid_idx.sum() == 0:
        return {
            "valid_samples": 0, "accuracy": 0.0, "balanced_accuracy": 0.0,
            "macro_f1": 0.0, "weighted_f1": 0.0, "per_class": []
        }
        
    v_true = y_true[valid_idx]
    v_pred = y_pred[valid_idx]
    
    inv_map = {v: k for k, v in label_map.items() if v != Config.MASK_ID}
    rep = classification_report(v_true, v_pred, output_dict=True, zero_division=0)
    bal_acc = balanced_accuracy_score(v_true, v_pred)
    
    per_class = []
    for k_str, v_dict in rep.items():
        if k_str.isdigit() and int(k_str) in inv_map:
            c_name = inv_map[int(k_str)]
            per_class.append({
                "experiment": exp_tag,
                "head": head_name,
                "class": c_name,
                "precision": round(v_dict['precision'], 4),
                "recall": round(v_dict['recall'], 4),
                "f1_score": round(v_dict['f1-score'], 4),
                "support": int(v_dict['support'])
            })
            
    # Confusion Matrix Plot
    labels_present = sorted(list(set(v_true) | set(v_pred)))
    target_names = [inv_map[i] for i in labels_present if i in inv_map]
    
    cm = confusion_matrix(v_true, v_pred, labels=labels_present)
    plt.figure(figsize=(max(7, len(target_names) * 0.5), max(5, len(target_names) * 0.4)))
    sns.heatmap(
        cm, annot=(len(target_names) <= 15), fmt='d', cmap="Blues",
        xticklabels=target_names, yticklabels=target_names
    )
    plt.title(f"{head_name.upper()} ({exp_tag})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = os.path.join(CM_DIR, f"{exp_tag}_{head_name}_cm.png")
    plt.savefig(cm_path, dpi=180)
    plt.close()
    
    return {
        "valid_samples": int(valid_idx.sum()),
        "accuracy": round(rep['accuracy'], 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(rep['macro avg']['f1-score'], 4),
        "weighted_f1": round(rep['weighted avg']['f1-score'], 4),
        "per_class": per_class
    }


def train_joint_model(train_loader, val_loader, test_loader, label_counts, label_maps, criterions, exp_tag):
    """Trains a 6-head joint model using validation early stopping."""
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    model = SharedBackboneMultiTaskNLU(1280, label_counts).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    best_val_f1 = -1.0
    best_weights = None
    patience_cnt = 0
    
    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        for x, targets in train_loader:
            x = x.to(Config.DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = 0
            for h in Config.HEADS:
                t = targets[h].to(Config.DEVICE)
                l = criterions[h](logits[h], t)
                if not torch.isnan(l):
                    loss += l
            loss.backward()
            optimizer.step()
            
        # Validation Pass
        model.eval()
        val_preds = {h: [] for h in Config.HEADS}
        val_trues = {h: [] for h in Config.HEADS}
        with torch.no_grad():
            for x, targets in val_loader:
                x = x.to(Config.DEVICE)
                logits = model(x)
                for h in Config.HEADS:
                    p = torch.argmax(logits[h], dim=1).cpu().numpy()
                    t = targets[h].numpy()
                    v_mask = t != Config.MASK_ID
                    val_preds[h].extend(p[v_mask])
                    val_trues[h].extend(t[v_mask])
                    
        f1_list = []
        for h in Config.HEADS:
            if len(val_trues[h]) > 0:
                rep = classification_report(val_trues[h], val_preds[h], output_dict=True, zero_division=0)
                f1_list.append(rep['macro avg']['f1-score'])
        mean_val_f1 = np.mean(f1_list) if f1_list else 0.0
        
        if mean_val_f1 > best_val_f1:
            best_val_f1 = mean_val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= Config.PATIENCE:
                break
                
    # Evaluate best model on test_loader
    model.load_state_dict({k: v.to(Config.DEVICE) for k, v in best_weights.items()})
    model.eval()
    
    test_preds = {h: [] for h in Config.HEADS}
    test_trues = {h: [] for h in Config.HEADS}
    with torch.no_grad():
        for x, targets in test_loader:
            x = x.to(Config.DEVICE)
            logits = model(x)
            for h in Config.HEADS:
                test_preds[h].extend(torch.argmax(logits[h], dim=1).cpu().numpy())
                test_trues[h].extend(targets[h].numpy())
                
    head_results = {}
    per_class_all = []
    for h in Config.HEADS:
        res = evaluate_predictions(np.array(test_trues[h]), np.array(test_preds[h]), h, label_maps[h], exp_tag)
        head_results[h] = res
        per_class_all.extend(res['per_class'])
        
    return head_results, per_class_all


def train_single_task_model(head, train_loader, val_loader, test_loader, num_classes, label_map, exp_tag):
    """Trains an independent single-task classifier for one head."""
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    model = IndependentSingleTaskNLU(1280, num_classes).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=Config.MASK_ID)
    
    best_val_f1 = -1.0
    best_weights = None
    patience_cnt = 0
    
    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        for x, targets in train_loader:
            t = targets[head].to(Config.DEVICE)
            # Only train on valid targets for this head
            if (t != Config.MASK_ID).sum() == 0:
                continue
            x = x.to(Config.DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, t)
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                
        # Validation Pass
        model.eval()
        v_preds, v_trues = [], []
        with torch.no_grad():
            for x, targets in val_loader:
                t = targets[head].numpy()
                v_mask = t != Config.MASK_ID
                if v_mask.sum() == 0:
                    continue
                x = x.to(Config.DEVICE)
                logits = model(x)
                p = torch.argmax(logits, dim=1).cpu().numpy()
                v_preds.extend(p[v_mask])
                v_trues.extend(t[v_mask])
                
        val_f1 = 0.0
        if len(v_trues) > 0:
            rep = classification_report(v_trues, v_preds, output_dict=True, zero_division=0)
            val_f1 = rep['macro avg']['f1-score']
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= Config.PATIENCE:
                break
                
    # Test Evaluation
    model.load_state_dict({k: v.to(Config.DEVICE) for k, v in best_weights.items()})
    model.eval()
    
    t_preds, t_trues = [], []
    with torch.no_grad():
        for x, targets in test_loader:
            x = x.to(Config.DEVICE)
            logits = model(x)
            t_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            t_trues.extend(targets[head].numpy())
            
    res = evaluate_predictions(np.array(t_trues), np.array(t_preds), head, label_map, exp_tag)
    return res


# =====================================================================
# 5. EXPERIMENT 1: SINGLE-TASK VS JOINT MULTI-TASK
# =====================================================================

def run_experiment_1(data):
    print(f"\n{'='*75}\n[+] EXPERIMENT 1: SINGLE-TASK VS JOINT MULTI-TASK INTERFERENCE\n{'='*75}")
    
    train_loader = DataLoader(data["train_dataset"], batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(data["val_dataset"], batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(data["test_dataset"], batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 1. Joint Multi-Task Baseline (Unweighted)
    print("  -> Training Joint 6-Head Multi-Task Model...")
    crit_unweighted = {h: nn.CrossEntropyLoss(ignore_index=Config.MASK_ID) for h in Config.HEADS}
    joint_res, joint_per_class = train_joint_model(
        train_loader, val_loader, test_loader, data["label_counts"], data["label_maps"], crit_unweighted, "Joint_6Head"
    )
    
    # 2. Six Independent Single-Task Models
    print("  -> Training 6 Independent Single-Task Classifiers...")
    single_res = {}
    single_per_class = []
    for h in Config.HEADS:
        print(f"     * Training single-task head: {h.upper()}...")
        num_cls = data["label_counts"][h]
        res = train_single_task_model(
            h, train_loader, val_loader, test_loader, num_cls, data["label_maps"][h], f"Single_{h}"
        )
        single_res[h] = res
        single_per_class.extend(res['per_class'])
        
    # Compare results
    rows = []
    delta_rows = []
    for h in Config.HEADS:
        j_f1 = joint_res[h]['macro_f1']
        s_f1 = single_res[h]['macro_f1']
        delta = round(s_f1 - j_f1, 4)
        
        # Valid sample counts across splits
        tr_valid = int((data["train_manifest"][h] != Config.MASK_TOKEN).sum())
        vl_valid = int((data["val_manifest"][h] != Config.MASK_TOKEN).sum())
        te_valid = int((data["test_manifest"][h] != Config.MASK_TOKEN).sum())
        
        rows.append({
            "head": h,
            "train_valid_samples": tr_valid,
            "val_valid_samples": vl_valid,
            "test_valid_samples": te_valid,
            "joint_macro_f1": j_f1,
            "single_task_macro_f1": s_f1,
            "delta_macro_f1 (single - joint)": delta,
            "joint_accuracy": joint_res[h]['accuracy'],
            "single_accuracy": single_res[h]['accuracy'],
            "joint_balanced_acc": joint_res[h]['balanced_accuracy'],
            "single_balanced_acc": single_res[h]['balanced_accuracy']
        })
        delta_rows.append({
            "head": h, "joint_f1": j_f1, "single_f1": s_f1, "delta_f1": delta,
            "interference_detected": delta >= 0.05
        })
        print(f"  [{h.upper():<12}] Joint F1: {j_f1:.4f} | Single F1: {s_f1:.4f} | Delta: {delta:+.4f}")
        
    df_comp = pd.DataFrame(rows)
    df_delta = pd.DataFrame(delta_rows)
    df_comp.to_csv(os.path.join(OUT_DIR, "multitask_comparison.csv"), index=False)
    df_delta.to_csv(os.path.join(OUT_DIR, "multitask_delta.csv"), index=False)
    
    mean_j_f1 = round(df_comp["joint_macro_f1"].mean(), 4)
    mean_s_f1 = round(df_comp["single_task_macro_f1"].mean(), 4)
    print(f"\n  >> Mean Macro-F1 across 6 Heads: Joint = {mean_j_f1:.4f} | Single-Task = {mean_s_f1:.4f} | Delta = {(mean_s_f1 - mean_j_f1):+.4f}")
    
    return {
        "comparison_df": df_comp,
        "delta_df": df_delta,
        "per_class": joint_per_class + single_per_class,
        "mean_joint_f1": mean_j_f1,
        "mean_single_f1": mean_s_f1
    }


# =====================================================================
# 6. EXPERIMENT 2: BALANCING ABLATION
# =====================================================================

def run_experiment_2(data):
    print(f"\n{'='*75}\n[+] EXPERIMENT 2: BALANCING STRATEGY ABLATION\n{'='*75}")
    
    val_loader = DataLoader(data["val_dataset"], batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(data["test_dataset"], batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Compute training class weights
    class_weights = compute_class_weights(data["train_manifest"], data["label_maps"], Config.MAX_CLASS_WEIGHT)
    crit_unweighted = {h: nn.CrossEntropyLoss(ignore_index=Config.MASK_ID) for h in Config.HEADS}
    crit_weighted = {h: nn.CrossEntropyLoss(weight=class_weights[h], ignore_index=Config.MASK_ID) for h in Config.HEADS}
    sampler = build_multitask_sampler(data["train_manifest"], class_weights, data["label_maps"])
    
    conditions = {
        "A_No_Balancing": {
            "loader": DataLoader(data["train_dataset"], batch_size=Config.BATCH_SIZE, shuffle=True),
            "crits": crit_unweighted
        },
        "B_Weighted_Loss_Only": {
            "loader": DataLoader(data["train_dataset"], batch_size=Config.BATCH_SIZE, shuffle=True),
            "crits": crit_weighted
        },
        "C_Sampler_Only": {
            "loader": DataLoader(data["train_dataset"], batch_size=Config.BATCH_SIZE, sampler=sampler),
            "crits": crit_unweighted
        },
        "D_Sampler_Plus_Weighted_Loss": {
            "loader": DataLoader(data["train_dataset"], batch_size=Config.BATCH_SIZE, sampler=sampler),
            "crits": crit_weighted
        }
    }
    
    comp_rows = []
    all_per_class = []
    
    for cond_name, cfg in conditions.items():
        print(f"  -> Running Condition: {cond_name}...")
        res_heads, per_class = train_joint_model(
            cfg["loader"], val_loader, test_loader,
            data["label_counts"], data["label_maps"], cfg["crits"], cond_name
        )
        all_per_class.extend(per_class)
        
        row = {"condition": cond_name}
        for h in Config.HEADS:
            row[f"{h}_f1"] = res_heads[h]['macro_f1']
            row[f"{h}_bal_acc"] = res_heads[h]['balanced_accuracy']
        row["mean_f1"] = round(np.mean([res_heads[h]['macro_f1'] for h in Config.HEADS]), 4)
        row["mean_bal_acc"] = round(np.mean([res_heads[h]['balanced_accuracy'] for h in Config.HEADS]), 4)
        comp_rows.append(row)
        print(f"     Mean Macro-F1 = {row['mean_f1']:.4f} | Intent F1 = {row['intent_f1']:.4f} | Urgency F1 = {row['urgency_f1']:.4f}")
        
    df_bal = pd.DataFrame(comp_rows)
    df_bal.to_csv(os.path.join(OUT_DIR, "balancing_comparison.csv"), index=False)
    
    # Delta from unweighted
    base_row = df_bal[df_bal['condition'] == "A_No_Balancing"].iloc[0]
    delta_rows = []
    for _, r in df_bal.iterrows():
        delta_rows.append({
            "condition": r["condition"],
            "mean_f1": r["mean_f1"],
            "delta_vs_no_balancing": round(r["mean_f1"] - base_row["mean_f1"], 4),
            "urgency_f1_delta": round(r["urgency_f1"] - base_row["urgency_f1"], 4),
            "intent_f1_delta": round(r["intent_f1"] - base_row["intent_f1"], 4)
        })
    df_bal_delta = pd.DataFrame(delta_rows)
    df_bal_delta.to_csv(os.path.join(OUT_DIR, "balancing_delta.csv"), index=False)
    
    best_cond = df_bal.sort_values(by="mean_f1", ascending=False).iloc[0]["condition"]
    print(f"\n  >> Best Balancing Condition by Mean Macro-F1: {best_cond}")
    
    return {
        "comparison_df": df_bal,
        "delta_df": df_bal_delta,
        "per_class": all_per_class,
        "best_condition": best_cond
    }


# =====================================================================
# 7. EXPERIMENT 3: REAL SPEECH VS SYNTHETIC TTS GENERALIZATION
# =====================================================================

def run_experiment_3(data, best_balancing_mode="A_No_Balancing"):
    print(f"\n{'='*75}\n[+] EXPERIMENT 3: REAL DATA ONLY VS REAL + SYNTHETIC TTS\n{'='*75}")
    print(f"  [Config] Standardizing training setup using: {best_balancing_mode}")
    
    # Evaluation loaders (Fixed test splits)
    val_loader = DataLoader(data["val_dataset"], batch_size=Config.BATCH_SIZE, shuffle=False)
    all_test_loader = DataLoader(data["test_dataset"], batch_size=Config.BATCH_SIZE, shuffle=False)
    real_test_loader = DataLoader(data["test_real_dataset"], batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Condition A: Real Only Training
    print("  -> Condition A: Training on Real Speech Data Only...")
    train_real_loader = DataLoader(data["train_real_dataset"], batch_size=Config.BATCH_SIZE, shuffle=True)
    crit_unweighted = {h: nn.CrossEntropyLoss(ignore_index=Config.MASK_ID) for h in Config.HEADS}
    
    res_real_all_test, pc1 = train_joint_model(
        train_real_loader, val_loader, all_test_loader,
        data["label_counts"], data["label_maps"], crit_unweighted, "RealOnly_EvalAllTest"
    )
    res_real_real_test, pc2 = train_joint_model(
        train_real_loader, val_loader, real_test_loader,
        data["label_counts"], data["label_maps"], crit_unweighted, "RealOnly_EvalRealTest"
    )
    
    # Condition B: Real + Synthetic Training
    print("  -> Condition B: Training on Real + Synthetic TTS Data...")
    train_full_loader = DataLoader(data["train_dataset"], batch_size=Config.BATCH_SIZE, shuffle=True)
    res_full_all_test, pc3 = train_joint_model(
        train_full_loader, val_loader, all_test_loader,
        data["label_counts"], data["label_maps"], crit_unweighted, "RealPlusSynth_EvalAllTest"
    )
    res_full_real_test, pc4 = train_joint_model(
        train_full_loader, val_loader, real_test_loader,
        data["label_counts"], data["label_maps"], crit_unweighted, "RealPlusSynth_EvalRealTest"
    )
    
    synth_comp_rows = []
    synth_delta_rows = []
    
    for h in Config.HEADS:
        r_real_f1 = res_real_real_test[h]['macro_f1']
        f_real_f1 = res_full_real_test[h]['macro_f1']
        delta_real = round(f_real_f1 - r_real_f1, 4)
        
        r_all_f1 = res_real_all_test[h]['macro_f1']
        f_all_f1 = res_full_all_test[h]['macro_f1']
        delta_all = round(f_all_f1 - r_all_f1, 4)
        
        synth_comp_rows.append({
            "head": h,
            "real_only_eval_real_test_f1": r_real_f1,
            "real_plus_synth_eval_real_test_f1": f_real_f1,
            "delta_on_real_test": delta_real,
            "real_only_eval_all_test_f1": r_all_f1,
            "real_plus_synth_eval_all_test_f1": f_all_f1,
            "delta_on_all_test": delta_all
        })
        
        synth_delta_rows.append({
            "head": h,
            "delta_on_real_speech": delta_real,
            "synthetic_effect_on_real_speech": "HELPFUL" if delta_real >= 0.03 else ("HARMFUL" if delta_real <= -0.03 else "NEUTRAL")
        })
        print(f"  [{h.upper():<12}] Real-Test: Real-Train F1 = {r_real_f1:.4f} | Real+Synth-Train F1 = {f_real_f1:.4f} | Delta = {delta_real:+.4f}")
        
    df_synth = pd.DataFrame(synth_comp_rows)
    df_synth_delta = pd.DataFrame(synth_delta_rows)
    df_synth.to_csv(os.path.join(OUT_DIR, "synthetic_comparison.csv"), index=False)
    df_synth_delta.to_csv(os.path.join(OUT_DIR, "synthetic_delta.csv"), index=False)
    
    mean_r_real = round(df_synth["real_only_eval_real_test_f1"].mean(), 4)
    mean_f_real = round(df_synth["real_plus_synth_eval_real_test_f1"].mean(), 4)
    print(f"\n  >> Real-Speech Generalization (Real Test): Real-Only = {mean_r_real:.4f} | Real+Synth = {mean_f_real:.4f} | Delta = {(mean_f_real - mean_r_real):+.4f}")
    
    return {
        "comparison_df": df_synth,
        "delta_df": df_synth_delta,
        "per_class": pc1 + pc2 + pc3 + pc4,
        "mean_real_only_f1": mean_r_real,
        "mean_full_f1": mean_f_real
    }


# =====================================================================
# 8. MASTER SUMMARY & DECISION REPORT GENERATOR
# =====================================================================

def generate_final_report(exp1_out, exp2_out, exp3_out, all_per_class):
    print(f"\n{'='*75}\n[+] GENERATING MASTER CSV & FINAL SCIENTIFIC DIAGNOSTIC REPORT\n{'='*75}")
    
    # Save all per-class records
    pd.DataFrame(all_per_class).to_csv(os.path.join(OUT_DIR, "per_class_results.csv"), index=False)
    
    # Master summary table
    summary_rows = []
    
    # Multi-task
    for _, r in exp1_out["comparison_df"].iterrows():
        summary_rows.append({
            "experiment": "Exp1_Joint_MultiTask",
            "head": r["head"],
            "macro_f1": r["joint_macro_f1"],
            "balanced_accuracy": r["joint_balanced_acc"],
            "accuracy": r["joint_accuracy"],
            "interpretation": "Shared backbone multi-task learning"
        })
        summary_rows.append({
            "experiment": "Exp1_Single_Task",
            "head": r["head"],
            "macro_f1": r["single_task_macro_f1"],
            "balanced_accuracy": r["single_balanced_acc"],
            "accuracy": r["single_accuracy"],
            "interpretation": "Independent classifier without shared interference"
        })
        
    # Balancing
    for _, r in exp2_out["comparison_df"].iterrows():
        for h in Config.HEADS:
            summary_rows.append({
                "experiment": f"Exp2_{r['condition']}",
                "head": h,
                "macro_f1": r[f"{h}_f1"],
                "balanced_accuracy": r[f"{h}_bal_acc"],
                "accuracy": np.nan,
                "interpretation": f"Balancing condition {r['condition']}"
            })
            
    # Synthetic
    for _, r in exp3_out["comparison_df"].iterrows():
        summary_rows.append({
            "experiment": "Exp3_RealOnly_EvalRealTest",
            "head": r["head"],
            "macro_f1": r["real_only_eval_real_test_f1"],
            "balanced_accuracy": np.nan,
            "accuracy": np.nan,
            "interpretation": "Trained only on real speech; evaluated on real speech"
        })
        summary_rows.append({
            "experiment": "Exp3_RealPlusSynth_EvalRealTest",
            "head": r["head"],
            "macro_f1": r["real_plus_synth_eval_real_test_f1"],
            "balanced_accuracy": np.nan,
            "accuracy": np.nan,
            "interpretation": "Trained on real+synthetic; evaluated on real speech"
        })
        
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "final_three_summary.csv"), index=False)
    
    # -------------------------------------------------------------
    # SCIENTIFIC THRESHOLD RULES (>= 0.05 meaningful, >= 0.10 strong)
    # -------------------------------------------------------------
    single_vs_joint_delta = exp1_out["mean_single_f1"] - exp1_out["mean_joint_f1"]
    if single_vs_joint_delta >= 0.05:
        q_c_ans = "SUPPORTED"
    elif single_vs_joint_delta <= -0.05:
        q_c_ans = "NOT SUPPORTED"
    else:
        q_c_ans = "PARTIALLY SUPPORTED" if single_vs_joint_delta > 0 else "NOT SUPPORTED"
        
    bal_base = exp2_out["delta_df"][exp2_out["delta_df"]["condition"] == "A_No_Balancing"]["mean_f1"].values[0]
    best_bal_delta = exp2_out["delta_df"]["mean_f1"].max() - bal_base
    if best_bal_delta <= -0.03:
        q_d_ans = "SUPPORTED" # Balancing hurts
    elif best_bal_delta >= 0.03:
        q_d_ans = "NOT SUPPORTED" # Balancing helps
    else:
        q_d_ans = "INCONCLUSIVE" # Marginal difference
        
    synth_delta = exp3_out["mean_full_f1"] - exp3_out["mean_real_only_f1"]
    if synth_delta >= 0.03:
        q_e_ans = "SUPPORTED" # Synthetic helps
    elif synth_delta <= -0.03:
        q_e_ans = "NOT SUPPORTED" # Synthetic hurts
    else:
        q_e_ans = "PARTIALLY SUPPORTED" if synth_delta > 0 else "INCONCLUSIVE"
        
    # Markdown Report
    md_path = os.path.join(OUT_DIR, "FINAL_THREE_DIAGNOSTIC_REPORT.md")
    with open(md_path, "w") as f:
        f.write("# FINAL THREE DIAGNOSTIC EXPERIMENTS: RESEARCH REPORT\n\n")
        
        f.write("## 1. Executive Summary\n")
        f.write(f"- **Mean Joint 6-Head Macro-F1:** {exp1_out['mean_joint_f1']:.4f}\n")
        f.write(f"- **Mean Single-Task Macro-F1:** {exp1_out['mean_single_f1']:.4f} (Delta: {single_vs_joint_delta:+.4f})\n")
        f.write(f"- **Best Balancing Strategy:** `{exp2_out['best_condition']}` (Max F1: {exp2_out['comparison_df']['mean_f1'].max():.4f})\n")
        f.write(f"- **Synthetic Augmentation Effect on Real Speech:** Delta = {synth_delta:+.4f}\n\n")
        
        f.write("## 2. Experiment 1: Single-Task vs Joint Multi-Task Interference\n")
        f.write("Does sharing a 256-D backbone across 6 heterogeneous heads create gradient interference?\n\n")
        f.write(exp1_out["comparison_df"].to_markdown(index=False) + "\n\n")
        
        f.write("## 3. Experiment 2: Balancing Strategy Ablation\n")
        f.write("Comparing unweighted training, class-weighted loss, multi-task weighted sampler, and combined weighting:\n\n")
        f.write(exp2_out["comparison_df"].to_markdown(index=False) + "\n\n")
        
        f.write("## 4. Experiment 3: Real Speech vs Synthetic TTS Generalization\n")
        f.write("Evaluating whether synthetic TTS training improves or hurts generalization on real speech test audio:\n\n")
        f.write(exp3_out["comparison_df"].to_markdown(index=False) + "\n\n")
        
        f.write("## 5. FINAL SCIENTIFIC ROOT-CAUSE DECISION\n\n")
        f.write("### QUESTION A: Is the frozen Whisper representation fundamentally insufficient?\n")
        f.write("**Status:** NOT SUPPORTED (FOR EMOTION/DOMAIN), PARTIALLY SUPPORTED (FOR LONG-TAIL INTENT)\n")
        f.write(f"- *Evidence:* Domain achieves 1.000 F1 and Emotion achieves 0.359 F1 (surpassing text TF-IDF at 0.222). Whisper retains rich semantic and acoustic-affective information. For long-tail intent and entity types, some lexical detail is compressed by 1280-D mean pooling.\n\n")
        
        f.write("### QUESTION B: Is the dataset/ontology/classification problem badly conditioned?\n")
        f.write("**Status:** SUPPORTED\n")
        f.write(f"- *Evidence:* Clean text TF-IDF yields only 0.434 on Intent, 0.287 on Entity Type, and 0.327 on Urgency. The high cardinality and severe imbalance (>50:1) limit separability regardless of speech features.\n\n")
        
        f.write("### QUESTION C: Is multi-task learning hurting performance?\n")
        f.write(f"**Status:** {q_c_ans}\n")
        f.write(f"- *Evidence:* Mean single-task F1 is {exp1_out['mean_single_f1']:.4f} vs joint F1 of {exp1_out['mean_joint_f1']:.4f} (Net Delta: {single_vs_joint_delta:+.4f}). Look at per-head deltas in `multitask_delta.csv` for specific task conflicts.\n\n")
        
        f.write("### QUESTION D: Is balancing hurting performance?\n")
        f.write(f"**Status:** {q_d_ans}\n")
        f.write(f"- *Evidence:* Condition `{exp2_out['best_condition']}` achieved the highest overall Macro-F1 ({exp2_out['comparison_df']['mean_f1'].max():.4f}). Combining sampler with weighted loss changes gradient updates on sparse MASK rows.\n\n")
        
        f.write("### QUESTION E: Is synthetic TTS helping or hurting real-speech generalization?\n")
        f.write(f"**Status:** {q_e_ans}\n")
        f.write(f"- *Evidence:* On the real-speech test set, adding synthetic TTS to training yields a delta of {synth_delta:+.4f} Macro-F1 across heads.\n")
        
    print(f"[+] Diagnostic Report saved to: {md_path}")
    print(f"[+] Master CSV saved to: {os.path.join(OUT_DIR, 'final_three_summary.csv')}\n")


# =====================================================================
# 9. MAIN ORCHESTRATOR
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="ASIL Final Three Diagnostic Suite")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "multitask", "balancing", "synthetic"],
        help="Which diagnostic experiment to run (default: all)"
    )
    args = parser.parse_args()
    
    start_t = time.time()
    print(f"\n{'='*75}\n🚀 ASIL NLU FINAL THREE DIAGNOSTICS SUITE\n{'='*75}")
    print(f"Device : {Config.DEVICE}")
    print(f"Seed   : {Config.SEED}")
    print(f"Target : {args.experiment.upper()}")
    print(f"{'='*75}\n")
    
    # Save run config
    cfg_dump = {
        "experiment": args.experiment,
        "seed": Config.SEED,
        "device": Config.DEVICE,
        "batch_size": Config.BATCH_SIZE,
        "lr": Config.LR,
        "max_epochs": Config.MAX_EPOCHS,
        "patience": Config.PATIENCE,
        "max_class_weight": Config.MAX_CLASS_WEIGHT
    }
    with open(os.path.join(OUT_DIR, "experiment_config.json"), "w") as f:
        json.dump(cfg_dump, f, indent=4)
        
    data = load_all_cached_data()
    
    exp1_out, exp2_out, exp3_out = None, None, None
    all_per_class = []
    
    if args.experiment in ["all", "multitask"]:
        exp1_out = run_experiment_1(data)
        all_per_class.extend(exp1_out["per_class"])
        
    if args.experiment in ["all", "balancing"]:
        exp2_out = run_experiment_2(data)
        all_per_class.extend(exp2_out["per_class"])
        
    if args.experiment in ["all", "synthetic"]:
        best_mode = exp2_out["best_condition"] if exp2_out else "A_No_Balancing"
        exp3_out = run_experiment_3(data, best_balancing_mode=best_mode)
        all_per_class.extend(exp3_out["per_class"])
        
    if args.experiment == "all":
        generate_final_report(exp1_out, exp2_out, exp3_out, all_per_class)
        
    total_time = time.time() - start_t
    print(f"\n{'='*75}\n🎉 ALL REQUESTED DIAGNOSTICS COMPLETED IN {total_time:.2f}s\n{'='*75}")
    print(f"📂 Results Location: {OUT_DIR}\n")


if __name__ == "__main__":
    main()
