import os
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from model import ASILNLU, get_label_counts


class NLUDataset(Dataset):
    """
    Multi-task Dataset supporting Whisper (mean/attention) and HuBERT acoustic representations.
    """
    def __init__(self, manifest_df: pd.DataFrame, mode: str = "mean"):
        self.df = manifest_df.reset_index(drop=True)
        self.mode = mode
        self.use_attention = "attention" in mode
        self.use_acoustic = "acoustic" in mode
        
        # Load fitted training label maps
        self.label_maps = {
            h: json.load(open(os.path.join(Config.ROOT_DIR, "results", "label_maps", f"{h}.json")))
            for h in Config.HEADS
        }
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load Whisper Embedding (1280-D mean or [T, 1280] attention sequence)
        w_pool = "attention_pool" if self.use_attention else "mean_pool"
        w_path = os.path.join(Config.ROOT_DIR, "embeddings", w_pool, f"{row['sample_id']}.npz")
        if not os.path.exists(w_path):
            raise FileNotFoundError(f"Whisper embedding file not found: {w_path}")
        whisper_emb = torch.from_numpy(np.load(w_path)['embedding']).float()
        
        # 2. Load HuBERT Acoustic Embedding (768-D) if acoustic mode enabled
        acoustic_emb = torch.zeros(Config.ACOUSTIC_DIM, dtype=torch.float32)
        if self.use_acoustic:
            if 'acoustic_path' not in row or pd.isna(row['acoustic_path']):
                raise ValueError(f"Sample {row['sample_id']} has no valid acoustic_path in manifest.")
            a_path = os.path.join(Config.ROOT_DIR, str(row['acoustic_path']).strip())
            if not os.path.exists(a_path):
                raise FileNotFoundError(f"Acoustic embedding file not found: {a_path}")
            acoustic_emb = torch.from_numpy(np.load(a_path)['embedding']).float()
            
        # 3. Encode Multi-Task Labels (MASK token -> Config.MASK_ID)
        labels = {}
        for head in Config.HEADS:
            raw_val = row[head]
            labels[head] = self.label_maps[head].get(raw_val, Config.MASK_ID)
            
        return whisper_emb, acoustic_emb, labels


def compute_class_weights(train_df: pd.DataFrame, label_maps: dict) -> dict:
    """
    Computes inverse class frequency weights strictly on the training split:
    w_c = N / (K * N_c), capped at Config.MAX_CLASS_WEIGHT.
    """
    weights = {}
    for head in Config.HEADS:
        valid_series = train_df[train_df[head] != Config.MASK_TOKEN][head]
        N = len(valid_series)
        K = len(label_maps[head]) - 1  # Exclude MASK from class count
        
        if N == 0 or K == 0:
            weights[head] = torch.ones(max(K, 1), device=Config.DEVICE)
            continue
            
        counts = valid_series.value_counts().to_dict()
        w_c = torch.ones(K, dtype=torch.float32)
        
        for label, count in counts.items():
            if label in label_maps[head] and count > 0:
                idx = label_maps[head][label]
                raw_weight = N / (K * count)
                w_c[idx] = min(raw_weight, Config.MAX_CLASS_WEIGHT)
                
        weights[head] = w_c.to(Config.DEVICE)
    return weights


def build_sampler(train_df: pd.DataFrame, class_weights: dict, label_maps: dict) -> WeightedRandomSampler:
    """
    Multi-task weighted sampler: computes mean minority weight across all valid heads per row.
    """
    sample_weights = []
    for _, row in train_df.iterrows():
        row_w = []
        for head in Config.HEADS:
            val = row[head]
            if val != Config.MASK_TOKEN and val in label_maps[head]:
                idx = label_maps[head][val]
                row_w.append(class_weights[head][idx].item())
        # If all heads are MASK for a row, assign minimum base weight 1.0
        sample_weights.append(float(np.mean(row_w)) if row_w else 1.0)
    
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)
    return WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )


def train_and_eval(mode: str = "mean", use_balanced: bool = False):
    # Unique experiment namespace
    exp_name = f"{mode}_balanced" if use_balanced else mode
    
    # -------------------------------------------------------------
    # Isolated Directory Setup
    # -------------------------------------------------------------
    exp_results_dir = os.path.join(Config.ROOT_DIR, "results", exp_name)
    exp_figures_dir = os.path.join(exp_results_dir, "figures")
    exp_models_dir = os.path.join(Config.ROOT_DIR, "models", exp_name)
    
    os.makedirs(exp_results_dir, exist_ok=True)
    os.makedirs(exp_figures_dir, exist_ok=True)
    os.makedirs(exp_models_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🚀 ASIL NLU EXPERIMENT: {exp_name.upper()}")
    print(f"📂 Results Output Directory: {exp_results_dir}")
    print(f"💾 Checkpoint Directory:    {exp_models_dir}")
    print(f"{'='*70}\n")
    
    manifest_path = os.path.join(Config.ROOT_DIR, "results", "embedding_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Embedding manifest not found at {manifest_path}. Run extraction scripts first.")
        
    manifest = pd.read_csv(manifest_path)
    train_df = manifest[manifest['split'] == 'train'].reset_index(drop=True)
    val_df = manifest[manifest['split'] == 'validation'].reset_index(drop=True)
    test_df = manifest[manifest['split'] == 'test'].reset_index(drop=True)
    
    label_maps = {
        h: json.load(open(os.path.join(Config.ROOT_DIR, "results", "label_maps", f"{h}.json")))
        for h in Config.HEADS
    }
    
    # -------------------------------------------------------------
    # Multi-Task Balancing Configuration
    # -------------------------------------------------------------
    sampler = None
    criterions = {h: nn.CrossEntropyLoss(ignore_index=Config.MASK_ID) for h in Config.HEADS}
    
    if use_balanced:
        class_weights = compute_class_weights(train_df, label_maps)
        sampler = build_sampler(train_df, class_weights, label_maps)
        criterions = {
            h: nn.CrossEntropyLoss(weight=class_weights[h], ignore_index=Config.MASK_ID)
            for h in Config.HEADS
        }
        
        # Save class weights to experiment directory
        weights_dump = {
            h: {
                label: float(class_weights[h][idx].item())
                for label, idx in label_maps[h].items()
                if idx != Config.MASK_ID
            }
            for h in Config.HEADS
        }
        with open(os.path.join(exp_results_dir, "class_weights.json"), "w") as f:
            json.dump(weights_dump, f, indent=4)
    
    # DataLoaders: Weighted sampling strictly on Train, sequential on Val & Test
    train_loader = DataLoader(
        NLUDataset(train_df, mode=mode),
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True if Config.DEVICE == "cuda" else False
    )
    val_loader = DataLoader(
        NLUDataset(val_df, mode=mode),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    test_loader = DataLoader(
        NLUDataset(test_df, mode=mode),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # Initialize Model & Optimizer
    model = ASILNLU(mode=mode, label_counts=get_label_counts()).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    best_val_f1 = -1.0
    patience_counter = 0
    training_history = []
    checkpoint_path = os.path.join(exp_models_dir, "best_model.pt")
    
    # -------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------
    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        
        for w_embs, a_embs, labels in train_loader:
            w_embs = w_embs.to(Config.DEVICE)
            a_embs = a_embs.to(Config.DEVICE) if "acoustic" in mode else None
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=Config.DTYPE) if Config.DEVICE == "cuda" else torch.no_grad():
                logits, _ = model(w_embs, a_embs)
                batch_loss = torch.tensor(0.0, device=Config.DEVICE)
                for h in Config.HEADS:
                    target = labels[h].to(Config.DEVICE)
                    loss = criterions[h](logits[h], target)
                    if not torch.isnan(loss):
                        batch_loss += loss
            
            if batch_loss.item() > 0:
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                
        # ---------------------------------------------------------
        # Validation Pass (MASK-aware Macro-F1)
        # ---------------------------------------------------------
        model.eval()
        val_preds = {h: [] for h in Config.HEADS}
        val_targets = {h: [] for h in Config.HEADS}
        
        with torch.no_grad():
            for w_embs, a_embs, labels in val_loader:
                w_embs = w_embs.to(Config.DEVICE)
                a_embs = a_embs.to(Config.DEVICE) if "acoustic" in mode else None
                logits, _ = model(w_embs, a_embs)
                
                for h in Config.HEADS:
                    preds = torch.argmax(logits[h], dim=1).cpu().numpy()
                    targets = labels[h].numpy()
                    valid_idx = targets != Config.MASK_ID
                    val_preds[h].extend(preds[valid_idx])
                    val_targets[h].extend(targets[valid_idx])
                    
        head_f1s = []
        for h in Config.HEADS:
            if len(val_targets[h]) > 0:
                rep = classification_report(val_targets[h], val_preds[h], output_dict=True, zero_division=0)
                head_f1s.append(rep['macro avg']['f1-score'])
                
        mean_macro_f1 = float(np.mean(head_f1s)) if head_f1s else 0.0
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(total_loss, 4),
            "val_mean_macro_f1": round(mean_macro_f1, 4)
        }
        training_history.append(epoch_record)
        
        print(f"Epoch {epoch+1:02d}/{Config.MAX_EPOCHS:02d} | Train Loss: {total_loss:.4f} | Val Mean Macro-F1: {mean_macro_f1:.4f}")
        
        # Early Stopping check
        if mean_macro_f1 > best_val_f1:
            best_val_f1 = mean_macro_f1
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"[!] Early stopping triggered at epoch {epoch+1}.")
                break

    # Save training curves to experiment folder
    pd.DataFrame(training_history).to_csv(os.path.join(exp_results_dir, "training_history.csv"), index=False)

    # -------------------------------------------------------------
    # Test Split Evaluation
    # -------------------------------------------------------------
    print(f"\n[+] Loading best checkpoint from {checkpoint_path} for test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    test_preds = {h: [] for h in Config.HEADS}
    test_targets = {h: [] for h in Config.HEADS}
    gate_stats = {h: [] for h in Config.HEADS}
    
    inference_start = time.time()
    with torch.no_grad():
        for w_embs, a_embs, labels in test_loader:
            w_embs = w_embs.to(Config.DEVICE)
            a_embs = a_embs.to(Config.DEVICE) if "acoustic" in mode else None
            logits, gates = model(w_embs, a_embs)
            
            for h in Config.HEADS:
                preds = torch.argmax(logits[h], dim=1).cpu().numpy()
                targets = labels[h].numpy()
                test_preds[h].extend(preds)
                test_targets[h].extend(targets)
                if gates is not None:
                    gate_stats[h].extend(gates[h].squeeze(-1).cpu().numpy())
                    
    inf_latency_ms = ((time.time() - inference_start) / len(test_df)) * 1000
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # -------------------------------------------------------------
    # Save Gate Statistics & Plots (If Acoustic Branch Active)
    # -------------------------------------------------------------
    if "acoustic" in mode and gate_stats[Config.HEADS[0]]:
        g_records = []
        for h in Config.HEADS:
            vals = np.array(gate_stats[h])
            g_records.append({
                "head": h,
                "mean_gate": float(np.mean(vals)),
                "std_gate": float(np.std(vals)),
                "min_gate": float(np.min(vals)),
                "max_gate": float(np.max(vals))
            })
        g_df = pd.DataFrame(g_records)
        g_df.to_csv(os.path.join(exp_results_dir, "gate_statistics.csv"), index=False)
        
        # Plot learned gate distribution
        plt.figure(figsize=(8, 4))
        sns.barplot(data=g_df, x="head", y="mean_gate", palette="viridis")
        plt.ylim(0, 1.0)
        plt.title(f"Learned Acoustic Gate Weight per Head ({exp_name})")
        plt.ylabel("Acoustic Weight ($g_h$)")
        plt.tight_layout()
        plt.savefig(os.path.join(exp_figures_dir, "gate_weights_per_head.png"), dpi=200)
        plt.close()

    # -------------------------------------------------------------
    # Compute Metrics, Save Per-Class Reports & Confusion Matrices
    # -------------------------------------------------------------
    final_metrics = []
    inv_label_maps = {
        h: {v: k for k, v in label_maps[h].items() if v != Config.MASK_ID}
        for h in Config.HEADS
    }
    
    summary_dict_for_ablation = {
        "experiment": exp_name,
        "mode": mode,
        "balanced": use_balanced,
        "trainable_parameters": param_count,
        "inference_latency_ms": round(inf_latency_ms, 2)
    }
    
    for head in Config.HEADS:
        tgts = np.array(test_targets[head])
        prds = np.array(test_preds[head])
        valid_idx = tgts != Config.MASK_ID
        
        if valid_idx.sum() > 0:
            v_tgts = tgts[valid_idx]
            v_prds = prds[valid_idx]
            
            report = classification_report(v_tgts, v_prds, output_dict=True, zero_division=0)
            bal_acc = balanced_accuracy_score(v_tgts, v_prds)
            
            final_metrics.append({
                "head": head,
                "valid_samples": int(valid_idx.sum()),
                "num_classes": len(np.unique(v_tgts)),
                "accuracy": round(report['accuracy'], 4),
                "balanced_accuracy": round(bal_acc, 4),
                "macro_precision": round(report['macro avg']['precision'], 4),
                "macro_recall": round(report['macro avg']['recall'], 4),
                "macro_f1": round(report['macro avg']['f1-score'], 4),
                "weighted_f1": round(report['weighted avg']['f1-score'], 4)
            })
            
            summary_dict_for_ablation[f"{head}_macro_f1"] = round(report['macro avg']['f1-score'], 4)
            
            # Save per-class metrics
            per_class_data = []
            for k, v in report.items():
                if k.isdigit() and int(k) in inv_label_maps[head]:
                    per_class_data.append({
                        "class": inv_label_maps[head][int(k)],
                        "precision": round(v['precision'], 4),
                        "recall": round(v['recall'], 4),
                        "f1_score": round(v['f1-score'], 4),
                        "support": int(v['support'])
                    })
            pd.DataFrame(per_class_data).to_csv(
                os.path.join(exp_results_dir, f"per_class_{head}.csv"), index=False
            )

            # Generate and save readable Confusion Matrix
            labels_present = sorted(list(set(v_tgts) | set(v_prds)))
            target_names = [inv_label_maps[head][i] for i in labels_present if i in inv_label_maps[head]]
            
            cm = confusion_matrix(v_tgts, v_prds, labels=labels_present)
            plt.figure(figsize=(max(8, len(target_names) * 0.5), max(6, len(target_names) * 0.4)))
            sns.heatmap(
                cm, annot=(len(target_names) <= 15), fmt='d', cmap="Blues",
                xticklabels=target_names, yticklabels=target_names
            )
            plt.title(f"{head.capitalize()} Confusion Matrix ({exp_name})")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(exp_figures_dir, f"{head}_confusion_matrix.png"), dpi=200)
            plt.close()
            
    # Save test metrics table to experiment folder
    metrics_df = pd.DataFrame(final_metrics)
    metrics_df.to_csv(os.path.join(exp_results_dir, "final_metrics.csv"), index=False)
    
    # -------------------------------------------------------------
    # Append to Master Ablation Comparison Table
    # -------------------------------------------------------------
    summary_dict_for_ablation["mean_macro_f1"] = round(metrics_df["macro_f1"].mean(), 4)
    summary_dict_for_ablation["mean_balanced_accuracy"] = round(metrics_df["balanced_accuracy"].mean(), 4)
    
    master_ablation_path = os.path.join(Config.ROOT_DIR, "results", "ablation_comparison.csv")
    master_row_df = pd.DataFrame([summary_dict_for_ablation])
    
    if os.path.exists(master_ablation_path):
        existing_df = pd.read_csv(master_ablation_path)
        existing_df = existing_df[existing_df["experiment"] != exp_name]
        updated_df = pd.concat([existing_df, master_row_df], ignore_index=True)
        updated_df.to_csv(master_ablation_path, index=False)
    else:
        master_row_df.to_csv(master_ablation_path, index=False)

    # -------------------------------------------------------------
    # Terminal Summary
    # -------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY: {exp_name.upper()}")
    print(f"{'='*70}")
    print(f"Trainable Parameters : {param_count:,}")
    print(f"Inference Latency    : {inf_latency_ms:.2f} ms/sample")
    print(f"Mean Macro-F1        : {summary_dict_for_ablation['mean_macro_f1']:.4f}")
    print(f"Mean Balanced Acc    : {summary_dict_for_ablation['mean_balanced_accuracy']:.4f}")
    print("-" * 70)
    for m in final_metrics:
        print(f"[{m['head'].upper():<12}] Valid: {m['valid_samples']:<5} | Macro-F1: {m['macro_f1']:.4f} | Balanced-Acc: {m['balanced_accuracy']:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASIL Multi-Task NLU Training & Evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        default="mean",
        choices=["mean", "attention", "mean_acoustic", "attention_acoustic"],
        help="Model architecture and pooling mode"
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Enable multi-task weighted sampling and class-weighted Cross-Entropy loss"
    )
    args = parser.parse_args()
    
    train_and_eval(mode=args.mode, use_balanced=args.balanced)
