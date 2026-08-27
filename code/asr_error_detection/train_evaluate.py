"""
Model Training, Cross-Entropy Loss Balancing, Evaluation & Ablation Orchestrator.
Supports:
- Dynamic Multi-Task Weighted Sampling
- Mask-Aware Class-Weighted Loss
- Ablation Execution across modes: mean, attention, mean_acoustic, attention_acoustic
- Latency, Parameter Reporting & Metric Diagnostics
"""
import argparse
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from config import Config
from model import ASILNLU, get_label_counts


class NLUDataset(Dataset):

    def __init__(self, manifest_df: pd.DataFrame, mode: str = "mean"):
        self.df = manifest_df
        self.mode = mode
        self.use_attention = "attention" in mode
        self.use_acoustic = "acoustic" in mode
        self.label_maps = {
            h: json.load(
                open(
                    os.path.join(Config.ROOT_DIR, "results", "label_maps", f"{h}.json")
                )
            )
            for h in Config.HEADS
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        w_pool = "attention_pool" if self.use_attention else "mean_pool"
        w_path = os.path.join(Config.ROOT_DIR, "embeddings", w_pool, f"{row['sample_id']}.npz")

        whisper_emb = torch.from_numpy(np.load(w_path)["embedding"]).float()

        acoustic_emb = torch.zeros(Config.ACOUSTIC_DIM).float()
        if self.use_acoustic and pd.notna(row.get("acoustic_path")):
            a_path = os.path.join(Config.ROOT_DIR, row["acoustic_path"])
            acoustic_emb = torch.from_numpy(np.load(a_path)["embedding"]).float()

        labels = {
            head: self.label_maps[head].get(row[head], Config.MASK_ID)
            for head in Config.HEADS
        }
        return whisper_emb, acoustic_emb, labels


def compute_class_weights(train_df: pd.DataFrame, label_maps: dict) -> dict:
    weights = {}
    for head in Config.HEADS:
        valid_series = train_df[train_df[head] != Config.MASK_TOKEN][head]
        N = len(valid_series)
        K = len(label_maps[head]) - 1
        if N == 0 or K == 0:
            weights[head] = torch.ones(K).to(Config.DEVICE)
            continue
        counts = valid_series.value_counts().to_dict()
        w_c = torch.ones(K)
        for label, count in counts.items():
            if label in label_maps[head] and count > 0:
                idx = label_maps[head][label]
                raw_weight = N / (K * count)
                w_c[idx] = min(raw_weight, Config.MAX_CLASS_WEIGHT)
        weights[head] = w_c.to(Config.DEVICE)
    return weights


def build_sampler(train_df: pd.DataFrame, class_weights: dict, label_maps: dict):
    sample_weights = []
    for _, row in train_df.iterrows():
        row_w = []
        for head in Config.HEADS:
            val = row[head]
            if val != Config.MASK_TOKEN and val in label_maps[head]:
                idx = label_maps[head][val]
                row_w.append(class_weights[head][idx].item())
        sample_weights.append(np.mean(row_w) if row_w else 1.0)

    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


def train_and_eval(mode: str = "mean", use_balanced: bool = False):
    manifest = pd.read_csv(
        os.path.join(Config.ROOT_DIR, "results", "embedding_manifest.csv")
    )
    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest[manifest["split"] == "validation"].reset_index(drop=True)
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)

    label_maps = {
        h: json.load(
            open(os.path.join(Config.ROOT_DIR, "results", "label_maps", f"{h}.json"))
        )
        for h in Config.HEADS
    }

    sampler = None
    criterions = {
        h: nn.CrossEntropyLoss(ignore_index=Config.MASK_ID) for h in Config.HEADS
    }

    if use_balanced:
        class_weights = compute_class_weights(train_df, label_maps)
        sampler = build_sampler(train_df, class_weights, label_maps)
        criterions = {
            h: nn.CrossEntropyLoss(
                weight=class_weights[h], ignore_index=Config.MASK_ID
            )
            for h in Config.HEADS
        }

    train_loader = DataLoader(
        NLUDataset(train_df, mode),
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
    )
    val_loader = DataLoader(
        NLUDataset(val_df, mode), batch_size=Config.BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        NLUDataset(test_df, mode), batch_size=Config.BATCH_SIZE, shuffle=False
    )

    model = ASILNLU(mode=mode, label_counts=get_label_counts()).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    save_dir = os.path.join(Config.ROOT_DIR, "models", mode)
    os.makedirs(save_dir, exist_ok=True)
    best_val_f1 = -1.0
    patience_counter = 0

    print(f"\n=======================================================")
    print(f" TRAINING: Mode = {mode} | Balanced = {use_balanced}")
    print(f"=======================================================")

    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        total_loss = 0.0

        for w_embs, a_embs, labels in train_loader:
            w_embs = w_embs.to(Config.DEVICE)
            a_embs = a_embs.to(Config.DEVICE)

            optimizer.zero_grad()
            logits, _ = model(w_embs, a_embs)

            batch_loss = sum(
                criterions[h](logits[h], labels[h].to(Config.DEVICE))
                for h in Config.HEADS
            )

            if not torch.isnan(batch_loss) and batch_loss.item() > 0:
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()

        # Validation Step
        model.eval()
        val_preds = {h: [] for h in Config.HEADS}
        val_targets = {h: [] for h in Config.HEADS}

        with torch.no_grad():
            for w_embs, a_embs, labels in val_loader:
                w_embs = w_embs.to(Config.DEVICE)
                a_embs = a_embs.to(Config.DEVICE)
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
                rep = classification_report(
                    val_targets[h], val_preds[h], output_dict=True, zero_division=0
                )
                head_f1s.append(rep["macro avg"]["f1-score"])

        mean_macro_f1 = float(np.mean(head_f1s)) if head_f1s else 0.0
        print(
            f"Epoch [{epoch+1:02d}/{Config.MAX_EPOCHS:02d}] "
            f"Loss: {total_loss:.4f} | Val Mean Macro-F1: {mean_macro_f1:.4f}"
        )

        if mean_macro_f1 > best_val_f1:
            best_val_f1 = mean_macro_f1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"[!] Early stopping triggered at epoch {epoch+1}")
                break

    # --- FINAL TEST EVALUATION ---
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    model.eval()

    test_preds = {h: [] for h in Config.HEADS}
    test_targets = {h: [] for h in Config.HEADS}
    gate_stats = {h: [] for h in Config.HEADS}
    inference_start = time.time()

    with torch.no_grad():
        for w_embs, a_embs, labels in test_loader:
            w_embs = w_embs.to(Config.DEVICE)
            a_embs = a_embs.to(Config.DEVICE)
            logits, gates = model(w_embs, a_embs)

            for h in Config.HEADS:
                preds = torch.argmax(logits[h], dim=1).cpu().numpy()
                targets = labels[h].numpy()
                test_preds[h].extend(preds)
                test_targets[h].extend(targets)
                if gates is not None:
                    gate_stats[h].extend(gates[h].squeeze(-1).cpu().numpy())

    total_samples = len(test_df)
    inf_latency = ((time.time() - inference_start) / total_samples) * 1000
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if gates is not None:
        g_df = pd.DataFrame([
            {
                "head": h,
                "mean_gate": np.mean(gate_stats[h]),
                "std_gate": np.std(gate_stats[h]),
                "min_gate": np.min(gate_stats[h]),
                "max_gate": np.max(gate_stats[h]),
            }
            for h in Config.HEADS
        ])
        g_df.to_csv(
            os.path.join(Config.ROOT_DIR, "results", f"gate_statistics_{mode}.csv"),
            index=False,
        )

    fig_dir = os.path.join(Config.ROOT_DIR, "results", "figures", mode)
    os.makedirs(fig_dir, exist_ok=True)
    inv_label_maps = {
        h: {v: k for k, v in label_maps[h].items()} for h in Config.HEADS
    }
    final_metrics = []

    for head in Config.HEADS:
        tgts = np.array(test_targets[head])
        prds = np.array(test_preds[head])
        valid_idx = tgts != Config.MASK_ID

        if valid_idx.sum() > 0:
            v_tgts, v_prds = tgts[valid_idx], prds[valid_idx]
            report = classification_report(
                v_tgts, v_prds, output_dict=True, zero_division=0
            )

            final_metrics.append({
                "head": head,
                "macro_f1": report["macro avg"]["f1-score"],
                "accuracy": report["accuracy"],
            })

            per_class = [
                {
                    "class": inv_label_maps[head][int(k)],
                    "precision": v["precision"],
                    "recall": v["recall"],
                    "f1": v["f1-score"],
                    "support": v["support"],
                }
                for k, v in report.items()
                if k.isdigit()
            ]
            pd.DataFrame(per_class).to_csv(
                os.path.join(
                    Config.ROOT_DIR, "results", f"per_class_{head}_{mode}.csv"
                ),
                index=False,
            )

            # Confusion Matrix
            cm = confusion_matrix(v_tgts, v_prds)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title(f"{head.capitalize()} Confusion Matrix ({mode})")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"{head}_confusion_matrix.png"))
            plt.close()

    print(f"\n" + "=" * 60)
    print(f" EVALUATION SUMMARY ({mode.upper()})")
    print(f" Parameters: {param_count:,} | Latency: {inf_latency:.2f} ms/sample")
    print("-" * 60)
    for m in final_metrics:
        print(f" Head: {m['head']:<15} | Macro-F1: {m['macro_f1']:.4f} | Accuracy: {m['accuracy']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASIL NLU Multi-Task Heads")
    parser.add_argument(
        "--mode",
        type=str,
        default="mean",
        choices=["mean", "attention", "mean_acoustic", "attention_acoustic"],
        help="Ablation configuration mode.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Enable multi-task weighted sampler and class-weighted loss.",
    )
    args = parser.parse_args()

    train_and_eval(mode=args.mode, use_balanced=args.balanced)
