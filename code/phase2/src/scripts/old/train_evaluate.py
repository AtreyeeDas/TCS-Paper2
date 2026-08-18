import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from model import ASILNLU, get_label_counts

os.makedirs("models/mean_pool",exist_ok=True)
os.makedirs("models/attention_pool",exist_ok=True)
os.makedirs("results/figures",exist_ok=True)
class NLUDataset(Dataset):
    def __init__(self, manifest_df, mode="mean"):
        self.df = manifest_df
        self.mode = mode
        self.label_maps = {h: json.load(open(os.path.join(Config.ROOT_DIR, "results", "label_maps", f"{h}.json"))) for h in Config.HEADS}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_path = os.path.join(Config.ROOT_DIR, "embeddings", f"{self.mode}_pool", f"{row['sample_id']}.npz")
        
        # Load embedding and convert strictly to PyTorch float32 for model computation
        emb = torch.from_numpy(np.load(npz_path)['embedding']).float()
        
        labels = {}
        for head in Config.HEADS:
            val = row[head]
            labels[head] = self.label_maps[head].get(val, Config.MASK_ID)
            
        return emb, labels

def train_and_eval(mode="mean"):
    manifest = pd.read_csv(os.path.join(Config.ROOT_DIR, "results", "embedding_manifest.csv"))
    train_df = manifest[manifest['split'] == 'train']
    val_df = manifest[manifest['split'] == 'validation']
    test_df = manifest[manifest['split'] == 'test']
    
    train_loader = DataLoader(NLUDataset(train_df, mode), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(NLUDataset(val_df, mode), batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(NLUDataset(test_df, mode), batch_size=Config.BATCH_SIZE)
    
    model = ASILNLU(mode=mode, label_counts=get_label_counts()).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    # Mask-Aware Loss
    criterions = {head: nn.CrossEntropyLoss(ignore_index=Config.MASK_ID) for head in Config.HEADS}
    
    best_val_f1 = 0
    patience_counter = 0
    
    print(f"\n[+] Training {mode.capitalize()} Pooling Model...")
    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        total_loss = 0
        for embs, labels in train_loader:
            embs = embs.to(Config.DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=Config.DTYPE):
                logits = model(embs)
                batch_loss = 0
                for head in Config.HEADS:
                    targets = labels[head].to(Config.DEVICE)
                    # Excludes MASK via ignore_index
                    loss = criterions[head](logits[head], targets)
                    if not torch.isnan(loss):
                        batch_loss += loss
            
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            
        # Validation
        model.eval()
        val_preds, val_targets = {h: [] for h in Config.HEADS}, {h: [] for h in Config.HEADS}
        with torch.no_grad():
            for embs, labels in val_loader:
                embs = embs.to(Config.DEVICE)
                logits = model(embs)
                for head in Config.HEADS:
                    preds = torch.argmax(logits[head], dim=1).cpu().numpy()
                    targets = labels[head].numpy()
                    
                    valid_idx = targets != Config.MASK_ID
                    val_preds[head].extend(preds[valid_idx])
                    val_targets[head].extend(targets[valid_idx])
                    
        # Compute Macro F1 for Model Selection
        head_f1s = []
        for head in Config.HEADS:
            if len(val_targets[head]) > 0:
                rep = classification_report(val_targets[head], val_preds[head], output_dict=True, zero_division=0)
                head_f1s.append(rep['macro avg']['f1-score'])
                
        mean_macro_f1 = np.mean(head_f1s) if head_f1s else 0
        print(f"Epoch {epoch+1}/{Config.MAX_EPOCHS} - Loss: {total_loss:.4f} - Val Mean Macro-F1: {mean_macro_f1:.4f}")
        
        if mean_macro_f1 > best_val_f1:
            best_val_f1 = mean_macro_f1
            torch.save(model.state_dict(), os.path.join(Config.ROOT_DIR, "models", f"{mode}_pool", "best_model.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("[!] Early stopping triggered.")
                break

    # Evaluate on Test
    model.load_state_dict(torch.load(os.path.join(Config.ROOT_DIR, "models", f"{mode}_pool", "best_model.pt")))
    model.eval()
    
    test_preds, test_targets = {h: [] for h in Config.HEADS}, {h: [] for h in Config.HEADS}
    inference_start = time.time()
    
    with torch.no_grad():
        for embs, labels in test_loader:
            embs = embs.to(Config.DEVICE)
            logits = model(embs)
            for head in Config.HEADS:
                preds = torch.argmax(logits[head], dim=1).cpu().numpy()
                targets = labels[head].numpy()
                test_preds[head].extend(preds)
                test_targets[head].extend(targets)
                
    inf_latency = (time.time() - inference_start) / len(test_df)
    
    # Generate Metrics and Reports
    final_metrics = []
    inv_label_maps = {h: {v: k for k, v in json.load(open(os.path.join(Config.ROOT_DIR, "results", "label_maps", f"{h}.json"))).items()} for h in Config.HEADS}
    
    for head in Config.HEADS:
        tgts = np.array(test_targets[head])
        prds = np.array(test_preds[head])
        valid_idx = tgts != Config.MASK_ID
        
        if valid_idx.sum() > 0:
            v_tgts = tgts[valid_idx]
            v_prds = prds[valid_idx]
            
            report = classification_report(v_tgts, v_prds, output_dict=True, zero_division=0)
            final_metrics.append({
                "head": head,
                "valid_test_samples": int(valid_idx.sum()),
                "num_classes": len(np.unique(v_tgts)),
                "accuracy": report['accuracy'],
                "macro_precision": report['macro avg']['precision'],
                "macro_recall": report['macro avg']['recall'],
                "macro_f1": report['macro avg']['f1-score']
            })
            
            # Confusion Matrix
            cm = confusion_matrix(v_tgts, v_prds)
            plt.figure(figsize=(10,8))
            sns.heatmap(cm, annot=False, cmap="Blues")
            plt.title(f"{head.capitalize()} Confusion Matrix")
            plt.savefig(os.path.join(Config.ROOT_DIR, "results", "figures", f"{head}_confusion_matrix.png"))
            plt.close()
            
    pd.DataFrame(final_metrics).to_csv(os.path.join(Config.ROOT_DIR, "results", f"final_metrics_{mode}.csv"), index=False)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== TERMINAL SUMMARY ({mode.upper()} POOLING) ===")
    print(f"Model Parameters: {param_count:,}")
    print(f"Inference Latency: {inf_latency*1000:.2f} ms/sample")
    for m in final_metrics:
        print(f"[{m['head'].upper()}] Valid Samples: {m['valid_test_samples']} | Macro-F1: {m['macro_f1']:.4f} | Acc: {m['accuracy']:.4f}")

if __name__ == "__main__":
    train_and_eval(mode="mean")
    train_and_eval(mode="attention")
