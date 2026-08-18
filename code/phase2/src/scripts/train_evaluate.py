import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from model import ASILNLU

class NLUDataset(Dataset):
    def __init__(self, manifest_df, scaler_dict):
        self.df = manifest_df
        self.label_maps = {h: json.load(open(os.path.join(Config.ROOT_DIR, "results", "label_maps", f"{h}.json"))) for h in Config.HEADS}
        self.ac_mean = np.array(scaler_dict['mean'], dtype=np.float32)
        self.ac_std = np.array(scaler_dict['std'], dtype=np.float32)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['sample_id']
        
        w_emb = torch.from_numpy(np.load(os.path.join(Config.ROOT_DIR, "embeddings", "mean_pool", f"{sample_id}.npz"))['embedding']).float()
        
        ac_feats = np.load(os.path.join(Config.ROOT_DIR, "embeddings", "acoustic", f"{sample_id}.npz"))['features']
        ac_feats = (ac_feats - self.ac_mean) / self.ac_std
        a_emb = torch.from_numpy(ac_feats).float()
        
        labels = {head: self.label_maps[head].get(row[head], Config.MASK_ID) for head in Config.HEADS}
        return w_emb, a_emb, labels

def get_class_weights(df, head, num_classes):
    counts = df[df[head] != Config.MASK_TOKEN][head].value_counts()
    N = counts.sum()
    weights = torch.ones(num_classes)
    label_map = json.load(open(os.path.join(Config.ROOT_DIR, "results", "label_maps", f"{head}.json")))
    for cls, count in counts.items():
        idx = label_map.get(cls, Config.MASK_ID)
        if idx != Config.MASK_ID:
            w = np.sqrt(N / (num_classes * count))
            weights[idx] = min(w, Config.MAX_CLASS_WEIGHT)
    return weights

def train_and_eval():
    res_dir = os.path.join(Config.ROOT_DIR, "results")
    model_dir = os.path.join(Config.ROOT_DIR, "models", "final")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(res_dir, "per_class_metrics"), exist_ok=True)
    os.makedirs(os.path.join(res_dir, "figures"), exist_ok=True)
    
    manifest = pd.read_csv(os.path.join(res_dir, "embedding_manifest.csv"))
    train_df = manifest[manifest['split'] == 'train'].reset_index(drop=True)
    val_df = manifest[manifest['split'] == 'validation'].reset_index(drop=True)
    test_df = manifest[manifest['split'] == 'test'].reset_index(drop=True)
    
    scaler_dict = json.load(open(os.path.join(res_dir, "acoustic_scaler.json")))
    
    label_counts = {h: len(json.load(open(os.path.join(res_dir, "label_maps", f"{h}.json")))) - 1 for h in Config.HEADS}
    
    # 1. Multi-Task Weighted Sampler
    sample_weights = []
    if Config.USE_WEIGHTED_SAMPLER:
        head_weights = {h: get_class_weights(train_df, h, label_counts[h]) for h in Config.HEADS}
        for _, row in train_df.iterrows():
            w_list = []
            for h in Config.HEADS:
                l_idx = json.load(open(os.path.join(res_dir, "label_maps", f"{h}.json"))).get(row[h], Config.MASK_ID)
                if l_idx != Config.MASK_ID:
                    w_list.append(head_weights[h][l_idx].item())
            sample_weights.append(np.mean(w_list) if w_list else 1.0)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(NLUDataset(train_df, scaler_dict), batch_size=Config.BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(NLUDataset(train_df, scaler_dict), batch_size=Config.BATCH_SIZE, shuffle=True)
        
    val_loader = DataLoader(NLUDataset(val_df, scaler_dict), batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(NLUDataset(test_df, scaler_dict), batch_size=Config.BATCH_SIZE)
    
    model = ASILNLU(label_counts).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    # 2. Class-Weighted Loss
    criterions = {}
    for h in Config.HEADS:
        w = get_class_weights(train_df, h, label_counts[h]).to(Config.DEVICE) if Config.USE_CLASS_WEIGHTS else None
        criterions[h] = nn.CrossEntropyLoss(weight=w, ignore_index=Config.MASK_ID)
    
    best_val_f1 = 0
    patience = 0
    history = []
    
    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        total_loss = 0
        
        for w_emb, a_emb, labels in train_loader:
            w_emb, a_emb = w_emb.to(Config.DEVICE), a_emb.to(Config.DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=Config.DTYPE):
                logits, _ = model(w_emb, a_emb)
                batch_loss = 0
                valid_heads = 0
                
                for head in Config.HEADS:
                    targets = labels[head].to(Config.DEVICE)
                    if (targets != Config.MASK_ID).any():
                        loss = criterions[head](logits[head], targets)
                        if not torch.isnan(loss):
                            batch_loss += loss
                            valid_heads += 1
                            
                if valid_heads > 0:
                    batch_loss = batch_loss / valid_heads
                    
            if valid_heads > 0:
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                
        # Validation
        model.eval()
        val_preds, val_targets = {h: [] for h in Config.HEADS}, {h: [] for h in Config.HEADS}
        with torch.no_grad():
            for w_emb, a_emb, labels in val_loader:
                w_emb, a_emb = w_emb.to(Config.DEVICE), a_emb.to(Config.DEVICE)
                logits, _ = model(w_emb, a_emb)
                for head in Config.HEADS:
                    preds = torch.argmax(logits[head], dim=1).cpu().numpy()
                    targets = labels[head].numpy()
                    valid_idx = targets != Config.MASK_ID
                    val_preds[head].extend(preds[valid_idx])
                    val_targets[head].extend(targets[valid_idx])
                    
        head_f1s = [classification_report(val_targets[h], val_preds[h], output_dict=True, zero_division=0)['macro avg']['f1-score'] 
                    for h in Config.HEADS if len(val_targets[h]) > 0]
        mean_macro_f1 = np.mean(head_f1s) if head_f1s else 0
        
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | Val Mean Macro-F1: {mean_macro_f1:.4f}")
        history.append({"epoch": epoch+1, "train_loss": total_loss, "val_mean_macro_f1": mean_macro_f1})
        
        if mean_macro_f1 > best_val_f1:
            best_val_f1 = mean_macro_f1
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            patience = 0
        else:
            patience += 1
            if patience >= Config.PATIENCE:
                print("[!] Early stopping.")
                break
                
    pd.DataFrame(history).to_csv(os.path.join(res_dir, "training_history.csv"), index=False)

    # 3. Final Evaluation
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt")))
    model.eval()
    
    test_preds, test_targets = {h: [] for h in Config.HEADS}, {h: [] for h in Config.HEADS}
    all_gates = []
    
    inf_start = time.time()
    with torch.no_grad():
        for w_emb, a_emb, labels in test_loader:
            w_emb, a_emb = w_emb.to(Config.DEVICE), a_emb.to(Config.DEVICE)
            logits, gates = model(w_emb, a_emb)
            if gates is not None:
                all_gates.extend(gates.mean(dim=1).cpu().numpy())
            
            for head in Config.HEADS:
                preds = torch.argmax(logits[head], dim=1).cpu().numpy()
                targets = labels[head].numpy()
                test_preds[head].extend(preds)
                test_targets[head].extend(targets)
                
    inf_latency = (time.time() - inf_start) / len(test_df)
    
    # Save Gate Analysis
    if all_gates:
        pd.DataFrame({"mean_gate": all_gates}).to_csv(os.path.join(res_dir, "gate_analysis.csv"), index=False)
        
    final_metrics = []
    for head in Config.HEADS:
        tgts = np.array(test_targets[head])
        prds = np.array(test_preds[head])
        valid_idx = tgts != Config.MASK_ID
        
        if valid_idx.sum() > 0:
            v_tgts, v_prds = tgts[valid_idx], prds[valid_idx]
            inv_map = {v: k for k, v in json.load(open(os.path.join(res_dir, "label_maps", f"{head}.json"))).items() if v != Config.MASK_ID}
            target_names = [inv_map[i] for i in sorted(inv_map.keys())]
            
            rep = classification_report(v_tgts, v_prds, target_names=target_names, output_dict=True, zero_division=0)
            
            # Save per-class
            class_recs = [{"class": c, "precision": rep[c]['precision'], "recall": rep[c]['recall'], "f1": rep[c]['f1-score'], "support": rep[c]['support']} for c in target_names]
            pd.DataFrame(class_recs).to_csv(os.path.join(res_dir, "per_class_metrics", f"{head}.csv"), index=False)
            
            final_metrics.append({
                "head": head,
                "valid_test_samples": int(valid_idx.sum()),
                "num_classes": len(np.unique(v_tgts)),
                "accuracy": rep['accuracy'],
                "macro_precision": rep['macro avg']['precision'],
                "macro_recall": rep['macro avg']['recall'],
                "macro_f1": rep['macro avg']['f1-score'],
                "weighted_f1": rep['weighted avg']['f1-score']
            })
            
            cm = confusion_matrix(v_tgts, v_prds)
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=target_names, yticklabels=target_names)
            plt.title(f"{head.capitalize()} Confusion Matrix")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(res_dir, "figures", f"{head}_confusion_matrix.png"))
            plt.close()
            
    pd.DataFrame(final_metrics).to_csv(os.path.join(res_dir, "final_metrics.csv"), index=False)
    
    # Save Latency & Params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with open(os.path.join(res_dir, "latency_report.json"), "w") as f:
        json.dump({"ms_per_sample": inf_latency*1000, "samples_per_second": 1/inf_latency, "trainable_parameters": params}, f, indent=4)
        
    print("\n=== FINAL TEST METRICS ===")
    for m in final_metrics:
        print(f"[{m['head'].upper()}] Valid: {m['valid_test_samples']} | Macro-F1: {m['macro_f1']:.4f} | Acc: {m['accuracy']:.4f}")

if __name__ == "__main__":
    train_and_eval()
