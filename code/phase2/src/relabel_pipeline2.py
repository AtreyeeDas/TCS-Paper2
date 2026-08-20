#!/usr/bin/env python3
"""
ASIL NLU Final Semantic Relabeling + Separability Pipeline
Empirical Learnability Gate + Fast Candidate Assignment + Instant Checkpointing
"""

import os
import re
import json
import time
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    INPUT_CSV = os.path.join(ROOT_DIR, "master_nlu_dataset_canonical_augmented.csv")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "ontology_relabel")
    DIAG_DIR = os.path.join(ROOT_DIR, "results", "relabel_diagnostics")
    PCA_DIR = os.path.join(DIAG_DIR, "pca")
    CM_DIR = os.path.join(DIAG_DIR, "learnability_confusion_matrices")
    
    # Strictly use attention_pool for frozen representations
    WHISPER_EMB_DIR = os.path.join(ROOT_DIR, "embeddings", "attention_pool")
    
    # Model Paths - LOCALLY MAPPED
    GEMMA_PATH = "/home/spark2/Models/gemma4-e4b-it"  
    EMBEDDING_MODEL = "/home/spark2/Models/all-MiniLM-L6-v2"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    LEARNABILITY_TARGETS = {
        "domain": 0.85, "subdomain": 0.75, "intent": 0.70,
        "entity_type": 0.70, "urgency": 0.75, "emotion": 0.65
    }
    
    ASSIGNMENT_SIMILARITY_THRESHOLD = 0.50
    ASSIGNMENT_MARGIN_THRESHOLD = 0.05

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.DIAG_DIR, exist_ok=True)
os.makedirs(Config.PCA_DIR, exist_ok=True)
os.makedirs(Config.CM_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==============================================================================
# HELPER: LLM INFERENCE
# ==============================================================================
class LLMManager:
    def __init__(self):
        logging.info(f"Loading Gemma from {Config.GEMMA_PATH}...")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.GEMMA_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.GEMMA_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        ).eval()

    def query_json(self, prompt: str) -> dict:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(Config.DEVICE)
        
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        
        text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group(0)) if match else json.loads(text)
        except:
            return {}

# ==============================================================================
# PIPELINE STAGES
# ==============================================================================
class RelabelPipeline:
    def __init__(self):
        self.df = None
        self.embs = None

    def stage_05_assignment(self):
        logging.info("[STAGE 5] Loading Pre-computed Artifacts and Locked Ontology...")
        
        # Load Artifacts
        onto_path = os.path.join(Config.OUTPUT_DIR, "approved_ontology_v1.json")
        if not os.path.exists(onto_path):
            raise FileNotFoundError(f"Missing LOCKED ontology: {onto_path}")
            
        with open(onto_path, "r") as f:
            approved_onto = json.load(f)
            
        self.df = pd.read_pickle(os.path.join(Config.OUTPUT_DIR, "01_normalized.pkl"))
        self.embs = np.load(os.path.join(Config.OUTPUT_DIR, "02_minilm_embeddings.npy"))
        embedder = SentenceTransformer(Config.EMBEDDING_MODEL, device=Config.DEVICE)
        
        # Build Prototypes & Authoritative Allowed Classes
        class_prototypes = {}
        allowed_classes = {}
        
        for c in approved_onto:
            head = c["head"]
            if head not in class_prototypes: 
                class_prototypes[head] = {"names": [], "embs": [], "defs": [], "pos": [], "neg": []}
                allowed_classes[head] = set()
            
            c_name = c["canonical_name"]
            class_prototypes[head]["names"].append(c_name)
            allowed_classes[head].add(c_name)
            
            class_prototypes[head]["defs"].append(c.get("definition", ""))
            class_prototypes[head]["pos"].append(", ".join(c.get("positive_criteria", [])))
            class_prototypes[head]["neg"].append(", ".join(c.get("negative_criteria", [])))
            
            # Embed definition + positive criteria for matching
            def_text = c.get("definition", "") + " " + " ".join(c.get("positive_criteria", []))
            class_prototypes[head]["embs"].append(embedder.encode(def_text, normalize_embeddings=True))
            
        for h in class_prototypes: 
            class_prototypes[h]["embs"] = np.array(class_prototypes[h]["embs"])
        
        # Vectorized similarity matrix pre-computation
        all_sims = {head: cosine_similarity(self.embs, class_prototypes[head]["embs"]) for head in class_prototypes}

        llm = None # Lazy load only if needed
        
        # Checkpoint Recovery
        ckpt_path = os.path.join(Config.OUTPUT_DIR, "05_assignment_checkpoint.jsonl")
        processed_indices = set()
        completed_rows = []
        
        if os.path.exists(ckpt_path):
            with open(ckpt_path, 'r') as f:
                for line in f:
                    row_data = json.loads(line)
                    completed_rows.append(row_data)
                    processed_indices.add(row_data["_idx"])
            logging.info(f"Resumed {len(processed_indices)} samples from checkpoint.")

        assignment_stats = {
            "total_rows": len(self.df),
            "total_head_decisions": 0,
            "fast_path_count": 0,
            "gemma_adjudicated_count": 0,
            "invalid_llm_fallback_count": 0,
            "mask_count": 0,
            "outliers_count": 0,
            "per_head": {h: {"fast": 0, "gemma": 0, "fallback": 0, "mask": 0, "outlier": 0} for h in class_prototypes.keys()}
        }

        start_time = time.time()

        with open(ckpt_path, 'a') as ckpt_file:
            for i in tqdm(range(len(self.df)), desc="Deterministic & Fast Assignment"):
                if i in processed_indices:
                    continue 

                row_dict = {"_idx": i}
                row_data = self.df.iloc[i]
                
                for head in class_prototypes.keys():
                    assignment_stats["total_head_decisions"] += 1
                    
                    # 1. Strict MASK Handling
                    if head in row_data and str(row_data[head]).strip() == "MASK":
                        row_dict[f"canonical_{head}"] = "MASK"
                        row_dict[f"{head}_status"] = "MASK"
                        row_dict[f"{head}_similarity"] = 0.0
                        row_dict[f"{head}_margin"] = 0.0
                        assignment_stats["mask_count"] += 1
                        assignment_stats["per_head"][head]["mask"] += 1
                        continue

                    # ==============================================================
                    # BULLETPROOF SORTING LOGIC TO PREVENT INDEX ERRORS
                    # ==============================================================
                    sims = all_sims[head][i]
                    sorted_idx = np.argsort(sims)[::-1] # Sort all indices highest to lowest
                    
                    if len(sorted_idx) == 0:
                        continue
                        
                    idx1 = sorted_idx[0]
                    sim1 = sims[idx1]
                    class1 = class_prototypes[head]["names"][idx1]
                    
                    if len(sorted_idx) > 1:
                        idx2 = sorted_idx[1]
                        sim2 = sims[idx2]
                        class2 = class_prototypes[head]["names"][idx2]
                        margin = float(sim1 - sim2)
                    else:
                        idx2 = None
                        sim2 = 0.0
                        class2 = None
                        margin = 1.0 # 100% margin if there is literally no second option

                    row_dict[f"{head}_similarity"] = float(sim1)
                    row_dict[f"{head}_margin"] = float(margin)

                    # Explicit Outlier Check
                    if sim1 < 0.20:
                        row_dict[f"canonical_{head}"] = "OUTLIER_REVIEW"
                        row_dict[f"{head}_status"] = "OUTLIER"
                        assignment_stats["outliers_count"] += 1
                        assignment_stats["per_head"][head]["outlier"] += 1
                        continue

                    # 2. Fast Path
                    if sim1 >= Config.ASSIGNMENT_SIMILARITY_THRESHOLD and margin >= Config.ASSIGNMENT_MARGIN_THRESHOLD:
                        row_dict[f"canonical_{head}"] = class1
                        row_dict[f"{head}_status"] = "FAST_PATH"
                        assignment_stats["fast_path_count"] += 1
                        assignment_stats["per_head"][head]["fast"] += 1
                    else:
                        # 3. Gemma Adjudication
                        if class2 is None:
                            # Edge case: Only 1 valid class exists, skip Gemma
                            row_dict[f"canonical_{head}"] = class1
                            row_dict[f"{head}_status"] = "FAST_PATH"
                            assignment_stats["fast_path_count"] += 1
                            assignment_stats["per_head"][head]["fast"] += 1
                        else:
                            if llm is None: llm = LLMManager() # Load Gemma only on first slow-path
                            
                            prompt = f"""Choose exactly one of these already-approved classes according to the definitions and acceptance/rejection criteria. Do not invent, rename, merge, or create a class.
                            
Transcript: "{row_data['transcript']}"
Head: {head}

Candidate 1: {class1}
Definition: {class_prototypes[head]['defs'][idx1]}
Positive: {class_prototypes[head]['pos'][idx1]}
Negative: {class_prototypes[head]['neg'][idx1]}

Candidate 2: {class2}
Definition: {class_prototypes[head]['defs'][idx2]}
Positive: {class_prototypes[head]['pos'][idx2]}
Negative: {class_prototypes[head]['neg'][idx2]}

Return ONLY valid JSON: {{"selected_class": "EXACT_APPROVED_CLASS_NAME", "reason": "brief reason"}}"""
                            
                            res = llm.query_json(prompt)
                            selected = res.get("selected_class", "").strip()
                            
                            # Authoritative Validation
                            if selected in allowed_classes[head]:
                                row_dict[f"canonical_{head}"] = selected
                                row_dict[f"{head}_status"] = "GEMMA_ADJUDICATED"
                                assignment_stats["gemma_adjudicated_count"] += 1
                                assignment_stats["per_head"][head]["gemma"] += 1
                            else:
                                row_dict[f"canonical_{head}"] = class1 # Fallback to mathematical best
                                row_dict[f"{head}_status"] = "INVALID_LLM_FALLBACK"
                                assignment_stats["invalid_llm_fallback_count"] += 1
                                assignment_stats["per_head"][head]["fallback"] += 1

                ckpt_file.write(json.dumps(row_dict) + "\n")
                ckpt_file.flush()
                completed_rows.append(row_dict)
                processed_indices.add(i)

        assignment_stats["runtime_seconds"] = round(time.time() - start_time, 2)
        
        # Merge outputs
        ckpt_df = pd.DataFrame(completed_rows).set_index("_idx")
        final_df = self.df.join(ckpt_df)
        final_df.to_csv(os.path.join(Config.OUTPUT_DIR, "master_nlu_dataset_relabelled.csv"), index=False)
        
        # Diagnostics
        with open(os.path.join(Config.DIAG_DIR, "stage05_assignment_summary.json"), "w") as f:
            json.dump(assignment_stats, f, indent=2)

        freq_data, low_support_data = [], []
        for head in class_prototypes.keys():
            counts = final_df[f"canonical_{head}"].value_counts()
            total_valid = counts.sum() - counts.get("MASK", 0)
            
            for cls_name, count in counts.items():
                if cls_name == "MASK": continue
                freq_data.append({"head": head, "canonical_class": cls_name, "count": count, "percentage": round(count/total_valid*100, 2)})
                if count < Config.MIN_CLASS_SIZE:
                    low_support_data.append({"head": head, "canonical_class": cls_name, "count": count, "flag": "< 75" if count >= 50 else ("< 50" if count >= 30 else "< 30")})

        pd.DataFrame(freq_data).to_csv(os.path.join(Config.DIAG_DIR, "stage05_class_frequency.csv"), index=False)
        pd.DataFrame(low_support_data).to_csv(os.path.join(Config.DIAG_DIR, "stage05_low_support_classes.csv"), index=False)
        logging.info("[STAGE 5] Assignment Complete. Results saved.")

    def stage_06_learnability_gate(self):
        logging.info("[STAGE 6] Whisper Learnability Gate Audit...")
        
        df = pd.read_csv(os.path.join(Config.OUTPUT_DIR, "master_nlu_dataset_relabelled.csv"))
        
        # 1. Explicit Check for Frozen Whisper Embeddings
        missing_files = []
        whisper_embs = []
        
        for path in tqdm(df["audio_path"], desc="Loading Attention-Pool Features"):
            base = os.path.basename(path).replace(".wav", ".npz")
            npz_path = os.path.join(Config.WHISPER_EMB_DIR, base)
            if not os.path.exists(npz_path):
                missing_files.append(npz_path)
            else:
                whisper_embs.append(np.load(npz_path)['embedding'])

        if missing_files:
            raise FileNotFoundError(f"Missing {len(missing_files)} Whisper embeddings! Ensure {Config.WHISPER_EMB_DIR} is populated. First missing: {missing_files[:5]}")
            
        whisper_embs = np.array(whisper_embs)
        if whisper_embs.shape[1] != 1280:
            logging.warning(f"Unexpected dimension: {whisper_embs.shape[1]} (Expected 1280 for Whisper-large-v3-turbo). Proceeding...")

        results, per_class_results = [], []
        has_splits = "split" in df.columns
        
        for head, target in Config.LEARNABILITY_TARGETS.items():
            col = f"canonical_{head}"
            if col not in df.columns: continue
            
            valid_idx = ~df[col].isin(["OUTLIER_REVIEW", "MASK"])
            X = whisper_embs[valid_idx]
            y = df.loc[valid_idx, col].values
            
            if len(np.unique(y)) < 2: continue
            
            clf = LinearSVC(class_weight='balanced', max_iter=3000, dual=False)
            y_pred, y_true = [], []
            
            if has_splits:
                # Use project splits
                splits = df.loc[valid_idx, "split"].values
                train_mask, val_mask = splits == "train", splits == "val"
                clf.fit(X[train_mask], y[train_mask])
                y_pred = clf.predict(X[val_mask])
                y_true = y[val_mask]
            else:
                # Fallback to CV for diagnostic mapping
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                y_pred = np.zeros_like(y)
                y_true = y
                for train_idx, test_idx in skf.split(X, y):
                    clf.fit(X[train_idx], y[train_idx])
                    y_pred[test_idx] = clf.predict(X[test_idx])
                    
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            w_f1 = f1_score(y_true, y_pred, average='weighted')
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            
            passed = "PASS" if macro_f1 >= target else "FAIL"
            results.append({
                "Head": head, "Classes": len(np.unique(y_true)), "Target": target, 
                "Macro_F1": macro_f1, "Weighted_F1": w_f1, "Bal_Acc": bal_acc, "Acc": acc, "Gate_Status": passed
            })
            
            # Detailed Class Metrics
            rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            for cls_name in np.unique(y_true):
                per_class_results.append({
                    "Head": head, "Class": cls_name, "Precision": rep[cls_name]['precision'],
                    "Recall": rep[cls_name]['recall'], "F1": rep[cls_name]['f1-score'], "Support": rep[cls_name]['support']
                })
            
            # Confusion Matrix Plot
            cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
            plt.title(f"{head} Learnability Confusion Matrix (Macro-F1: {macro_f1:.2f})")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(Config.CM_DIR, f"{head}_confusion_matrix.png"))
            plt.close()
            
            # PCA Plot
            pca = PCA(n_components=2).fit_transform(X)
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=pca[:, 0], y=pca[:, 1], hue=y, palette="tab20", s=15, alpha=0.8)
            plt.title(f"{head} Frozen Whisper Space (PCA)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
            plt.tight_layout()
            plt.savefig(os.path.join(Config.PCA_DIR, f"{head}_whisper_pca.png"))
            plt.close()

        pd.DataFrame(results).to_csv(os.path.join(Config.DIAG_DIR, "learnability_gate_summary.csv"), index=False)
        pd.DataFrame(per_class_results).to_csv(os.path.join(Config.DIAG_DIR, "learnability_gate_per_class.csv"), index=False)
        
        # 3. Generate Cautious Scientific Markdown Report
        with open(os.path.join(Config.DIAG_DIR, "FINAL_RELABEL_DIAGNOSTIC_REPORT.md"), "w") as f:
            f.write("# ASIL NLU Relabeling & Empirical Learnability Report\n\n")
            f.write("### Objective\nThis report empirically demonstrates whether the final canonical ontology is learnable from the frozen Whisper representations actually used by the downstream NLU. It evaluates semantic coherence against acoustic representation geometry.\n\n")
            f.write("### Learnability Gate Results (Linear SVM on Whisper `attention_pool`)\n")
            f.write(pd.DataFrame(results).to_markdown(index=False) + "\n\n")
            f.write("### Interpretation\n* Pass/Fail denotes empirical learnability based on the development targets, **not** a mathematical guarantee of perfect classification for the final MLP.\n")
            f.write("* Check `learnability_gate_per_class.csv` for problematic overlapping classes.\n")
            f.write("* Confusion matrices and PCA plots are available in the diagnostic directories.\n")

        logging.info("[STAGE 6] Gate Evaluation Complete. Check FINAL_RELABEL_DIAGNOSTIC_REPORT.md.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=["5", "6", "all"], required=True, help="Note: Stages 1-4 are pre-computed and locked.")
    args = parser.parse_args()
    
    p = RelabelPipeline()
    
    if args.stage in ["1", "2", "3", "4"]:
        logging.warning("Stages 1-4 have been locked out to prevent overwriting existing artifacts. Please run --stage 5 or --stage 6.")
        
    if args.stage in ["5", "all"]: 
        p.stage_05_assignment()
        
    if args.stage in ["6", "all"]: 
        p.stage_06_learnability_gate()
