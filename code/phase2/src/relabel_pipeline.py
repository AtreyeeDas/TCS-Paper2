#!/usr/bin/env python3
"""
ASIL NLU Final Semantic Relabeling + Separability Pipeline
Empirical Learnability Gate + Fast Candidate Assignment + Instant Checkpointing
"""

import os
import re
import json
import glob
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, f1_score
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    INPUT_CSV = os.path.join(ROOT_DIR, "master_nlu_dataset_canonical_augmented.csv")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "ontology_relabel")
    DIAG_DIR = os.path.join(ROOT_DIR, "results", "relabel_diagnostics")
    
    # Using attention_pool for stronger baseline representations
    WHISPER_EMB_DIR = os.path.join(ROOT_DIR, "embeddings", "attention_pool")
    
    # Model Paths - LOCALLY MAPPED
    GEMMA_PATH = "/home/spark2/Models/gemma4-e4b-it"  
    EMBEDDING_MODEL = "/home/spark2/Models/all-MiniLM-L6-v2" # Updated local path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Constraints & Targets
    MIN_CLASS_SIZE = 75
    MAX_CLASSES = {"domain": 3, "subdomain": 6, "intent": 15, "entity_type": 20}
    FIXED_HEADS = ["urgency", "emotion"]
    FIXED_CLASSES = {
        "urgency": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "emotion": ["ANGER", "DISGUST", "FEAR", "JOY", "NEUTRAL", "SADNESS", "SURPRISE"]
    }
    
    LEARNABILITY_TARGETS = {
        "domain": 0.85, "subdomain": 0.75, "intent": 0.70,
        "entity_type": 0.70, "urgency": 0.75, "emotion": 0.65
    }
    
    ASSIGNMENT_SIMILARITY_THRESHOLD = 0.60
    ASSIGNMENT_MARGIN_THRESHOLD = 0.05

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.DIAG_DIR, exist_ok=True)
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
            outputs = self.model.generate(**inputs, max_new_tokens=400, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        
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
        self.start_time = time.time()

    def save_diag(self, filename, data):
        path = os.path.join(Config.DIAG_DIR, filename)
        if filename.endswith(".json"):
            with open(path, "w") as f: json.dump(data, f, indent=2)
        elif filename.endswith(".csv"):
            pd.DataFrame(data).to_csv(path, index=False)

    def stage_01_normalize(self):
        logging.info("[STAGE 1] Loading and Normalizing...")
        self.df = pd.read_csv(Config.INPUT_CSV)
        self.df["norm_transcript"] = self.df["transcript"].astype(str).str.lower().apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        self.df.to_pickle(os.path.join(Config.OUTPUT_DIR, "01_normalized.pkl"))
        logging.info("[STAGE 1] Saved fast 01_normalized.pkl checkpoint.")

    def stage_02_minilm(self):
        logging.info("[STAGE 2] Extracting MiniLM Embeddings...")
        self.df = pd.read_pickle(os.path.join(Config.OUTPUT_DIR, "01_normalized.pkl"))
        
        emb_path = os.path.join(Config.OUTPUT_DIR, "02_minilm_embeddings.npy")
        if os.path.exists(emb_path):
            logging.info("Embeddings already found. Skipping extraction.")
            return

        unique_texts = self.df["norm_transcript"].unique().tolist()
        embedder = SentenceTransformer(Config.EMBEDDING_MODEL, device=Config.DEVICE)
        
        unique_embs = embedder.encode(unique_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        text_to_emb = {t: e for t, e in zip(unique_texts, unique_embs)}
        
        self.embs = np.array([text_to_emb[t] for t in self.df["norm_transcript"]])
        np.save(emb_path, self.embs)
        logging.info("[STAGE 2] Saved 02_minilm_embeddings.npy checkpoint.")

    def stage_03_clustering(self):
        logging.info("[STAGE 3] Semantic Candidate Clustering...")
        self.df = pd.read_pickle(os.path.join(Config.OUTPUT_DIR, "01_normalized.pkl"))
        self.embs = np.load(os.path.join(Config.OUTPUT_DIR, "02_minilm_embeddings.npy"))
        
        cluster_path = os.path.join(Config.OUTPUT_DIR, "03_candidate_clusters.npy")
        if os.path.exists(cluster_path):
            logging.info("Clusters already found. Skipping.")
            return
            
        cluster_summary = []
        cluster_labels_dict = {}
        
        for head, max_k in Config.MAX_CLASSES.items():
            best_k, best_score, best_labels = 2, -1, None
            for k in range(3, max_k + 1):
                labels = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average').fit_predict(self.embs)
                score = silhouette_score(self.embs, labels, metric='cosine')
                if score > best_score: best_k, best_score, best_labels = k, score, labels
                    
            cluster_labels_dict[head] = best_labels
            
            for c_id in range(best_k):
                c_idx = np.where(best_labels == c_id)[0]
                examples = self.df.iloc[c_idx]["transcript"].sample(min(5, len(c_idx))).tolist()
                cluster_summary.append({
                    "head": head, "cluster_id": c_id, "cluster_size": len(c_idx), 
                    "representative_examples": " | ".join(examples)
                })
                
        np.save(cluster_path, cluster_labels_dict)
        self.save_diag("stage03_cluster_summary.csv", cluster_summary)
        logging.info("[STAGE 3] Saved 03_candidate_clusters.npy checkpoint.")

    def stage_04_gemma_ontology(self):
        logging.info("[STAGE 4] Gemma Ontology Proposal with Checkpointing...")
        
        # Checkpoint mapping
        ckpt_path = os.path.join(Config.OUTPUT_DIR, "04_ontology_checkpoint.jsonl")
        processed_clusters = set()
        candidate_ontology = []
        
        if os.path.exists(ckpt_path):
            with open(ckpt_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    candidate_ontology.append(data)
                    processed_clusters.add(f"{data['head']}_{data['cluster_id']}")
            logging.info(f"Resumed {len(processed_clusters)} completed clusters from checkpoint.")

        cluster_summary = pd.read_csv(os.path.join(Config.DIAG_DIR, "stage03_cluster_summary.csv"))
        llm = LLMManager()

        with open(ckpt_path, 'a') as ckpt_file:
            for _, row in tqdm(cluster_summary.iterrows(), total=len(cluster_summary), desc="Gemma Proposing"):
                c_key = f"{row['head']}_{row['cluster_id']}"
                if c_key in processed_clusters:
                    continue # Skip already processed
                    
                prompt = f"""You are building a semantic ontology for NLU. Analyze this cluster for the head '{row['head']}'.
Size: {row['cluster_size']} samples. 
Examples: {row['representative_examples']}
Return STRICT JSON defining this class:
{{"canonical_name": "NAME_IN_CAPS", "definition": "short def", "positive_criteria": ["...", "..."], "negative_criteria": ["...", "..."]}}"""
                
                res = llm.query_json(prompt)
                if "canonical_name" in res:
                    res["head"] = row["head"]
                    res["cluster_id"] = row["cluster_id"]
                    res["support"] = row["cluster_size"]
                    
                    candidate_ontology.append(res)
                    processed_clusters.add(c_key)
                    
                    # Instant Save
                    ckpt_file.write(json.dumps(res) + "\n")
                    ckpt_file.flush()

        # Add fixed heads and export final CSV
        for head, classes in Config.FIXED_CLASSES.items():
            for c in classes:
                candidate_ontology.append({"head": head, "canonical_name": c, "definition": f"Standard {c}", "positive_criteria": [], "negative_criteria": []})
                
        pd.DataFrame(candidate_ontology).to_csv(os.path.join(Config.OUTPUT_DIR, "FINAL_ONTOLOGY_REVIEW.csv"), index=False)
        logging.info("PAUSING. Review FINAL_ONTOLOGY_REVIEW.csv and save as approved_ontology_v1.json")

    def stage_05_assignment(self):
        logging.info("[STAGE 5] Fast Assignment with Row-by-Row Checkpointing...")
        onto_path = os.path.join(Config.OUTPUT_DIR, "approved_ontology_v1.json")
        if not os.path.exists(onto_path): raise FileNotFoundError(f"Missing {onto_path}")
        
        with open(onto_path, "r") as f: approved_onto = json.load(f)
            
        self.df = pd.read_pickle(os.path.join(Config.OUTPUT_DIR, "01_normalized.pkl"))
        self.embs = np.load(os.path.join(Config.OUTPUT_DIR, "02_minilm_embeddings.npy"))
        embedder = SentenceTransformer(Config.EMBEDDING_MODEL, device=Config.DEVICE)
        
        # Build Prototypes
        class_prototypes = {}
        for c in approved_onto:
            head = c["head"]
            if head not in class_prototypes: class_prototypes[head] = {"names": [], "embs": []}
            class_prototypes[head]["names"].append(c["canonical_name"])
            def_text = c["definition"] + " " + " ".join(c.get("positive_criteria", []))
            class_prototypes[head]["embs"].append(embedder.encode(def_text, normalize_embeddings=True))
            
        for h in class_prototypes: class_prototypes[h]["embs"] = np.array(class_prototypes[h]["embs"])
        
        # Pre-compute all similarities to save time inside the loop
        all_sims = {head: cosine_similarity(self.embs, class_prototypes[head]["embs"]) for head in class_prototypes}

        # Checkpoint recovery
        ckpt_path = os.path.join(Config.OUTPUT_DIR, "05_assignment_checkpoint.jsonl")
        processed_indices = set()
        completed_rows = []
        
        if os.path.exists(ckpt_path):
            with open(ckpt_path, 'r') as f:
                for line in f:
                    row_data = json.loads(line)
                    completed_rows.append(row_data)
                    processed_indices.add(row_data["_idx"])
            logging.info(f"Resumed {len(processed_indices)} samples from previous run.")

        llm = LLMManager()
        assignment_stats = {"fast_path": 0, "gemma_adjudicated": 0, "outliers": 0}

        with open(ckpt_path, 'a') as ckpt_file:
            for i in tqdm(range(len(self.df)), desc="Row-by-Row Assignment"):
                if i in processed_indices:
                    continue # Instantly skip what's already saved

                row_dict = {"_idx": i}
                
                for head in class_prototypes.keys():
                    sims = all_sims[head][i]
                    top_2_idx = np.argsort(sims)[-2:][::-1]
                    idx1, idx2 = top_2_idx[0], top_2_idx[1]
                    sim1, sim2 = sims[idx1], sims[idx2]
                    margin = sim1 - sim2
                    class1, class2 = class_prototypes[head]["names"][idx1], class_prototypes[head]["names"][idx2]
                    
                    if sim1 < 0.2:
                        row_dict[f"canonical_{head}"] = "OUTLIER_REVIEW"
                        row_dict[f"{head}_status"] = "OUTLIER"
                        assignment_stats["outliers"] += 1
                    elif margin >= Config.ASSIGNMENT_MARGIN_THRESHOLD and sim1 >= Config.ASSIGNMENT_SIMILARITY_THRESHOLD:
                        row_dict[f"canonical_{head}"] = class1
                        row_dict[f"{head}_status"] = "FAST_PATH"
                        assignment_stats["fast_path"] += 1
                    else:
                        prompt = f"Transcript: '{self.df.iloc[i]['transcript']}'. Best classes for {head}: 1. {class1} 2. {class2}. Select the best ONE. Return JSON: {{\"selected_class\": \"CLASS\"}}"
                        res = llm.query_json(prompt)
                        row_dict[f"canonical_{head}"] = res.get("selected_class", class1)
                        row_dict[f"{head}_status"] = "GEMMA_ADJUDICATED"
                        assignment_stats["gemma_adjudicated"] += 1
                        
                    row_dict[f"{head}_margin"] = float(margin)

                # INSTANT DISK FLUSH: Save progress for this specific transcript
                ckpt_file.write(json.dumps(row_dict) + "\n")
                ckpt_file.flush()
                completed_rows.append(row_dict)
                processed_indices.add(i)

        # Merge check-pointed columns back into main DataFrame
        ckpt_df = pd.DataFrame(completed_rows).set_index("_idx")
        final_df = self.df.join(ckpt_df)
        final_df.to_csv(os.path.join(Config.OUTPUT_DIR, "master_nlu_dataset_relabelled.csv"), index=False)
        self.save_diag("stage06_assignment_summary.json", assignment_stats)
        logging.info("Stage 5 fully completed and merged.")

    def stage_06_learnability_gate(self):
        logging.info("[STAGE 6] Whisper Learnability Gate Audit...")
        self.df = pd.read_csv(os.path.join(Config.OUTPUT_DIR, "master_nlu_dataset_relabelled.csv"))
        
        emb_list = []
        for path in self.df["audio_path"]:
            base = os.path.basename(path).replace(".wav", ".npz")
            npz_path = os.path.join(Config.WHISPER_EMB_DIR, base)
            if os.path.exists(npz_path):
                emb_list.append(np.load(npz_path)['embedding'])
            else:
                emb_list.append(np.zeros(1280))
        self.whisper_embs = np.array(emb_list)

        results = []
        for head, target in Config.LEARNABILITY_TARGETS.items():
            col = f"canonical_{head}"
            if col not in self.df.columns: continue
            
            valid_idx = ~self.df[col].isin(["OUTLIER_REVIEW", "MASK"])
            X = self.whisper_embs[valid_idx]
            y = self.df.loc[valid_idx, col].values
            
            if len(np.unique(y)) < 2: continue
            
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            clf = LinearSVC(class_weight='balanced', max_iter=2000)
            
            y_pred = np.zeros_like(y)
            for train_idx, test_idx in skf.split(X, y):
                clf.fit(X[train_idx], y[train_idx])
                y_pred[test_idx] = clf.predict(X[test_idx])
                
            macro_f1 = f1_score(y, y_pred, average='macro')
            passed = "PASS" if macro_f1 >= target else "FAIL"
            
            results.append({"Head": head, "Classes": len(np.unique(y)), "Macro_F1": macro_f1, "Target": target, "Gate_Status": passed})

        res_df = pd.DataFrame(results)
        res_df.to_csv(os.path.join(Config.DIAG_DIR, "learnability_gate_summary.csv"), index=False)
        logging.info("\n" + res_df.to_markdown(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=["1", "2", "3", "4", "5", "6", "all"], required=True)
    args = parser.parse_args()
    
    p = RelabelPipeline()
    if args.stage in ["1", "all"]: p.stage_01_normalize()
    if args.stage in ["2", "all"]: p.stage_02_minilm()
    if args.stage in ["3", "all"]: p.stage_03_clustering()
    if args.stage in ["4", "all"]: p.stage_04_gemma_ontology()
    
    if args.stage == "all" and not os.path.exists(os.path.join(Config.OUTPUT_DIR, "approved_ontology_v1.json")):
        print("\n[!] PAUSED: Human Review Required. Edit 'FINAL_ONTOLOGY_REVIEW.csv', save as 'approved_ontology_v1.json', then resume with --stage 5.\n")
        exit(0)
        
    if args.stage in ["5", "all"]: p.stage_05_assignment()
    if args.stage in ["6", "all"]: p.stage_06_learnability_gate()
