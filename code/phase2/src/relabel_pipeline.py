#!/usr/bin/env python3
"""
ASIL NLU Final Semantic Relabeling + Separability Pipeline
Empirical Learnability Gate + Fast Candidate Assignment + Gemma Adjudication
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
from sklearn.metrics import silhouette_score, f1_score, accuracy_score, confusion_matrix
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
    WHISPER_EMB_DIR = os.path.join(ROOT_DIR, "embeddings", "mean_pool")
    
    # Model Paths
    GEMMA_PATH = "/home/spark2/Models/gemma4-e4b-it"  # Adjust as needed
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
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
        self.whisper_embs = None
        self.start_time = time.time()

    def save_diag(self, filename, data):
        path = os.path.join(Config.DIAG_DIR, filename)
        if filename.endswith(".json"):
            with open(path, "w") as f: json.dump(data, f, indent=2)
        elif filename.endswith(".csv"):
            pd.DataFrame(data).to_csv(path, index=False)
        logging.info(f"Saved diagnostic: {filename}")

    def stage_01_normalize(self):
        logging.info("[STAGE 1] Loading and Normalizing...")
        self.df = pd.read_csv(Config.INPUT_CSV)
        self.df["norm_transcript"] = self.df["transcript"].astype(str).str.lower().apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        
        self.df.to_pickle(os.path.join(Config.OUTPUT_DIR, "01_normalized.pkl"))
        
        diag = {
            "total_rows": len(self.df),
            "unique_transcripts": self.df["norm_transcript"].nunique(),
            "empty_transcripts": int((self.df["norm_transcript"] == "").sum())
        }
        self.save_diag("stage01_normalization_summary.json", diag)

    def stage_02_minilm(self):
        logging.info("[STAGE 2] Extracting MiniLM Embeddings...")
        self.df = pd.read_pickle(os.path.join(Config.OUTPUT_DIR, "01_normalized.pkl"))
        
        # Deduplication to save time
        unique_texts = self.df["norm_transcript"].unique().tolist()
        embedder = SentenceTransformer(Config.EMBEDDING_MODEL, device=Config.DEVICE)
        
        unique_embs = embedder.encode(unique_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        text_to_emb = {t: e for t, e in zip(unique_texts, unique_embs)}
        
        self.embs = np.array([text_to_emb[t] for t in self.df["norm_transcript"]])
        np.save(os.path.join(Config.OUTPUT_DIR, "02_minilm_embeddings.npy"), self.embs)
        
        diag = {"embedding_dim": self.embs.shape[1], "unique_embeddings": len(unique_embs), "mean_norm": float(np.linalg.norm(self.embs, axis=1).mean())}
        self.save_diag("stage02_embedding_summary.json", diag)

    def stage_03_clustering(self):
        logging.info("[STAGE 3] Semantic Candidate Clustering...")
        self.df = pd.read_pickle(os.path.join(Config.OUTPUT_DIR, "01_normalized.pkl"))
        self.embs = np.load(os.path.join(Config.OUTPUT_DIR, "02_minilm_embeddings.npy"))
        
        cluster_summary = []
        cluster_labels_dict = {}
        
        for head, max_k in Config.MAX_CLASSES.items():
            best_k, best_score, best_labels = 2, -1, None
            
            # Bounded search for best silhouette
            for k in range(3, max_k + 1):
                clustering = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
                labels = clustering.fit_predict(self.embs)
                score = silhouette_score(self.embs, labels, metric='cosine')
                if score > best_score:
                    best_k, best_score, best_labels = k, score, labels
                    
            cluster_labels_dict[head] = best_labels
            
            # PCA Visualization
            pca = PCA(n_components=2).fit_transform(self.embs)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=pca[:, 0], y=pca[:, 1], hue=best_labels, palette="tab10", s=10)
            plt.title(f"{head.capitalize()} Candidate Clusters (MiniLM)")
            plt.savefig(os.path.join(Config.DIAG_DIR, f"stage03_pca_{head}.png"))
            plt.close()
            
            for c_id in range(best_k):
                c_idx = np.where(best_labels == c_id)[0]
                examples = self.df.iloc[c_idx]["transcript"].sample(min(5, len(c_idx))).tolist()
                cluster_summary.append({
                    "head": head, "cluster_id": c_id, "cluster_size": len(c_idx), 
                    "representative_examples": " | ".join(examples)
                })
                
        np.save(os.path.join(Config.OUTPUT_DIR, "03_candidate_clusters.npy"), cluster_labels_dict)
        self.save_diag("stage03_cluster_summary.csv", cluster_summary)

    def stage_04_gemma_ontology(self):
        logging.info("[STAGE 4] Gemma Ontology Proposal...")
        llm = LLMManager()
        cluster_summary = pd.read_csv(os.path.join(Config.DIAG_DIR, "stage03_cluster_summary.csv"))
        
        candidate_ontology = []
        for _, row in tqdm(cluster_summary.iterrows(), total=len(cluster_summary), desc="Gemma Proposing"):
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
                
        # Append fixed heads
        for head, classes in Config.FIXED_CLASSES.items():
            for c in classes:
                candidate_ontology.append({"head": head, "canonical_name": c, "definition": f"Standard {c}", "positive_criteria": [], "negative_criteria": []})
                
        pd.DataFrame(candidate_ontology).to_csv(os.path.join(Config.OUTPUT_DIR, "FINAL_ONTOLOGY_REVIEW.csv"), index=False)
        self.save_diag("stage04_ontology_summary.csv", candidate_ontology)
        logging.info("PAUSING. Review FINAL_ONTOLOGY_REVIEW.csv and save as approved_ontology_v1.json")

    def stage_05_assignment(self):
        logging.info("[STAGE 5] Fast Candidate Retrieval & Adjudication...")
        onto_path = os.path.join(Config.OUTPUT_DIR, "approved_ontology_v1.json")
        if not os.path.exists(onto_path): raise FileNotFoundError(f"Missing {onto_path}")
        
        with open(onto_path, "r") as f: approved_onto = json.load(f)
            
        self.df = pd.read_pickle(os.path.join(Config.OUTPUT_DIR, "01_normalized.pkl"))
        self.embs = np.load(os.path.join(Config.OUTPUT_DIR, "02_minilm_embeddings.npy"))
        embedder = SentenceTransformer(Config.EMBEDDING_MODEL, device=Config.DEVICE)
        llm = LLMManager()
        
        # Build Prototypes
        class_prototypes = {}
        for c in approved_onto:
            head = c["head"]
            if head not in class_prototypes: class_prototypes[head] = {"names": [], "embs": []}
            class_prototypes[head]["names"].append(c["canonical_name"])
            def_text = c["definition"] + " " + " ".join(c.get("positive_criteria", []))
            class_prototypes[head]["embs"].append(embedder.encode(def_text, normalize_embeddings=True))
            
        for h in class_prototypes: class_prototypes[h]["embs"] = np.array(class_prototypes[h]["embs"])
        
        assignment_stats = {"fast_path": 0, "gemma_adjudicated": 0, "outliers": 0}
        
        # Batch Assignment
        for head in class_prototypes.keys():
            sims = cosine_similarity(self.embs, class_prototypes[head]["embs"])
            top_2_idx = np.argsort(sims, axis=1)[:, -2:][:, ::-1]
            
            new_labels, confs, statuses, margins = [], [], [], []
            for i in tqdm(range(len(self.df)), desc=f"Assigning {head}"):
                idx1, idx2 = top_2_idx[i][0], top_2_idx[i][1]
                sim1, sim2 = sims[i][idx1], sims[i][idx2]
                margin = sim1 - sim2
                class1, class2 = class_prototypes[head]["names"][idx1], class_prototypes[head]["names"][idx2]
                
                if sim1 < 0.2: # Outlier
                    new_labels.append("OUTLIER_REVIEW")
                    statuses.append("OUTLIER")
                    assignment_stats["outliers"] += 1
                elif margin >= Config.ASSIGNMENT_MARGIN_THRESHOLD and sim1 >= Config.ASSIGNMENT_SIMILARITY_THRESHOLD:
                    new_labels.append(class1)
                    statuses.append("FAST_PATH")
                    assignment_stats["fast_path"] += 1
                else:
                    # Slow path Adjudication
                    prompt = f"Transcript: '{self.df.iloc[i]['transcript']}'. Best classes for {head}: 1. {class1} 2. {class2}. Select the best ONE. Return JSON: {{\"selected_class\": \"CLASS\"}}"
                    res = llm.query_json(prompt)
                    new_labels.append(res.get("selected_class", class1))
                    statuses.append("GEMMA_ADJUDICATED")
                    assignment_stats["gemma_adjudicated"] += 1
                    
                confs.append(float(sim1))
                margins.append(float(margin))
                
            self.df[f"canonical_{head}"] = new_labels
            self.df[f"{head}_margin"] = margins
            self.df[f"{head}_status"] = statuses

        self.df.to_csv(os.path.join(Config.OUTPUT_DIR, "master_nlu_dataset_relabelled.csv"), index=False)
        self.save_diag("stage06_assignment_summary.json", assignment_stats)

    def stage_06_learnability_gate(self):
        logging.info("[STAGE 6] Whisper Learnability Gate Audit...")
        self.df = pd.read_csv(os.path.join(Config.OUTPUT_DIR, "master_nlu_dataset_relabelled.csv"))
        
        # Load Whisper Embeddings (Assuming user has them named by audio_path in .npz)
        # Mocking loader for this script's robustness; replace with your actual loader
        emb_list = []
        for path in self.df["audio_path"]:
            base = os.path.basename(path).replace(".wav", ".npz")
            npz_path = os.path.join(Config.WHISPER_EMB_DIR, base)
            if os.path.exists(npz_path):
                emb_list.append(np.load(npz_path)['embedding']) # Adjust key as per your extract script
            else:
                emb_list.append(np.zeros(1280)) # Fallback
        self.whisper_embs = np.array(emb_list)

        results = []
        for head, target in Config.LEARNABILITY_TARGETS.items():
            col = f"canonical_{head}"
            if col not in self.df.columns: continue
            
            valid_idx = ~self.df[col].isin(["OUTLIER_REVIEW", "MASK"])
            X = self.whisper_embs[valid_idx]
            y = self.df.loc[valid_idx, col].values
            
            if len(np.unique(y)) < 2: continue
            
            # Linear SVM 5-Fold Evaluation
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            clf = LinearSVC(class_weight='balanced', max_iter=2000)
            
            y_pred = np.zeros_like(y)
            for train_idx, test_idx in skf.split(X, y):
                clf.fit(X[train_idx], y[train_idx])
                y_pred[test_idx] = clf.predict(X[test_idx])
                
            macro_f1 = f1_score(y, y_pred, average='macro')
            passed = "PASS" if macro_f1 >= target else "FAIL"
            
            results.append({"Head": head, "Classes": len(np.unique(y)), "Macro_F1": macro_f1, "Target": target, "Gate_Status": passed})
            
            # PCA for Whisper Space Separability
            pca = PCA(n_components=2).fit_transform(X)
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=pca[:, 0], y=pca[:, 1], hue=y, palette="tab20", s=15, alpha=0.7)
            plt.title(f"Whisper Separability PCA: {head.capitalize()} (Macro F1: {macro_f1:.2f})")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.tight_layout()
            plt.savefig(os.path.join(Config.DIAG_DIR, f"stage07_pca_whisper_{head}.png"))
            plt.close()

        res_df = pd.DataFrame(results)
        res_df.to_csv(os.path.join(Config.DIAG_DIR, "learnability_gate_summary.csv"), index=False)
        
        # Dashboard Markdown
        with open(os.path.join(Config.DIAG_DIR, "FINAL_RELABELING_DASHBOARD.md"), "w") as f:
            f.write("# ASIL NLU Relabeling & Learnability Dashboard\n\n")
            f.write("### Empirical Whisper Learnability Gate\n")
            f.write(res_df.to_markdown(index=False))
            f.write("\n\n*Check the `results/relabel_diagnostics/` folder for PCA visualizations.*")
            
        logging.info("PIPELINE COMPLETE. Dashboards generated.")

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
        print("\n[!] PAUSED: Human Review Required.")
        print("Please edit 'FINAL_ONTOLOGY_REVIEW.csv', save as 'approved_ontology_v1.json', then resume with --stage 5.\n")
        exit(0)
        
    if args.stage in ["5", "all"]: p.stage_05_assignment()
    if args.stage in ["6", "all"]: p.stage_06_learnability_gate()
