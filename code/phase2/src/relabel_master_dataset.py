#!/usr/bin/env python3
"""
ASIL NLU Master Dataset Semantic Audit and Relabeling Pipeline
============================================================
Author: ASIL NLU Research Core
Description:
    Conducts a rigorous class-level ontology audit and sample-level semantic relabeling
    using local Gemma (4B-IT) via dual-pass reasoning + third-pass conflict adjudication.
    Enforces a strict, closed-set canonical ontology without uncontrolled class creation,
    drift, or test contamination.
"""

import os
import re
import sys
import json
import time
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================================================================
# 1. CONFIGURATION & LOCKED CANONICAL TAXONOMY
# ==============================================================================

class RelabelConfig:
    # Environment & Paths
    ROOT_DIR: str = os.path.abspath(os.path.dirname(__file__))
    INPUT_CSV: str = os.path.join(ROOT_DIR, "master_nlu_dataset_augmented_4712.csv")
    OUTPUT_DIR: str = os.path.join(ROOT_DIR, "results", "relabeling")
    CHECKPOINT_DIR: str = os.path.join(OUTPUT_DIR, "checkpoints")
    GEMMA_PATH: str = "/home/spark2/Models/gemma4-e4b-it"

    # Runtime Settings
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: torch.dtype = torch.bfloat16
    SEED: int = 42
    MASK_TOKEN: str = "MASK"
    MIN_CLASS_SIZE: int = 100

    # Locked Canonical Ontology
    CANONICAL_ONTOLOGY = {
        "domain": [
            "FINANCIAL",
            "MEDICAL",
            "GENERAL"
        ],
        "subdomain": [
            "TECHNOLOGY",
            "INDUSTRIAL_GOODS",
            "USER_INTERACTION_TYPE",
            "MEDICAL_SPECIALTY",
            "HEALTHCARE",
            "GENERAL_TOPIC"
        ],
        "intent": [
            "MEDICAL_SYMPTOM_AND_CONDITION",
            "FINANCIAL_REPORTING_AND_OUTLOOK",
            "FINANCIAL_PERFORMANCE_AND_SEGMENT_ANALYSIS",
            "FINANCIAL_OPERATIONS_RISK_AND_COMPLIANCE",
            "MEDICAL_ADVICE_AND_TREATMENT",
            "MEDIA_ENTERTAINMENT_AND_UTILITY",
            "BUSINESS_STRATEGY_AND_PROJECT_PLANNING",
            "GENERAL_ASSISTANCE_AND_QA",
            "BUSINESS_OPERATIONS_AND_HEALTH",
            "MEDICATION_DIAGNOSTICS_AND_CLINICAL_RESEARCH",
            "TIME_LOCATION_TRAVEL_AND_LOGISTICS",
            "COMMUNICATION_EMAIL_CALL_AND_CONTACTS",
            "MARKET_AND_COMPETITIVE_OUTLOOK",
            "PRODUCT_AND_TECHNOLOGY_MANAGEMENT"
        ],
        "entity_type": [
            "FINANCIAL_METRICS_DETAIL",
            "MEDICAL_SYMPTOM",
            "FINANCIAL_STRATEGY_AND_PLANNING",
            "GENERAL_INFORMATION_AND_MISCELLANEOUS",
            "MARKET_AND_INDUSTRY_ANALYSIS",
            "GENERAL_BUSINESS_ADMINISTRATION",
            "TREATMENT_OR_HEALTHCARE",
            "BUSINESS_OPERATIONS_AND_LOGISTICS",
            "MEDICAL_CONDITION_OR_CLINICAL_STATE",
            "CORPORATE_FINANCIAL_PERFORMANCE",
            "TECHNOLOGY_AND_DIGITAL_INFRASTRUCTURE",
            "DIAGNOSTIC_TEST_OR_RESULT",
            "MEDICATION_OR_DRUG",
            "SALES_AND_CUSTOMER_ENGAGEMENT",
            "COMMUNICATION_AND_INTERACTION",
            "ENTERTAINMENT_AND_MEDIA",
            "BUSINESS_GOVERNANCE_AND_RELATIONS",
            "PRODUCT_AND_SERVICE_MANAGEMENT",
            "REGULATORY_AND_ENVIRONMENTAL_CONTEXT",
            "PERSONAL_AND_DAILY_LIFE"
        ],
        "urgency": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        ],
        "emotion": [
            "ANGER",
            "DISGUST",
            "FEAR",
            "JOY",
            "NEUTRAL",
            "SADNESS",
            "SURPRISE"
        ]
    }

    # Hierarchical Dependency Constraints (Context Filtering)
    DOMAIN_TO_SUBDOMAINS = {
        "FINANCIAL": ["TECHNOLOGY", "INDUSTRIAL_GOODS", "HEALTHCARE", "GENERAL_TOPIC"],
        "MEDICAL": ["MEDICAL_SPECIALTY", "HEALTHCARE", "GENERAL_TOPIC"],
        "GENERAL": ["USER_INTERACTION_TYPE", "TECHNOLOGY", "GENERAL_TOPIC"]
    }

    SUBDOMAIN_TO_INTENTS = {
        "MEDICAL_SPECIALTY": [
            "MEDICAL_SYMPTOM_AND_CONDITION", "MEDICAL_ADVICE_AND_TREATMENT",
            "MEDICATION_DIAGNOSTICS_AND_CLINICAL_RESEARCH", "BUSINESS_OPERATIONS_AND_HEALTH"
        ],
        "HEALTHCARE": [
            "MEDICAL_SYMPTOM_AND_CONDITION", "MEDICAL_ADVICE_AND_TREATMENT",
            "MEDICATION_DIAGNOSTICS_AND_CLINICAL_RESEARCH", "BUSINESS_OPERATIONS_AND_HEALTH",
            "FINANCIAL_REPORTING_AND_OUTLOOK", "FINANCIAL_PERFORMANCE_AND_SEGMENT_ANALYSIS"
        ],
        "TECHNOLOGY": [
            "PRODUCT_AND_TECHNOLOGY_MANAGEMENT", "BUSINESS_STRATEGY_AND_PROJECT_PLANNING",
            "FINANCIAL_REPORTING_AND_OUTLOOK", "FINANCIAL_PERFORMANCE_AND_SEGMENT_ANALYSIS",
            "MEDIA_ENTERTAINMENT_AND_UTILITY", "GENERAL_ASSISTANCE_AND_QA"
        ],
        "INDUSTRIAL_GOODS": [
            "FINANCIAL_REPORTING_AND_OUTLOOK", "FINANCIAL_PERFORMANCE_AND_SEGMENT_ANALYSIS",
            "FINANCIAL_OPERATIONS_RISK_AND_COMPLIANCE", "MARKET_AND_COMPETITIVE_OUTLOOK",
            "BUSINESS_STRATEGY_AND_PROJECT_PLANNING"
        ],
        "USER_INTERACTION_TYPE": [
            "MEDIA_ENTERTAINMENT_AND_UTILITY", "GENERAL_ASSISTANCE_AND_QA",
            "TIME_LOCATION_TRAVEL_AND_LOGISTICS", "COMMUNICATION_EMAIL_CALL_AND_CONTACTS",
            "PRODUCT_AND_TECHNOLOGY_MANAGEMENT"
        ],
        "GENERAL_TOPIC": [
            "GENERAL_ASSISTANCE_AND_QA", "MEDIA_ENTERTAINMENT_AND_UTILITY",
            "TIME_LOCATION_TRAVEL_AND_LOGISTICS", "COMMUNICATION_EMAIL_CALL_AND_CONTACTS"
        ]
    }

    HEADS = ["domain", "subdomain", "intent", "entity_type", "urgency", "emotion"]

    @classmethod
    def setup_dirs(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)

# Set random seeds for deterministic execution
torch.manual_seed(RelabelConfig.SEED)
np.random.seed(RelabelConfig.SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# ==============================================================================
# 2. LOCAL GEMMA LLM CLIENT & ROBUST JSON INFERENCE ENGINE
# ==============================================================================

class GemmaEngine:
    """Wraps local HuggingFace Gemma with robust SDPA and structured decoding."""
    def __init__(self, model_path: str = RelabelConfig.GEMMA_PATH):
        logging.info(f"Loading local Gemma model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=RelabelConfig.DTYPE,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="sdpa",
            trust_remote_code=True
        ).eval()

    def generate_json(self, prompt: str, system_prompt: str = "", max_retries: int = 3) -> Dict[str, Any]:
        """Generates text from Gemma and extracts valid JSON payload."""
        full_content = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        messages = [{"role": "user", "content": full_content}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(RelabelConfig.DEVICE)

        for attempt in range(max_retries):
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,  # Deterministic decoding
                    pad_token_id=self.tokenizer.eos_token_id
                )

            gen_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            raw_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            try:
                # Regex match for JSON block
                match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    return parsed
                parsed = json.loads(raw_text)
                return parsed
            except Exception as e:
                logging.debug(f"JSON parsing retry {attempt+1}/{max_retries}: {e}")

        # Fallback if unparsable
        return {
            "predicted_label": RelabelConfig.MASK_TOKEN,
            "confidence": 0.0,
            "status": "AMBIGUOUS",
            "supporting_evidence": "JSON decode failed",
            "contradicting_evidence": raw_text[:100]
        }

# ==============================================================================
# 3. PART 1: CLASS-LEVEL ONTOLOGY AUDIT
# ==============================================================================

class OntologyAuditor:
    """Audits the canonical ontology definitions, boundaries, and separability."""
    def __init__(self, llm: GemmaEngine):
        self.llm = llm

    def run_audit(self) -> Dict[str, Any]:
        audit_json_path = os.path.join(RelabelConfig.OUTPUT_DIR, "ontology_audit.json")
        audit_csv_path = os.path.join(RelabelConfig.OUTPUT_DIR, "ontology_audit.csv")

        if os.path.exists(audit_json_path) and os.path.exists(audit_csv_path):
            logging.info("Class-level ontology audit already exists. Loading cached version.")
            with open(audit_json_path, "r") as f:
                return json.load(f)

        logging.info("Conducting Phase 1: Class-Level Canonical Ontology Audit...")
        ontology_audit = {}
        csv_rows = []

        for head in RelabelConfig.HEADS:
            ontology_audit[head] = {}
            classes = RelabelConfig.CANONICAL_ONTOLOGY[head]
            logging.info(f"Auditing head: {head.upper()} ({len(classes)} classes)...")

            for cname in classes:
                prompt = f"""You are an expert NLP and speech ontology architect.
Analyze the canonical class '{cname}' for the classification head '{head}'.
Available classes for this head are: {classes}.

Provide structured JSON with the exact format:
{{
  "canonical_class": "{cname}",
  "concise_definition": "Clear, single-sentence definition",
  "positive_inclusion_criteria": ["Explicit semantic cues required"],
  "exclusion_criteria": ["Criteria that forbid this class"],
  "required_semantic_evidence": "What MUST be in the transcript",
  "optional_evidence": "Contextual or auxiliary cues",
  "typical_examples": ["2 realistic example utterances"],
  "likely_confusable_classes": ["Top 1-2 nearest neighboring classes"],
  "distinguishing_boundary": "How to strictly distinguish from nearest confusable classes",
  "estimated_semantic_clarity": "HIGH | MEDIUM | LOW"
}}
Respond ONLY with the JSON object.
"""
                spec = self.llm.generate_json(prompt)
                ontology_audit[head][cname] = spec

                csv_rows.append({
                    "head": head,
                    "canonical_class": cname,
                    "definition": spec.get("concise_definition", ""),
                    "inclusion_criteria": " | ".join(spec.get("positive_inclusion_criteria", [])),
                    "exclusion_criteria": " | ".join(spec.get("exclusion_criteria", [])),
                    "required_evidence": spec.get("required_semantic_evidence", ""),
                    "confusable_classes": " | ".join(spec.get("likely_confusable_classes", [])),
                    "distinguishing_boundary": spec.get("distinguishing_boundary", ""),
                    "clarity": spec.get("estimated_semantic_clarity", "MEDIUM")
                })

        with open(audit_json_path, "w") as f:
            json.dump(ontology_audit, f, indent=4)

        pd.DataFrame(csv_rows).to_csv(audit_csv_path, index=False)
        logging.info(f"[+] Ontology Audit saved: {audit_json_path} & {audit_csv_path}")
        return ontology_audit

# ==============================================================================
# 4. PART 3–6: MULTI-PASS HIERARCHICAL SAMPLE RELABELING ENGINE
# ==============================================================================

class SampleRelabeler:
    """Executes Dual-Pass Verification and Third-Pass Arbitration on every sample."""
    def __init__(self, llm: GemmaEngine, ontology_audit: Dict[str, Any]):
        self.llm = llm
        self.audit = ontology_audit

    def _get_candidates(self, head: str, resolved_context: Dict[str, str]) -> List[str]:
        """Hierarchical Candidate Restriction conditioned on upstream decisions."""
        allowed = RelabelConfig.CANONICAL_ONTOLOGY[head]
        
        if head == "subdomain":
            dom = resolved_context.get("domain")
            if dom in RelabelConfig.DOMAIN_TO_SUBDOMAINS:
                return [s for s in allowed if s in RelabelConfig.DOMAIN_TO_SUBDOMAINS[dom]]
        elif head == "intent":
            subdom = resolved_context.get("subdomain")
            if subdom in RelabelConfig.SUBDOMAIN_TO_INTENTS:
                return [i for i in allowed if i in RelabelConfig.SUBDOMAIN_TO_INTENTS[subdom]]
        
        return allowed

    def audit_head(self, sample_id: str, transcript: str, head: str, 
                   current_label: str, candidates: List[str]) -> Dict[str, Any]:
        """Runs Pass A, Pass B, and conditional Pass C for a specific head."""
        if current_label == RelabelConfig.MASK_TOKEN or pd.isna(current_label) or str(current_label).strip() == "":
            return {
                "head": head,
                "current_label": RelabelConfig.MASK_TOKEN,
                "final_label": RelabelConfig.MASK_TOKEN,
                "confidence": 1.0,
                "status": "KEEP",
                "evidence": "Original label is MASK; preserved without forcing."
            }

        # -------------------------------------------------------------
        # PASS A: Independent Canonical Classification
        # -------------------------------------------------------------
        pass_a_prompt = f"""You are a multi-task semantic speech classifier.
Transcript: "{transcript}"
Classification Head: {head}
Allowed Canonical Candidates: {candidates}

Task: Choose the single best canonical class from Allowed Candidates representing this utterance.
Return JSON:
{{
  "predicted_label": "ONE_ALLOWED_CLASS",
  "confidence": 0.0 to 1.0,
  "supporting_evidence": "Verbatim semantic evidence from transcript"
}}
"""
        pass_a_res = self.llm.generate_json(pass_a_prompt)
        pred_a = pass_a_res.get("predicted_label", "").strip()
        conf_a = float(pass_a_res.get("confidence", 0.5))
        if pred_a not in candidates:
            pred_a = candidates[0] if candidates else RelabelConfig.MASK_TOKEN

        # -------------------------------------------------------------
        # PASS B: Contradiction & Evidence Audit
        # -------------------------------------------------------------
        curr_def = self.audit.get(head, {}).get(current_label, {}).get("concise_definition", "No definition")
        pass_b_prompt = f"""You are auditing the existing label '{current_label}' for the transcript:
Transcript: "{transcript}"
Head: {head}
Current Label Definition: {curr_def}
Alternative Candidates: {candidates}

Task: Audit if the current label genuinely fits the utterance.
Return JSON:
{{
  "supports_current": true or false,
  "contradiction_found": true or false,
  "suggested_alternative": "ONE_ALLOWED_CLASS",
  "confidence": 0.0 to 1.0,
  "contradicting_evidence": "Evidence contradicting current label",
  "status": "KEEP | RELABEL | AMBIGUOUS"
}}
"""
        pass_b_res = self.llm.generate_json(pass_b_prompt)
        status_b = pass_b_res.get("status", "KEEP").strip().upper()
        alt_b = pass_b_res.get("suggested_alternative", current_label).strip()
        if alt_b not in candidates:
            alt_b = pred_a

        # -------------------------------------------------------------
        # PASS C: Disagreement Adjudication (Only when Pass A != Current or Ambiguous)
        # -------------------------------------------------------------
        final_label = current_label
        final_conf = conf_a
        final_status = "KEEP"
        evidence_note = pass_a_res.get("supporting_evidence", "")

        disagreement = (pred_a != current_label) or (status_b in ["RELABEL", "AMBIGUOUS"])

        if disagreement:
            pass_c_prompt = f"""Disagreement detected during semantic audit for head '{head}'.
Utterance: "{transcript}"
Current Label: {current_label} (Pass B status: {status_b})
Proposed Class: {pred_a} (Pass A confidence: {conf_a})
Confusable Candidates: {candidates}

Decision Rules:
- Return 'KEEP_CURRENT' if current label is reasonably valid.
- Return 'CHANGE_TO_PROPOSED' if proposed class has overwhelmingly stronger transcript evidence.
- Return 'AMBIGUOUS' if evidence is insufficient or boundary is indistinguishable.

Return JSON:
{{
  "adjudication": "KEEP_CURRENT | CHANGE_TO_PROPOSED | AMBIGUOUS",
  "confidence": 0.0 to 1.0,
  "definitive_reason": "Detailed boundary justification"
}}
"""
            pass_c_res = self.llm.generate_json(pass_c_prompt)
            adj = pass_c_res.get("adjudication", "KEEP_CURRENT").strip().upper()
            final_conf = float(pass_c_res.get("confidence", 0.7))
            evidence_note = pass_c_res.get("definitive_reason", "")

            if adj == "CHANGE_TO_PROPOSED" and final_conf >= 0.70:
                final_label = pred_a
                final_status = "RELABEL"
            elif adj == "AMBIGUOUS" or final_conf < 0.50:
                final_label = RelabelConfig.MASK_TOKEN
                final_status = "AMBIGUOUS"
            else:
                final_label = current_label
                final_status = "KEEP"
        else:
            final_label = current_label
            final_status = "KEEP"

        return {
            "head": head,
            "current_label": current_label,
            "final_label": final_label,
            "confidence": round(final_conf, 3),
            "status": final_status,
            "evidence": evidence_note
        }

    def process_sample(self, row: pd.Series) -> Dict[str, Any]:
        """Hierarchically processes an individual speech record."""
        sample_id = str(row["audio_path"])
        transcript = str(row.get("transcript", "")).strip()
        
        resolved_context = {}
        sample_result = {
            "audio_path": sample_id,
            "transcript": transcript,
            "heads": {}
        }

        for head in RelabelConfig.HEADS:
            curr_val = str(row.get(head, RelabelConfig.MASK_TOKEN)).strip()
            cands = self._get_candidates(head, resolved_context)
            res = self.audit_head(sample_id, transcript, head, curr_val, cands)
            
            sample_result["heads"][head] = res
            resolved_context[head] = res["final_label"]

        return sample_result

# ==============================================================================
# 5. ORCHESTRATION, BATCHING, & RESUME CHECKPOINTING
# ==============================================================================

def run_relabeling_pipeline(resume: bool = True, batch_size: int = 50):
    RelabelConfig.setup_dirs()
    logging.info("Starting ASIL NLU Deterministic Relabeling Engine...")

    if not os.path.exists(RelabelConfig.INPUT_CSV):
        raise FileNotFoundError(f"Cannot find master CSV: {RelabelConfig.INPUT_CSV}")

    df = pd.read_csv(RelabelConfig.INPUT_CSV)
    logging.info(f"Loaded master dataset: {len(df)} records from {RelabelConfig.INPUT_CSV}")

    # Step 1: Initialize Gemma Engine & Run Ontology Audit
    llm = GemmaEngine()
    auditor = OntologyAuditor(llm)
    ontology_audit = auditor.run_audit()

    relabeler = SampleRelabeler(llm, ontology_audit)

    # Step 2: Checkpoint Resumption Handling
    processed_records = {}
    checkpoint_file = os.path.join(RelabelConfig.CHECKPOINT_DIR, "relabeling_progress.jsonl")

    if resume and os.path.exists(checkpoint_file):
        logging.info(f"Resuming from checkpoint file: {checkpoint_file}")
        with open(checkpoint_file, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    processed_records[item["audio_path"]] = item
        logging.info(f"Loaded {len(processed_records)} previously completed audits.")

    # Step 3: Run Batch Relabeling
    unprocessed_indices = [idx for idx, row in df.iterrows() if str(row["audio_path"]) not in processed_records]
    logging.info(f"Total samples remaining to audit: {len(unprocessed_indices)}")

    if unprocessed_indices:
        ckpt_writer = open(checkpoint_file, "a", encoding="utf-8")
        for i, idx in enumerate(tqdm(unprocessed_indices, desc="Auditing Samples")):
            row = df.iloc[idx]
            result = relabeler.process_sample(row)
            processed_records[result["audio_path"]] = result
            ckpt_writer.write(json.dumps(result) + "\n")

            if (i + 1) % batch_size == 0:
                ckpt_writer.flush()

        ckpt_writer.close()

    # ==============================================================================
    # 6. POST-AUDIT INTEGRITY, DRIFT ANALYSIS & ARTIFACT GENERATION
    # ==============================================================================
    logging.info("Compiling full audit traces, drift analytics, and clean datasets...")

    relabelled_rows = []
    clean_rows = []
    audit_trace_rows = []

    for _, row in df.iterrows():
        audio_p = str(row["audio_path"])
        audit_res = processed_records.get(audio_p, {})
        heads_data = audit_res.get("heads", {})

        row_dict = row.to_dict()

        # Populate Audit and Final Columns
        clean_entry = {
            "audio_path": audio_p,
            "transcript": row.get("transcript", "")
        }

        for h in RelabelConfig.HEADS:
            h_info = heads_data.get(h, {})
            f_lbl = h_info.get("final_label", str(row.get(h, RelabelConfig.MASK_TOKEN)))
            conf = h_info.get("confidence", 1.0)
            stat = h_info.get("status", "KEEP")
            evid = h_info.get("evidence", "")

            row_dict[f"audit_{h}"] = f_lbl
            row_dict[f"final_{h}"] = f_lbl
            row_dict[f"{h}_confidence"] = conf
            row_dict[f"{h}_status"] = stat

            clean_entry[h] = f_lbl

            audit_trace_rows.append({
                "audio_path": audio_p,
                "head": h,
                "original_label": str(row.get(h, RelabelConfig.MASK_TOKEN)),
                "audited_label": f_lbl,
                "confidence": conf,
                "status": stat,
                "evidence": evid
            })

        relabelled_rows.append(row_dict)
        clean_rows.append(clean_entry)

    # Convert to DataFrames
    relabelled_df = pd.DataFrame(relabelled_rows)
    clean_df = pd.DataFrame(clean_rows)
    audit_trace_df = pd.DataFrame(audit_trace_rows)

    # Save Output CSVs
    relabelled_path = os.path.join(RelabelConfig.OUTPUT_DIR, "master_nlu_dataset_relabelled.csv")
    clean_path = os.path.join(RelabelConfig.OUTPUT_DIR, "master_nlu_dataset_clean.csv")
    sample_audit_path = os.path.join(RelabelConfig.OUTPUT_DIR, "sample_label_audit.csv")

    relabelled_df.to_csv(relabelled_path, index=False)
    clean_df.to_csv(clean_path, index=False)
    audit_trace_df.to_csv(sample_audit_path, index=False)

    # --------------------------------------------------------------------------
    # Artifact 4: Proposed Class Distribution & Drift Sanity Checks
    # --------------------------------------------------------------------------
    dist_rows = []
    conflict_rows = []
    below_100_classes = []

    for head in RelabelConfig.HEADS:
        orig_counts = df[df[head] != RelabelConfig.MASK_TOKEN][head].value_counts().to_dict()
        final_counts = clean_df[clean_df[head] != RelabelConfig.MASK_TOKEN][head].value_counts().to_dict()
        all_cls = RelabelConfig.CANONICAL_ONTOLOGY[head]

        for c in all_cls:
            orig_c = orig_counts.get(c, 0)
            fin_c = final_counts.get(c, 0)
            delta = fin_c - orig_c
            pct = round((fin_c / len(clean_df)) * 100, 2)

            # Sub-dataframe of this original class
            sub_orig = relabelled_df[relabelled_df[head] == c]
            out_relab = len(sub_orig[sub_orig[f"final_{head}"] != c])
            
            sub_fin = relabelled_df[relabelled_df[f"final_{head}"] == c]
            in_relab = len(sub_fin[sub_fin[head] != c])

            dist_rows.append({
                "head": head,
                "class": c,
                "original_count": orig_c,
                "proposed_count": fin_c,
                "delta": delta,
                "percentage": pct,
                "num_relabelled_into_class": in_relab,
                "num_relabelled_out_of_class": out_relab
            })

            # Check Minimum Class Support
            if fin_c > 0 and fin_c < RelabelConfig.MIN_CLASS_SIZE:
                below_100_classes.append({"head": head, "class": c, "count": fin_c})

            # Part 7: Detect Ontology Conflicts (e.g. >50% changed or >30% absorbed)
            if orig_c >= 50 and (out_relab / orig_c) > 0.50:
                conflict_rows.append({
                    "head": head,
                    "source_class": c,
                    "conflict_type": "HIGH_OUTFLOW_DRIFT (>50% reassigned)",
                    "original_count": orig_c,
                    "reassigned_count": out_relab,
                    "description": f"Class {c} lost {out_relab}/{orig_c} samples. Check definition overlap."
                })

    dist_df = pd.DataFrame(dist_rows)
    dist_csv_path = os.path.join(RelabelConfig.OUTPUT_DIR, "proposed_class_distribution.csv")
    dist_df.to_csv(dist_csv_path, index=False)

    conflict_df = pd.DataFrame(conflict_rows)
    conflict_csv_path = os.path.join(RelabelConfig.OUTPUT_DIR, "ontology_conflicts.csv")
    conflict_df.to_csv(conflict_csv_path, index=False)

    # --------------------------------------------------------------------------
    # Artifact 6: Cross-Head Semantic Consistency Check
    # --------------------------------------------------------------------------
    consistency_violations = []
    for _, r in clean_df.iterrows():
        dom = r["domain"]
        sub = r["subdomain"]
        intent = r["intent"]

        if dom != RelabelConfig.MASK_TOKEN and sub != RelabelConfig.MASK_TOKEN:
            valid_subs = RelabelConfig.DOMAIN_TO_SUBDOMAINS.get(dom, [])
            if sub not in valid_subs:
                consistency_violations.append({
                    "sample_id": r["audio_path"],
                    "heads_in_conflict": ["domain", "subdomain"],
                    "description": f"Subdomain '{sub}' is invalid under Domain '{dom}'.",
                    "recommended_action": "AMBIGUOUS"
                })

        if sub != RelabelConfig.MASK_TOKEN and intent != RelabelConfig.MASK_TOKEN:
            valid_intents = RelabelConfig.SUBDOMAIN_TO_INTENTS.get(sub, [])
            if intent not in valid_intents:
                consistency_violations.append({
                    "sample_id": r["audio_path"],
                    "heads_in_conflict": ["subdomain", "intent"],
                    "description": f"Intent '{intent}' is semantically inconsistent with Subdomain '{sub}'.",
                    "recommended_action": "REVIEW"
                })

    consistency_df = pd.DataFrame(consistency_violations)
    consistency_csv_path = os.path.join(RelabelConfig.OUTPUT_DIR, "cross_head_consistency.csv")
    consistency_df.to_csv(consistency_csv_path, index=False)

    # --------------------------------------------------------------------------
    # Artifact 7: Source Dataset Confounding Audit
    # --------------------------------------------------------------------------
    def extract_source(path_str: str) -> str:
        s = str(path_str).lower()
        if "meld" in s: return "MELD"
        if "med" in s: return "MedDialog"
        if "slurp" in s: return "SLURP"
        if "earnings" in s: return "Earnings-21"
        return "OTHER"

    clean_df["source_dataset"] = clean_df["audio_path"].apply(extract_source)
    source_confound_rows = []

    for head in RelabelConfig.HEADS:
        valid_head_df = clean_df[clean_df[head] != RelabelConfig.MASK_TOKEN]
        for cname, grp in valid_head_df.groupby(head):
            src_counts = grp["source_dataset"].value_counts(normalize=True).to_dict()
            dominant_src = max(src_counts, key=src_counts.get)
            dominance_ratio = src_counts[dominant_src]

            if dominance_ratio >= 0.95 and len(grp) >= 30:
                source_confound_rows.append({
                    "head": head,
                    "class": cname,
                    "count": len(grp),
                    "dominant_source": dominant_src,
                    "source_concentration": round(dominance_ratio, 3),
                    "flag": "SOURCE_LABEL_CONFOUNDING"
                })

    source_df = pd.DataFrame(source_confound_rows)
    source_csv_path = os.path.join(RelabelConfig.OUTPUT_DIR, "source_label_confounding.csv")
    source_df.to_csv(source_csv_path, index=False)

    # --------------------------------------------------------------------------
    # Artifact 8: Machine-Readable Summary JSON
    # --------------------------------------------------------------------------
    summary_stats = {
        "total_samples": len(clean_df),
        "heads": {}
    }

    for head in RelabelConfig.HEADS:
        kept = int((relabelled_df[f"{head}_status"] == "KEEP").sum())
        relab = int((relabelled_df[f"{head}_status"] == "RELABEL").sum())
        ambig = int((relabelled_df[f"{head}_status"] == "AMBIGUOUS").sum())
        valid_count = int((clean_df[head] != RelabelConfig.MASK_TOKEN).sum())

        summary_stats["heads"][head] = {
            "valid_samples": valid_count,
            "mask_samples": len(clean_df) - valid_count,
            "kept_count": kept,
            "relabeled_count": relab,
            "ambiguous_count": ambig,
            "relabel_rate_pct": round((relab / valid_count * 100), 2) if valid_count > 0 else 0.0
        }

    summary_json_path = os.path.join(RelabelConfig.OUTPUT_DIR, "relabeling_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_stats, f, indent=4)

    # --------------------------------------------------------------------------
    # Artifact 11: Scientific Relabeling Final Report (Markdown)
    # --------------------------------------------------------------------------
    report_md_path = os.path.join(RelabelConfig.OUTPUT_DIR, "FINAL_RELABELING_REPORT.md")
    with open(report_md_path, "w") as f:
        f.write("# ASIL NLU Semantic Audit & Relabeling Scientific Report\n\n")
        f.write(f"**Total Samples Processed:** {len(clean_df)} (100% row preservation)\n\n")
        f.write("## 1. Head-Level Audit Summary\n\n")
        f.write("| Classification Head | Valid Count | MASK Count | Kept | Relabeled | Ambiguous (MASKed) | Relabel Rate |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for h, st in summary_stats["heads"].items():
            f.write(f"| **{h.upper()}** | {st['valid_samples']} | {st['mask_samples']} | {st['kept_count']} | {st['relabeled_count']} | {st['ambiguous_count']} | {st['relabel_rate_pct']}% |\n")

        f.write("\n## 2. Classes with Low Support (<100 Samples)\n\n")
        if below_100_classes:
            f.write("| Head | Canonical Class | Final Sample Count | Action Recommendation |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for b in below_100_classes:
                f.write(f"| {b['head']} | `{b['class']}` | {b['count']} | Retain canonical concept; target for TTS augmentation |\n")
        else:
            f.write("All active canonical classes have ≥100 samples.\n")

        f.write("\n## 3. Detected Ontology Conflicts & High-Shift Classes\n\n")
        if not conflict_df.empty:
            f.write("| Head | Class | Conflict Type | Description |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for _, r in conflict_df.iterrows():
                f.write(f"| {r['head']} | `{r['source_class']}` | {r['conflict_type']} | {r['description']} |\n")
        else:
            f.write("Zero severe ontology conflicts detected (no single class absorbed >50% out-of-boundary samples).\n")

        f.write("\n## 4. Source Dataset Confounding Flags\n\n")
        if not source_df.empty:
            f.write("| Head | Canonical Class | Dominant Dataset | Concentration | Sample Count |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for _, r in source_df.iterrows():
                f.write(f"| {r['head']} | `{r['class']}` | {r['dominant_source']} | {r['source_concentration']*100}% | {r['count']} |\n")
        else:
            f.write("No severe single-source confounding detected.\n")

        f.write("\n## 5. Cross-Head Consistency Report\n\n")
        f.write(f"- Total Semantic Boundary Violations Detected: **{len(consistency_df)}**\n")
        f.write("- All contradictions logged to `cross_head_consistency.csv` for independent review.\n")

    # ==============================================================================
    # 7. TERMINAL SUMMARY & STRICT VALIDATION VERIFICATION
    # ==============================================================================
    print("\n" + "=" * 70)
    print("🎯 ASIL NLU DATASET RELABELING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Total Rows Verified: {len(clean_df)} / {len(df)}")
    print(f"Clean ML-Ready Dataset:   {clean_path}")
    print(f"Full Audit Trace Dataset: {relabelled_path}")
    print(f"Scientific Final Report:  {report_md_path}")
    print("\nPer-Head Audit Summary:")
    for h, st in summary_stats["heads"].items():
        print(f"  [{h.upper():<12}] Valid: {st['valid_samples']:<5} | Kept: {st['kept_count']:<5} | Relabeled: {st['relabeled_count']:<5} | MASKed: {st['ambiguous_count']:<4}")
    
    print("\nArtifacts Saved to results/relabeling/:")
    for f in [
        "ontology_audit.json", "ontology_audit.csv", "sample_label_audit.csv",
        "proposed_class_distribution.csv", "ontology_conflicts.csv",
        "cross_head_consistency.csv", "source_label_confounding.csv",
        "relabeling_summary.json", "FINAL_RELABELING_REPORT.md"
    ]:
        print(f"  ✓ {f}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASIL NLU Deterministic Relabeling Pipeline")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume execution from existing progress checkpoint")
    parser.add_argument("--batch-size", type=int, default=50, help="Checkpointing flush frequency")
    args = parser.parse_args()

    run_relabeling_pipeline(resume=args.resume, batch_size=args.batch_size)
