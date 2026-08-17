"""
Queries local Gemma model to propose semantic groupings and canonical classes.
Implements domain-specific rules (Emotion 7-class, Urgency 4-level, Intent scenario_action).
Validates JSON schemas and retries malformed outputs.
"""
import os
import re
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from ontology_config import OntologyConfig

class GemmaOntologyProposer:
    def __init__(self):
        print(f"[+] Loading Local Gemma from {OntologyConfig.GEMMA_PATH} on {OntologyConfig.DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(OntologyConfig.GEMMA_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            OntologyConfig.GEMMA_PATH,
            torch_dtype=OntologyConfig.DTYPE,
            device_map=OntologyConfig.DEVICE,
            attn_implementation="sdpa",
            trust_remote_code=True
        ).eval()

    def query_gemma(self, prompt: str, max_retries: int = 3) -> list:
        """Executes inference and strictly parses the JSON array response with retries."""
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(OntologyConfig.DEVICE)

        for attempt in range(max_retries):
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

            # Attempt JSON array parsing
            try:
                match = re.search(r"\[.*\]", text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, list):
                        return parsed
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except Exception as e:
                print(f"[!] JSON parse retry {attempt+1}/{max_retries} due to: {e}")
        
        return []

    def get_head_instructions(self, head: str) -> str:
        """Domain-specific constraints passed to Gemma."""
        if head == "emotion":
            return f"""SPECIAL RULE FOR EMOTION:
You must ONLY normalize labels into these EXACT 7 canonical classes: {OntologyConfig.LOCKED_EMOTIONS}.
Do NOT merge any of these 7 emotions together. Every member must map to one of these 7 classes."""
        elif head == "urgency":
            return f"""SPECIAL RULE FOR URGENCY:
You must ONLY normalize labels into these EXACT 4 severity levels: {OntologyConfig.LOCKED_URGENCIES}.
Do NOT collapse severity levels (e.g. HIGH != CRITICAL, MEDIUM != HIGH)."""
        elif head == "intent":
            return """SPECIAL RULE FOR INTENT:
Preserve both SCENARIO and ACTION (e.g., SYMPTOM_REPORT vs MEDICATION_QUERY).
Do NOT merge distinct operational actions into broad generic buckets."""
        elif head == "domain":
            return "Domain represents the top-level broad context (e.g., MEDICAL, FINANCE, GENERAL). Keep high-level boundaries clear."
        elif head == "subdomain":
            return "Subdomain represents specialized domain sectors (e.g., CARDIOLOGY, HEALTHCARE, TECHNOLOGY, INDUSTRIAL_GOODS, VOICE_COMMAND)."
        elif head == "entity_type":
            return "Entity Type represents the specific informational entity being discussed. Only merge true synonyms."
        return ""

    def propose_ontology_for_head(self, head: str):
        norm_csv = os.path.join(OntologyConfig.NORM_STATS_DIR, f"{head}.csv")
        if not os.path.exists(norm_csv):
            print(f"[-] Normalized CSV missing for {head}. Run extract_raw_statistics.py first.")
            return

        df = pd.read_csv(norm_csv)
        if df.empty:
            print(f"[-] No valid records for {head}.")
            return

        # Prepare compact payload for Gemma
        payload_items = []
        for _, row in df.iterrows():
            payload_items.append({
                "normalized_label": row["normalized_label"],
                "original_labels": json.loads(row["original_labels"]),
                "frequency": int(row["total_frequency"]),
                "source_datasets": json.loads(row["source_datasets"]),
                "examples": json.loads(row["example_transcripts"])
            })

        rule_instruction = self.get_head_instructions(head)

        prompt = f"""You are an expert NLP ontology architect.
Your task is to analyze candidate labels for the classification head: '{head}' and propose a finite, canonical ontology.

{rule_instruction}

GUIDELINES:
1. Propose meaningful canonical class names in UPPER_SNAKE_CASE.
2. Group ONLY genuinely semantically equivalent labels together. Do NOT merge merely related concepts.
3. Frequency is provided for context only; do not merge rare labels simply because they are rare.
4. If a label is ambiguous or contradictory, set decision to "REVIEW_REQUIRED".
5. Every input 'normalized_label' must be accounted for in exactly one proposed class members list.

INPUT LABELS WITH EXAMPLES:
{json.dumps(payload_items, indent=2)}

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects with the exact schema:
[
  {{
    "canonical_label": "CANONICAL_NAME",
    "definition": "Short concise definition of this category",
    "members": ["NORM_LABEL_1", "NORM_LABEL_2"],
    "reason": "Clear linguistic/semantic rationale for grouping",
    "confidence": 0.95,
    "decision": "MERGE" or "KEEP" or "REVIEW_REQUIRED"
  }}
]
"""
        print(f"\n[+] Requesting Gemma ontology proposals for '{head}' ({len(payload_items)} labels)...")
        proposals = self.query_gemma(prompt)

        # Fallback / Verification check: ensure all labels are accounted for
        accounted_members = set()
        for p in proposals:
            for m in p.get("members", []):
                accounted_members.add(m)

        all_norm_labels = set(df["normalized_label"].tolist())
        missing_labels = all_norm_labels - accounted_members

        # If Gemma missed any label, create explicit fallback singletons for human review
        if missing_labels:
            print(f"[!] Warning: Gemma missed {len(missing_labels)} labels for {head}. Adding as REVIEW_REQUIRED singletons.")
            for missing in missing_labels:
                proposals.append({
                    "canonical_label": missing,
                    "definition": f"Self-mapped category for {missing}",
                    "members": [missing],
                    "reason": "Auto-added singleton because not grouped by LLM proposal",
                    "confidence": 0.50,
                    "decision": "REVIEW_REQUIRED"
                })

        # Enrich proposals with frequencies and metadata
        norm_dict = df.set_index("normalized_label").to_dict("index")
        enriched_proposals = []

        for prop in proposals:
            canon = prop["canonical_label"]
            members = prop.get("members", [])
            
            tot_freq = sum(norm_dict[m]["total_frequency"] for m in members if m in norm_dict)
            orig_labels_grouped = []
            sources_grouped = set()
            examples_grouped = []
            member_freq_breakdown = {}

            for m in members:
                if m in norm_dict:
                    orig_list = json.loads(norm_dict[m]["original_labels"])
                    orig_labels_grouped.extend(orig_list)
                    sources_grouped.update(json.loads(norm_dict[m]["source_datasets"]))
                    examples_grouped.extend(json.loads(norm_dict[m]["example_transcripts"]))
                    m_freqs = json.loads(norm_dict[m]["member_frequencies"])
                    member_freq_breakdown.update(m_freqs)

            enriched_proposals.append({
                "canonical_label": canon,
                "definition": prop.get("definition", "No definition provided"),
                "normalized_members": members,
                "original_labels": orig_labels_grouped,
                "member_frequencies": member_freq_breakdown,
                "total_frequency": tot_freq,
                "source_datasets": sorted(list(sources_grouped)),
                "representative_examples": examples_grouped[:5],
                "gemma_confidence": prop.get("confidence", 1.0),
                "reason": prop.get("reason", "Direct mapping"),
                "decision": prop.get("decision", "KEEP")
            })

        out_json = os.path.join(OntologyConfig.PROPOSED_DIR, f"{head}.json")
        with open(out_json, "w") as f:
            json.dump(enriched_proposals, f, indent=4)
        print(f"[+] Saved proposed ontology for '{head}' ({len(enriched_proposals)} canonical classes) to {out_json}")

def run_all_proposals():
    proposer = GemmaOntologyProposer()
    for head in OntologyConfig.HEADS:
        proposer.propose_ontology_for_head(head)

if __name__ == "__main__":
    run_all_proposals()
