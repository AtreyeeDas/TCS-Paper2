"""
Queries local Gemma model to propose semantic groupings and canonical classes.
Implements Two-Pass Hierarchical Clustering (Map-Reduce) for high-cardinality heads:
- Pass 1 (Map): Batch-wise local grouping (35 labels/batch)
- Pass 2 (Reduce): Global cross-batch consolidation
- Pass 3: Deterministic code aggregation and duplicate merging
"""
import os
import re
import gc
import json
import torch
import pandas as pd
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

    def _clean_and_repair_json(self, json_str: str) -> list:
        """Repairs markdown blocks, trailing commas, and token limit truncations."""
        json_str = re.sub(r"```(?:json)?", "", json_str).strip()
        start_idx = json_str.find("[")
        if start_idx == -1:
            return []
        json_str = json_str[start_idx:]
        json_str = json_str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        json_str = re.sub(r",\s*([\}\]])", r"\1", json_str)

        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        # Repair truncated JSON if cut off
        last_brace = json_str.rfind("}")
        if last_brace != -1:
            truncated = json_str[:last_brace + 1].strip()
            truncated = re.sub(r",\s*$", "", truncated) + "]"
            try:
                parsed = json.loads(truncated)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return []

    def query_gemma(self, prompt: str, max_retries: int = 2) -> list:
        """Executes inference and strictly parses JSON output with retries."""
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
            parsed = self._clean_and_repair_json(text)
            if parsed:
                return parsed
            print(f"[!] JSON parse retry {attempt+1}/{max_retries}...")
        return []

    def get_head_instructions(self, head: str) -> str:
        if head == "emotion":
            return f"ONLY normalize into these 7 classes: {OntologyConfig.LOCKED_EMOTIONS}. Do not merge them."
        elif head == "urgency":
            return f"ONLY normalize into these 4 severity levels: {OntologyConfig.LOCKED_URGENCIES}."
        elif head == "intent":
            return "Preserve SCENARIO and ACTION (e.g., SYMPTOM_REPORT vs MEDICATION_QUERY). Do not over-generalize."
        elif head == "domain":
            return "Domain represents the top-level broad context (MEDICAL, FINANCE, GENERAL)."
        elif head == "subdomain":
            return "Subdomain represents specialized sectors (CARDIOLOGY, HEALTHCARE, TECHNOLOGY, etc.)."
        elif head == "entity_type":
            return "Entity Type represents the specific informational subject. Only merge true synonyms."
        return ""

    def _process_batch(self, items: list, head: str, rule_instruction: str) -> list:
        compact_items = [
            {
                "label": it["normalized_label"],
                "frequency": it["frequency"],
                "sources": it["source_datasets"],
                "samples": [ex[:70] for ex in it.get("examples", [])[:2]]
            }
            for it in items
        ]

        prompt = f"""You are an expert NLP ontology architect. Group candidate labels for '{head}' into canonical classes.
{rule_instruction}
1. Propose canonical names in UPPER_SNAKE_CASE.
2. Group ONLY genuinely semantically equivalent labels together.
3. Every input 'label' must appear in exactly one 'members' list.

INPUT:
{json.dumps(compact_items, indent=1)}

OUTPUT JSON ARRAY ONLY:
[
  {{
    "canonical_label": "CANONICAL_NAME",
    "definition": "Short definition",
    "members": ["LABEL_1", "LABEL_2"],
    "reason": "Semantic rationale",
    "confidence": 0.95,
    "decision": "MERGE"
  }}
]"""
        return self.query_gemma(prompt)

    def _reconcile_cross_batch_clusters(self, intermediate_clusters: list, head: str, rule_instruction: str) -> list:
        """Pass 2 (Reduce): Feeds all intermediate candidate classes back to Gemma to correlate cross-batch synonyms."""
        print(f"\n[+] Running Pass 2 (Global Reconciliation) across {len(intermediate_clusters)} intermediate clusters...")
        
        # Prepare lightweight summary of intermediate classes
        summary_payload = [
            {
                "candidate_canonical": c["canonical_label"],
                "definition": c["definition"],
                "raw_members_count": len(c["members"])
            }
            for c in intermediate_clusters
        ]

        prompt = f"""You are consolidating intermediate ontology clusters for the head: '{head}'.
Some clusters were generated in separate batches and may be identical or semantically synonymous across batches.

{rule_instruction}

TASK:
Review these intermediate candidate classes and merge any synonymous clusters formed across batches into a final consolidated ontology.

INPUT CANDIDATES:
{json.dumps(summary_payload, indent=1)}

OUTPUT FORMAT (JSON ARRAY ONLY):
[
  {{
    "canonical_label": "FINAL_CONSOLIDATED_NAME",
    "definition": "Consolidated definition",
    "merged_candidates": ["CANDIDATE_1", "CANDIDATE_2"],
    "reason": "Cross-batch correlation rationale"
  }}
]"""
        reconciled = self.query_gemma(prompt)
        if not reconciled:
            print("[!] Pass 2 reconciliation fallback: keeping intermediate clusters as-is.")
            return intermediate_clusters

        # Map intermediate clusters to final reconciled classes
        intermediate_map = {c["canonical_label"]: c for c in intermediate_clusters}
        final_clusters = []
        accounted_intermediates = set()

        for group in reconciled:
            final_canon = group.get("canonical_label", "").strip().upper()
            merged_cands = group.get("merged_candidates", [])
            
            combined_members = []
            for cand in merged_cands:
                if cand in intermediate_map:
                    combined_members.extend(intermediate_map[cand]["members"])
                    accounted_intermediates.add(cand)

            if combined_members:
                final_clusters.append({
                    "canonical_label": final_canon,
                    "definition": group.get("definition", "Consolidated category"),
                    "members": list(set(combined_members)),
                    "reason": group.get("reason", "Global cross-batch consolidation"),
                    "confidence": 0.95,
                    "decision": "MERGE"
                })

        # Add any intermediate clusters that were missed in Pass 2
        for cand, c_obj in intermediate_map.items():
            if cand not in accounted_intermediates:
                final_clusters.append(c_obj)

        return final_clusters

    def propose_ontology_for_head(self, head: str, batch_size: int = 35):
        norm_csv = os.path.join(OntologyConfig.NORM_STATS_DIR, f"{head}.csv")
        if not os.path.exists(norm_csv):
            print(f"[-] Normalized CSV missing for {head}. Run extract_raw_statistics.py first.")
            return

        df = pd.read_csv(norm_csv)
        if df.empty:
            print(f"[-] No valid records for {head}.")
            return

        payload_items = [
            {
                "normalized_label": row["normalized_label"],
                "original_labels": json.loads(row["original_labels"]),
                "frequency": int(row["total_frequency"]),
                "source_datasets": json.loads(row["source_datasets"]),
                "examples": json.loads(row["example_transcripts"])
            }
            for _, row in df.iterrows()
        ]

        rule_instruction = self.get_head_instructions(head)
        total_items = len(payload_items)
        intermediate_proposals = []

        print(f"\n[+] Processing '{head}' ({total_items} labels in batches of {batch_size})...")

        # --- PASS 1: MAP (Local Mini-Batching) ---
        for i in range(0, total_items, batch_size):
            batch = payload_items[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_items + batch_size - 1) // batch_size
            
            print(f"  -> Pass 1: Batch {batch_num}/{total_batches} ({len(batch)} labels)...")
            batch_res = self._process_batch(batch, head, rule_instruction)

            if batch_res:
                intermediate_proposals.extend(batch_res)
            else:
                for b_item in batch:
                    intermediate_proposals.append({
                        "canonical_label": b_item["normalized_label"],
                        "definition": f"Self-mapped category for {b_item['normalized_label']}",
                        "members": [b_item["normalized_label"]],
                        "reason": "Batch parsing fallback",
                        "confidence": 0.50,
                        "decision": "REVIEW_REQUIRED"
                    })

            torch.cuda.empty_cache()
            gc.collect()

        # --- PASS 2: REDUCE (Global Cross-Batch Reconciliation) ---
        if len(intermediate_proposals) > 20:
            final_raw_proposals = self._reconcile_cross_batch_clusters(intermediate_proposals, head, rule_instruction)
        else:
            final_raw_proposals = intermediate_proposals

        # --- PASS 3: DETERMINISTIC AGGREGATION & ENRICHMENT ---
        # Ensure 100% label accountability
        accounted_members = {m for p in final_raw_proposals for m in p.get("members", [])}
        all_norm_labels = set(df["normalized_label"].tolist())
        missing_labels = all_norm_labels - accounted_members

        for missing in missing_labels:
            final_raw_proposals.append({
                "canonical_label": missing,
                "definition": f"Self-mapped category for {missing}",
                "members": [missing],
                "reason": "Unaccounted member fallback",
                "confidence": 0.50,
                "decision": "REVIEW_REQUIRED"
            })

        # Merge clusters that share the exact same canonical_label
        merged_by_canonical = {}
        norm_dict = df.set_index("normalized_label").to_dict("index")

        for prop in final_raw_proposals:
            canon = prop.get("canonical_label", "UNKNOWN").strip().upper()
            members = prop.get("members", [])
            if not members:
                continue

            if canon not in merged_by_canonical:
                merged_by_canonical[canon] = {
                    "canonical_label": canon,
                    "definition": prop.get("definition", "No definition"),
                    "normalized_members": set(),
                    "original_labels": [],
                    "member_frequencies": {},
                    "total_frequency": 0,
                    "source_datasets": set(),
                    "representative_examples": [],
                    "gemma_confidence": prop.get("confidence", 1.0),
                    "reason": prop.get("reason", "Direct mapping"),
                    "decision": prop.get("decision", "KEEP")
                }

            entry = merged_by_canonical[canon]
            for m in members:
                if m in norm_dict and m not in entry["normalized_members"]:
                    entry["normalized_members"].add(m)
                    entry["original_labels"].extend(json.loads(norm_dict[m]["original_labels"]))
                    entry["source_datasets"].update(json.loads(norm_dict[m]["source_datasets"]))
                    entry["representative_examples"].extend(json.loads(norm_dict[m]["example_transcripts"]))
                    entry["member_frequencies"].update(json.loads(norm_dict[m]["member_frequencies"]))
                    entry["total_frequency"] += norm_dict[m]["total_frequency"]

        # Final serialization list
        enriched_proposals = []
        for canon, data in merged_by_canonical.items():
            enriched_proposals.append({
                "canonical_label": canon,
                "definition": data["definition"],
                "normalized_members": list(data["normalized_members"]),
                "original_labels": sorted(list(set(data["original_labels"]))),
                "member_frequencies": data["member_frequencies"],
                "total_frequency": data["total_frequency"],
                "source_datasets": sorted(list(data["source_datasets"])),
                "representative_examples": data["representative_examples"][:3],
                "gemma_confidence": data["gemma_confidence"],
                "reason": data["reason"],
                "decision": data["decision"]
            })

        enriched_proposals.sort(key=lambda x: x["total_frequency"], reverse=True)
        out_json = os.path.join(OntologyConfig.PROPOSED_DIR, f"{head}.json")
        with open(out_json, "w") as f:
            json.dump(enriched_proposals, f, indent=4)
        print(f"[+] Saved fully reconciled ontology for '{head}' ({len(enriched_proposals)} canonical classes) to {out_json}")

def run_all_proposals():
    proposer = GemmaOntologyProposer()
    for head in OntologyConfig.HEADS:
        proposer.propose_ontology_for_head(head)

if __name__ == "__main__":
    run_all_proposals()

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

            # Attempt JSON array parsing with robust cleaning
            try:
                clean_text = self._clean_json_string(text)
                
                # Extract the array block if there's conversational wrapper text
                match = re.search(r"\[.*\]", clean_text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, list):
                        return parsed
                        
                # Fallback: try parsing the whole cleaned string
                parsed = json.loads(clean_text)
                if isinstance(parsed, list):
                    return parsed
                    
            except Exception as e:
                print(f"[!] JSON parse retry {attempt+1}/{max_retries} due to: {e}")
                if attempt == max_retries - 1:
                    print(f"[-] Final failed raw output snippet: {text[:300]}...")
        
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
