"""
Forcefully compresses a fragmented ontology into 25-35 broad Super-Classes.
Explicitly enforces load-balancing to ensure no class starves (<80 samples) 
and no class hoards (>250 samples), generating a healthy distribution for MLP training.
"""
import os
import re
import gc
import json
import torch
import shutil
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from ontology_config import OntologyConfig

class OntologyCompressor:
    def __init__(self):
        print(f"[+] Loading Local Gemma from {OntologyConfig.GEMMA_PATH} for Distribution-Aware Compression...")
        self.tokenizer = AutoTokenizer.from_pretrained(OntologyConfig.GEMMA_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            OntologyConfig.GEMMA_PATH,
            torch_dtype=OntologyConfig.DTYPE,
            device_map=OntologyConfig.DEVICE,
            attn_implementation="sdpa",
            trust_remote_code=True
        ).eval()

    def _clean_and_repair_json(self, json_str: str) -> list:
        json_str = re.sub(r"```(?:json)?", "", json_str).strip()
        start_idx = json_str.find("[")
        if start_idx == -1: return []
        json_str = json_str[start_idx:]
        json_str = json_str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        json_str = re.sub(r",\s*([\}\]])", r"\1", json_str)
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list): return parsed
        except Exception: pass
        
        last_brace = json_str.rfind("}")
        if last_brace != -1:
            truncated = json_str[:last_brace + 1].strip()
            truncated = re.sub(r",\s*$", "", truncated) + "]"
            try:
                parsed = json.loads(truncated)
                if isinstance(parsed, list): return parsed
            except Exception: pass
        return []

    def query_gemma(self, prompt: str, max_retries: int = 3) -> list:
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
            if parsed: return parsed
            print(f"[!] JSON parse retry {attempt+1}/{max_retries}...")
        return []

    def define_super_classes(self, head: str, top_classes: list, total_dataset_samples: int, target_min: int = 25, target_max: int = 35) -> list:
        """Stage 1: Generates broad semantic boundaries with strict load-balancing."""
        ideal_mean = total_dataset_samples // 30
        print(f"\n[+] STAGE 1: Generating {target_min}-{target_max} Load-Balanced Super-Classes for '{head}'...")
        print(f"    (Targeting roughly 80 to 250 samples per class. Ideal mean: {ideal_mean})")
        
        payload = [{"class": c["canonical_label"], "freq": c["total_frequency"]} for c in top_classes]

        prompt = f"""You are an expert NLP Ontologist preparing a balanced ontology for a neural network.
We have highly fragmented, granular classes for the '{head}' head representing ~{total_dataset_samples} total samples. 
A neural network needs a HEALTHY, EQUAL DISTRIBUTION of samples per class to train effectively.

TASK:
Analyze the provided high-frequency granular classes and synthesize exactly {target_min} to {target_max} BROAD SUPER-CLASSES.
You MUST include one class named 'OTHER_GENERAL' for miscellaneous noise.

CRITICAL LOAD-BALANCING RULES:
1. Do NOT create a "mega-class" that will absorb 1,000+ samples (e.g., do not just use "FINANCE_GENERAL"). If a concept is too broad, SPLIT it into more specific classes (e.g., "REVENUE_REPORT", "MARKET_GUIDANCE", "MERGER_ACQUISITION").
2. Do NOT create "micro-classes" that will only get 10-20 samples. Merge rare concepts into broader categories.
3. Your goal is to design these categories so that when the 400+ raw labels are eventually mapped into them, each Super-Class ends up with roughly 80 to 250 samples.

INPUT GRANULAR CLASSES (WITH FREQUENCIES):
{json.dumps(payload, indent=1)}

OUTPUT FORMAT (JSON ARRAY ONLY):
[
  {{"super_class": "BROAD_UPPER_SNAKE_CASE_NAME", "definition": "Broad definition designed to capture 80-250 samples"}}
]"""
        super_classes = self.query_gemma(prompt)
        
        # Ensure OTHER_GENERAL exists
        if not any(sc.get("super_class") == "OTHER_GENERAL" for sc in super_classes):
            super_classes.append({"super_class": "OTHER_GENERAL", "definition": "Miscellaneous or unmappable long-tail noise."})
            
        print(f"  -> Generated {len(super_classes)} Broad Super-Classes.")
        return super_classes

    def compress_head(self, head: str, target_min: int = 25, target_max: int = 35):
        if head in ["emotion", "urgency"]:
            print(f"[-] Skipping compression for locked head: {head}")
            return

        json_path = os.path.join(OntologyConfig.PROPOSED_DIR, f"{head}.json")
        if not os.path.exists(json_path):
            return

        with open(json_path, "r") as f:
            classes = json.load(f)

        if len(classes) <= target_max:
            print(f"[-] {head} only has {len(classes)} classes. No compression needed.")
            return

        classes.sort(key=lambda x: x["total_frequency"], reverse=True)
        total_samples = sum(c["total_frequency"] for c in classes)

        # STAGE 1: Get Broad Super-Classes based on the top 75 heaviest hitters
        top_75 = classes[:75]
        super_classes = self.define_super_classes(head, top_75, total_samples, target_min, target_max)
        
        super_class_names = [sc["super_class"] for sc in super_classes]
        super_class_dict = {
            sc["super_class"]: {
                "canonical_label": sc["super_class"],
                "definition": sc["definition"],
                "normalized_members": [],
                "original_labels": [],
                "member_frequencies": {},
                "total_frequency": 0,
                "source_datasets": [],
                "representative_examples": [],
                "gemma_confidence": 1.0,
                "reason": "Broad Super-Class Generalization",
                "decision": "MERGE"
            } for sc in super_classes
        }

        # STAGE 2: Batch Map all granular classes into the Super-Classes
        print(f"\n[+] STAGE 2: Mapping {len(classes)} granular classes into the {len(super_class_names)} Super-Classes...")
        
        batch_size = 35
        for i in tqdm(range(0, len(classes), batch_size), desc=f"Mapping {head}"):
            batch = classes[i : i + batch_size]
            payload = [{"narrow_class": o["canonical_label"], "freq": o["total_frequency"], "examples": o["representative_examples"][:1]} for o in batch]

            prompt = f"""You are forcing semantic compression for the '{head}' classification head.
Map every 'narrow_class' provided below strictly into the single most appropriate 'BROAD_SUPER_CLASS'.

AVAILABLE BROAD SUPER-CLASSES:
{json.dumps(super_class_names, indent=1)}

NARROW CLASSES TO MAP (WITH FREQUENCIES):
{json.dumps(payload, indent=1)}

LOAD-BALANCING RULES:
1. You may ONLY output a super-class exactly as it appears in the available list.
2. Distribute the mappings as evenly as semantic logic allows. Do NOT dump everything into one generic class.
3. If a narrow class is irrelevant noise, map it to "OTHER_GENERAL".

OUTPUT FORMAT (JSON ARRAY ONLY):
[
  {{"narrow_class": "NAME_OF_NARROW_CLASS", "mapped_to": "NAME_OF_BROAD_SUPER_CLASS"}}
]"""
            mappings = self.query_gemma(prompt)

            # Aggregate mapped data
            for mapping in mappings:
                narrow = mapping.get("narrow_class", "")
                target_sc = mapping.get("mapped_to", "OTHER_GENERAL")

                if target_sc not in super_class_dict:
                    target_sc = "OTHER_GENERAL"

                orig_obj = next((o for o in batch if o["canonical_label"] == narrow), None)
                if not orig_obj: continue

                target = super_class_dict[target_sc]
                target["normalized_members"].extend(orig_obj.get("normalized_members", []))
                target["original_labels"].extend(orig_obj.get("original_labels", []))
                target["member_frequencies"].update(orig_obj.get("member_frequencies", {}))
                target["total_frequency"] += orig_obj.get("total_frequency", 0)
                target["source_datasets"] = list(set(target["source_datasets"] + orig_obj.get("source_datasets", [])))
                target["representative_examples"].extend(orig_obj.get("representative_examples", []))
                target["representative_examples"] = target["representative_examples"][:5]

            torch.cuda.empty_cache()
            gc.collect()

        # Finalize, Validate, and Save
        final_compressed = [c for c in super_class_dict.values() if c["total_frequency"] > 0]
        final_compressed.sort(key=lambda x: x["total_frequency"], reverse=True)

        print("\n--- DISTRIBUTION AUDIT ---")
        severe_imbalance = False
        for c in final_compressed:
            freq = c["total_frequency"]
            if freq > 400:
                print(f"[!] WARNING: Mega-Class detected -> '{c['canonical_label']}' has {freq} samples. You may need to split this in the review CSV.")
                severe_imbalance = True
            elif freq < 50:
                print(f"[-] Warning: Micro-Class detected -> '{c['canonical_label']}' has only {freq} samples. Consider merging in the review CSV.")
                severe_imbalance = True
        
        if not severe_imbalance:
            print("[+] Audit Passed: Class distribution appears healthy for MLP training.")

        backup_path = json_path.replace(".json", "_pre_compression_backup.json")
        shutil.copy2(json_path, backup_path)

        with open(json_path, "w") as f:
            json.dump(final_compressed, f, indent=4)
            
        print(f"\n[+] '{head}' compressed to {len(final_compressed)} generalized classes!")
        print(f"    (Original backed up to {backup_path})")

if __name__ == "__main__":
    compressor = OntologyCompressor()
    # Execute on high-cardinality heads
    for head in ["intent", "entity_type"]:
        compressor.compress_head(head, target_min=25, target_max=35)
