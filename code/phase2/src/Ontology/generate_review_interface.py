"""
Converts proposed ontologies into human-reviewable CSV tables (Class-Level Only).
Human reviewers edit the CSV (approving, renaming, or rejecting proposed classes)
without needing to inspect 4,000 rows.
"""
import os
import json
import pandas as pd
from ontology_config import OntologyConfig

def generate_review_tables():
    OntologyConfig.make_dirs()
    print("[+] Generating Class-Level Review Files in results/ontology/final_class_review/...")

    for head in OntologyConfig.HEADS:
        json_path = os.path.join(OntologyConfig.PROPOSED_DIR, f"{head}.json")
        if not os.path.exists(json_path):
            print(f"[-] Proposals not found for {head}. Skipping.")
            continue

        with open(json_path, "r") as f:
            proposals = json.load(f)

        review_rows = []
        for prop in proposals:
            review_rows.append({
                "canonical_label": prop["canonical_label"],
                "definition": prop["definition"],
                "original_labels": " | ".join(prop["original_labels"]),
                "member_frequencies": json.dumps(prop["member_frequencies"]),
                "total_frequency": prop["total_frequency"],
                "source_datasets": "|".join(prop["source_datasets"]),
                "representative_examples": " || ".join(prop["representative_examples"]),
                "gemma_confidence": prop["gemma_confidence"],
                "gemma_reason": prop["reason"],
                "review_status": "APPROVE" if prop["decision"] in ["MERGE", "KEEP"] else "REVIEW_REQUIRED",
                "human_notes": ""
            })

        review_df = pd.DataFrame(review_rows).sort_values(by="total_frequency", ascending=False)
        out_csv = os.path.join(OntologyConfig.REVIEW_DIR, f"{head}_review.csv")
        review_df.to_csv(out_csv, index=False)
        print(f"[+] Created review interface for '{head}': {out_csv} ({len(review_df)} proposed classes)")

    # Generate Markdown instructions
    guide_md = os.path.join(OntologyConfig.REVIEW_DIR, "REVIEW_INSTRUCTIONS.md")
    with open(guide_md, "w") as f:
        f.write("""# ASIL Final Ontology Human Review Instructions

Review the generated `{head}_review.csv` files in this directory. 
Each row represents **ONE CANONICAL CLASS**, not an individual dataset row.

### How to Review:
1. Open the CSV file (in Excel, VS Code, or text editor).
2. Inspect `canonical_label`, `definition`, and `original_labels`.
3. If satisfied, leave `review_status` as **`APPROVE`**.
4. If you want to rename a class, edit the text in `canonical_label` directly and set `review_status` to **`APPROVE`**.
5. If you want to merge two proposed classes, give them the exact same `canonical_label` and set `review_status` to **`APPROVE`**.
6. If a proposed group is invalid, set `review_status` to **`REJECT`** or **`SPLIT`**.
7. Once finished, run `lock_and_apply_ontology.py` to compile `ontology_v1.json` and automatically remap the entire master dataset.
""")
    print(f"[+] Review guide saved to: {guide_md}")

if __name__ == "__main__":
    generate_review_tables()
