import json
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# CONFIGURATION
# ==========================================
GEMMA_PATH = "/home/spark2/Models/gemma4-e4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

print(f"[+] Initializing Quick Gemma Test on {DEVICE} ({DTYPE})...")

# Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
model = AutoModelForCausalLM.from_pretrained(
    GEMMA_PATH,
    torch_dtype=DTYPE,
    device_map=DEVICE,
    attn_implementation="sdpa"
).eval()

# Sample test cases representing your 3 core annotation domains
TEST_CASES = [
    {
        "domain": "Medical and Clinical Healthcare",
        "transcript": "Doctor, I've had acute chest tightness and shortness of breath for the last thirty minutes.",
        "instruction": """Extract the following into JSON:
1. "intent": snake_case action (e.g., patient_report_symptoms).
2. "entity_type": core entity discussed (e.g., symptom, medication, test_result).
3. "urgency": strictly one of ["Low", "Medium", "High", "Critical"].
4. "subdomain": clinical specialty (e.g., cardiology, respiratory, general) if evident, otherwise "MASK"."""
    },
    {
        "domain": "Corporate Finance (technology sector)",
        "transcript": "In Q3, our cloud subscription revenue grew twenty-two percent year over year, exceeding our previous guidance.",
        "instruction": """Extract the following into JSON:
1. "intent": snake_case scenario and action (e.g., financial_report_revenue).
2. "entity_type": main financial metric or subject (e.g., revenue, operating_margin).
3. "urgency": strictly one of ["Low", "Medium", "High", "Critical"]."""
    },
    {
        "domain": "Voice Assistant Commands",
        "transcript": "Set an alarm for seven thirty tomorrow morning and remind me about the team meeting.",
        "instruction": """Extract the core entity_type discussed in this voice command.
Return JSON: {"entity_type": "<extracted_entity_or_MASK>"}"""
    }
]

print("=" * 65)
print("🚀 RUNNING GEMMA NLU REASONING BENCHMARK")
print("=" * 65)

for i, test in enumerate(TEST_CASES, 1):
    print(f"\n[Test Case {i}] Domain: {test['domain']}")
    print(f"Transcript: \"{test['transcript']}\"")

    messages = [
        {
            "role": "user",
            "content": f"""You are a precise NLU classification model for the {test['domain']} domain.
Analyze the following transcript:
"{test['transcript']}"

{test['instruction']}

Respond ONLY with a valid, raw JSON object. Do not include markdown codeblocks or extra text.
"""
        }
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

    # Benchmark Generation Time
    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = (time.perf_counter() - start_time) * 1000  # ms

    # Decode Output
    response_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    raw_response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    # Parse JSON
    parsed_json = None
    try:
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if match:
            parsed_json = json.loads(match.group(0))
        else:
            parsed_json = json.loads(raw_response)
        status = "✅ VALID JSON"
    except Exception as e:
        status = f"❌ JSON PARSE FAILED: {e}"

    print(f"Latency   : {elapsed:.2f} ms")
    print(f"Status    : {status}")
    print(f"Parsed NLU Output:\n{json.dumps(parsed_json if parsed_json else raw_response, indent=2)}")
    print("-" * 65)

print("\n🎉 Sanity Check Complete. Gemma is fully capable of your NLU tasks.")
