import json
import torch
import gc
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------
# CONFIG (Hackathon-compliant)
# --------------------------------------------------

# 🔁 Automatically switch model based on environment
USE_MEDGEMMA = os.getenv("USE_MEDGEMMA", "false").lower() == "true"

MODEL_ID = (
    "google/medgemma-1.5-4b-it"
    if USE_MEDGEMMA
    else "google/gemma-2b-it"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATA"


STAGE_TO_FILE = {
    "post_transplant": "patient_timeline_post_transplant.json",
    "advanced_ckd": "patient_timeline_advanced_ckd.json",
    "dialysis": "patient_timeline_dialysis.json",
}


# --------------------------------------------------
# Prompt Builder (stage-aware, reusable agent)
# --------------------------------------------------
def build_user_prompt(stage: str, labs: list) -> str:
    lab_block = "\n".join(
        f"- {r['date']}: {r['lab_name']} = {r['value']} {r['unit']} "
        f"(ref {r['reference_range']})"
        for r in labs
    )

    stage_phrase = {
        "post_transplant": "after a kidney transplant",
        "advanced_ckd": "with advanced chronic kidney disease",
        "dialysis": "while on dialysis",
    }[stage]

    return f"""
You are explaining kidney lab trends to a patient {stage_phrase}.

LAB RESULTS (use exactly as provided):
{lab_block}

RULES:
- Do NOT diagnose or give medical instructions.
- Do NOT comment on data formatting or correctness.
- Do NOT explain your reasoning.
- Be calm, supportive, and patient-friendly.

Write ONLY patient-facing content using EXACTLY this structure
and FILL ALL SECTIONS with real text:

🧠 Key takeaways
- 

🧬 What changed in your labs

🔍 Common reasons this can happen {stage_phrase}
- 

💬 Helpful questions to ask your care team
- 

🛟 Safety note
""".strip()


# --------------------------------------------------
# Sanitizer (final safety layer)
# --------------------------------------------------
def sanitize(text: str) -> str:
    forbidden = [
        "<thought>", "<unused", "analysis", "reasoning",
        "The user wants", "Model:", "Confidence"
    ]
    for bad in forbidden:
        if bad.lower() in text.lower():
            text = text[: text.lower().find(bad.lower())]
    return text.strip()


# --------------------------------------------------
# Agent Runner
# --------------------------------------------------
def run_lab_agent(stage: str):
    timeline_path = DATA_DIR / STAGE_TO_FILE[stage]

    with open(timeline_path) as f:
        timeline = json.load(f)

    prompt = build_user_prompt(stage, timeline["labs_over_time"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if USE_MEDGEMMA else torch.float32,
        device_map="auto" if USE_MEDGEMMA else {"": "cpu"},
        low_cpu_mem_usage=True,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are NephroBridge, a patient-facing explanation assistant. "
                "You explain medical information clearly without giving advice."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 🔑 Force continuation
    chat += "\n🧠 Key takeaways\n- "

    inputs = tokenizer(chat, return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=180 if USE_MEDGEMMA else 250,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    text = sanitize(text)

    idx = text.find("🧠 Key takeaways")
    if idx != -1:
        text = text[idx:]

    del model
    gc.collect()

    return text


# --------------------------------------------------
# CLI ENTRYPOINT
# --------------------------------------------------
if __name__ == "__main__":
    print("\nWhich kidney journey stage are you in?")
    print("Options: post_transplant | advanced_ckd | dialysis")

    stage = input("Enter stage: ").strip().lower()

    if stage not in STAGE_TO_FILE:
        raise ValueError("Invalid stage")

    print("\n" + "=" * 60)
    print(f"NEPHROBRIDGE — {stage.upper()}")
    print("=" * 60 + "\n")

    print(run_lab_agent(stage))
