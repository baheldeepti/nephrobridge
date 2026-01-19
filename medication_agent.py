import torch
import gc
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------
# MODEL SELECTION (Hackathon-compliant)
# --------------------------------------------------

USE_MEDGEMMA = os.getenv("USE_MEDGEMMA", "false").lower() == "true"

MODEL_ID = (
    "google/medgemma-1.5-4b-it"
    if USE_MEDGEMMA
    else "google/gemma-2b-it"
)

# --------------------------------------------------
# Prompt Builder — Patient-first, safety-aligned
# --------------------------------------------------

def build_medication_prompt(stage: str, meds_before: list, meds_after: list) -> str:
    before_block = ", ".join(meds_before) if meds_before else "No medications listed"
    after_block = ", ".join(meds_after) if meds_after else "No medications listed"

    stage_phrase = {
        "post_transplant": "after a kidney transplant",
        "advanced_ckd": "with advanced chronic kidney disease",
        "dialysis": "while on dialysis",
    }.get(stage, "during kidney care")

    return f"""
You are NephroBridge, a calm and supportive medical explanation assistant.

Your role is NOT to diagnose, NOT to give medical advice, and NOT to replace a clinician.
Your only goal is to help a patient understand medication changes in general terms
so they feel less confused and less anxious.

PATIENT CONTEXT:
The patient is currently {stage_phrase}.

MEDICATIONS BEFORE:
{before_block}

MEDICATIONS AFTER:
{after_block}

STRICT SAFETY RULES:
- Do NOT give dosing advice or instructions.
- Do NOT recommend starting or stopping medications.
- Do NOT predict outcomes or complications.
- Do NOT assign blame or suggest mistakes.
- Do NOT list rare or severe side effects.
- Do NOT tell the patient to seek urgent or emergency care.
- Do NOT explain your reasoning or how you generated the answer.

TONE & STYLE RULES:
- Calm, supportive, non-alarmist.
- Normalize that medication changes are common.
- Clear, simple language suitable for a patient reading on their phone.
- Avoid medical jargon when possible.

OUTPUT FORMAT RULES:
- Write ONLY patient-facing content.
- Use EXACTLY the following section headers.
- Fill in ALL sections with real text.
- Do NOT add extra sections.
- Do NOT rename or reorder sections.

💊 What changed in your medications

🔍 Why clinicians commonly make changes like this

💬 Helpful questions to ask your care team

🛟 Safety note
""".strip()


# --------------------------------------------------
# Output Sanitizer — Final safety net
# --------------------------------------------------

def sanitize_output(text: str) -> str:
    forbidden = [
        "<thought>", "<unused", "analysis", "reasoning",
        "The user wants", "Model:", "Confidence",
        "I am an AI", "I cannot diagnose"
    ]

    for bad in forbidden:
        idx = text.lower().find(bad.lower())
        if idx != -1:
            text = text[:idx]

    return text.strip()


# --------------------------------------------------
# Public Agent API — called from Streamlit
# --------------------------------------------------

def run_medication_agent(timeline: dict) -> str:
    stage = timeline.get("kidney_journey_stage", "post_transplant")
    meds_before = timeline.get("medications_before", [])
    meds_after = timeline.get("medications_after", [])

    # Graceful fallback when no medication data is provided
    if not meds_before and not meds_after:
        return (
            "💊 What changed in your medications\n"
            "No medication changes were provided, so there is nothing to explain yet.\n\n"
            "🔍 Why clinicians commonly make changes like this\n"
            "Medication adjustments are often made to keep treatment aligned with lab results, symptoms, and recovery needs.\n\n"
            "💬 Helpful questions to ask your care team\n"
            "- Have any of my medications changed recently?\n"
            "- Should I expect any adjustments at my next visit?\n\n"
            "🛟 Safety note\n"
            "This tool cannot replace medical advice. Always follow your care team’s guidance."
        )

    prompt = build_medication_prompt(stage, meds_before, meds_after)

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
                "You are NephroBridge. "
                "You explain kidney-related medication changes in a calm, supportive, "
                "non-diagnostic, patient-friendly way."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Force safe continuation into patient text
    chat += "\n💊 What changed in your medications\n"

    inputs = tokenizer(chat, return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=180 if USE_MEDGEMMA else 240,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    text = sanitize_output(text)

    idx = text.find("💊 What changed in your medications")
    if idx != -1:
        text = text[idx:]

    del model
    gc.collect()

    return text
