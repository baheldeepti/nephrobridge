import json
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

def build_lab_prompt(stage: str, labs: list) -> str:
    lab_block = "\n".join(
        f"- {r['date']}: {r['lab_name']} = {r['value']} {r.get('unit','')} "
        f"(ref {r.get('reference_range','')})"
        for r in labs
    )

    stage_phrase = {
        "post_transplant": "after a kidney transplant",
        "advanced_ckd": "with advanced chronic kidney disease",
        "dialysis": "while on dialysis",
    }.get(stage, "during kidney care")

    return f"""
You are NephroBridge, a calm and supportive medical explanation assistant.

Your role is NOT to diagnose, NOT to give medical advice, and NOT to replace a clinician.
Your only goal is to help a patient understand what their lab trends *might mean in general terms*
so they feel less confused and less anxious.

PATIENT CONTEXT:
The patient is currently {stage_phrase}.

LAB RESULTS (use exactly as provided, do not reinterpret or correct):
{lab_block}

STRICT SAFETY RULES:
- Do NOT diagnose any condition.
- Do NOT speculate about organ failure, rejection, or emergencies.
- Do NOT assign blame or fault.
- Do NOT give medical instructions or treatment advice.
- Do NOT tell the patient to go to the ER or seek urgent care.
- Do NOT explain your reasoning or how you generated the answer.
- Do NOT mention uncertainty about the data format.

TONE & STYLE RULES:
- Calm, supportive, non-alarmist.
- Reassuring but honest.
- Clear, simple language suitable for a patient reading on their phone.
- Avoid medical jargon when possible.

OUTPUT FORMAT RULES:
- Write ONLY patient-facing content.
- Use EXACTLY the following section headers.
- Fill in ALL sections with real text.
- Do NOT add extra sections.
- Do NOT rename or reorder sections.

🧠 Key takeaways

🧬 What changed in your labs

🔍 Common reasons this can happen {stage_phrase}

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

def run_lab_agent_from_timeline(timeline: dict) -> str:
    stage = timeline.get("kidney_journey_stage", "post_transplant")
    labs = timeline.get("labs_over_time", [])

    if not labs:
        return (
            "🧠 Key takeaways\n"
            "- No lab results were provided, so there is nothing to interpret yet.\n\n"
            "🧬 What changed in your labs\n"
            "No lab trends are available.\n\n"
            "🔍 Common reasons this can happen\n"
            "Sometimes lab results are still pending or not yet entered.\n\n"
            "💬 Helpful questions to ask your care team\n"
            "- Are there recent lab results I should review?\n\n"
            "🛟 Safety note\n"
            "This tool cannot replace medical advice. Always follow your care team’s guidance."
        )

    prompt = build_lab_prompt(stage, labs)

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
                "You explain kidney-related lab information in a calm, supportive, "
                "non-diagnostic, patient-friendly way."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Force safe continuation into patient text
    chat += "\n🧠 Key takeaways\n- "

    inputs = tokenizer(chat, return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200 if USE_MEDGEMMA else 260,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    text = sanitize_output(text)

    idx = text.find("🧠 Key takeaways")
    if idx != -1:
        text = text[idx:]

    del model
    gc.collect()

    return text
