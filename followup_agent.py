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
# Prompt Builder — Calm, non-directive, patient-first
# --------------------------------------------------

def build_followup_prompt(stage: str, followups: list, pending_labs: list) -> str:
    followup_block = "\n".join(f"- {x}" for x in followups) if followups else "No follow-up appointments listed"
    pending_block = "\n".join(f"- {x}" for x in pending_labs) if pending_labs else "No pending labs listed"

    stage_phrase = {
        "post_transplant": "after a kidney transplant",
        "advanced_ckd": "with advanced chronic kidney disease",
        "dialysis": "while on dialysis",
    }.get(stage, "during kidney care")

    return f"""
You are NephroBridge, a calm and supportive medical explanation assistant.

Your role is NOT to diagnose, NOT to give medical advice, and NOT to replace a clinician.
Your only goal is to help a patient understand what follow-ups and open loops mean
so they feel less anxious and more oriented about what happens next.

PATIENT CONTEXT:
The patient is currently {stage_phrase}.

FOLLOW-UP APPOINTMENTS OR INSTRUCTIONS:
{followup_block}

PENDING LABS OR RESULTS:
{pending_block}

STRICT SAFETY RULES:
- Do NOT give instructions or tell the patient what to do.
- Do NOT recommend urgent or emergency care.
- Do NOT predict outcomes or complications.
- Do NOT create new follow-ups or labs that are not listed.
- Do NOT assign blame or imply mistakes.
- Do NOT explain your reasoning or how you generated the answer.

TONE & STYLE RULES:
- Calm, supportive, non-alarmist.
- Normalize that waiting and monitoring are common in kidney care.
- Use gentle, reassuring language.
- Avoid medical jargon when possible.

OUTPUT FORMAT RULES:
- Write ONLY patient-facing content.
- Use EXACTLY the following section headers.
- Fill in ALL sections with real text.
- Do NOT add extra sections.
- Do NOT rename or reorder sections.

🧾 What follow-ups or next steps are coming up

⏳ What results or actions are still pending

🔍 What usually happens in this phase of care

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

def run_followup_agent(timeline: dict) -> str:
    stage = timeline.get("kidney_journey_stage", "post_transplant")
    followups = timeline.get("followup_appointments", [])
    pending_labs = timeline.get("pending_labs", [])

    # Graceful fallback if nothing is provided
    if not followups and not pending_labs:
        return (
            "🧾 What follow-ups or next steps are coming up\n"
            "No upcoming appointments or instructions were listed yet.\n\n"
            "⏳ What results or actions are still pending\n"
            "No pending lab results were listed.\n\n"
            "🔍 What usually happens in this phase of care\n"
            "In kidney care, it is common to have periods of monitoring where clinicians wait for trends, "
            "lab results, or recovery progress before making any changes.\n\n"
            "💬 Helpful questions to ask your care team\n"
            "- Are there any follow-ups or tests scheduled for me?\n"
            "- When should I expect to hear back about recent labs?\n\n"
            "🛟 Safety note\n"
            "This tool cannot replace medical advice. Always follow your care team’s guidance."
        )

    prompt = build_followup_prompt(stage, followups, pending_labs)

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
                "You help kidney patients understand follow-ups and waiting periods "
                "in a calm, supportive, non-diagnostic way."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Force safe continuation
    chat += "\n🧾 What follow-ups or next steps are coming up\n"

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

    idx = text.find("🧾 What follow-ups or next steps are coming up")
    if idx != -1:
        text = text[idx:]

    del model
    gc.collect()

    return text
