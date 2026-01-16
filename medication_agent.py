import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/medgemma-1.5-4b-it"
TIMELINE_PATH = "data/patient_timeline_post_transplant.json"


def build_medication_prompt(stage: str, meds: dict, hospital_stay: dict) -> str:
    return f"""
You are explaining medication changes to a kidney patient.

PATIENT CONTEXT (use exactly as given):
- Kidney journey stage: {stage}
- Recent hospital reason: {hospital_stay.get('reason_for_admission')}

MEDICATION CHANGES (use exactly as provided):
- Previous medications: {", ".join(meds.get("previous", []))}
- Current medications: {", ".join(meds.get("current", []))}

IMPORTANT RULES:
- Do NOT give medical advice, dosing, or instructions
- Do NOT diagnose or predict outcomes
- Do NOT list side effects unless common and general
- Do NOT comment on data quality or assumptions
- Be calm, supportive, and non-alarmist

STRUCTURE RULES:
- Use the EXACT section headers below
- Do NOT add or remove sections
- Fill in ALL sections with patient-facing content

Write ONLY the explanation using EXACTLY this structure:

💊 What changed in your medications

🔍 Why clinicians commonly make changes like this

💬 Helpful questions to ask your care team

🛟 Safety note
""".strip()


def main():
    with open(TIMELINE_PATH, "r") as f:
        timeline = json.load(f)

    prompt = build_medication_prompt(
        timeline["kidney_journey_stage"],
        timeline["medications"],
        timeline["recent_hospital_stay"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are NephroBridge, a patient-facing explanation assistant "
                "for people living with kidney disease."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(chat, return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=400,
            num_beams=2,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    print("=" * 60)
    print("Medication Change Explainer Output")
    print("=" * 60 + "\n")

    if "💊 What changed in your medications" in response:
        response = response[response.find("💊 What changed in your medications") :]

    print(response.strip())
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
