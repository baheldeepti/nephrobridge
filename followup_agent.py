import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/medgemma-1.5-4b-it"
TIMELINE_PATH = "data/patient_timeline_post_transplant.json"


def build_followup_prompt(stage: str, instructions: list, pending: dict) -> str:
    return f"""
You are helping a kidney patient understand follow-up items after a hospital stay.

PATIENT CONTEXT (use exactly as given):
- Kidney journey stage: {stage}

DISCHARGE INSTRUCTIONS:
{chr(10).join("- " + i for i in instructions)}

PENDING ITEMS:
- Labs: {", ".join(pending.get("labs", []))}
- Appointments: {", ".join(pending.get("appointments", []))}

IMPORTANT RULES:
- Do NOT give instructions or medical advice
- Phrase everything as awareness or questions
- Do NOT add urgency unless explicitly stated
- Be reassuring and supportive

STRUCTURE RULES:
- Use EXACT section headers below
- Do NOT add or remove sections
- Fill in ALL sections

Write ONLY the patient-facing explanation using EXACTLY this structure:

🧾 Things to be aware of after discharge

💬 Helpful questions to ask your care team

🛟 Safety note
""".strip()


def main():
    with open(TIMELINE_PATH, "r") as f:
        timeline = json.load(f)

    prompt = build_followup_prompt(
        timeline["kidney_journey_stage"],
        timeline.get("discharge_instructions", []),
        timeline.get("pending_items", {}),
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
                "focused on follow-up awareness, not instructions."
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
            max_new_tokens=300,
            num_beams=2,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    print("=" * 60)
    print("Follow-up & Open-Loops Tracker Output")
    print("=" * 60 + "\n")

    if "🧾 Things to be aware of after discharge" in response:
        response = response[
            response.find("🧾 Things to be aware of after discharge") :
        ]

    print(response.strip())
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
