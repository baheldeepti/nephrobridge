import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/medgemma-1.5-4b-it"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="auto"
)


print("Model loaded successfully.")

prompt = (
    "You are a medical assistant. Explain in simple language what a rising "
    "creatinine level might indicate for a kidney patient. "
    "Do not give medical advice."
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Running inference...")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False
    )

response = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n--- Model response ---\n")
print(response)
