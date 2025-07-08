
from transformers import pipeline


qa_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", max_new_tokens=256)

def generate_answer(prompt: str) -> str:
    response = qa_pipeline(prompt)[0]["generated_text"]
    return response.split("Answer:")[-1].strip()
