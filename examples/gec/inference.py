from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_DIR = "final-model/final"
PREFIX = "Correct grammar, agreement, tense, articles, and punctuation in this text: "

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

examples = [
    "I has been working here since two years.",
    "The report were submitted by the team yesterday.",
    "He don't know nothing about the project deadline.",
]

for text in examples:
    inputs = tokenizer(PREFIX + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=80, num_beams=4, do_sample=False)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Input:  {text}")
    print(f"Output: {corrected}")
    print()
