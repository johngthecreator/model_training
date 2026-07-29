from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as hf_logging


MODEL_DIR = "final-gec-model/final"
TASK_PROMPT = "Correct the grammar, agreement, tense, articles, and punctuation in this text: "


hf_logging.set_verbosity_error()


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.config.tie_word_embeddings = False
    model.eval()
    return tokenizer, model


def generate_correction(tokenizer, model, text: str, device: torch.device) -> str:
    inputs = tokenizer(
        TASK_PROMPT + text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=4,
            do_sample=False,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with the final GEC model.")
    parser.add_argument(
        "--text",
        action="append",
        help="Input text to correct. Pass multiple times for multiple examples.",
    )
    args = parser.parse_args()

    examples = args.text or [
        "I has been working here since two years.",
        "The report were submitted by the team yesterday.",
        "He don't know nothing about the project deadline.",
    ]

    print(f"Loading model and tokenizer from {MODEL_DIR}")
    tokenizer, model = load_model_and_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for text in examples:
        corrected = generate_correction(tokenizer, model, text, device)
        print(f"Input:  {text}")
        print(f"Output: {corrected}")
        print()


if __name__ == "__main__":
    main()
