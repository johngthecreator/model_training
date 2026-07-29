"""
GEC single-stage example: fine-tune FLAN-T5-base on a combined GEC + coherence dataset.

Contrasts with examples/gec/train_gec.py (two-stage curriculum).

See the root train.py for the generic template.
"""
import modal


app = modal.App("gec-single-stage-example")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers",
        "datasets",
        "accelerate",
        "sentencepiece",
        "safetensors",
        "torch",
    )
)

volume = modal.Volume.from_name("gec-training-artifacts", create_if_missing=True)


@app.function(image=image, gpu="H100", timeout=60 * 60 * 4, volumes={"/mnt/model": volume})
def train():
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model.config.tie_word_embeddings = False

    dataset = load_dataset("JohnGorri/gec-coherence-coedit-synth")

    def preprocess_function(examples):
        inputs = [src for src in examples["src"]]
        targets = [tgt for tgt in examples["tgt"]]

        model_inputs = tokenizer(
            inputs, max_length=512, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            targets, max_length=160, truncation=True, padding="max_length"
        )

        labels["input_ids"] = [
            [tok if tok != tokenizer.pad_token_id else -100 for tok in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="/mnt/model/checkpoints",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=500,
        predict_with_generate=True,
        generation_max_length=160,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )
    trainer.train()

    model.config.tie_word_embeddings = False
    model.config.use_cache = True
    trainer.save_model("/mnt/model/final")
    tokenizer.save_pretrained("/mnt/model/final")
    volume.commit()


@app.local_entrypoint()
def main():
    path = train.spawn()
    print(path)
