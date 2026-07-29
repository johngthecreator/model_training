"""
Two-stage curriculum training on Modal.

Stage 1: Train on a large, noisier dataset (e.g. synthetic data)
Stage 2: Fine-tune on a smaller, higher-quality dataset

Adapt the normalize_* functions and dataset names for your own task.
"""
import modal


# -- Adjust these for your project -------------------------------------------
APP_NAME = "my-two-stage-training"
MODEL_ID = "google/flan-t5-small"          # base model from HuggingFace
GPU = "L40S"
TIMEOUT = 60 * 60 * 4                       # 4 hours
VOLUME_NAME = "my-training-artifacts"
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

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

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, volumes={"/mnt/model": volume})
def train():
    from datasets import Dataset, load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.config.tie_word_embeddings = False

    # -- Stage 1 dataset (noisy / synthetic) ---------------------------------
    # Replace with your own dataset and normalization logic.
    # Expected output columns: "src" (input text), "tgt" (target text).
    noise_dataset = load_dataset("your-username/your-dataset")
    noise_train = noise_dataset["train"].shuffle(seed=42)
    noise_eval = noise_dataset["validation"].shuffle(seed=42).select(range(2000))

    def normalize_noise(examples):
        return {
            "src": examples["src"],
            "tgt": examples["tgt"],
        }

    normalized_noise_train = Dataset.from_list(
        [normalize_noise(row) for row in noise_train]
    )
    normalized_noise_eval = Dataset.from_list(
        [normalize_noise(row) for row in noise_eval]
    )

    # -- Stage 2 dataset (high-quality) --------------------------------------
    # Replace with your own dataset and normalization logic.
    quality_dataset = load_dataset("your-username/your-quality-dataset")
    quality_train = quality_dataset["train"].shuffle(seed=42)

    def normalize_quality(examples):
        return {
            "src": examples["src"],
            "tgt": examples["tgt"],
        }

    normalized_quality_train = Dataset.from_list(
        [normalize_quality(row) for row in quality_train]
    )

    # -- Shared preprocessing -------------------------------------------------
    def preprocess_function(examples):
        # If using an instruct-tuned model (e.g. FLAN-T5), prepend a task
        # instruction to each input. Skip this step for base (non-instruct)
        # variants like t5-base or bart-base.
        #   inputs = ["correct grammar: " + src for src in examples["src"]]
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

    tokenized_noise_train = normalized_noise_train.map(
        preprocess_function, batched=True
    )
    tokenized_noise_eval = normalized_noise_eval.map(
        preprocess_function, batched=True
    )
    tokenized_quality_train = normalized_quality_train.map(
        preprocess_function, batched=True
    )

    # -- Stage 1: train on noisy data ----------------------------------------
    stage_1_args = Seq2SeqTrainingArguments(
        output_dir="/mnt/model/checkpoints/stage_1",
        eval_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=1000,
        predict_with_generate=True,
        generation_max_length=160,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
    )

    trainer_stage_1 = Seq2SeqTrainer(
        model=model,
        args=stage_1_args,
        train_dataset=tokenized_noise_train,
        eval_dataset=tokenized_noise_eval,
    )
    trainer_stage_1.train()
    trainer_stage_1.save_model("/mnt/model/stage_1")

    # -- Stage 2: fine-tune on high-quality data -----------------------------
    stage_2_args = Seq2SeqTrainingArguments(
        output_dir="/mnt/model/checkpoints/stage_2",
        eval_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=1000,
        predict_with_generate=True,
        generation_max_length=160,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
    )

    model_stage_1 = AutoModelForSeq2SeqLM.from_pretrained("/mnt/model/stage_1")

    trainer_stage_2 = Seq2SeqTrainer(
        model=model_stage_1,
        args=stage_2_args,
        train_dataset=tokenized_quality_train,
        eval_dataset=tokenized_noise_eval,
    )
    trainer_stage_2.train()

    # FLAN-T5 uses an untied LM head. If this is saved as true, transformers
    # ties lm_head to shared embeddings on load and generation collapses into
    # repeated multilingual tokens.
    model_stage_1.config.tie_word_embeddings = False
    model_stage_1.config.use_cache = True
    trainer_stage_2.save_model("/mnt/model/final")
    tokenizer.save_pretrained("/mnt/model/final")
    volume.commit()


@app.local_entrypoint()
def main():
    path = train.spawn()
    print(path)
