import modal


app = modal.App("flan-t5-small-gec-gramercy")

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

volume = modal.Volume.from_name("flan-t5-small-gec-gramercy-artifacts", create_if_missing=True)

@app.function(image=image, gpu="L40S", timeout=60 * 60 * 4, volumes={"/mnt/model": volume})
def train():
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model.config.tie_word_embeddings = False

    coedit = load_dataset("grammarly/coedit")
    coedit_train = coedit["train"].shuffle(seed=42)

    c4_200m_gec_stream = load_dataset("martinsr/c4_200m", split="train", streaming=True).shuffle(seed=42)
    c4_200m_gec = c4_200m_gec_stream.take(35000)
    c4_200m_gec_eval = c4_200m_gec_stream.skip(35000).take(2000)

    # jfleg_gec = load_dataset("jhu-clsp/jfleg")

    def normalize_coedit_train(dataset):
        rows = [
            {
                "src": "Correct grammar, agreement, tense, articles, and punctuation in this text: " + row["src"].split(": ", 1)[1],
                "tgt": row["tgt"],
                "task": row["task"],
                "source": "coedit",
            }
            for row in dataset
            if row["task"] == "gec"
        ]
        return Dataset.from_list(rows)

    def normalize_c4_train(dataset, limit):
        rows = []
        for row in dataset:
            rows.append(
                {
                    "src": "Correct grammar, agreement, tense, articles, and punctuation in this text: " + row["input"],
                    "tgt": row["output"],
                    "task": "gec",
                    "source": "c4_200m",
                }
            )
            if len(rows) >= limit:
                break
        return Dataset.from_list(rows)

    # def normalize_jfleg_split(dataset):
    #     rows = [
    #         {
    #             "src": row["sentence"],
    #             "tgt": correction,
    #             "task": "gec",
    #             "source": "jfleg",
    #         }
    #         for row in dataset
    #         for correction in row["corrections"]
    #     ]
    #     return Dataset.from_list(rows)

    def preprocess_function(examples):
        inputs = [input for input in examples["src"]]
        targets = [answer for answer in examples["tgt"]]

        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
        labels = tokenizer(targets, max_length=160, truncation=True, padding='max_length')

        labels["input_ids"] = [
            [tok if tok != tokenizer.pad_token_id else -100 for tok in label]
            for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    normalized_coedit_train = normalize_coedit_train(coedit_train)
    normalized_c4_200m_train = normalize_c4_train(c4_200m_gec, limit=35000)
    # normalized_jfleg_val = normalize_jfleg_split(jfleg_gec["validation"])
    normalized_c4_200m_eval = normalize_c4_train(c4_200m_gec_eval, limit=2000)

    tokenized_coedit = normalized_coedit_train.map(preprocess_function, batched=True)
    tokenized_c4_200m = normalized_c4_200m_train.map(preprocess_function, batched=True)
    tokenized_c4_200m_eval = normalized_c4_200m_eval.map(preprocess_function, batched=True)
    # tokenized_jfleg_val = normalized_jfleg_val.map(preprocess_function, batched=True)

    training_stage_1_args = Seq2SeqTrainingArguments(
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
        save_strategy="epoch"
    )

    # Initialize the Trainer
    trainer_stage_1 = Seq2SeqTrainer(
        model=model,
        args=training_stage_1_args,
        train_dataset=tokenized_c4_200m,
        eval_dataset=tokenized_c4_200m_eval,
    )

    trainer_stage_1.train()
    trainer_stage_1.save_model("/mnt/model/stage_1")

    training_stage_2_args = Seq2SeqTrainingArguments(
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
        save_strategy="epoch"
    )


    model_stage_1 = AutoModelForSeq2SeqLM.from_pretrained("/mnt/model/stage_1")

    trainer_stage_2 = Seq2SeqTrainer(
        model=model_stage_1,
        args=training_stage_2_args,
        train_dataset=tokenized_coedit,
        eval_dataset=tokenized_c4_200m_eval,
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
