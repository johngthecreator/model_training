import modal

app = modal.App("grammary-heldout-eval")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers", "datasets", "evaluate", "torch", "sacrebleu", "rouge-score")
)

volume = modal.Volume.from_name("flan-t5-base-coedit-artifacts", create_if_missing=True)

GEC_PREFIX = "Correct grammar, agreement, tense, articles, and punctuation in this text: "
COH_PREFIX = "Choose the most coherent ending for this story: "


@app.function(image=image, gpu="A100-80GB", timeout=60 * 60, volumes={"/mnt/model": volume})
def evaluate_heldout(model_id: str, batch_size: int = 8) -> dict:
    import torch
    from datasets import load_dataset
    from evaluate import load as load_metric
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"[{model_id}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")

    def generate(src_texts: list[str]) -> list[str]:
        preds = []
        for i in range(0, len(src_texts), batch_size):
            batch = src_texts[i : i + batch_size]
            inputs = tokenizer(batch, max_length=512, truncation=True, padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=160, num_beams=4)
            preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return preds

    results = {}

    def pick_split(dataset_obj):
        if hasattr(dataset_obj, "keys"):
            for split_name in ("test", "validation", "train"):
                if split_name in dataset_obj:
                    return dataset_obj[split_name]
        return dataset_obj

    def load_coherence_dataset():
        candidates = [
            ("Rowan/hellaswag", "validation"),
            ("Rowan/hellaswag", "train"),
            ("Rowan/hellaswag", None),
        ]

        errors = []
        for dataset_name, split_name in candidates:
            try:
                ds = load_dataset(dataset_name, split=split_name) if split_name else load_dataset(dataset_name)
                return dataset_name, pick_split(ds)
            except Exception as exc:
                errors.append(f"{dataset_name!r}({split_name!r}): {exc}")

        raise RuntimeError(
            "Could not load a coherence dataset from Hugging Face. Tried:\n"
            + "\n".join(errors)
        )

    def extract_hellaswag_examples(dataset):
        columns = set(dataset.column_names)
        examples = []

        if not {"ctx_a", "ctx_b", "endings"}.issubset(columns):
            raise ValueError(
                "The loaded dataset did not match a HellaSwag schema with ctx_a, ctx_b, and endings."
            )

        for row in dataset:
            context = " ".join(
                part for part in (str(row["ctx_a"]).strip(), str(row["ctx_b"]).strip()) if part
            )
            endings = [str(x).strip() for x in row["endings"]]
            gold = None
            for key in ("label", "answer_key", "correct_answer", "target", "answer_idx"):
                if key not in row:
                    continue
                value = row[key]
                if value in ("", None):
                    continue
                if isinstance(value, str):
                    if value in {"A", "a", "0"}:
                        gold = 0
                        break
                    if value in {"B", "b", "1"}:
                        gold = 1
                        break
                    if value in {"C", "c", "2"}:
                        gold = 2
                        break
                    if value in {"D", "d", "3"}:
                        gold = 3
                        break
                    if value.isdigit():
                        gold = int(value)
                        break
                else:
                    try:
                        gold = int(value)
                        break
                    except Exception:
                        continue
            if context and len(endings) == 4:
                if gold is None:
                    continue
                examples.append(
                    {
                        "context": context,
                        "options": endings,
                        "gold": gold,
                    }
                )

        if not examples:
            raise ValueError(
                "The loaded dataset did not produce any HellaSwag examples."
            )

        return examples

    def score_choice(contexts, endings, probe_batch_size: int = 8):
        scores = []
        for i in range(0, len(contexts), probe_batch_size):
            context_batch = contexts[i : i + probe_batch_size]
            ending_batch = endings[i : i + probe_batch_size]
            inputs = tokenizer(
                [COH_PREFIX + c for c in context_batch],
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            labels = tokenizer(
                ending_batch,
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).input_ids.to("cuda")
            labels = labels.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            with torch.no_grad():
                outputs = model(**inputs, labels=labels)
                logits = outputs.logits

            token_log_probs = torch.log_softmax(logits, dim=-1)
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0
            gathered = token_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            mask = labels != -100
            summed = (gathered * mask).sum(dim=-1)
            lengths = mask.sum(dim=-1).clamp(min=1)
            scores.append(summed / lengths)
        return torch.cat(scores, dim=0)

    # ---- GEC: JFLEG ----
    print(f"[{model_id}] GEC: JFLEG test...")
    jfleg = load_dataset("jhu-clsp/jfleg", split="test")
    src_texts = [GEC_PREFIX + s for s in jfleg["sentence"]]
    refs = jfleg["corrections"]
    preds = generate(src_texts)

    bleu = load_metric("sacrebleu").compute(predictions=preds, references=refs)
    rouge = load_metric("rouge").compute(predictions=preds, references=[r[0] for r in refs])
    results["gec_jfleg"] = {
        "sacreBLEU": round(bleu["score"], 1),
        "rouge1": round(rouge["rouge1"], 4),
        "rouge2": round(rouge["rouge2"], 4),
        "rougeL": round(rouge["rougeL"], 4),
    }

    # ---- Coherence: HellaSwag probing ----
    dataset_name, coherence_dataset = load_coherence_dataset()
    print(f"[{model_id}] Coherence: {dataset_name}...")
    examples = extract_hellaswag_examples(coherence_dataset)

    # Cap at 1000 for speed.
    examples = examples[:1000]
    contexts = [ex["context"] for ex in examples]
    option_scores = [
        score_choice(contexts, [ex["options"][i] for ex in examples], probe_batch_size=4)
        for i in range(4)
    ]
    stacked_scores = torch.stack(option_scores, dim=1)
    chosen = torch.argmax(stacked_scores, dim=1).tolist()
    gold = [ex["gold"] for ex in examples]
    correct = sum(int(p == y) for p, y in zip(chosen, gold))
    top2 = torch.topk(stacked_scores, k=2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).mean().item()

    results["coherence_hellaswag"] = {
        "accuracy": round(correct / len(examples), 4),
        "avg_margin": round(margin, 4),
        "num_examples": len(examples),
    }

    return results


@app.local_entrypoint()
def main():
    models = [
        ("/mnt/model/final", 4),
        ("grammarly/coedit-large", 4),
    ]

    for model, batch_size in models:
        handle = evaluate_heldout.spawn(model, batch_size=batch_size)
        results = handle.get()
        print(f"\n{'='*55}")
        print(f"Model: {model}")
        for task, metrics in results.items():
            print(f"  {task}:")
            for k, v in metrics.items():
                print(f"    {k}: {v}")
