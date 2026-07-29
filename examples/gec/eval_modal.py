import modal

app = modal.App("grammary-eval")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers", "datasets", "evaluate", "torch", "sacrebleu", "rouge-score", "errant", "spacy")
    .run_commands("python -m spacy download en_core_web_sm")
)

volume = modal.Volume.from_name("flan-t5-small-gec-gramercy-artifacts", create_if_missing=True)

GEC_PREFIX = "Correct grammar, agreement, tense, articles, and punctuation in this text: "


@app.function(image=image, gpu="L4", timeout=60 * 60, volumes={"/mnt/model": volume})
def evaluate(model_id: str) -> dict:
    import torch
    import re
    import subprocess
    import tempfile
    from pathlib import Path

    from datasets import load_dataset
    from evaluate import load as load_metric
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    import spacy

    spacy.load("en_core_web_sm")

    print(f"[{model_id}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    model = model.to("cuda")

    print(f"[{model_id}] Loading test dataset...")
    jfleg = load_dataset("jhu-clsp/jfleg", split="test")
    raw_src_texts = [row["sentence"] for row in jfleg]
    src_texts = [GEC_PREFIX + sentence for sentence in raw_src_texts]
    refs = [row["corrections"][0] for row in jfleg]
    all_refs = [row["corrections"] for row in jfleg]

    print(f"[{model_id}] Generating predictions...")
    preds = []
    BATCH_SIZE = 8

    for i in range(0, len(src_texts), BATCH_SIZE):
        batch = src_texts[i : i + BATCH_SIZE]
        inputs = tokenizer(batch, max_length=512, truncation=True, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4)
        preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

        if (i // BATCH_SIZE) % 25 == 0:
            print(f"  [{model_id}] {min(i + BATCH_SIZE, len(src_texts))}/{len(src_texts)}")

    sacrebleu = load_metric("sacrebleu")
    rouge = load_metric("rouge")

    single_bleu = sacrebleu.compute(predictions=preds, references=[[r] for r in refs])
    single_rouge = rouge.compute(predictions=preds, references=refs)

    expanded_preds = []
    expanded_refs = []
    for pred, ref_group in zip(preds, all_refs):
        for ref in ref_group:
            expanded_preds.append(pred)
            expanded_refs.append(ref)

    multi_bleu = sacrebleu.compute(predictions=expanded_preds, references=[[r] for r in expanded_refs])
    multi_rouge = rouge.compute(predictions=expanded_preds, references=expanded_refs)

    def _write_lines(path: Path, lines: list[str]) -> None:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _run_errant_compare(srcs: list[str], hyps: list[str], ref_groups: list[list[str]]) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            orig_path = tmp / "orig.txt"
            hyp_txt_path = tmp / "hyp.txt"
            hyp_m2_path = tmp / "hyp.m2"
            ref_m2_path = tmp / "ref.m2"

            _write_lines(orig_path, srcs)
            _write_lines(hyp_txt_path, hyps)

            subprocess.run(
                ["errant_parallel", "-orig", str(orig_path), "-cor", str(hyp_txt_path), "-out", str(hyp_m2_path)],
                check=True,
                capture_output=True,
                text=True,
            )

            max_refs = max(len(group) for group in ref_groups)
            ref_paths = []
            for ref_idx in range(max_refs):
                ref_path = tmp / f"ref_{ref_idx}.txt"
                ref_lines = [
                    group[ref_idx] if ref_idx < len(group) else (group[-1] if group else src)
                    for src, group in zip(srcs, ref_groups)
                ]
                _write_lines(ref_path, ref_lines)
                ref_paths.append(str(ref_path))

            subprocess.run(
                ["errant_parallel", "-orig", str(orig_path), "-cor", *ref_paths, "-out", str(ref_m2_path)],
                check=True,
                capture_output=True,
                text=True,
            )

            compare = subprocess.run(
                ["errant_compare", "-hyp", str(hyp_m2_path), "-ref", str(ref_m2_path)],
                check=True,
                capture_output=True,
                text=True,
            )

        text = compare.stdout + "\n" + compare.stderr
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        score_line = None
        for line in lines:
            if re.fullmatch(r"[0-9]+\s+[0-9]+\s+[0-9]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+", line):
                score_line = line
                break
        if score_line is None:
            raise ValueError(f"Could not find ERRANT score line in output:\n{text}")

        tp, fp, fn, prec, rec, f05 = score_line.split()

        return {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f0.5": round(float(f05), 4),
        }

    single_errant = _run_errant_compare(raw_src_texts, preds, [[r] for r in refs])
    multi_errant = _run_errant_compare(raw_src_texts, preds, all_refs)

    return {
        "gec_jfleg_single": {
            "sacreBLEU": round(single_bleu["score"], 1),
            "rouge1": round(single_rouge["rouge1"], 4),
            "rouge2": round(single_rouge["rouge2"], 4),
            "rougeL": round(single_rouge["rougeL"], 4),
            "precision": single_errant["precision"],
            "recall": single_errant["recall"],
            "f0.5": single_errant["f0.5"],
            "num_examples": len(src_texts),
        },
        "gec_jfleg_multi": {
            "sacreBLEU": round(multi_bleu["score"], 1),
            "rouge1": round(multi_rouge["rouge1"], 4),
            "rouge2": round(multi_rouge["rouge2"], 4),
            "rougeL": round(multi_rouge["rougeL"], 4),
            "precision": multi_errant["precision"],
            "recall": multi_errant["recall"],
            "f0.5": multi_errant["f0.5"],
            "num_examples": len(expanded_preds),
        }
    }


@app.local_entrypoint()
def main():
    models = [
        # "/mnt/model/final",           # final gec model in the volume
        "grammarly/coedit-large",     # baseline
        # "your-username/your-model-name",       # untuned baseline
    ]

    # Run both evaluations in parallel
    handles = [evaluate.spawn(model) for model in models]

    for model, handle in zip(models, handles):
        results = handle.get()
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        for task, metrics in results.items():
            print(f"  {task}:")
            for k, v in metrics.items():
                print(f"    {k}: {v}")
