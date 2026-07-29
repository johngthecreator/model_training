import argparse
import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="final-model/final", help="Model path or HF model ID")
args = parser.parse_args()

BATCH_SIZE = 8

print(f"Loading model and tokenizer: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Loading test dataset...")
# Replace with your Hugging Face dataset path
dataset = load_dataset("your-username/your-dataset-name")
test = dataset["test"]

print(f"Generating predictions for {len(test)} examples (batch_size={BATCH_SIZE})...")
preds, refs, tasks = [], [], []

for i in range(0, len(test), BATCH_SIZE):
    batch = test[i : i + BATCH_SIZE]
    inputs = tokenizer(batch["src"], max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4)
    preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    refs.extend(batch["tgt"])
    tasks.extend(batch["task"])

    if (i // BATCH_SIZE) % 10 == 0:
        print(f"  {i + len(batch)}/{len(test)}")

# Compute metrics
sacrebleu = load("sacrebleu")
rouge = load("rouge")

refs_wrapped = [[r] for r in refs]

overall_bleu = sacrebleu.compute(predictions=preds, references=refs_wrapped)
overall_rouge = rouge.compute(predictions=preds, references=refs)

print(f"\n{'='*50}")
print("Overall (all tasks)")
print(f"{'='*50}")
print(f"sacreBLEU:  {overall_bleu['score']:.1f}")
print(f"ROUGE-1:   {overall_rouge['rouge1']:.4f}")
print(f"ROUGE-2:   {overall_rouge['rouge2']:.4f}")
print(f"ROUGE-L:   {overall_rouge['rougeL']:.4f}")

# Per-task breakdown
for task_name in ["gec", "coherence"]:
    idxs = [i for i, t in enumerate(tasks) if t == task_name]
    if not idxs:
        continue
    p = [preds[i] for i in idxs]
    r = [[refs[i]] for i in idxs]
    r_flat = [refs[i] for i in idxs]
    bleu = sacrebleu.compute(predictions=p, references=r)
    rg = rouge.compute(predictions=p, references=r_flat)
    print(f"\n{'='*50}")
    print(f"{task_name} ({len(idxs)} examples)")
    print(f"{'='*50}")
    print(f"sacreBLEU:  {bleu['score']:.1f}")
    print(f"ROUGE-1:   {rg['rouge1']:.4f}")
    print(f"ROUGE-2:   {rg['rouge2']:.4f}")
    print(f"ROUGE-L:   {rg['rougeL']:.4f}")
