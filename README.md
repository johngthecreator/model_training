# ML Training

A template for fine-tuning HuggingFace models on Modal cloud GPUs. The core pipeline handles dataset loading, tokenization, training, and checkpointing — adapt it to any text-to-text task.

## Project Structure

```
├── train.py              # Core training script (Modal + HuggingFace Trainer)
├── train_gec.py          # Core training script (two-stage curriculum variant)
├── push_dataset.py       # Push local CSVs to Hugging Face Hub
├── push_model.py         # Push trained models to Hugging Face Hub
└── examples/
    └── gec/              # Grammar Error Correction example
        ├── data/                   # Raw CSV exports from grammarly/coedit
        ├── training_data/          # Train/val/test splits
        ├── dataset.py             # Dataset exploration & normalization
        ├── eval.py                # Local BLEU/ROUGE evaluation
        ├── eval_modal.py          # Modal evaluation (BLEU/ROUGE/ERRANT)
        ├── eval_heldout.py        # Held-out eval (JFLEG + HellaSwag)
        ├── inference.py           # Local inference
        └── test_inference.py      # CLI inference tool
```

## Getting Started

### Install dependencies

```bash
pip install torch transformers datasets evaluate sacrebleu rouge_score nltk
```

### Train on Modal

```bash
modal run train.py
modal run train_gec.py
```

### Prepare and upload a dataset

```bash
python push_dataset.py
```

### Push a trained model to Hugging Face

```bash
python push_model.py --repo-id your-username/your-model-name
```

## How to adapt for your own task

1. Replace the dataset path in `train.py` with your Hugging Face dataset
2. Update the instruction prefix in the `preprocess_function` to match your task
3. Adjust model architecture, batch size, and training args as needed
4. See `examples/gec/` for a complete end-to-end grammar error correction pipeline

## Modal Volumes

Training and evaluation use Modal volumes for persistent storage:

| Volume | Used by |
|---|---|
| `flan-t5-small-gec-gramercy-artifacts` | GEC training, GEC evaluation |
| `flan-t5-base-coedit-artifacts` | Combined training, held-out eval |

```bash
modal volume create flan-t5-small-gec-gramercy-artifacts
modal volume create flan-t5-base-coedit-artifacts
```
