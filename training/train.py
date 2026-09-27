"""Fine-tune a transformer for Marathi NER on L3Cube-MahaNER.

Runnable independently:

    python train.py                 # uses defaults from backend/app/config.py
    EPOCHS=1 python train.py        # override via env

Saves the model, tokenizer and label maps to OUTPUT_DIR and writes evaluation
metrics (seqeval) to METRICS_PATH.
"""

from __future__ import annotations

import json
import os
import sys

# Make backend config importable when run from the training/ dir.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from app.config import get_settings  # noqa: E402
from dataset import ENTITY_FULL_NAMES, build_label_maps, load_dataset  # noqa: E402
from preprocess import tokenize_and_align_labels  # noqa: E402
from evaluate import compute_seqeval_metrics  # noqa: E402


class ListDataset:
    """Minimal torch-free dataset wrapper over a list of feature dicts."""

    def __init__(self, features: list[dict]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        return self.features[idx]


def _encode(split, tokenizer, label2id, max_length) -> ListDataset:
    enc = tokenize_and_align_labels(
        split.tokens, split.tags, tokenizer, label2id, max_length
    )
    keys = list(enc.keys())
    features = [
        {k: enc[k][i] for k in keys} for i in range(len(enc["input_ids"]))
    ]
    return ListDataset(features)


def main() -> int:
    s = get_settings()
    set_seed(s.SEED)

    print(f"Loading dataset from {s.DATA_DIR} ...")
    data = load_dataset(s.DATA_DIR)
    label2id, id2label = build_label_maps(list(data.values()))
    print(f"Labels ({len(label2id)}): {list(label2id)}")

    print(f"Loading base model: {s.BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(s.BASE_MODEL)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for word_ids() alignment.")

    model = AutoModelForTokenClassification.from_pretrained(
        s.BASE_MODEL,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = _encode(data["train"], tokenizer, label2id, s.MAX_LENGTH)
    eval_ds = _encode(data["validation"], tokenizer, label2id, s.MAX_LENGTH)
    test_ds = _encode(data["test"], tokenizer, label2id, s.MAX_LENGTH)

    collator = DataCollatorForTokenClassification(tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_seqeval_metrics(preds, labels, id2label)["summary"]

    args = TrainingArguments(
        output_dir=os.path.join(s.OUTPUT_DIR, "_checkpoints"),
        num_train_epochs=s.EPOCHS,
        per_device_train_batch_size=s.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=s.EVAL_BATCH_SIZE,
        learning_rate=s.LEARNING_RATE,
        weight_decay=s.WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        seed=s.SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training ...")
    trainer.train()

    print("Evaluating on test split ...")
    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    metrics = compute_seqeval_metrics(preds, predictions.label_ids, id2label)
    metrics["model"] = s.BASE_MODEL
    metrics["trained"] = True
    metrics["dataset"] = "L3Cube-MahaNER"
    metrics["counts"] = {
        "train": len(train_ds),
        "validation": len(eval_ds),
        "test": len(test_ds),
        "num_labels": len(label2id),
    }

    os.makedirs(s.OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(s.OUTPUT_DIR)
    tokenizer.save_pretrained(s.OUTPUT_DIR)
    with open(os.path.join(s.OUTPUT_DIR, "label_maps.json"), "w", encoding="utf-8") as fh:
        json.dump({"label2id": label2id, "id2label": id2label}, fh, ensure_ascii=False, indent=2)
    with open(s.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    print(f"Saved model to {s.OUTPUT_DIR}")
    print(f"Saved metrics to {s.METRICS_PATH}")
    print("Test summary:", json.dumps(metrics["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
