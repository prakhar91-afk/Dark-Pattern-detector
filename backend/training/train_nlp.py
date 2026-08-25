"""
training/train_nlp.py — Fine-tune DistilBERT on ec-darkpattern dataset
=======================================================================
Trains a binary dark-pattern text classifier using HuggingFace Trainer.
Run data_prep.py first to generate train.csv / val.csv.

Usage:
    python training/train_nlp.py [--model distilbert-base-uncased] [--epochs 3]
"""

from __future__ import annotations
import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune NLP model for dark pattern detection")
    parser.add_argument("--model",      default="distilbert-base-uncased",
                        help="HuggingFace model name or path")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max-length", type=int,   default=128)
    parser.add_argument("--data-dir",   type=str,   default="data")
    parser.add_argument("--output-dir", type=str,   default="checkpoints/nlp")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback,
        )
        from datasets import Dataset, DatasetDict
    except ImportError as e:
        print(f"[train_nlp] Missing dependency: {e}")
        print("  Install with: pip install transformers datasets torch accelerate")
        return

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────
    for split in ("train", "val", "test"):
        if not (data_dir / f"{split}.csv").exists():
            print(f"[train_nlp] Missing {split}.csv — run data_prep.py first")
            return

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df   = pd.read_csv(data_dir / "val.csv")
    test_df  = pd.read_csv(data_dir / "test.csv")

    # Ensure integer labels
    for df in (train_df, val_df, test_df):
        df["label"] = df["label"].astype(int)

    print(f"[train_nlp] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"[train_nlp] Dark pattern rate: train={train_df['label'].mean():.1%}")

    # ── Tokenizer ─────────────────────────────────────────────────
    print(f"\n[train_nlp] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df[["text", "label"]]),
        "val":   Dataset.from_pandas(val_df[["text", "label"]]),
        "test":  Dataset.from_pandas(test_df[["text", "label"]]),
    })
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])

    # ── Model ─────────────────────────────────────────────────────
    print(f"[train_nlp] Loading model: {args.model}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label={0: "clean", 1: "dark_pattern"},
        label2id={"clean": 0, "dark_pattern": 1},
    )

    # ── Metrics ───────────────────────────────────────────────────
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc = accuracy_score(labels, preds)
        return {
            "accuracy":  round(acc, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        }

    # ── Training arguments ────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = str(output_dir / "checkpoints"),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size * 2,
        learning_rate               = args.lr,
        weight_decay                = 0.01,
        warmup_ratio                = 0.1,
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        logging_steps               = 50,
        report_to                   = "none",
        fp16                        = torch.cuda.is_available(),
        dataloader_num_workers      = 0,
    )

    # ── Trainer ───────────────────────────────────────────────────
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = ds["train"],
        eval_dataset    = ds["val"],
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\n[train_nlp] Starting training ({args.epochs} epochs)...")
    trainer.train()

    # ── Evaluate on test set ──────────────────────────────────────
    print("\n[train_nlp] Evaluating on test set...")
    test_results = trainer.evaluate(ds["test"])
    print(f"[train_nlp] Test results: {test_results}")

    # ── Save best model ───────────────────────────────────────────
    best_model_path = output_dir / "best_model"
    trainer.save_model(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))

    # Save metrics
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"   Best model saved to: {best_model_path}")
    print(f"   Test F1: {test_results.get('eval_f1', 'N/A')}")


if __name__ == "__main__":
    main()
