#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


LABELS = {"safe": 0, "unsafe": 1}


def load_jsonl(path: Path) -> List[Dict]:
    # Reads a JSONL dataset file into memory for preprocessing.
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def normalize_rows(rows: List[Dict]) -> List[Dict]:
    # Validates labels and converts them into the numeric format the model expects.
    normalized = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        label = str(row.get("label", "")).strip().lower()
        if not text:
            continue
        if label not in LABELS:
            raise ValueError(f"unsupported label '{label}', expected one of {sorted(LABELS)}")
        normalized.append({"text": text, "label": LABELS[label]})
    if not normalized:
        raise ValueError("dataset contains no valid rows")
    return normalized


def build_dataset(path: Path) -> Dataset:
    # Wraps the normalized rows as a Hugging Face dataset.
    return Dataset.from_list(normalize_rows(load_jsonl(path)))


def main() -> None:
    # Fine-tunes a DistilBERT classifier and writes a reusable local checkpoint.
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for prompt security classification.")
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--validation-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--learning-rate", default=5e-5, type=float)
    args = parser.parse_args()

    train_dataset = build_dataset(args.train_file)
    validation_dataset = build_dataset(args.validation_file)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "SAFE", 1: "UNSAFE"},
        label2id={"SAFE": 0, "UNSAFE": 1},
    )

    def tokenize(batch: Dict) -> Dict:
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_dataset = train_dataset.map(tokenize, batched=True)
    validation_dataset = validation_dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=20,
        save_total_limit=2,
        report_to="none",
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
