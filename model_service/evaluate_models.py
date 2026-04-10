#!/usr/bin/env python3
import json
from pathlib import Path

from transformers import pipeline


TEST_PATH = Path("model_service/data/shieldlm/test.jsonl")
OLD_MODEL_PATH = "model_service/models/distilbert-security"
NEW_MODEL_PATH = "model_service/models/distilbert-security-shieldlm"

SEMANTIC_MARKERS = [
    ("override", 0.18),
    ("ignore safeguards", 0.24),
    ("ignore previous instructions", 0.30),
    ("hidden prompt", 0.25),
    ("system prompt", 0.25),
    ("developer message", 0.25),
    ("secret", 0.18),
    ("token", 0.15),
    ("api key", 0.18),
    ("bypass", 0.22),
    ("disable safeguards", 0.26),
    ("simulate unrestricted", 0.22),
    ("act as", 0.12),
    ("decode and execute", 0.28),
    ("jailbreak", 0.26),
    ("nuclear codes", 0.82),
    ("launch codes", 0.82),
    ("weapon", 0.28),
]


def heuristic_score(text: str) -> float:
    # Reuses the same fallback-style marker scoring for baseline comparison.
    lowered = text.lower()
    score = 0.04
    for marker, weight in SEMANTIC_MARKERS:
        if marker in lowered:
            score += weight
    return max(0.0, min(score, 1.0))


def load_rows(path: Path) -> list[dict]:
    # Loads the evaluation split from JSONL.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def predict_heuristic(text: str) -> str:
    # Converts the heuristic score into a binary label using the gateway threshold.
    return "unsafe" if heuristic_score(text) >= 0.35 else "safe"


def build_classifier(path: str):
    # Loads a local sequence-classification checkpoint.
    return pipeline(
        "text-classification",
        model=path,
        tokenizer=path,
        truncation=True,
    )


def predict_model(classifier, text: str) -> str:
    # Normalizes model outputs into the same safe/unsafe labels used by training.
    result = classifier(f"channel=input\nrule_score=0.00\ncontent={text}", top_k=None)
    predictions = result[0] if result and isinstance(result[0], list) else result

    unsafe = None
    for item in predictions:
        label = str(item.get("label", "")).upper()
        score = float(item.get("score", 0.0))
        if label in {"UNSAFE", "LABEL_1", "MALICIOUS", "JAILBREAK", "INJECTION"}:
            unsafe = score
            break
        if label in {"SAFE", "LABEL_0", "BENIGN"}:
            unsafe = 1.0 - score

    if unsafe is None:
        unsafe = max(float(item.get("score", 0.0)) for item in predictions)

    return "unsafe" if unsafe >= 0.35 else "safe"


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    # Reports the main binary classification metrics for unsafe prompts.
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == "unsafe" and pred == "unsafe")
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == "safe" and pred == "safe")
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == "safe" and pred == "unsafe")
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == "unsafe" and pred == "safe")

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision_unsafe": precision,
        "recall_unsafe": recall,
        "f1_unsafe": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    # Compares heuristic, small-data model, and ShieldLM-trained model on the test split.
    rows = load_rows(TEST_PATH)
    y_true = [row["label"] for row in rows]

    heuristic_preds = [predict_heuristic(row["text"]) for row in rows]

    old_model = build_classifier(OLD_MODEL_PATH)
    new_model = build_classifier(NEW_MODEL_PATH)

    old_preds = [predict_model(old_model, row["text"]) for row in rows]
    new_preds = [predict_model(new_model, row["text"]) for row in rows]

    report = {
        "rows": len(rows),
        "heuristic": metrics(y_true, heuristic_preds),
        "old_distilbert": metrics(y_true, old_preds),
        "shieldlm_distilbert": metrics(y_true, new_preds),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
