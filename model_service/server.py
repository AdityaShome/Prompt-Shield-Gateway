#!/usr/bin/env python3
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Any, Optional, Tuple

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None


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
]


class DistilBertClassifier:
    def __init__(self) -> None:
        self.model_id = os.getenv("DISTILBERT_MODEL_ID", "distilbert-security-classifier")
        self.unsafe_label = os.getenv("DISTILBERT_UNSAFE_LABEL", "LABEL_1")
        self.classifier = self._load_pipeline()

    def _load_pipeline(self):
        # Loads the fine-tuned classifier and drops to heuristic mode if unavailable.
        if pipeline is None:
            print("transformers not installed, falling back to heuristic scorer")
            return None

        try:
            return pipeline(
                "text-classification",
                model=self.model_id,
                tokenizer=self.model_id,
                truncation=True,
            )
        except Exception as exc:  # pragma: no cover
            print(f"failed to load DistilBERT model '{self.model_id}': {exc}")
            print("falling back to heuristic scorer")
            return None

    def score(self, content: str, metadata: Dict[str, Any], channel: str, rule_score: float, encoding_suspected: bool) -> Tuple[float, str]:
        # Returns a semantic risk score plus the backend that produced it.
        if self.classifier is None:
            return (
                semantic_score(content, metadata, channel, rule_score, encoding_suspected),
                "heuristic_fallback",
            )

        enriched = f"channel={channel}\nrule_score={rule_score:.2f}\ncontent={content}"
        result = self.classifier(enriched, top_k=None)
        if isinstance(result, list) and result and isinstance(result[0], list):
            predictions = result[0]
        else:
            predictions = result

        unsafe_probability = self._unsafe_probability(predictions)
        if encoding_suspected:
            unsafe_probability = min(1.0, unsafe_probability + 0.08)
        return unsafe_probability, "distilbert"

    def _unsafe_probability(self, predictions: Any) -> float:
        if isinstance(predictions, dict):
            predictions = [predictions]
        if not isinstance(predictions, list):
            return 0.0

        normalized = []
        for item in predictions:
            label = str(item.get("label", "")).upper()
            score = float(item.get("score", 0.0))
            normalized.append((label, score))

        for label, score in normalized:
            if label == self.unsafe_label.upper():
                return max(0.0, min(score, 1.0))

        for label, score in normalized:
            if label in {"UNSAFE", "INJECTION", "JAILBREAK", "MALICIOUS"}:
                return max(0.0, min(score, 1.0))

        for label, score in normalized:
            if label in {"SAFE", "BENIGN", "LABEL_0"}:
                return max(0.0, min(1.0 - score, 1.0))

        return max(score for _, score in normalized) if normalized else 0.0


CLASSIFIER = DistilBertClassifier()


def semantic_score(content: str, metadata: Dict[str, Any], channel: str, rule_score: float, encoding_suspected: bool) -> float:
    # Keeps the service usable when the trained classifier cannot be loaded.
    text = content.lower()
    score = 0.04

    for marker, weight in SEMANTIC_MARKERS:
        if marker in text:
            score += weight

    if re.search(r"(base64|rot13|hex|decode)", text):
        score += 0.12

    if channel == "context" and "tool" in text:
        score += 0.10

    if channel == "output" and re.search(r"(system prompt|api key|secret|token)", text):
        score += 0.35

    if metadata.get("sensitive_action"):
        score += 0.10

    if encoding_suspected:
        score += 0.18

    score += min(rule_score * 0.25, 0.20)
    return max(0.0, min(score, 1.0))


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        # Scores a prompt and returns the model probability expected by Rust.
        if self.path != "/score":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)

        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"invalid_json"}')
            return

        model_score, backend = CLASSIFIER.score(
            payload.get("content", ""),
            payload.get("metadata", {}) or {},
            payload.get("channel", "input"),
            float(payload.get("rule_score", 0.0) or 0.0),
            bool(payload.get("encoding_suspected", False)),
        )

        result = {
            "model_score": model_score,
            "backend": backend,
        }

        body = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        # Reports whether the service is running with DistilBERT or fallback heuristics.
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return

        body = json.dumps(
            {
                "status": "ok",
                "backend": "distilbert" if CLASSIFIER.classifier is not None else "heuristic_fallback",
                "model_id": CLASSIFIER.model_id,
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    # Starts the local model-scoring service on the configured port.
    port = int(os.getenv("MODEL_SERVICE_PORT", "8090"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"python model service listening on 0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
