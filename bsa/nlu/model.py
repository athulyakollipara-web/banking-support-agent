"""Intent model: word and character n-gram TF-IDF features with a multinomial logistic regression.

It predicts one of Banking77's 77 intents or ``out_of_scope``. The decision policy on top is what the
agent acts on:

* ``accept``        top intent is confident enough (threshold tuned on validation data for a target precision)
* ``clarify``       not confident: ask the customer to pick from the top 3 intents (or "something else")
* ``out_of_scope``  the request is confidently not a banking request (a weak out-of-scope guess is clarified instead)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

OOS = "out_of_scope"


def model_dir() -> Path:
    return Path(os.getenv("BSA_MODEL_DIR", Path(os.getenv("BSA_HOME", Path.home() / ".bsa")) / "model"))


@dataclass
class Decision:
    action: str                                  # accept | clarify | out_of_scope
    intent: str | None                           # set when action == accept
    confidence: float
    ranked: list[tuple[str, float]] = field(default_factory=list)

    @property
    def options(self) -> list[str]:
        return [i for i, _ in self.ranked if i != OOS][:3]


class IntentModel:
    def __init__(self, pipeline, labels: list[str], accept_threshold: float, oos_threshold: float = 0.5,
                 meta: dict | None = None):
        self.pipeline = pipeline
        self.labels = labels
        self.accept_threshold = accept_threshold
        self.oos_threshold = oos_threshold
        self.meta = meta or {}

    @classmethod
    def load(cls, directory: Path | None = None) -> "IntentModel":
        directory = directory or model_dir()
        if not (directory / "intent.joblib").exists():
            from .train import train

            train(directory)
        meta = json.loads((directory / "intent_meta.json").read_text())
        return cls(joblib.load(directory / "intent.joblib"), meta["labels"], meta["accept_threshold"],
                   meta.get("oos_threshold", 0.5), meta)

    def rank(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def decide_from_probs(self, probs: np.ndarray) -> Decision:
        order = np.argsort(-probs)
        ranked = [(self.labels[i], float(probs[i])) for i in order[:5]]
        top, p = ranked[0]
        if top == OOS:
            if p >= self.oos_threshold:
                return Decision("out_of_scope", None, p, ranked)
            return Decision("clarify", None, p, ranked)
        if p >= self.accept_threshold:
            return Decision("accept", top, p, ranked)
        return Decision("clarify", None, p, ranked)

    def decide(self, text: str) -> Decision:
        return self.decide_from_probs(self.rank([text])[0])
