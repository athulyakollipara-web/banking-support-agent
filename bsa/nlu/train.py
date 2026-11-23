"""Train the intent model. All choices (regularization, accept threshold) are made on a validation split;
the Banking77 test set and CLINC out-of-scope test set are only used by eval/intent_eval.py.

    bsa train            # ~30 s on a laptop; logs the run to MLflow if it is installed
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from .. import data
from .model import OOS, model_dir

SEED = 42
C_GRID = [5.0, 20.0, 50.0]
TARGET_PRECISION = 0.97      # auto-accepted intents must be right at least this often (on validation)
OOS_TRAIN_CAP = 1500          # keep out-of-scope from dominating the 77 banking classes


def build_pipeline(C: float) -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=30000)),
    ])
    return Pipeline([("features", features), ("clf", LogisticRegression(C=C, max_iter=3000))])


def splits():
    train, _ = data.banking77()
    texts, labels = zip(*train)
    tr_x, va_x, tr_y, va_y = train_test_split(list(texts), list(labels), test_size=0.15, stratify=labels, random_state=SEED)
    oos = data.out_of_scope()
    rng = random.Random(SEED)
    oos_train = rng.sample(oos["train"], min(OOS_TRAIN_CAP, len(oos["train"])))
    return (tr_x + oos_train, tr_y + [OOS] * len(oos_train)), (va_x + oos["val"], va_y + [OOS] * len(oos["val"]))


def pick_threshold(probs: np.ndarray, labels: list[str], y: list[str], target: float) -> tuple[float, dict]:
    """Lowest threshold whose auto-accepted banking predictions reach the target precision on validation."""
    top = probs.argmax(1)
    conf = probs.max(1)
    pred = np.array([labels[i] for i in top])
    gold = np.array(y)
    banking_pred = pred != OOS
    best = (1.0, {})
    for t in np.round(np.arange(0.95, 0.04, -0.01), 2):
        accepted = banking_pred & (conf >= t)
        if accepted.sum() == 0:
            continue
        precision = float((pred[accepted] == gold[accepted]).mean())
        if precision >= target:
            best = (float(t), {"precision": round(precision, 4), "accept_rate": round(float(accepted.mean()), 4)})
    return best


def pick_oos_threshold(probs: np.ndarray, labels: list[str], y: list[str]) -> tuple[float, dict]:
    """When the top guess is out-of-scope but weak, ask the customer instead of turning them away.
    Choose the threshold on validation that best balances banking queries wrongly turned away against
    out-of-scope queries that end up in a clarification question."""
    oos = labels.index(OOS)
    top = probs.argmax(1)
    gold = np.array(y)
    is_oos = gold == OOS
    best = (0.5, {}, -1.0)
    for t in np.round(np.arange(0.3, 0.91, 0.05), 2):
        turned_away = (top == oos) & (probs[:, oos] >= t)
        banking_lost = float(turned_away[~is_oos].mean())
        oos_caught = float(turned_away[is_oos].mean())
        score = oos_caught - 5 * banking_lost            # turning away a real customer is 5x worse than one extra question
        if score > best[2]:
            best = (float(t), {"banking_turned_away": round(banking_lost, 4), "oos_recognised": round(oos_caught, 4)}, score)
    return best[0], best[1]


def train(out_dir: Path | None = None, verbose: bool = True) -> dict:
    out_dir = out_dir or model_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (tr_x, tr_y), (va_x, va_y) = splits()
    results = []
    for C in C_GRID:
        pipe = build_pipeline(C).fit(tr_x, tr_y)
        acc = float((pipe.predict(va_x) == np.array(va_y)).mean())
        results.append((acc, C, pipe))
        if verbose:
            print(f"C={C:<5} validation accuracy {acc:.4f}", flush=True)
    acc, C, pipe = max(results, key=lambda r: r[0])
    labels = list(pipe.classes_)
    probs = pipe.predict_proba(va_x)
    threshold, at = pick_threshold(probs, labels, va_y, TARGET_PRECISION)
    oos_threshold, oos_at = pick_oos_threshold(probs, labels, va_y)
    meta = {"labels": labels, "C": C, "accept_threshold": threshold, "oos_threshold": oos_threshold,
            "target_precision": TARGET_PRECISION,
            "validation": {"accuracy": round(acc, 4), **at, **oos_at}, "train_size": len(tr_x), "validation_size": len(va_x),
            "data_sha256": {n: data.SHA256.get(n) for n in data.SOURCES}}
    joblib.dump(pipe, out_dir / "intent.joblib", compress=3)
    (out_dir / "intent_meta.json").write_text(json.dumps(meta, indent=2))
    _log_mlflow(meta, out_dir)
    if verbose:
        print(f"chose C={C}; accept threshold {threshold}, out-of-scope threshold {oos_threshold} -> validation {at} {oos_at}")
    return meta


def _log_mlflow(meta: dict, out_dir: Path) -> None:
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri(f"sqlite:///{out_dir.parent / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("intent-model") is None:       # keep artifacts next to the model, not in the cwd
        mlflow.create_experiment("intent-model", artifact_location=(out_dir.parent / "mlartifacts").as_uri())
    mlflow.set_experiment("intent-model")
    with mlflow.start_run(run_name=f"logreg-C{meta['C']:g}"):
        mlflow.log_params({"C": meta["C"], "accept_threshold": meta["accept_threshold"], "oos_threshold": meta["oos_threshold"],
                           "target_precision": meta["target_precision"], "train_size": meta["train_size"]})
        mlflow.log_metrics({f"val_{k}": v for k, v in meta["validation"].items()})
        mlflow.log_artifact(str(out_dir / "intent_meta.json"))


if __name__ == "__main__":
    train()
