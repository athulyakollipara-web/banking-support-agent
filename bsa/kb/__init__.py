"""Help-center knowledge base: one article per topic, each mapped to the intents it answers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    intents: tuple[str, ...]
    text: str


def load(path: Path | None = None) -> dict[str, Article]:
    raw = yaml.safe_load((path or Path(__file__).with_name("articles.yaml")).read_text())
    return {a["id"]: Article(a["id"], a["title"], tuple(a["intents"]), a["text"].strip()) for a in raw}


def by_intent(articles: dict[str, Article]) -> dict[str, Article]:
    out = {}
    for a in articles.values():
        for intent in a.intents:
            if intent in out:
                raise ValueError(f"intent {intent} is in two articles")
            out[intent] = a
    return out
