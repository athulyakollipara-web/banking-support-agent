"""Public datasets, downloaded on first use and checked against pinned hashes.

* Banking77 (PolyAI, CC BY 4.0): 13,083 online-banking customer queries labeled with 77 intents.
  Casanueva et al., "Efficient Intent Detection with Dual Sentence Encoders", 2020.
* CLINC150 out-of-scope queries (CC BY 3.0): requests that belong to no supported intent.
  Larson et al., "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction", 2019.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import urllib.request
from pathlib import Path

SOURCES = {
    "banking77_train.csv": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "banking77_test.csv": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
    "clinc150_full.json": "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json",
}
# Filled in from the first verified download; a changed upstream file fails loudly instead of silently changing results.
SHA256 = {}
_HASH_FILE = Path(__file__).with_name("data_hashes.json")
if _HASH_FILE.exists():
    SHA256 = json.loads(_HASH_FILE.read_text())

# CLINC150 in-scope intents that are clearly not banking. Used as extra "out_of_scope" training examples.
# CLINC's banking and credit-card intents are deliberately excluded because they overlap with Banking77.
NON_BANKING_CLINC = [
    "weather", "travel_alert", "book_flight", "book_hotel", "car_rental", "restaurant_reservation", "recipe",
    "cook_time", "calories", "nutrition_info", "play_music", "change_volume", "tell_joke", "fun_fact", "meaning_of_life",
    "who_made_you", "what_is_your_name", "timer", "alarm", "calendar", "todo_list", "shopping_list", "traffic",
    "directions", "gas", "oil_change_how", "tire_pressure", "jump_start", "uber", "flight_status", "translate",
    "definition", "spelling", "time", "date", "timezone", "measurement_conversion", "smart_home", "reminder",
    "text", "make_call", "find_phone", "sync_device", "roll_dice", "flip_coin", "next_song", "plug_type", "vaccines",
]


def data_dir() -> Path:
    d = Path(os.getenv("BSA_HOME", Path.home() / ".bsa")) / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch(name: str) -> Path:
    path = data_dir() / name
    if not path.exists():
        with urllib.request.urlopen(SOURCES[name], timeout=60) as r:
            path.write_bytes(r.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if name in SHA256 and SHA256[name] != digest:
        raise RuntimeError(f"{name} does not match its pinned hash; upstream changed. Delete {path} to re-download and review.")
    return path


def _csv(path: Path) -> list[tuple[str, str]]:
    return [(r["text"].strip(), r["category"]) for r in csv.DictReader(io.StringIO(path.read_text()))]


def banking77() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    return _csv(fetch("banking77_train.csv")), _csv(fetch("banking77_test.csv"))


def clinc() -> dict:
    return json.loads(fetch("clinc150_full.json").read_text())


def out_of_scope() -> dict[str, list[str]]:
    """Out-of-scope examples: CLINC's own OOS split plus non-banking CLINC intents (train only)."""
    d = clinc()
    extra = [t for t, intent in d["train"] if intent in NON_BANKING_CLINC]
    return {"train": [t for t, _ in d["oos_train"]] + extra,
            "val": [t for t, _ in d["oos_val"]] + [t for t, i in d["val"] if i in NON_BANKING_CLINC],
            "test": [t for t, _ in d["oos_test"]]}
