"""Held-out evaluation of the intent model and its decision policy.

Test data never used for training or tuning:
  * Banking77 test set: 3,080 banking queries, 77 intents
  * CLINC150 out-of-scope test set: 1,000 requests that are not banking requests

    python eval/intent_eval.py          # writes eval/results/intent_eval.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from bsa import data
from bsa.nlu.model import OOS, IntentModel
from bsa.skills import INTENT_SKILL, SKILLS


def main(out="eval/results/intent_eval.json"):
    model = IntentModel.load()
    _, test = data.banking77()
    texts, gold = zip(*test)
    oos_texts = data.out_of_scope()["test"]

    probs = model.rank(list(texts))
    decisions = [model.decide_from_probs(p) for p in probs]
    top1 = [model.labels[i] for i in probs.argmax(1)]
    top3 = [[model.labels[i] for i in np.argsort(-p)[:4] if model.labels[i] != OOS][:3] for p in probs]

    n = len(gold)
    acc = sum(t == g for t, g in zip(top1, gold)) / n
    acts = Counter(d.action for d in decisions)
    accepted = [(d, g) for d, g in zip(decisions, gold) if d.action == "accept"]
    clarified = [(d, g) for d, g in zip(decisions, gold) if d.action == "clarify"]
    accepted_right = sum(d.intent == g for d, g in accepted)
    clarify_hit = sum(g in d.options for d, g in clarified)

    def skill(i):
        return INTENT_SKILL.get(i, "answer")
    wrong_skill = [(d, g) for d, g in accepted if skill(d.intent) != skill(g)]
    wrong_write = [(d, g) for d, g in wrong_skill if SKILLS[skill(d.intent)].writes]

    oos_dec = [model.decide_from_probs(p) for p in model.rank(oos_texts)]
    oos_acts = Counter(d.action for d in oos_dec)

    confusions = Counter((g, t) for g, t in zip(gold, top1) if g != t).most_common(8)
    per_intent = {}
    for intent in sorted(set(gold)):
        idx = [i for i, g in enumerate(gold) if g == intent]
        per_intent[intent] = round(sum(top1[i] == intent for i in idx) / len(idx), 3)
    worst = sorted(per_intent.items(), key=lambda kv: kv[1])[:8]

    result = {
        "model": {k: model.meta.get(k) for k in ("C", "accept_threshold", "oos_threshold", "target_precision")},
        "banking77_test": {
            "queries": n,
            "top1_accuracy": round(acc, 4),
            "top3_accuracy": round(sum(g in t for g, t in zip(gold, top3)) / n, 4),
            "policy": {
                "auto_accepted": round(acts["accept"] / n, 4),
                "accuracy_when_auto_accepted": round(accepted_right / max(1, len(accepted)), 4),
                "clarified": round(acts["clarify"] / n, 4),
                "right_intent_among_clarify_options": round(clarify_hit / max(1, len(clarified)), 4),
                "wrongly_called_out_of_scope": round(acts["out_of_scope"] / n, 4),
                "resolved_correctly": round((accepted_right + clarify_hit) / n, 4),
                "wrong_intent_acted_on": round((len(accepted) - accepted_right) / n, 4),
                "wrong_skill_started": round(len(wrong_skill) / n, 4),
                "wrong_write_flow_started": round(len(wrong_write) / n, 4),
            },
            "wrong_write_flow_examples": [{"gold": g, "predicted": d.intent}
                                          for d, g in wrong_write][:10],
            "most_confused_pairs": [{"gold": g, "predicted": p, "count": c} for (g, p), c in confusions],
            "lowest_accuracy_intents": dict(worst),
        },
        "clinc_out_of_scope_test": {
            "queries": len(oos_texts),
            "recognised_as_out_of_scope": round(oos_acts["out_of_scope"] / len(oos_texts), 4),
            "clarified": round(oos_acts["clarify"] / len(oos_texts), 4),
            "wrongly_accepted_as_banking": round(oos_acts["accept"] / len(oos_texts), 4),
            "examples_wrongly_accepted": [(t, d.intent) for t, d in zip(oos_texts, oos_dec) if d.action == "accept"][:8],
        },
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main(*sys.argv[1:])
