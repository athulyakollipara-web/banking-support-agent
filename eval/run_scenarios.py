"""Run the multi-turn scenarios in eval/scenarios.yaml and report which pass.

    python eval/run_scenarios.py        # writes eval/results/scenarios.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from bsa.config import build_agent
from bsa.sessions import MemorySessions
from bsa.tools import DemoBank

WRITES = {"freeze_card", "activate_card", "cancel_transfer", "open_dispute"}
HERE = Path(__file__).resolve().parent


def check_turn(t: dict, r: dict, calls: list[str], session: dict) -> list[str]:
    errors = []
    d = r["decision"]
    if "action" in t and d.get("action") != t["action"]:
        errors.append(f"action {d.get('action')} != {t['action']}")
    if "intent" in t and d.get("intent") != t["intent"]:
        errors.append(f"intent {d.get('intent')} != {t['intent']} (top: {d.get('top')})")
    if "skill" in t and r["skill"] != t["skill"]:
        errors.append(f"skill {r['skill']} != {t['skill']}")
    if "awaiting" in t and r["awaiting"] != t["awaiting"]:
        errors.append(f"awaiting {r['awaiting']} != {t['awaiting']}")
    for s in t.get("says", []):
        if s not in r["reply"]:
            errors.append(f"reply lacks {s!r}")
    for c in t.get("calls", []):
        if c not in calls:
            errors.append(f"tool {c} not called (called {calls})")
    if t.get("no_calls") and calls:
        errors.append(f"unexpected tool calls {calls}")
    if t.get("no_writes") and WRITES & set(calls):
        errors.append(f"write tool called {WRITES & set(calls)}")
    if "handoff" in t and (r["handoff"] or {}).get("queue") != t["handoff"]:
        errors.append(f"handoff {r['handoff']} != {t['handoff']}")
    if t.get("no_handoff") and r["handoff"]:
        errors.append(f"unexpected handoff {r['handoff']}")
    if "masked_in_transcript" in t and t["masked_in_transcript"] in json.dumps(session["turns"]):
        errors.append("raw value stored in transcript")
    return errors


def run(path: Path = HERE / "scenarios.yaml", model=None) -> list[dict]:
    results = []
    for sc in yaml.safe_load(path.read_text()):
        bank = DemoBank(otp_code="246810")
        agent = build_agent(bank=bank, sessions=MemorySessions(), **({"model": model} if model else {}))
        sid, errors = None, []
        for n, t in enumerate(sc["turns"], 1):
            before = len(bank.calls)
            r = agent.turn(t["say"], sid)
            sid = r["session_id"]
            calls = [name for name, _ in bank.calls[before:]]
            errors += [f"turn {n} ({t['say']!r}): {e}" for e in check_turn(t, r, calls, agent.sessions.get(sid))]
        results.append({"name": sc["name"], "turns": len(sc["turns"]), "passed": not errors, "errors": errors})
    return results


def main(out="eval/results/scenarios.json"):
    results = run()
    passed = sum(r["passed"] for r in results)
    for r in results:
        print(("PASS " if r["passed"] else "FAIL ") + r["name"])
        for e in r["errors"]:
            print("     " + e)
    summary = {"scenarios": len(results), "passed": passed, "turns": sum(r["turns"] for r in results), "results": results}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"\n{passed}/{len(results)} scenarios passed ({summary['turns']} turns)")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
