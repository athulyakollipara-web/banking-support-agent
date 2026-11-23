"""One conversation turn as a LangGraph state graph.

    guard ──► resume (a pending step: choice, yes/no, passcode) ──► respond
      │          └─ reply doesn't fit the pending step ──┐
      ├──► human hand-off ──► respond                    ▼
      └──────────────────────────────────────────► understand ──► skill ──► respond
                                                      ├─ clarify (top 3 options) ──► respond
                                                      └─ out of scope ──► respond

The intent model and the policy decide what happens. The language model (when enabled) only words the reply.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .. import kb
from ..guard import is_injection, mask, wants_human
from ..nlu import IntentModel
from ..responder import Responder, TemplateResponder
from ..skills import INTENT_SKILL, SKILLS, label
from ..tools import BankApi
from .flows import Flows, Step, option_text

YES = re.compile(r"^\s*(y|yes|yeah|yep|yup|sure|ok|okay|please do|please|go ahead|do it|confirm|correct)\b", re.I)
NO = re.compile(r"^\s*(n|no|nope|nah|cancel|stop|don't|do not|never ?mind)\b", re.I)
OTP = re.compile(r"^\s*(\d{6})\s*$")
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "last": -1}
SOMETHING_ELSE = re.compile(r"\b(something else|none|neither|other|not (any|one) of (these|those))\b", re.I)
MAX_MISSES = 2
OUT_OF_SCOPE = ("I can only help with your Harbor Bank account: cards, payments, transfers, top-ups, fees and "
                "account questions. What can I help you with?")


class Turn(TypedDict, total=False):
    session: dict
    message: str
    masked: str
    route: str
    decision: dict
    step: Any
    result: dict


def parse_choice(text: str, n: int) -> int | None:
    t = text.strip().lower()
    m = re.fullmatch(r"(?:option|number|#)?\s*(\d)\s*[.)]?", t)
    if m and 1 <= int(m.group(1)) <= n:
        return int(m.group(1)) - 1
    for word, k in ORDINALS.items():
        if re.search(rf"\b{word}\b", t):
            return n - 1 if k == -1 else (k - 1 if k <= n else None)
    return None


class Agent:
    def __init__(self, model: IntentModel, bank: BankApi, sessions, responder: Responder | None = None):
        self.model, self.bank, self.sessions = model, bank, sessions
        self.responder = responder or TemplateResponder()
        self.flows = Flows(bank)
        self.articles = kb.by_intent(kb.load())
        self.graph = self._build()

    # ------------------------------------------------------------------------------------
    def _build(self):
        g = StateGraph(Turn)
        g.add_node("guard", self._guard)
        g.add_node("resume", self._resume)
        g.add_node("understand", self._understand)
        g.add_node("skill", self._skill)
        g.add_node("human", self._human)
        g.add_node("respond", self._respond)
        g.set_entry_point("guard")
        route = lambda s: s["route"]  # noqa: E731
        g.add_conditional_edges("guard", route, {"resume": "resume", "understand": "understand", "human": "human",
                                                 "respond": "respond"})
        g.add_conditional_edges("resume", route, {"understand": "understand", "skill": "skill", "respond": "respond",
                                                  "human": "human"})
        g.add_conditional_edges("understand", route, {"skill": "skill", "respond": "respond"})
        g.add_edge("skill", "respond")
        g.add_edge("human", "respond")
        g.add_edge("respond", END)
        return g.compile()

    def turn(self, message: str, session_id: str | None = None, customer_id: str = "cust_001") -> dict:
        session = (self.sessions.get(session_id) if session_id else None) or {
            "id": session_id or uuid.uuid4().hex, "customer_id": customer_id, "turns": [], "events": [],
            "pending": None, "misses": 0, "handoffs": []}
        if session["customer_id"] != customer_id:
            raise PermissionError("session belongs to another customer")
        out = self.graph.invoke({"session": session, "message": message})
        return out["result"]

    # ------------------------------------------------------------------------------------
    def _guard(self, s: Turn) -> Turn:
        masked, found = mask(s["message"])
        s["masked"] = masked
        if found:
            s["session"]["events"].append({"type": "pii_masked", "kinds": found})
        if is_injection(s["message"]):
            s["session"]["events"].append({"type": "injection_blocked"})
            return {**s, "route": "respond", "step": Step("I can't do that. I can help with your cards, payments, "
                                                          "transfers and account. What do you need?")}
        if wants_human(s["message"]):
            return {**s, "route": "human"}
        return {**s, "route": "resume" if s["session"].get("pending") else "understand"}

    def _resume(self, s: Turn) -> Turn:
        session, pending, text = s["session"], s["session"]["pending"], s["message"]
        kind = pending["type"]
        if kind == "clarify":
            options = pending["options"]
            choice = parse_choice(text, len(options) + 1)
            if choice is not None and choice < len(options):
                intent = options[choice]
                return {**s, "route": "skill", "decision": {"action": "accept", "intent": intent, "confidence": None,
                                                             "via": "clarification"}}
            if choice == len(options) or SOMETHING_ELSE.search(text):
                session["misses"] += 1
                session["pending"] = None
                if session["misses"] >= MAX_MISSES:
                    return {**s, "route": "human", "decision": {"action": "handoff", "reason": "not understood twice"}}
                return {**s, "route": "respond", "step": Step("Sorry about that. Could you tell me in a few words what "
                                                              "you need help with?")}
        elif kind == "select":
            choice = parse_choice(text, len(pending["candidates"]))
            if choice is None:
                digits = re.findall(r"\b\d{4}\b", text)
                matches = [i for i, c in enumerate(pending["candidates"]) if any(d in c["label"] for d in digits)]
                choice = matches[0] if len(matches) == 1 else None
            if choice is not None:
                return {**s, "route": "respond", "step": self.flows.select(pending, choice)}
            if NO.search(text):
                return {**s, "route": "respond", "step": Step("No problem, I've stopped there. Anything else?")}
        elif kind == "confirm":
            if YES.search(text):
                return {**s, "route": "respond", "step": self.flows.confirmed(session, pending)}
            if NO.search(text):
                return {**s, "route": "respond", "step": Step("Okay, I haven't changed anything. Anything else?")}
        elif kind == "otp":
            if OTP.match(text):
                return {**s, "route": "respond", "step": self.flows.otp(session, pending, OTP.match(text).group(1))}
            if NO.search(text):
                return {**s, "route": "respond", "step": Step("Okay, I haven't changed anything. Anything else?")}
        if kind != "clarify" and len(text.split()) <= 3:   # short reply that doesn't fit: ask again, keep the step
            again = {"select": "Please reply with the number from the list, or say cancel.",
                     "confirm": "Please reply yes or no.",
                     "otp": "Please type the 6-digit code we sent, or say cancel."}[kind]
            return {**s, "route": "respond", "step": Step(again, pending=pending)}
        session["pending"] = None          # a longer reply that doesn't fit: treat it as a new request
        return {**s, "route": "understand"}

    def _understand(self, s: Turn) -> Turn:
        d = self.model.decide(s["masked"])
        decision = {"action": d.action, "intent": d.intent, "confidence": round(d.confidence, 3),
                    "top": [(i, round(p, 3)) for i, p in d.ranked[:3]]}
        if d.action == "accept":
            return {**s, "route": "skill", "decision": decision}
        if d.action == "out_of_scope":
            return {**s, "route": "respond", "decision": decision, "step": Step(OUT_OF_SCOPE)}
        options = d.options
        step = Step("I want to make sure I get this right. Is it about: " + option_text(options),
                    pending={"type": "clarify", "options": options})
        return {**s, "route": "respond", "decision": decision, "step": step}

    def _skill(self, s: Turn) -> Turn:
        intent = s["decision"]["intent"]
        skill = INTENT_SKILL.get(intent, "answer")
        step = self.flows.start(skill, intent, s["session"], s["message"], self.articles[intent])
        s["session"]["misses"] = 0
        return {**s, "step": step}

    def _human(self, s: Turn) -> Turn:
        reason = s.get("decision", {}).get("reason", "customer asked for a person")
        step = Step("I'm connecting you with a member of our team. They'll see this conversation, so you won't need to "
                    "repeat yourself.", events=[{"type": "handoff_requested", "queue": "support", "priority": "normal",
                                                "reason": reason}])
        return {**s, "step": step}

    def _respond(self, s: Turn) -> Turn:
        session, step = s["session"], s["step"]
        decision = s.get("decision", {})
        intent = decision.get("intent")
        session["turns"].append({"role": "customer", "text": s.get("masked", s["message"]), "at": _now()})
        ticket = None
        for e in step.events:
            if e.get("type") == "handoff_requested":
                ticket = self.flows.handoff(session, e.get("queue") or SKILLS[INTENT_SKILL.get(intent, "answer")].handoff_queue
                                            or "support", e.get("priority", "normal"), e.get("reason", e.get("queue", "")),
                                            intent)
                session["handoffs"].append(ticket)
        if ticket:
            step.draft += f" Your reference is {ticket['ticket_id']}."
        session["events"] += step.events
        session["pending"] = step.pending
        reply, how = self.responder.write(step.draft, step.article.text if step.article else None)
        session["turns"].append({"role": "agent", "text": reply, "at": _now()})
        self.sessions.put(session)
        result = {"session_id": session["id"], "reply": reply, "decision": decision,
                  "skill": next((e["skill"] for e in step.events if e.get("type") == "skill"), None),
                  "awaiting": (step.pending or {}).get("type"), "handoff": ticket,
                  "events": step.events, "written_by": how,
                  "article": step.article.id if step.article else None}
        return {**s, "result": result}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
