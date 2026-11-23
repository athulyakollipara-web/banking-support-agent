"""Skill flows. Each returns a draft reply and, for multi-step flows, the next pending step.

Write actions always follow the same path, enforced here and covered by tests:
    choose the item (select)  ->  customer says yes (confirm)  ->  one-time passcode (otp)  ->  act
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from ..kb import Article
from ..skills import SKILLS, label
from ..tools import BankApi, ToolError

CURRENCY_WORDS = {"euro": "EUR", "euros": "EUR", "pound": "GBP", "pounds": "GBP", "sterling": "GBP", "rupee": "INR",
                  "rupees": "INR", "peso": "MXN", "pesos": "MXN", "yen": "JPY", "canadian": "CAD"}
MAX_OTP_ATTEMPTS = 3


@dataclass
class Step:
    draft: str
    pending: dict | None = None
    events: list[dict] = field(default_factory=list)
    article: Article | None = None
    handoff: dict | None = None


def money(x: float) -> str:
    return f"${x:,.2f}"


def nice_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d:%b} {d.day}"


def card_name(c: dict) -> str:
    return f"{c['label']} ending {c['last4']}"


def payment_name(p: dict) -> str:
    return f"{money(p['amount'])} to {p['merchant']} on {nice_date(p['date'])}"


def transfer_name(t: dict) -> str:
    return f"{money(t['amount'])} to {t['to']} on {nice_date(t['created'])}"


class Flows:
    def __init__(self, bank: BankApi):
        self.bank = bank

    # ---- entry -------------------------------------------------------------------------
    def start(self, skill: str, intent: str, session: dict, message: str, article: Article) -> Step:
        handler = getattr(self, f"_start_{skill}")
        step = handler(session["customer_id"], intent, message, article)
        step.article = step.article or article
        step.events.insert(0, {"type": "skill", "skill": skill, "intent": intent})
        return step

    # ---- read-only skills ---------------------------------------------------------------
    def _start_answer(self, cid, intent, message, article):
        return Step(article.text)

    def _start_track_card(self, cid, intent, message, article):
        shipped = [c for c in self.bank.cards(cid) if c["status"] == "shipped"]
        if not shipped:
            return Step("I can't see a card on its way to you right now. " + article.text)
        lines = []
        for c in shipped:
            d = self.bank.card_delivery(cid, c["id"])
            lines.append(f"Your {card_name(c).lower()} was shipped by {d['carrier']} on {nice_date(d['shipped_on'])} "
                         f"and should arrive by {nice_date(d['eta'])}.")
        return Step(" ".join(lines) + " If it hasn't arrived after 10 business days, tell me and we'll send a replacement for free.")

    def _start_transfer_status(self, cid, intent, message, article):
        ts = [t for t in self.bank.transfers(cid) if t["status"] in ("pending", "failed")]
        if not ts:
            return Step("All your recent transfers have completed. " + article.text)
        parts = []
        for t in ts:
            why = f": {t['reason']}" if t.get("reason") else ""
            parts.append(f"Your transfer of {transfer_name(t)} is {t['status']}{why}.")
        tail = " Money from a failed transfer returns to your balance within 2 business days." if any(
            t["status"] == "failed" for t in ts) else ""
        return Step(" ".join(parts) + tail)

    def _start_declined_payment(self, cid, intent, message, article):
        ps = self.bank.card_payments(cid, "declined")
        if not ps:
            return Step("I can't see any declined card payments recently. " + article.text)
        return Step(" ".join(f"Your payment of {payment_name(p)} was declined because of an {p['reason']}."
                             if p["reason"][0] in "aeiou" else f"Your payment of {payment_name(p)} was declined: {p['reason']}."
                             for p in ps) + " You can try again; after 3 wrong PIN attempts the PIN is blocked.")

    def _start_pending_payment(self, cid, intent, message, article):
        ps = self.bank.card_payments(cid, "pending")
        if not ps:
            return Step("You have no pending card payments. " + article.text)
        return Step(" ".join(f"Your payment of {payment_name(p)} is pending." for p in ps) + " " + article.text)

    def _start_top_up_status(self, cid, intent, message, article):
        tus = [t for t in self.bank.top_ups(cid) if t["status"] != "completed"]
        if not tus:
            return Step("Your recent top-ups have all completed. " + article.text)
        return Step(" ".join(f"Your {money(t['amount'])} top-up from your {t['method']} on {nice_date(t['date'])} "
                             f"{t['status']}" + (f": {t['reason']}." if t.get("reason") else ".") for t in tus)
                    + " A failed top-up is returned to the card you used within 5 business days.")

    def _start_exchange_rate(self, cid, intent, message, article):
        codes = [c for c in re.findall(r"\b[A-Z]{3}\b", message.upper()) if c != "USD"]
        codes += [CURRENCY_WORDS[w] for w in re.findall(r"[a-z]+", message.lower()) if w in CURRENCY_WORDS]
        codes = list(dict.fromkeys(codes)) or ["EUR", "GBP"]
        lines = []
        for code in codes[:3]:
            try:
                r = self.bank.exchange_rate("USD", code)
                lines.append(f"1 USD = {r['rate']} {code}")
            except ToolError:
                lines.append(f"I don't have a rate for {code}")
        return Step("Today's rates: " + "; ".join(lines) + ". Exchanging in the app costs 0.5% above the interbank rate.")

    # ---- hand-offs ----------------------------------------------------------------------
    def handoff(self, session: dict, queue: str, priority: str, reason: str, intent: str | None = None) -> dict:
        summary = {"queue": queue, "priority": priority, "reason": reason, "intent": intent,
                   "customer_id": session["customer_id"], "recent_turns": session["turns"][-6:],
                   "actions": [e for e in session.get("events", []) if e.get("type") == "action"]}
        return self.bank.create_handoff(session["customer_id"], summary)

    def _start_handoff_closure(self, cid, intent, message, article):
        return Step(article.text + " I've passed this to our team and someone will contact you.",
                    events=[{"type": "handoff_requested", "queue": "account-closure", "priority": "normal"}])

    def _start_handoff_security(self, cid, intent, message, article):
        return Step(article.text + " I've flagged this to our security team as urgent.",
                    events=[{"type": "handoff_requested", "queue": "security", "priority": "high"}])

    # ---- write skills: choose -> confirm -> OTP -> act ----------------------------------
    def _start_freeze_card(self, cid, intent, message, article):
        cards = [c for c in self.bank.cards(cid) if c["status"] in ("active", "shipped")]
        return self._choose("freeze_card", cards, card_name, "Which card should I freeze?", "You have no cards that can be frozen.")

    def _start_activate_card(self, cid, intent, message, article):
        cards = [c for c in self.bank.cards(cid) if c["status"] == "shipped"]
        return self._choose("activate_card", cards, card_name, "Which card would you like to activate?",
                            "You don't have a card waiting to be activated. " + article.text)

    def _start_cancel_transfer(self, cid, intent, message, article):
        ts = self.bank.transfers(cid, "pending")
        return self._choose("cancel_transfer", ts, transfer_name, "Which transfer should I cancel?",
                            "You have no pending transfers, and completed transfers can't be cancelled. " + article.text)

    def _start_dispute_unrecognised(self, cid, intent, message, article):
        ps = [p for p in self.bank.card_payments(cid) if p["status"] == "completed"][:5]
        return self._choose("dispute_unrecognised", ps, payment_name, "Which payment don't you recognise?",
                            "I can't see recent completed payments to report. " + article.text)

    def _start_dispute_duplicate(self, cid, intent, message, article):
        ps = [p for p in self.bank.card_payments(cid) if p["status"] == "completed"][:5]
        return self._choose("dispute_duplicate", ps, payment_name, "Which payment was charged twice?",
                            "I can't see recent completed payments. " + article.text)

    def _choose(self, skill, items, name, question, none_text) -> Step:
        if not items:
            return Step(none_text)
        candidates = [{"id": i["id"], "label": name(i)} for i in items]
        if len(candidates) == 1:
            return self._confirm(skill, candidates[0])
        listing = " ".join(f"{n}) {c['label']}." for n, c in enumerate(candidates, 1))
        return Step(f"{question} {listing} Reply with the number.", pending={"type": "select", "skill": skill, "candidates": candidates})

    CONFIRM = {
        "freeze_card": "I can freeze your {label_lower}. A frozen card can't be used for new payments, and you can unfreeze it in the app. Shall I freeze it?",
        "activate_card": "I can activate your {label_lower}. Shall I go ahead?",
        "cancel_transfer": "I can cancel the transfer of {label}. Shall I cancel it?",
        "dispute_unrecognised": "I'll report the payment of {label} as not recognised and send it to our fraud team. Shall I go ahead?",
        "dispute_duplicate": "I'll open a dispute for the payment of {label} as a duplicate charge. Shall I go ahead?",
    }

    def _confirm(self, skill, target) -> Step:
        text = self.CONFIRM[skill].format(label=target["label"], label_lower=target["label"][0].lower() + target["label"][1:])
        return Step(text + " (yes / no)",
                    pending={"type": "confirm", "skill": skill, "target": target})

    def select(self, pending: dict, choice: int) -> Step:
        return self._confirm(pending["skill"], pending["candidates"][choice])

    def confirmed(self, session: dict, pending: dict) -> Step:
        where = self.bank.send_otp(session["customer_id"])
        return Step(f"To keep your account safe, I've sent a 6-digit code by {where}. Please type it here.",
                    pending={**pending, "type": "otp", "attempts": 0}, events=[{"type": "otp_sent"}])

    def otp(self, session: dict, pending: dict, code: str) -> Step:
        cid = session["customer_id"]
        if not self.bank.check_otp(cid, code):
            attempts = pending["attempts"] + 1
            if attempts >= MAX_OTP_ATTEMPTS:
                return Step("That code didn't match again, so I've stopped here for your security and passed this to our "
                            "security team.", events=[{"type": "otp_failed"}, {"type": "handoff_requested", "queue": "security",
                                                                               "priority": "high"}])
            left = MAX_OTP_ATTEMPTS - attempts
            return Step(f"That code didn't match. Please try again ({left} {'try' if left == 1 else 'tries'} left).",
                        pending={**pending, "attempts": attempts}, events=[{"type": "otp_failed"}])
        return self._act(session, pending["skill"], pending["target"])

    def _act(self, session, skill, target) -> Step:
        cid = session["customer_id"]
        event = {"type": "action", "skill": skill, "target": target["id"]}
        try:
            if skill == "freeze_card":
                r = self.bank.freeze_card(cid, target["id"])
                return Step(f"Done: your card ending {r['last4']} is frozen. You can unfreeze it in the app if you find it. "
                            "A replacement card is free if it was lost, stolen or compromised, and if you see payments you "
                            "don't recognise, tell me and I'll report them.", events=[event])
            if skill == "activate_card":
                r = self.bank.activate_card(cid, target["id"])
                return Step(f"Your card ending {r['last4']} is now active. Make one chip-and-PIN payment to turn on contactless.",
                            events=[event])
            if skill == "cancel_transfer":
                r = self.bank.cancel_transfer(cid, target["id"])
                return Step(f"The {money(r['amount'])} transfer to {r['to']} is cancelled and the money stays in your account.",
                            events=[event])
            if skill in ("dispute_unrecognised", "dispute_duplicate"):
                reason = "not recognised" if skill == "dispute_unrecognised" else "duplicate charge"
                r = self.bank.open_dispute(cid, target["id"], reason)
                text = (f"I've opened dispute {r['case_id']} for the {money(r['amount'])} payment to {r['merchant']}. "
                        f"We'll update you within {r['decision_within_business_days']} business days.")
                events = [event]
                if skill == "dispute_unrecognised":
                    text += (" Because you don't recognise it, our fraud team will review it, and I recommend freezing your "
                             "card: just say \"freeze my card\".")
                    events.append({"type": "handoff_requested", "queue": "fraud", "priority": "high"})
                return Step(text, events=events)
        except ToolError as e:
            return Step(f"I couldn't complete that: {e}.", events=[{**event, "error": str(e)}])
        raise ValueError(skill)


def writes(skill: str) -> bool:
    return SKILLS[skill].writes


def option_text(intents: list[str]) -> str:
    return " ".join(f"{n}) {label(i)}." for n, i in enumerate(intents, 1)) + f" {len(intents) + 1}) Something else."
