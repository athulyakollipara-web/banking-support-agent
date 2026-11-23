"""The bank's systems, as the agent sees them: a small, typed set of tools.

``BankApi`` is the contract. ``DemoBank`` is an in-memory implementation with one demo customer, used locally
and in tests. In a real deployment this is the piece a forward-deployed engineer wires to the client's
core-banking and card-processor APIs; nothing else in the agent changes.

Read tools return data. Write tools (freeze, activate, cancel, dispute) change state; the agent only calls
them after the customer has confirmed and passed a one-time-passcode check.
"""
from __future__ import annotations

import copy
import hmac
import secrets
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol


class ToolError(Exception):
    pass


class BankApi(Protocol):
    def cards(self, customer_id: str) -> list[dict]: ...
    def card_delivery(self, customer_id: str, card_id: str) -> dict: ...
    def transfers(self, customer_id: str, status: str | None = None) -> list[dict]: ...
    def card_payments(self, customer_id: str, status: str | None = None) -> list[dict]: ...
    def top_ups(self, customer_id: str) -> list[dict]: ...
    def exchange_rate(self, base: str, quote: str) -> dict: ...
    def send_otp(self, customer_id: str) -> str: ...
    def check_otp(self, customer_id: str, code: str) -> bool: ...
    def freeze_card(self, customer_id: str, card_id: str) -> dict: ...
    def activate_card(self, customer_id: str, card_id: str) -> dict: ...
    def cancel_transfer(self, customer_id: str, transfer_id: str) -> dict: ...
    def open_dispute(self, customer_id: str, payment_id: str, reason: str) -> dict: ...
    def create_handoff(self, customer_id: str, summary: dict) -> dict: ...


def _seed(today: date) -> dict:
    d = lambda n: (today - timedelta(days=n)).isoformat()  # noqa: E731
    return {
        "cust_001": {
            "name": "Alex Rivera",
            "cards": [
                {"id": "card_4821", "last4": "4821", "type": "physical", "status": "active", "label": "Debit card"},
                {"id": "card_7310", "last4": "7310", "type": "virtual", "status": "active", "label": "Virtual card"},
                {"id": "card_9902", "last4": "9902", "type": "physical", "status": "shipped", "label": "Replacement card",
                 "shipped_on": d(3), "carrier": "USPS", "eta": (today + timedelta(days=2)).isoformat()},
            ],
            "transfers": [
                {"id": "tr_1001", "to": "Jordan Smith", "amount": 250.00, "currency": "USD", "status": "pending",
                 "created": d(0), "reason": "routine security check"},
                {"id": "tr_1000", "to": "Riverside Apartments", "amount": 1450.00, "currency": "USD", "status": "completed",
                 "created": d(2)},
                {"id": "tr_0999", "to": "M. Okafor", "amount": 80.00, "currency": "USD", "status": "failed", "created": d(4),
                 "reason": "recipient bank rejected the account number"},
            ],
            "card_payments": [
                {"id": "pay_501", "merchant": "SQ *BLUE BOTTLE COFFEE", "amount": 6.75, "status": "completed", "date": d(1),
                 "card_id": "card_4821"},
                {"id": "pay_502", "merchant": "AMZN Mktp US", "amount": 42.18, "status": "pending", "date": d(1), "card_id": "card_4821"},
                {"id": "pay_503", "merchant": "STEAMGAMES.COM 4259522", "amount": 59.99, "status": "completed", "date": d(2),
                 "card_id": "card_4821"},
                {"id": "pay_504", "merchant": "SHELL OIL 5744", "amount": 38.40, "status": "declined", "date": d(3),
                 "card_id": "card_4821", "reason": "incorrect PIN"},
                {"id": "pay_505", "merchant": "UBER *TRIP", "amount": 23.10, "status": "completed", "date": d(5), "card_id": "card_7310"},
            ],
            "top_ups": [
                {"id": "tu_31", "method": "debit card ending 5566", "amount": 200.00, "status": "failed", "date": d(1),
                 "reason": "card issuer declined the top-up"},
                {"id": "tu_30", "method": "bank transfer", "amount": 500.00, "status": "completed", "date": d(6)},
            ],
        }
    }


RATES = {("USD", "EUR"): 0.9214, ("USD", "GBP"): 0.7861, ("USD", "INR"): 84.12, ("USD", "MXN"): 18.07,
         ("USD", "CAD"): 1.3962, ("USD", "JPY"): 154.3}


@dataclass
class DemoBank:
    """In-memory core banking for one demo customer. ``otp_code`` fixes the passcode in tests and demos."""

    today: date | None = None
    otp_code: str | None = None

    def __post_init__(self):
        self.today = self.today or date.today()
        self.db = _seed(self.today)
        self._otp: dict[str, str] = {}
        self.calls: list[tuple[str, dict]] = []
        self.handoffs: list[dict] = []
        self.disputes: list[dict] = []

    def _customer(self, customer_id: str) -> dict:
        if customer_id not in self.db:
            raise ToolError(f"unknown customer {customer_id}")
        return self.db[customer_id]

    def _log(self, name, **args):
        self.calls.append((name, args))

    # --- read ---------------------------------------------------------------------------
    def cards(self, customer_id):
        self._log("cards", customer_id=customer_id)
        return copy.deepcopy(self._customer(customer_id)["cards"])

    def card_delivery(self, customer_id, card_id):
        self._log("card_delivery", card_id=card_id)
        card = next((c for c in self._customer(customer_id)["cards"] if c["id"] == card_id), None)
        if not card:
            raise ToolError("card not found")
        if card["status"] != "shipped":
            return {"card_id": card_id, "status": card["status"]}
        return {k: card[k] for k in ("id", "last4", "status", "shipped_on", "carrier", "eta")}

    def transfers(self, customer_id, status=None):
        self._log("transfers", status=status)
        return [copy.deepcopy(t) for t in self._customer(customer_id)["transfers"] if status in (None, t["status"])]

    def card_payments(self, customer_id, status=None):
        self._log("card_payments", status=status)
        return [copy.deepcopy(p) for p in self._customer(customer_id)["card_payments"] if status in (None, p["status"])]

    def top_ups(self, customer_id):
        self._log("top_ups")
        return copy.deepcopy(self._customer(customer_id)["top_ups"])

    def exchange_rate(self, base, quote):
        self._log("exchange_rate", base=base, quote=quote)
        rate = RATES.get((base, quote))
        if rate is None:
            raise ToolError(f"no rate for {base}/{quote}")
        return {"base": base, "quote": quote, "rate": rate, "as_of": self.today.isoformat()}

    # --- step-up authentication --------------------------------------------------------
    def send_otp(self, customer_id):
        self._customer(customer_id)
        code = self.otp_code or f"{secrets.randbelow(10**6):06d}"
        self._otp[customer_id] = code
        self._log("send_otp")
        return "sms to phone ending 0132"

    def check_otp(self, customer_id, code):
        expected = self._otp.get(customer_id)
        ok = bool(expected) and hmac.compare_digest(expected, code.strip())
        if ok:
            del self._otp[customer_id]
        return ok

    # --- write (only after confirmation + OTP) -------------------------------------------
    def _card(self, customer_id, card_id):
        card = next((c for c in self._customer(customer_id)["cards"] if c["id"] == card_id), None)
        if not card:
            raise ToolError("card not found")
        return card

    def freeze_card(self, customer_id, card_id):
        self._log("freeze_card", card_id=card_id)
        card = self._card(customer_id, card_id)
        card["status"] = "frozen"
        return {"card_id": card_id, "last4": card["last4"], "status": "frozen"}

    def activate_card(self, customer_id, card_id):
        self._log("activate_card", card_id=card_id)
        card = self._card(customer_id, card_id)
        if card["status"] not in ("shipped", "delivered"):
            raise ToolError(f"card is {card['status']}, not waiting for activation")
        card["status"] = "active"
        return {"card_id": card_id, "last4": card["last4"], "status": "active"}

    def cancel_transfer(self, customer_id, transfer_id):
        self._log("cancel_transfer", transfer_id=transfer_id)
        t = next((t for t in self._customer(customer_id)["transfers"] if t["id"] == transfer_id), None)
        if not t:
            raise ToolError("transfer not found")
        if t["status"] != "pending":
            raise ToolError(f"transfer is {t['status']} and can no longer be cancelled")
        t["status"] = "cancelled"
        return {"transfer_id": transfer_id, "status": "cancelled", "amount": t["amount"], "to": t["to"]}

    def open_dispute(self, customer_id, payment_id, reason):
        self._log("open_dispute", payment_id=payment_id, reason=reason)
        p = next((p for p in self._customer(customer_id)["card_payments"] if p["id"] == payment_id), None)
        if not p:
            raise ToolError("payment not found")
        case = {"case_id": f"DSP-{len(self.disputes) + 7001}", "payment_id": payment_id, "merchant": p["merchant"],
                "amount": p["amount"], "reason": reason, "status": "under review", "decision_within_business_days": 10}
        self.disputes.append(case)
        return case

    def create_handoff(self, customer_id, summary):
        self._log("create_handoff", queue=summary.get("queue"))
        ticket = {"ticket_id": f"HO-{len(self.handoffs) + 3001}", "queue": summary.get("queue", "support"),
                  "priority": summary.get("priority", "normal"), "summary": summary}
        self.handoffs.append(ticket)
        return {k: ticket[k] for k in ("ticket_id", "queue", "priority")}
