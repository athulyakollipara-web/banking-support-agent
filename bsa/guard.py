"""Checks on every customer message before the agent does anything with it."""
from __future__ import annotations

import re

INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)"
    r"|disregard (your|the|all) (instructions|rules|guidelines)"
    r"|(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)"
    r"|you are now (a|an|in) |developer mode|jailbreak|\bDAN\b|pretend (that )?you (are|have) no (rules|restrictions)"
    r"|act as (an? )?(admin|administrator|bank employee|developer)",
    re.I)
HUMAN = re.compile(
    r"\b(talk|speak|chat)\s+(to|with)\s+(a\s+|an\s+|real\s+|someone|somebody)*(human|person|agent|representative|someone|somebody|staff)\b"
    r"|^\s*(human|agent|representative|real person|operator)\s*[.!?]*\s*$"
    r"|\b(get|want|need)\s+(me\s+)?(a\s+)?(human|real person|representative)\b",
    re.I)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch) * (2 if alt else 1)
        total += d - 9 if d > 9 else d
        alt = not alt
    return total % 10 == 0


def mask(text: str) -> tuple[str, list[str]]:
    """Mask full card numbers and SSNs. Returns the masked text and what was found."""
    found = []

    def card(m):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            found.append("card number")
            return f"[card ending {digits[-4:]}]"
        return m.group()

    text = _CARD.sub(card, text)
    if _SSN.search(text):
        found.append("SSN")
        text = _SSN.sub("[SSN]", text)
    return text, found


def is_injection(text: str) -> bool:
    return bool(INJECTION.search(text))


def wants_human(text: str) -> bool:
    return bool(HUMAN.search(text))
