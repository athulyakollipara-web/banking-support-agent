import json

import pytest

from bsa.agent import Agent
from bsa.responder import TemplateResponder
from bsa.sessions import MemorySessions

from .conftest import OTP, writes


def talk(agent, *messages):
    sid, out = None, []
    for m in messages:
        r = agent.turn(m, sid)
        sid = r["session_id"]
        out.append(r)
    return out


@pytest.mark.parametrize("opening,choice", [("my card was stolen", "1"), ("I want to stop the transfer I sent", None),
                                            ("how do I activate my card", None),
                                            ("there is a transaction on my card that I did not make", "2")])
def test_no_write_happens_before_yes_and_passcode(agent, bank, opening, choice):
    steps = [opening] + ([choice] if choice else []) + ["yes"]
    rs = talk(agent, *steps)
    assert rs[-1]["awaiting"] == "otp" and writes(bank) == []
    r = agent.turn(OTP, rs[-1]["session_id"])
    assert r["awaiting"] is None and len(writes(bank)) == 1


def test_passcode_is_single_use(agent, bank):
    rs = talk(agent, "my card was stolen", "1", "yes", OTP)
    assert writes(bank) == ["freeze_card"]
    assert not bank.check_otp("cust_001", OTP)


def test_sessions_are_isolated_per_customer(agent):
    r = agent.turn("my card was stolen")
    with pytest.raises(PermissionError):
        agent.turn("1", r["session_id"], customer_id="cust_999")


def test_handoff_carries_context_without_raw_card_numbers(agent, bank):
    talk(agent, "my card 4111 1111 1111 1111 isn't working", "I want to speak to a human")
    ticket = bank.handoffs[-1]
    dumped = json.dumps(ticket)
    assert ticket["queue"] == "support" and "card ending 1111" in dumped and "4111 1111" not in dumped


def test_replies_cite_the_help_article(agent):
    r = agent.turn("can I add my card to apple pay")
    assert r["article"] == "virtual-cards" and r["written_by"] == "template"


class Rewriter:
    def __init__(self, text):
        self.text = text

    def write(self, draft, article):
        from bsa.responder import numbers
        if numbers(self.text) <= numbers(draft) | numbers(article or ""):
            return self.text, "llm"
        return draft, "template (rewrite rejected: unsupported number)"


def test_rewrite_with_invented_number_is_rejected(model, bank):
    agent = Agent(model, bank, MemorySessions(), Rewriter("Good news, your card arrives in 2 days and the fee is $15!"))
    r = agent.turn("my new card still hasn't arrived")
    assert r["written_by"].startswith("template") and "$15" not in r["reply"]


def test_template_responder_returns_draft():
    assert TemplateResponder().write("x", None) == ("x", "template")
