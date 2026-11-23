import pytest

from bsa.agent import Agent
from bsa.nlu import IntentModel
from bsa.sessions import MemorySessions
from bsa.tools import DemoBank

OTP = "246810"


@pytest.fixture(scope="session")
def model():
    return IntentModel.load()       # trains on first use (downloads Banking77 + CLINC150 from GitHub)


@pytest.fixture
def bank():
    return DemoBank(otp_code=OTP)


@pytest.fixture
def agent(model, bank):
    return Agent(model, bank, MemorySessions())


def writes(bank):
    return [name for name, _ in bank.calls if name in {"freeze_card", "activate_card", "cancel_transfer", "open_dispute"}]
