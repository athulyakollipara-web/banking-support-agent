"""Settings from the environment.

    BSA_MODE           local (default: in-memory sessions, demo bank, template replies) | aws
    BSA_HOME           where datasets, the trained model and MLflow runs live (default ~/.bsa)
    BSA_MODEL_DIR      trained model location (default $BSA_HOME/model)
    AWS_REGION, SESSIONS_TABLE, BEDROCK_MODEL_ID   (aws mode)
    BSA_LLM            on | off: let Claude word replies (aws mode, default on)
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mode: str = "local"
    region: str = "us-east-1"
    sessions_table: str | None = None
    bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    llm: bool = True


def get_settings() -> Settings:
    return Settings(mode=os.getenv("BSA_MODE", "local"), region=os.getenv("AWS_REGION", "us-east-1"),
                    sessions_table=os.getenv("SESSIONS_TABLE"),
                    bedrock_model_id=os.getenv("BEDROCK_MODEL_ID", Settings.bedrock_model_id),
                    llm=os.getenv("BSA_LLM", "on") == "on")


def build_agent(settings: Settings | None = None, **overrides):
    from .agent import Agent
    from .nlu import IntentModel
    from .responder import BedrockResponder, TemplateResponder
    from .sessions import DynamoSessions, MemorySessions
    from .tools import DemoBank

    settings = settings or get_settings()
    if settings.mode == "aws":
        sessions = DynamoSessions(settings.sessions_table, region=settings.region)
        responder = BedrockResponder(settings.bedrock_model_id, region=settings.region) if settings.llm else TemplateResponder()
    else:
        sessions, responder = MemorySessions(), TemplateResponder()
    # The demo bank stands in for the client's core-banking API in both modes (see bsa/tools/bank.py).
    return Agent(overrides.pop("model", None) or IntentModel.load(), overrides.pop("bank", None) or DemoBank(),
                 overrides.pop("sessions", None) or sessions, overrides.pop("responder", None) or responder)
