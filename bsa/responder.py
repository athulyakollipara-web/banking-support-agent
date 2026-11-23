"""Turns the agent's draft reply into the message the customer sees.

The draft is built deterministically from tool results and the help-center article, so it is always correct
and complete. With Bedrock enabled, Claude rewrites it to sound natural, but its rewrite is only used if every
number in it (amounts, dates, days, card digits) also appears in the draft or the article. Otherwise the draft
is sent. The language model never decides what action to take.
"""
from __future__ import annotations

import re
from typing import Protocol

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

SYSTEM = ("You are the support assistant of Harbor Bank. Rewrite DRAFT as one short, warm, plain-English reply to the "
          "customer. Keep every fact, amount, date and question in DRAFT. Use only facts from DRAFT and ARTICLE. Do not add "
          "advice, numbers, fees or promises that are not in them. Do not mention DRAFT or ARTICLE. Reply with the message only.")


class Responder(Protocol):
    def write(self, draft: str, article: str | None) -> tuple[str, str]: ...


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER.findall(text)}


class TemplateResponder:
    def write(self, draft, article):
        return draft, "template"


class BedrockResponder:
    def __init__(self, model_id: str, client=None, region: str | None = None):
        import boto3

        self.model_id = model_id
        self.client = client or boto3.client("bedrock-runtime", region_name=region)

    def write(self, draft, article):
        from botocore.exceptions import BotoCoreError, ClientError

        prompt = f"DRAFT:\n{draft}\n\nARTICLE:\n{article or '(none)'}"
        try:
            out = self.client.converse(modelId=self.model_id, system=[{"text": SYSTEM}],
                                       messages=[{"role": "user", "content": [{"text": prompt}]}],
                                       inferenceConfig={"maxTokens": 400, "temperature": 0.2})
            text = "".join(b.get("text", "") for b in out["output"]["message"]["content"]).strip()
        except (BotoCoreError, ClientError):
            return draft, "template (model unavailable)"
        if not text or not numbers(text) <= numbers(draft) | numbers(article or ""):
            return draft, "template (rewrite rejected: unsupported number)"
        return text, "llm"
