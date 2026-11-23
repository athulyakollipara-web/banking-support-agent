"""HTTP API. Locally the customer is identified by an ``x-customer-id`` header; on AWS, API Gateway verifies the
customer's Cognito token and the customer id is the token's ``sub``, read from the request context."""
from __future__ import annotations

import json

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from . import __version__
from .config import build_agent, get_settings


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)


def create_app(agent=None) -> FastAPI:
    settings = get_settings()
    agent = agent or build_agent(settings)
    app = FastAPI(title="Banking Support Agent", version=__version__)

    def customer_id(request: Request, x_customer_id: str | None) -> str:
        if settings.mode == "aws":
            ctx = json.loads(request.headers.get("x-amzn-request-context", "{}"))
            sub = ctx.get("authorizer", {}).get("jwt", {}).get("claims", {}).get("sub")
            if not sub:
                raise HTTPException(401, "missing authenticated customer")
            return sub
        return x_customer_id or "cust_001"

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"status": "ok", "version": __version__, "mode": settings.mode,
                "model": {k: agent.model.meta.get(k) for k in ("C", "accept_threshold", "oos_threshold")}}

    @app.post("/v1/chat")
    def chat(body: ChatIn, request: Request, x_customer_id: str | None = Header(default=None)):
        try:
            return agent.turn(body.message, body.session_id, customer_id(request, x_customer_id))
        except PermissionError:
            raise HTTPException(403, "this session belongs to another customer")

    @app.get("/v1/sessions/{session_id}")
    def transcript(session_id: str, request: Request, x_customer_id: str | None = Header(default=None)):
        session = agent.sessions.get(session_id)
        if not session or session["customer_id"] != customer_id(request, x_customer_id):
            raise HTTPException(404, "session not found")
        return {"session_id": session_id, "turns": session["turns"], "handoffs": session["handoffs"]}

    return app


def __getattr__(name):
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
