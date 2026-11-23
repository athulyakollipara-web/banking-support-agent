import json
import time

import boto3
from botocore.stub import ANY, Stubber
from fastapi.testclient import TestClient

from bsa.api import create_app
from bsa.responder import BedrockResponder
from bsa.sessions import DynamoSessions


def client(name):
    return boto3.client(name, region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="x")


def test_chat_api_and_transcript(agent):
    c = TestClient(create_app(agent))
    r = c.post("/v1/chat", json={"message": "where is my new card"}, headers={"x-customer-id": "cust_001"}).json()
    assert "9902" in r["reply"]
    t = c.get(f"/v1/sessions/{r['session_id']}", headers={"x-customer-id": "cust_001"})
    assert t.status_code == 200 and len(t.json()["turns"]) == 2
    assert c.get(f"/v1/sessions/{r['session_id']}", headers={"x-customer-id": "cust_002"}).status_code == 404
    assert c.post("/v1/chat", json={"message": ""}).status_code == 422


def test_aws_mode_takes_customer_from_verified_token(agent, monkeypatch):
    monkeypatch.setenv("BSA_MODE", "aws")
    c = TestClient(create_app(agent))
    assert c.post("/v1/chat", json={"message": "hi"}).status_code == 401
    ctx = json.dumps({"authorizer": {"jwt": {"claims": {"sub": "cust_001"}}}})
    r = c.post("/v1/chat", json={"message": "where is my new card"}, headers={"x-amzn-request-context": ctx})
    assert r.status_code == 200 and "9902" in r.json()["reply"]


def converse_response(text):
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}, "stopReason": "end_turn",
            "usage": {"inputTokens": 50, "outputTokens": 20, "totalTokens": 70}, "metrics": {"latencyMs": 300}}


def test_bedrock_rewrite_is_grounded():
    c = client("bedrock-runtime")
    draft = "Your card ending 9902 should arrive by Nov 25."
    with Stubber(c) as stub:
        stub.add_response("converse", converse_response("Good news! Your card ending 9902 should be with you by Nov 25."), {
            "modelId": "m", "system": [{"text": ANY}], "messages": [{"role": "user", "content": [{"text": ANY}]}],
            "inferenceConfig": {"maxTokens": 400, "temperature": 0.2}})
        stub.add_response("converse", converse_response("Your card ending 9902 arrives Nov 25, and we'll waive the $5 fee."))
        r = BedrockResponder("m", client=c)
        assert r.write(draft, None) == ("Good news! Your card ending 9902 should be with you by Nov 25.", "llm")
        text, how = r.write(draft, None)
    assert text == draft and "rejected" in how


def test_bedrock_failure_falls_back_to_draft():
    c = client("bedrock-runtime")
    with Stubber(c) as stub:
        stub.add_client_error("converse", service_error_code="ThrottlingException", http_status_code=429)
        assert BedrockResponder("m", client=c).write("draft", None) == ("draft", "template (model unavailable)")


def test_dynamo_sessions_roundtrip_and_expiry():
    c = client("dynamodb")
    s = DynamoSessions("sessions", client=c)
    session = {"id": "s1", "customer_id": "cust_001", "turns": []}
    with Stubber(c) as stub:
        stub.add_response("put_item", {}, {"TableName": "sessions", "Item": {
            "session_id": {"S": "s1"}, "customer_id": {"S": "cust_001"}, "body": {"S": json.dumps(session)}, "ttl": ANY}})
        stub.add_response("get_item", {"Item": {"body": {"S": json.dumps(session)}, "ttl": {"N": str(int(time.time()) + 60)}}},
                          {"TableName": "sessions", "Key": {"session_id": {"S": "s1"}}, "ConsistentRead": True})
        stub.add_response("get_item", {"Item": {"body": {"S": json.dumps(session)}, "ttl": {"N": str(int(time.time()) - 1)}}})
        s.put(session)
        assert s.get("s1") == session
        assert s.get("s1") is None
