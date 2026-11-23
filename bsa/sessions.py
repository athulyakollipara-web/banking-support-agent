"""Conversation state between turns: in memory locally, DynamoDB in AWS (with a TTL so sessions expire)."""
from __future__ import annotations

import json
import time


class MemorySessions:
    def __init__(self):
        self._data: dict[str, dict] = {}

    def get(self, session_id: str) -> dict | None:
        s = self._data.get(session_id)
        return json.loads(json.dumps(s)) if s else None

    def put(self, session: dict) -> None:
        self._data[session["id"]] = json.loads(json.dumps(session))


class DynamoSessions:
    def __init__(self, table: str, client=None, region: str | None = None, ttl_s: int = 24 * 3600):
        import boto3

        self.table, self.ttl_s = table, ttl_s
        self.ddb = client or boto3.client("dynamodb", region_name=region)

    def get(self, session_id: str) -> dict | None:
        item = self.ddb.get_item(TableName=self.table, Key={"session_id": {"S": session_id}}, ConsistentRead=True).get("Item")
        if not item or int(item["ttl"]["N"]) < time.time():
            return None
        return json.loads(item["body"]["S"])

    def put(self, session: dict) -> None:
        self.ddb.put_item(TableName=self.table, Item={
            "session_id": {"S": session["id"]}, "customer_id": {"S": session["customer_id"]},
            "body": {"S": json.dumps(session)}, "ttl": {"N": str(int(time.time()) + self.ttl_s)}})
