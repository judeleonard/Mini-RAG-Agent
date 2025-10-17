import pytest
from unittest.mock import patch
import json

@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert "model_backend" in r.json()

@pytest.mark.asyncio
async def test_query_respond_flow(client, monkeypatch):
    from app.agents import Agent
    async def fake_decide(self, user_text):
        return {"action": "RESPOND", "why": "unit test"}
    monkeypatch.setattr(Agent, "decide", fake_decide)
    
    async def fake_synth(self, user_text, contexts):
        self.trace.final_answer = "This is a test answer"
        self.trace.provenance = "internal"
        return self.trace.final_answer
    monkeypatch.setattr(Agent, "synthesize", fake_synth)
    payload = {"text": "Hello, what's up?"}
    r = await client.post("/query", json=payload)
    assert r.status_code == 200
    t = r.json()["trace"]
    assert t["final_answer"] == "This is a test answer"


# test query tool endpoint with retrieval
@pytest.mark.asyncio
async def test_list_tools(client):
    r = await client.get("/tools")
    assert r.status_code == 200
    body = r.json()
    assert "tools" in body
    assert any(t["name"] == "date_diff" for t in body["tools"])
