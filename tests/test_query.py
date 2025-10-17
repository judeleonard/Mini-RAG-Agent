import pytest
from unittest.mock import patch
import json

@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert "model_backend" in r.json()

# test query tool endpoint with retrieval
@pytest.mark.asyncio
async def test_list_tools(client):
    r = await client.get("/tools")
    assert r.status_code == 200
    body = r.json()
    assert "tools" in body
    assert any(t["name"] == "date_diff" for t in body["tools"])
