from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from typing import List, Dict, Any
from app.utils.config import settings
import asyncio


async def embed_texts(texts: List[str]) -> List[List[float]]:
    def fake_embed(s):
        v = [float(ord(c) % 10) for c in s[:32]]
        v = (v + [0]*32)[:32]
        return v
    return [fake_embed(t) for t in texts]

class QdrantRetriever:
    def __init__(self, url: str = settings.QDRANT_URL, api_key: str = settings.QDRANT_API_KEY):
        self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False)

    async def search(self, query: str, top_k: int = 3, filter: Dict[str,Any]=None):
        vecs = await embed_texts([query])
        vector = vecs[0]
        res = self.client.search(collection_name="default", query_vector=vector, limit=top_k, with_payload=True)
        hits = []
        for r in res:
            payload = r.payload or {}
            txt = payload.get("text") or payload.get("content") or "<no-text>"
            hits.append({"id": r.id, "score": r.score, "payload": payload, "text": (txt[:500])})
        return hits
