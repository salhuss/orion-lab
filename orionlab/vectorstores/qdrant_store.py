from __future__ import annotations
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

class QdrantStore:
    def __init__(self, url: str, collection: str, dim: int = 384):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.dim = dim
        self._ensure()

    def _ensure(self):
        cols = [c.name for c in self.client.get_collections().collections]
        if self.collection in cols: return
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
        )

    def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        points = [qm.PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(collection_name=self.collection, points=points)

    def query(self, query_vec: List[float], top_k: int = 8, where: Optional[Dict] = None):
        filt = qm.Filter(**where) if where else None
        res = self.client.search(collection_name=self.collection, query_vector=query_vec, limit=top_k, query_filter=filt)
        return [{"id": r.id, "score": float(r.score), "payload": r.payload} for r in res]
