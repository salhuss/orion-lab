# orionlab/cache/semantic_cache.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json
import time
import math
import numpy as np

try:
    import faiss  # optional
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


@dataclass
class CacheItem:
    q: str
    a: str
    vec: List[float]
    meta: Dict[str, Any]


class SemanticCache:
    """
    Lightweight semantic cache with optional FAISS index.
    - Stores (question, answer, embedding, meta) in a JSONL file.
    - On load, builds FAISS (if available) or uses numpy cosine search.
    - query(vec, thresh) -> (answer, score, meta) or (None, None, None)
    """
    def __init__(self, path: str | Path, dim: int, use_faiss: bool = True):
        self.path = Path(path)
        self.dim = dim
        self.use_faiss = use_faiss and _HAS_FAISS
        self.items: List[CacheItem] = []
        self._faiss_index = None
        self._vecs: Optional[np.ndarray] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._load()
        else:
            self.path.write_text("", encoding="utf-8")

    # ---------- public ----------
    def query(self, qvec: List[float], threshold: float = 0.90) -> Tuple[Optional[str], Optional[float], Optional[Dict[str, Any]]]:
        if not self.items:
            return None, None, None
        v = np.array(qvec, dtype=np.float32).reshape(1, -1)
        if self.use_faiss and self._faiss_index is not None:
            D, I = self._faiss_index.search(v, 1)
            idx = int(I[0, 0])
            score = float(1 - D[0, 0] / 2)  # convert L2 on normalized vecs to ~cosine
        else:
            # cosine on normalized embeddings
            M = self._vecs  # (N, D)
            sims = (M @ v.T).ravel()  # dot since normalized
            idx = int(np.argmax(sims))
            score = float(sims[idx])
        if score >= threshold:
            it = self.items[idx]
            return it.a, score, dict(it.meta)
        return None, None, None

    def upsert(self, question: str, answer: str, vec: List[float], meta: Optional[Dict[str, Any]] = None) -> None:
        meta = meta or {}
        rec = {"q": question, "a": answer, "vec": vec, "meta": meta, "ts": time.time()}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.items.append(CacheItem(q=question, a=answer, vec=vec, meta=meta))
        self._rebuild_index(incremental=True, new_vec=np.array(vec, dtype=np.float32))

    # ---------- internal ----------
    def _load(self):
        items: List[CacheItem] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                    items.append(CacheItem(q=j["q"], a=j["a"], vec=j["vec"], meta=j.get("meta", {})))
                except Exception:
                    continue
        self.items = items
        self._rebuild_index()

    def _rebuild_index(self, incremental: bool = False, new_vec: Optional[np.ndarray] = None):
        if not self.items:
            self._faiss_index = None
            self._vecs = None
            return
        V = np.array([it.vec for it in self.items], dtype=np.float32)
        # normalize (cosine space)
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        V = V / norms
        self._vecs = V
        if self.use_faiss:
            if incremental and self._faiss_index is not None and new_vec is not None:
                v = new_vec.astype(np.float32).reshape(1, -1)
                v = v / (np.linalg.norm(v) + 1e-9)
                self._faiss_index.add(v)
            else:
                index = faiss.IndexFlatIP(self.dim)
                index.add(V)
                self._faiss_index = index
        else:
            self._faiss_index = None
