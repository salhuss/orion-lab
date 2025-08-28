# orionlab/monitoring/metrics.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import json, time

# Prometheus is optional
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False

# ---------- JSONL tracing ----------
class JSONTraceLogger:
    def __init__(self, path: str | Path = "orionlab_traces.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def log(self, record: Dict[str, Any]):
        rec = dict(record)
        rec["ts"] = time.time()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------- Prometheus metrics ----------
class Metrics:
    def __init__(self, port: Optional[int] = None):
        self.enabled = _HAS_PROM and (port is not None)
        if self.enabled:
            start_http_server(port)
            self.q_latency = Histogram("orionlab_query_latency_seconds", "End-to-end query latency (s)")
            self.q_total = Counter("orionlab_queries_total", "Total queries")
            self.cache_hits = Counter("orionlab_cache_hits_total", "Cache hits")
            self.tokens_used = Counter("orionlab_tokens_used_total", "Total output characters (proxy for tokens)")
        else:
            self.q_latency = None
            self.q_total = None
            self.cache_hits = None
            self.tokens_used = None

    def record(self, latency_s: float, cache_hit: bool, out_chars: int):
        if not self.enabled:
            return
        self.q_total.inc()
        self.q_latency.observe(latency_s)
        if cache_hit:
            self.cache_hits.inc()
        self.tokens_used.inc(max(0, out_chars))
