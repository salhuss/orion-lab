# orionlab/cli.py
import typer
from pathlib import Path
from typing import Optional
import time

from .config import settings
from .providers import Embedder, get_llm
from .vectorstores.qdrant_store import QdrantStore
from .ingest.ingest import _read_text, _chunks, fingerprint
from .agents.rag_graph import build_rag_graph
from .agents.planner_graph import build_planner_rag_graph
from .cache.semantic_cache import SemanticCache
from .monitoring.metrics import JSONTraceLogger, Metrics

app = typer.Typer(help="OrionLab CLI")

@app.command()
def ingest(path: str = typer.Argument(..., help="File or directory"),
           collection: str = typer.Option(settings.collection),
           embed_model: str = typer.Option(settings.embed_model),
           qdrant_url: str = typer.Option(settings.qdrant_url)):
    emb = Embedder(embed_model)
    store = QdrantStore(url=qdrant_url, collection=collection, dim=len(emb.embed(['x'])[0]))
    p = Path(path)
    files = [p] if p.is_file() else list(p.rglob("*"))
    texts = []
    payloads = []
    ids = []
    for f in files:
        if not f.is_file(): 
            continue
        try:
            txt = _read_text(f)
        except Exception:
            continue
        for ch in _chunks(txt):
            texts.append(ch)
            payloads.append({"text": ch, "source": str(f)})
            ids.append(f"{fingerprint(str(f))}-{fingerprint(ch)}")
    vecs = emb.embed(texts) if texts else []
    if vecs:
        store.upsert(ids, vecs, payloads)
        typer.echo(f"✅ Ingested {len(vecs)} chunks into '{collection}'")
    else:
        typer.echo("No chunks found.")

@app.command()
def query(question: str = typer.Argument(...),
          collection: str = typer.Option(settings.collection),
          provider: str = typer.Option("openai"),
          llm: str = typer.Option("gpt-4o-mini"),
          endpoint: Optional[str] = typer.Option(None),
          embed_model: str = typer.Option(settings.embed_model),
          qdrant_url: str = typer.Option(settings.qdrant_url),
          planner: bool = typer.Option(True, help="Use planner to choose Vector vs Web vs Hybrid"),
          use_cache: bool = typer.Option(True, help="Use semantic cache to short-circuit repeated queries"),
          cache_path: str = typer.Option(".orion_cache/cache.jsonl"),
          cache_thresh: float = typer.Option(0.90),
          metrics_port: Optional[int] = typer.Option(None, help="Expose Prometheus metrics on this port"),
          traces_path: str = typer.Option("orionlab_traces.jsonl")):
    t0 = time.time()
    tracer = JSONTraceLogger(traces_path)
    metrics = Metrics(metrics_port)

    emb = Embedder(embed_model)
    store = QdrantStore(url=qdrant_url, collection=collection, dim=len(emb.embed(['x'])[0]))
    model = get_llm(provider, llm, endpoint)

    cache_hit = False
    answer = None
    retrieved = None

    if use_cache:
        cache = SemanticCache(path=cache_path, dim=len(emb.embed(['cache'])[0]), use_faiss=True)
        qvec = emb.embed([question])[0]
        cached, score, meta = cache.query(qvec, threshold=cache_thresh)
        if cached is not None:
            cache_hit = True
            answer = cached
            retrieved = meta.get("retrieved")
    
    if answer is None:
        graph = build_planner_rag_graph(emb, store, model) if planner else build_rag_graph(emb, store, model)
        result = graph.invoke({"question": question})
        answer = result.answer
        retrieved = result.retrieved
        if use_cache:
            cache.upsert(
                question=question,
                answer=answer,
                vec=emb.embed([question])[0],
                meta={"retrieved": retrieved}
            )

    # output
    print("\n=== Answer ===\n")
    print(answer)
    print("\n=== Sources ===")
    for i,h in enumerate(retrieved or []):
        src = h.get("payload", {}).get("source") or h.get("payload", {}).get("text","")[:80].replace("\n"," ")
        print(f"[{i+1}] {src}")

    # metrics
    latency = time.time() - t0
    out_chars = len(answer or "")
    tracer.log({
        "question": question,
        "provider": provider,
        "llm": llm,
        "planner": planner,
        "cache_hit": cache_hit,
        "latency_s": latency,
        "out_chars": out_chars,
        "n_sources": len(retrieved or []),
    })
    metrics.record(latency_s=latency, cache_hit=cache_hit, out_chars=out_chars)
    typer.echo(f"\n⏱  {latency:.2f}s   {'(cache hit)' if cache_hit else ''}")

if __name__ == "__main__":
    app()
