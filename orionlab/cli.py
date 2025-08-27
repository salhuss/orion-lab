import typer
from pathlib import Path
from typing import Optional
from .config import settings
from .providers import Embedder, get_llm
from .vectorstores.qdrant_store import QdrantStore
from .ingest.ingest import _read_text, _chunks, fingerprint, build_payloads
from .agents.rag_graph import build_rag_graph

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
        if not f.is_file(): continue
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
          qdrant_url: str = typer.Option(settings.qdrant_url)):
    emb = Embedder(embed_model)
    store = QdrantStore(url=qdrant_url, collection=collection, dim=len(emb.embed(['x'])[0]))
    model = get_llm(provider, llm, endpoint)
    graph = build_rag_graph(emb, store, model)
    result = graph.invoke({"question": question})
    print("\n=== Answer ===\n")
    print(result.answer)
    print("\n=== Sources ===")
    for i,h in enumerate(result.retrieved or []):
        print(f"[{i+1}] {h['payload']['source']}")
