# 🌌 OrionLab — Agentic RAG Framework

**OrionLab** is a lightweight but powerful framework for building **LLM + Agents + Vector DB** applications.  
It combines:

- **LangGraph** for agent orchestration
- **Qdrant** for vector storage
- **Sentence-Transformers** for embeddings
- **Pluggable LLM providers** (OpenAI, Hugging Face Inference, or custom HTTP endpoints)
- **Agentic Planner**: choose Vector DB, Web Search, or Hybrid
- **Semantic Cache**: prevent repeated queries from hitting the LLM
- **Observability**: JSONL traces + optional Prometheus metrics

👉 Think of it as a **research lab** where agents can hunt, gather, and synthesize knowledge.

---

## ✨ Features

- **RAG Agent (LangGraph)**  
  Vector retrieval + synthesis with inline citations.
- **Planner Agent**  
  Dynamically decides between **Vector DB**, **Web Search**, or **Hybrid**.
- **Semantic Cache**  
  Saves embeddings + answers → avoids redundant LLM calls.
- **DuckDuckGo WebSearch Tool**  
  Pulls fresh information directly into the agent pipeline.
- **Monitoring**  
  - JSONL traces of every query (`orionlab_traces.jsonl`)  
  - Optional Prometheus metrics (latency, cache hits, output size)

---

## 🚀 Quickstart

### 1. Install

```bash
git clone https://github.com/<your-username>/orionlab
cd orionlab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

2. Start Qdrant (Vector DB)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

3. Ingest your documents
# put some .txt or .md files under ./docs
python -m orionlab.cli ingest ./docs --collection demo

4. Ask a question
# with OpenAI
export OPENAI_API_KEY=sk-...
python -m orionlab.cli query "Who discovered penicillin and when?" \
  --collection demo --provider openai --llm gpt-4o-mini

# with Hugging Face Inference API
export HF_API_TOKEN=hf_...
python -m orionlab.cli query "What is CRISPR used for?" \
  --collection demo --provider hf --llm mistralai/Mixtral-8x7B-Instruct-v0.1

# with custom HTTP endpoint
python -m orionlab.cli query "Tell me about Mars' moons" \
  --collection demo --provider http --endpoint http://localhost:8000/generate

Agent Modes
	•	RAG Mode: always use your vector DB
	•	Planner Mode (default):
	•	If the question looks like current events → WEB
	•	If it’s private knowledge → VECTOR
	•	If both help → HYBRID

⸻

⚡ Semantic Cache
	•	Stores Q&A with embeddings in .orion_cache/cache.jsonl
	•	Avoids repeated LLM calls when similarity ≥ threshold (default 0.90)
	•	Uses FAISS if available, otherwise numpy cosine search

⸻

🌐 Web Search Tool

DuckDuckGo integration (duckduckgo-search):
	•	Fetch top-k results with title, snippet, URL
	•	Inject into the context with [W1], [W2] citations

⸻

📊 Monitoring & Traces

JSONL Traces

Every query logs a record to orionlab_traces.jsonl:

{"question": "Who discovered penicillin?",
 "provider": "openai",
 "llm": "gpt-4o-mini",
 "planner": true,
 "cache_hit": false,
 "latency_s": 2.15,
 "out_chars": 124,
 "n_sources": 3,
 "ts": 1735490342.201}

Prometheus Metrics

Optionally start metrics endpoint:
python -m orionlab.cli query "Who discovered penicillin?" --metrics-port 9090

Exposed metrics:
	•	orionlab_query_latency_seconds
	•	orionlab_queries_total
	•	orionlab_cache_hits_total
	•	orionlab_tokens_used_total


Project Layout

orionlab/
  cli.py                  # Typer CLI: ingest/query
  config.py               # Settings
  providers.py            # Embeddings + LLM adapters
  vectorstores/qdrant_store.py
  ingest/ingest.py        # Chunking + ingest
  agents/
    rag_graph.py          # Simple RAG agent (LangGraph)
    planner_graph.py      # Planner agent (Vector/Web/Hybrid)
    tools/websearch.py    # DuckDuckGo search
  cache/semantic_cache.py # Semantic cache (FAISS/numpy)
  monitoring/metrics.py   # JSONL + Prometheus

Roadmap
	•	Multi-agent orchestration (Planner → Tools → Synthesizer)
	•	Advanced evaluation harness (fact coverage, entailment checks)
	•	Frontend (Streamlit/Gradio) for demoing agents
	•	Docker Compose with Qdrant + Prometheus + Grafana

License

MIT — free to use, modify, and extend.

Acknowledgements
	•	LangGraph for agent orchestration
	•	Qdrant for vector search
	•	Sentence-Transformers for embeddings
	•	DuckDuckGo Search for web search integration
