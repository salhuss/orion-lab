# OrionLab — Agentic RAG Framework (Model & Vector DB Agnostic)

- LangGraph agents
- Qdrant vector store
- Sentence-Transformers embeddings
- Pluggable LLM providers (OpenAI / HF Inference / custom HTTP)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# start qdrant (local)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# ingest a folder of docs
python -m orionlab.cli ingest ./docs --collection demo

# ask a question
export OPENAI_API_KEY=sk-...
python -m orionlab.cli query "Who discovered penicillin and when?" --collection demo --provider openai --llm gpt-4o-mini

Env vars
	•	OPENAI_API_KEY (for provider=openai)
	•	HF_API_TOKEN (for provider=hf)

`orionlab/__init__.py`
```python
__all__ = []
