from __future__ import annotations
import os, requests
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

# ---------- Embeddings ----------
class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.m = SentenceTransformer(model_name)
    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.m.encode(texts, normalize_embeddings=True).tolist()

# ---------- LLM adapters ----------
class OpenAIAdapter:
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        api = os.getenv("OPENAI_API_KEY")
        if not api:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api)
        self.model = model
    def generate(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[{"role":"user","content":prompt}]
        )
        return r.choices[0].message.content.strip()

class HFAdapter:
    """HF Inference API (or your own TGI)."""
    def __init__(self, model: str, endpoint: Optional[str] = None):
        self.model = model
        self.endpoint = endpoint or f"https://api-inference.huggingface.co/models/{model}"
        token = os.getenv("HF_API_TOKEN","").strip()
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
    def generate(self, prompt: str) -> str:
        r = requests.post(self.endpoint, headers=self.headers,
                          json={"inputs": prompt, "parameters":{"max_new_tokens":256,"temperature":0.2}})
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and j and "generated_text" in j[0]: return j[0]["generated_text"]
        if isinstance(j, dict) and "generated_text" in j: return j["generated_text"]
        return str(j)

class HTTPAdapter:
    """Custom JSON API: POST {prompt} -> {text}"""
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
    def generate(self, prompt: str) -> str:
        r = requests.post(self.endpoint, json={"prompt": prompt})
        r.raise_for_status()
        return r.json().get("text","")

def get_llm(provider: str, llm: str, endpoint: Optional[str] = None):
    if provider == "openai": return OpenAIAdapter(llm)
    if provider == "hf":     return HFAdapter(llm, endpoint)
    if provider == "http":
        if not endpoint: raise ValueError("--endpoint required for provider=http")
        return HTTPAdapter(endpoint)
    raise ValueError("provider must be one of: openai|hf|http")
