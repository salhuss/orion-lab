from __future__ import annotations
from typing import Dict, List
from langgraph.graph import StateGraph, END
from dataclasses import dataclass

@dataclass
class RAGState:
    question: str
    retrieved: List[Dict] | None = None
    answer: str | None = None

def build_rag_graph(emb, vstore, llm):
    g = StateGraph(RAGState)

    def retrieve_node(state: RAGState):
        qvec = emb.embed([state.question])[0]
        hits = vstore.query(qvec, top_k=6)
        return RAGState(question=state.question, retrieved=hits, answer=state.answer)

    def generate_node(state: RAGState):
        context = "\n\n".join([f"[{i+1}] {h['payload']['text'][:500]}" for i,h in enumerate(state.retrieved or [])])
        prompt = f"""You are a helpful assistant. Use ONLY the context to answer.

Question: {state.question}

Context:
{context}

Answer in 3-5 sentences, cite sources like [1], [2] inline when used."""
        ans = llm.generate(prompt)
        return RAGState(question=state.question, retrieved=state.retrieved, answer=ans)

    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()
