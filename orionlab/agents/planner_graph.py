# orionlab/agents/planner_graph.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Literal, Optional
from langgraph.graph import StateGraph, END
from .tools.websearch import web_search, render_search_context

Decision = Literal["vector", "web", "hybrid"]

@dataclass
class PlannerState:
    question: str
    decision: Decision | None = None
    retrieved: List[Dict] | None = None   # vector DB hits
    web_results: List[Dict] | None = None
    answer: str | None = None

def _simple_decider(llm, question: str) -> Decision:
    """
    LLM-driven binary/ternary decision. Falls back to heuristic if parsing fails.
    """
    prompt = f"""Classify the question into one of: VECTOR, WEB, HYBRID.

Rules:
- If it asks about current events, news, prices, or "today/this year", prefer WEB.
- If it's general knowledge or about a private knowledge base, prefer VECTOR.
- If both are helpful, choose HYBRID.

Question: {question}

Answer with only one word: VECTOR or WEB or HYBRID."""
    txt = llm.generate(prompt).strip().upper()
    if "HYBRID" in txt: return "hybrid"
    if "WEB" in txt: return "web"
    if "VECTOR" in txt: return "vector"
    # fallback heuristic
    t = question.lower()
    if any(k in t for k in ["today", "yesterday", "price", "latest", "news", "this year", "2024", "2025"]):
        return "web"
    return "vector"

def build_planner_rag_graph(emb, vstore, llm):
    g = StateGraph(PlannerState)

    def plan_node(state: PlannerState):
        dec = _simple_decider(llm, state.question)
        return PlannerState(question=state.question, decision=dec)

    def vector_node(state: PlannerState):
        qvec = emb.embed([state.question])[0]
        hits = vstore.query(qvec, top_k=6)
        return PlannerState(question=state.question, decision=state.decision, retrieved=hits, web_results=state.web_results)

    def web_node(state: PlannerState):
        results = web_search(state.question, max_results=5)
        return PlannerState(question=state.question, decision=state.decision, retrieved=state.retrieved, web_results=results)

    def generate_node(state: PlannerState):
        vctx = ""
        if state.retrieved:
            vctx = "\n\n".join([f"[V{i+1}] {h['payload']['text'][:500]}" for i,h in enumerate(state.retrieved)])
        wctx = ""
        if state.web_results:
            wctx = render_search_context(state.web_results, char_limit=1200)

        context = ""
        if state.decision == "vector":
            context = vctx
        elif state.decision == "web":
            context = wctx
        else:
            context = (vctx + "\n\n" + wctx).strip()

        prompt = f"""You are a careful assistant. Use ONLY the provided context.
Cite sources inline: [V1], [V2] for vector sources; [W1], [W2] for web results.

Question:
{state.question}

Context:
{context}

Answer in 3-6 sentences with inline citations."""
        ans = llm.generate(prompt)
        return PlannerState(question=state.question, decision=state.decision, retrieved=state.retrieved, web_results=state.web_results, answer=ans)

    # graph wiring
    g.add_node("plan", plan_node)
    g.add_node("vector", vector_node)
    g.add_node("web", web_node)
    g.add_node("generate", generate_node)

    g.set_entry_point("plan")
    # conditional branches
    def branch(state: PlannerState):
        if state.decision == "vector":
            return "vector"
        if state.decision == "web":
            return "web"
        return "vector"  # hybrid does both: vector then web

    g.add_conditional_edges("plan", branch, {
        "vector": "vector",
        "web": "web",
    })
    # for vector path, if HYBRID we still want web after vector
    def after_vector(state: PlannerState):
        return "web" if state.decision == "hybrid" else "generate"

    g.add_conditional_edges("vector", after_vector, {"web": "web", "generate": "generate"})
    g.add_edge("web", "generate")
    g.add_edge("generate", END)
    return g.compile()
