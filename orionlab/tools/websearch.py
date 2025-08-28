# orionlab/agents/tools/websearch.py
from __future__ import annotations
from typing import List, Dict, Any
from duckduckgo_search import DDGS

def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts: {title, url, snippet}
    """
    out: List[Dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                out.append({
                    "title": r.get("title") or "",
                    "url": r.get("href") or r.get("url") or "",
                    "snippet": r.get("body") or r.get("snippet") or "",
                })
    except Exception:
        pass
    return out

def render_search_context(results: List[Dict[str, Any]], char_limit: int = 1200) -> str:
    """
    Compact textual context to feed the LLM.
    """
    buf = []
    remain = char_limit
    for i, r in enumerate(results, 1):
        block = f"[W{i}] {r['title']}\n{r['snippet']}\nSource: {r['url']}\n"
        if len(block) <= remain:
            buf.append(block)
            remain -= len(block)
        else:
            if remain > 0:
                buf.append(block[:remain])
            break
    return "\n".join(buf)
