from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import hashlib

def _read_text(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    # simple fallback: treat unknown as text
    return path.read_text(encoding="utf-8", errors="ignore")

def _chunks(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    words = text.split()
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+size]))
        i += (size - overlap)
    return out

def fingerprint(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def build_payloads(chunks: List[str], path: Path) -> List[Dict]:
    return [{"text": c, "source": str(path)} for c in chunks]
