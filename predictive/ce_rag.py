"""
ce_rag.py — literature knowledge base + retrieval for the CE workflow.
=====================================================================
A tiny, dependency-light RAG store over `text-embedding-3-large` (llm.embed). The user
grows it over time with their own papers/notes; ce_design retrieves relevant passages to
(a) ground the mechanistic rationale in real literature and (b) flag novelty (a candidate
whose nearest literature match is very similar is "already explored").

Storage (molvae_artifacts/ce/kb/):
  chunks.jsonl   one {id, source, text} per line
  emb.npy        float32 [N, dim] aligned to chunks order

CLI:
  python molvae/ce_rag.py --add paper.txt --source "Zhang 2024 JACS"   # add a document
  python molvae/ce_rag.py --add-text "TFE additive raises Zn CE to 99.4% by ..." --source note
  python molvae/ce_rag.py --query "amide additives for zinc anode SEI" --k 5
  python molvae/ce_rag.py --list
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import config
import llm

KB_DIR = config.CKPT_DIR.parent / "ce" / "kb"
CHUNKS = KB_DIR / "chunks.jsonl"
EMB = KB_DIR / "emb.npy"

# Abuse guards (esp. for the public web app): cap documents and per-doc size so the
# embedding API can't be hammered. Override via env for trusted/local use.
MAX_SOURCES = int(os.getenv("CE_RAG_MAX_SOURCES", "10"))
MAX_CHUNKS_PER_DOC = int(os.getenv("CE_RAG_MAX_CHUNKS_PER_DOC", "40"))


def _chunk_text(text: str, words: int = 220, overlap: int = 40):
    toks = text.split()
    if not toks:
        return []
    out, i = [], 0
    while i < len(toks):
        out.append(" ".join(toks[i:i + words]))
        i += max(1, words - overlap)
    return out


def _load():
    chunks = []
    if CHUNKS.exists():
        with open(CHUNKS, encoding="utf-8") as f:
            chunks = [json.loads(l) for l in f if l.strip()]
    emb = np.load(EMB) if EMB.exists() else np.zeros((0, 1), np.float32)
    return chunks, emb


def add_document(text: str, source: str) -> int:
    """Chunk -> embed -> append to the store. Returns #chunks added (0 = rejected/empty/no key).
    Enforces MAX_SOURCES distinct documents and MAX_CHUNKS_PER_DOC to bound LLM usage."""
    chunks, _ = _load()
    existing_sources = {c["source"] for c in chunks}
    if source not in existing_sources and len(existing_sources) >= MAX_SOURCES:
        print(f"[rag] limit reached: {MAX_SOURCES} documents already in the KB. "
              f"Remove some (or raise CE_RAG_MAX_SOURCES) before adding more.")
        return 0
    pieces = _chunk_text(text)[:MAX_CHUNKS_PER_DOC]
    if not pieces:
        return 0
    vecs = llm.embed(pieces)
    if vecs is None:
        print("[rag] embeddings unavailable (set FOUNDRY_API_KEY) — nothing added.")
        return 0
    KB_DIR.mkdir(parents=True, exist_ok=True)
    chunks, emb = _load()
    start = len(chunks)
    new = np.asarray(vecs, np.float32)
    new /= (np.linalg.norm(new, axis=1, keepdims=True) + 1e-9)  # store normalized
    emb = new if emb.shape[0] == 0 else np.vstack([emb, new])
    np.save(EMB, emb)
    with open(CHUNKS, "a", encoding="utf-8") as f:
        for j, p in enumerate(pieces):
            f.write(json.dumps({"id": start + j, "source": source, "text": p}) + "\n")
    print(f"[rag] added {len(pieces)} chunks from '{source}' (KB now {emb.shape[0]} chunks)")
    return len(pieces)


def query(text: str, k: int = 5):
    """Return up to k {source, text, score} most similar passages. [] if empty/no keys."""
    chunks, emb = _load()
    if not chunks or emb.shape[0] == 0:
        return []
    qv = llm.embed([text])
    if qv is None:
        return []
    q = np.asarray(qv[0], np.float32); q /= (np.linalg.norm(q) + 1e-9)
    sims = emb @ q
    idx = np.argsort(sims)[::-1][:k]
    return [{"source": chunks[i]["source"], "text": chunks[i]["text"], "score": float(sims[i])}
            for i in idx]


def context_for(smiles: str, mechanism: str = "", k: int = 4):
    """Retrieve KB passages relevant to a candidate; returns (context_text, top_score)."""
    hits = query(f"zinc battery additive {smiles} {mechanism} coulombic efficiency SEI", k)
    if not hits:
        return "", 0.0
    ctx = "\n".join(f"[{h['source']}] {h['text'][:400]}" for h in hits)
    return ctx, hits[0]["score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", help="path to a .txt/.md document to add")
    ap.add_argument("--add-text", help="raw text to add")
    ap.add_argument("--source", default="user", help="citation/source label")
    ap.add_argument("--query", help="search the KB")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.add:
        p = Path(args.add)
        add_document(p.read_text(encoding="utf-8", errors="ignore"), args.source or p.name)
    if args.add_text:
        add_document(args.add_text, args.source)
    if args.query:
        for h in query(args.query, args.k):
            print(f"\n[{h['score']:.3f}] ({h['source']})\n  {h['text'][:300]}")
    if args.list:
        chunks, emb = _load()
        srcs = {}
        for c in chunks:
            srcs[c["source"]] = srcs.get(c["source"], 0) + 1
        print(f"KB: {len(chunks)} chunks, {emb.shape} embeddings, {len(srcs)} sources")
        for s, n in sorted(srcs.items(), key=lambda kv: -kv[1]):
            print(f"  {n:4d}  {s}")


if __name__ == "__main__":
    main()
