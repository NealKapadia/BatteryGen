"""Production server (FastAPI + Uvicorn) for the molvae inverse-design web app.

    pip install fastapi "uvicorn[standard]" python-multipart
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1

Endpoints (prompt-driven; no sliders):
    GET  /                 the single-page UI
    GET  /health
    POST /generate         {prompt, n, session_id?}  -> bulk molecules + features + IUPAC
    POST /train            multipart: file=CSV, objective=text [, smiles_col] -> session_id + CV R2
    POST /train_pkl        multipart: file=.pkl  (DISABLED unless MOLVAE_ALLOW_PKL=1; RCE risk)
    POST /molecule         {smiles} -> features + IUPAC + 3D
    POST /literature       {text, source} -> add to RAG KB (capped)
    GET  /literature        -> KB summary

User data is processed in memory and never written to disk (only the RAG KB persists).
Blocking model work runs in a threadpool so the event loop stays responsive.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import design
import webengine

STATIC = Path(__file__).resolve().parent / "static"
_device = os.getenv("MOLVAE_DEVICE", "cpu")
_ckpt = os.getenv("MOLVAE_CKPT")
ALLOW_PKL = os.getenv("MOLVAE_ALLOW_PKL", "0") == "1"
MAX_N = int(os.getenv("MOLVAE_MAX_N", "30"))
MAX_UPLOAD_MB = int(os.getenv("MOLVAE_MAX_UPLOAD_MB", "25"))

ENGINE: Optional[design.DesignEngine] = None
app = FastAPI(title="molvae inverse design", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class GenReq(BaseModel):
    prompt: str = ""
    n: int = 12
    session_id: Optional[str] = None


class MolReq(BaseModel):
    smiles: str


class LitReq(BaseModel):
    text: str
    source: str = "user"


@app.on_event("startup")
def _load():
    global ENGINE
    print(f"[server] loading model on {_device} ...")
    ENGINE = design.DesignEngine(ckpt=_ckpt, device=_device)
    print("[server] ready")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"ok": True, "device": str(ENGINE.device) if ENGINE else None,
            "electrolyte": bool(ENGINE and ENGINE.elec is not None),
            "llm": __import__("llm").available("fast"), "allow_pkl": ALLOW_PKL}


# Generation (esp. Bayesian optimization) can take 1-4 min on CPU — run it as a background
# job so the HTTP request returns instantly and the browser polls /job/{id}.
_JOBS: Dict[str, dict] = {}


def _run_job(job_id: str, prompt: str, n: int, session_id: Optional[str]):
    try:
        _JOBS[job_id] = {"status": "running", "ts": time.time()}
        res = webengine.generate(ENGINE, prompt, n, session_id)
        _JOBS[job_id] = {"status": "done", "result": res, "ts": time.time()}
    except Exception as e:  # surface errors to the client instead of hanging
        _JOBS[job_id] = {"status": "error", "result": {"ok": False, "error": str(e)}, "ts": time.time()}


@app.post("/generate")
def generate_ep(req: GenReq):
    n = max(1, min(req.n, MAX_N))
    # drop jobs older than 30 min
    for k in [k for k, v in _JOBS.items() if time.time() - v.get("ts", 0) > 1800]:
        _JOBS.pop(k, None)
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {"status": "queued", "ts": time.time()}
    threading.Thread(target=_run_job, args=(job_id, req.prompt, n, req.session_id), daemon=True).start()
    return {"ok": True, "job_id": job_id}


@app.get("/job/{job_id}")
def job_status(job_id: str):
    j = _JOBS.get(job_id)
    if not j:
        return JSONResponse({"ok": False, "error": "unknown job"}, 404)
    return {"ok": True, "status": j["status"], "result": j.get("result")}


@app.post("/train")
async def train_ep(file: UploadFile = File(...), objective: str = Form(...),
                   smiles_col: str = Form(None)):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        return JSONResponse({"ok": False, "error": f"file too large (>{MAX_UPLOAD_MB} MB)"}, 413)
    return await run_in_threadpool(webengine.train_from_csv, raw, objective, smiles_col)


@app.post("/train_pkl")
async def train_pkl_ep(file: UploadFile = File(...)):
    if not ALLOW_PKL:
        return JSONResponse({"ok": False, "error": "pkl upload disabled (set MOLVAE_ALLOW_PKL=1; "
                             "unpickling untrusted files is a security risk)"}, 403)
    raw = await file.read()
    try:
        model = await run_in_threadpool(webengine.load_pkl_model, raw)
        return {"ok": True, "note": "loaded; wire-up of custom-pkl scoring is experimental",
                "type": type(model).__name__}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, 400)


@app.post("/molecule")
async def molecule_ep(req: MolReq):
    return await run_in_threadpool(webengine.molecule_info, req.smiles)


@app.post("/literature")
async def literature_ep(req: LitReq):
    n = await run_in_threadpool(webengine.add_literature, req.text, req.source)
    return {"ok": n > 0, "chunks_added": n}


@app.get("/literature")
def literature_list():
    import ce_rag
    chunks, emb = ce_rag._load()
    srcs: Dict[str, int] = {}
    for c in chunks:
        srcs[c["source"]] = srcs.get(c["source"], 0) + 1
    return {"ok": True, "documents": len(srcs), "chunks": len(chunks),
            "max_documents": ce_rag.MAX_SOURCES, "sources": srcs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), workers=1)
