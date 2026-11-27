# server.py
# FastAPI Web Server for Knowledge RAG

import os
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List

from src.ingestion import ingest_pipeline, load_vectorstore, load_bm25, get_new_files, get_indexed_files
from src.markdown_ingestion import (
    ingest_llamaparse_pipeline,
    load_llamaparse_vectorstore,
    load_llamaparse_bm25,
    get_new_llamaparse_files,
    get_indexed_llamaparse_files
)
from src.retrieval import retrieve
from src.generation import generate_answer, get_sources
from src.config import (
    CHROMA_DIR, BM25_PKL,
    CHROMA_DIR_LLAMAPARSE, BM25_PKL_LLAMAPARSE,
    DOCS_DIR
)
from src.eval_ragas import evaluate_query_ragas
from src.error_handler import setup_logging, log_api_error
from src.reviewer import scan_telemetry, scan_corpus_health, generate_daily_briefing

logger = setup_logging()

app = FastAPI(title="Knowledge RAG API", version="2.5.0")

# In-memory store cache to prevent reloading per request
_stores = {
    "llamaparse": {"vs": None, "bm25": None},
    "pypdf": {"vs": None, "bm25": None},
}
_ragas_session_history = []


def get_store(mode: str):
    """Lazily loads vectorstore and BM25 index for the chosen mode."""
    if mode not in _stores:
        mode = "llamaparse"

    cache = _stores[mode]
    if cache["vs"] is None:
        if mode == "llamaparse":
            if os.path.exists(CHROMA_DIR_LLAMAPARSE) and os.path.exists(BM25_PKL_LLAMAPARSE):
                cache["vs"] = load_llamaparse_vectorstore()
                cache["bm25"] = load_llamaparse_bm25()
        else:
            if os.path.exists(CHROMA_DIR) and os.path.exists(BM25_PKL):
                cache["vs"] = load_vectorstore()
                cache["bm25"] = load_bm25()
    return cache["vs"], cache["bm25"]


class QueryRequest(BaseModel):
    question: str
    mode: Optional[str] = "llamaparse"
    enable_eval: Optional[bool] = False


@app.post("/api/chat")
def chat_endpoint(req: QueryRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    mode = req.mode or "llamaparse"
    vs, bm25 = None, None
    try:
        vs, bm25 = get_store(mode)
    except Exception as e:
        log_api_error("Vectorstore initialization", e)

    if vs is None or bm25 is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "is_timeout": False,
                    "is_overloaded": False,
                    "stage": "Index Loading",
                    "details": f"No active index found for mode '{mode}'. Please run indexing first."
                }
            }
        )

    start = time.time()
    docs = None
    answer = None
    error_info = None

    try:
        docs = retrieve(question, vs, bm25)
    except Exception as e:
        error_info = log_api_error("Retrieval (Embedding / Search)", e, query=question)

    if error_info is None and docs is not None:
        try:
            answer = generate_answer(question, docs)
        except Exception as e:
            error_info = log_api_error("Answer Generation", e, query=question)

    if error_info:
        return JSONResponse(status_code=500, content={"error": error_info})

    latency_ms = round((time.time() - start) * 1000)
    sources = get_sources(docs) if docs else []

    # Optional RAGAS 0.4.3 Evaluation
    ragas_scores = None
    if req.enable_eval and docs and answer:
        try:
            context_texts = [d.page_content for d in docs]
            ragas_scores = evaluate_query_ragas(question, answer, context_texts)
            if ragas_scores:
                _ragas_session_history.append({
                    "query": question,
                    "latency_ms": latency_ms,
                    **ragas_scores
                })
        except Exception as eval_err:
            log_api_error("RAGAS Evaluation", eval_err, query=question)

    return {
        "answer": answer,
        "sources": sources,
        "evaluation": ragas_scores,
        "mode": mode,
        "latency_ms": latency_ms
    }


@app.get("/api/status")
def status_endpoint(mode: str = "llamaparse"):
    if mode == "llamaparse":
        index_ready = os.path.exists(CHROMA_DIR_LLAMAPARSE) and os.path.exists(BM25_PKL_LLAMAPARSE)
    else:
        index_ready = os.path.exists(CHROMA_DIR) and os.path.exists(BM25_PKL)
    corpus = scan_corpus_health()
    telemetry = scan_telemetry()
    return {
        "mode": mode,
        "index_ready": index_ready,
        "unindexed_pdfs": len(corpus["unindexed_pdfs"]),
        "unindexed_mds": len(corpus["unindexed_mds"]),
        "telemetry": telemetry
    }


@app.post("/api/index")
def index_endpoint(mode: str = "llamaparse"):
    try:
        if mode == "pypdf":
            new_files = get_new_files()
            if not new_files:
                return {"message": "All PDFs are already indexed.", "count": 0}
            vs, bm25, n = ingest_pipeline()
            _stores["pypdf"]["vs"] = vs
            _stores["pypdf"]["bm25"] = bm25
            return {"message": f"Successfully indexed {n} chunks from {len(new_files)} PDF(s).", "count": n}
        else:
            new_mds = get_new_llamaparse_files()
            if not new_mds:
                return {"message": "All parsed markdown documents are already indexed.", "count": 0}
            vs, bm25, n = ingest_llamaparse_pipeline()
            _stores["llamaparse"]["vs"] = vs
            _stores["llamaparse"]["bm25"] = bm25
            return {"message": f"Successfully indexed {n} chunks from {len(new_mds)} markdown file(s).", "count": n}
    except Exception as e:
        err = log_api_error(f"{mode} Indexing", e)
        return JSONResponse(status_code=500, content={"error": err})


@app.post("/api/upload")
async def upload_endpoint(files: List[UploadFile] = File(...)):
    os.makedirs(DOCS_DIR, exist_ok=True)
    saved = []
    for f in files:
        if f.filename.endswith(".pdf"):
            dest = os.path.join(DOCS_DIR, f.filename)
            if not os.path.exists(dest):
                content = await f.read()
                with open(dest, "wb") as out:
                    out.write(content)
                saved.append(f.filename)
    return {"saved": saved, "count": len(saved)}


@app.get("/api/evaluation/history")
def eval_history_endpoint():
    return {"history": _ragas_session_history}


@app.get("/api/reviewer/briefing")
def reviewer_briefing_endpoint():
    return {
        "briefing": generate_daily_briefing(_ragas_session_history),
        "telemetry": scan_telemetry(),
        "corpus": scan_corpus_health()
    }


# Mount static frontend
os.makedirs("frontend", exist_ok=True)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")
