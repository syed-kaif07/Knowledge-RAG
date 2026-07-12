"""
rag_observability.py

Drop-in retrieval + faithfulness logging for a LangChain RAG pipeline.
Framework-agnostic: works with any retriever (Chroma, BM25, EnsembleRetriever,
custom RRF-fused hybrid retriever) as long as you can hand it a list of
retrieved chunks with scores.

Usage pattern (see example_integration.py for a full worked example):

    from rag_observability import RAGLogger

    logger = RAGLogger(db_path="rag_logs.db")

    # ... run your existing retrieval + rerank + generation ...

    log_id = logger.log_query(
        query=user_query,
        retrieved_chunks=retrieved_chunks,   # list[dict] before rerank
        reranked_chunks=reranked_chunks,     # list[dict] after Cohere rerank
        final_context_chunks=final_chunks,   # list[dict] actually sent to LLM
        answer=answer_text,
        hyde_used=hyde_toggle_state,
    )

    faithfulness = logger.score_faithfulness(
        log_id=log_id,
        answer=answer_text,
        cited_chunks=final_chunks,
        llm_judge=your_llm_instance,   # any LangChain-compatible chat model
    )
"""

import sqlite3
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    """Normalized representation of a retrieved chunk for logging purposes."""
    chunk_id: str
    text_preview: str          # first ~200 chars, not full text (keep logs light)
    source: str                # filename
    page: Optional[int] = None
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None

    @staticmethod
    def from_langchain_doc(doc, **scores) -> "ChunkRecord":
        """Build a ChunkRecord from a LangChain Document + optional score kwargs."""
        meta = doc.metadata or {}
        return ChunkRecord(
            chunk_id=meta.get("chunk_id", str(uuid.uuid4())[:8]),
            text_preview=(doc.page_content or "")[:200],
            source=meta.get("source", "unknown"),
            page=meta.get("page"),
            bm25_score=scores.get("bm25_score"),
            dense_score=scores.get("dense_score"),
            rrf_score=scores.get("rrf_score"),
            rerank_score=scores.get("rerank_score"),
        )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    log_id TEXT PRIMARY KEY,
    timestamp REAL,
    query TEXT,
    hyde_used INTEGER,
    retrieved_chunks TEXT,      -- JSON: list of ChunkRecord (pre-rerank)
    reranked_chunks TEXT,       -- JSON: list of ChunkRecord (post-rerank)
    final_context_chunks TEXT,  -- JSON: list of ChunkRecord (sent to LLM)
    answer TEXT,
    faithfulness_score REAL,    -- 0.0 - 1.0, NULL until scored
    faithfulness_detail TEXT,   -- JSON: per-claim support breakdown
    latency_ms REAL
);
"""


class RAGLogger:
    def __init__(self, db_path: str = "rag_logs.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(SCHEMA)
        conn.commit()
        conn.close()

    def log_query(
        self,
        query: str,
        retrieved_chunks: list[ChunkRecord],
        final_context_chunks: list[ChunkRecord],
        answer: str,
        reranked_chunks: Optional[list[ChunkRecord]] = None,
        hyde_used: bool = False,
        latency_ms: Optional[float] = None,
    ) -> str:
        """Log a single query's full retrieval + generation trace. Returns log_id."""
        log_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO queries
               (log_id, timestamp, query, hyde_used, retrieved_chunks,
                reranked_chunks, final_context_chunks, answer,
                faithfulness_score, faithfulness_detail, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)""",
            (
                log_id,
                time.time(),
                query,
                int(hyde_used),
                json.dumps([_chunk_to_dict(c) for c in retrieved_chunks]),
                json.dumps([_chunk_to_dict(c) for c in (reranked_chunks or [])]),
                json.dumps([_chunk_to_dict(c) for c in final_context_chunks]),
                answer,
                latency_ms,
            ),
        )
        conn.commit()
        conn.close()
        return log_id

    def score_faithfulness(
        self,
        log_id: str,
        answer: str,
        cited_chunks: list[ChunkRecord],
        llm_judge: Any,
    ) -> float:
        """
        Ask an LLM to judge whether each claim in `answer` is supported by
        the cited chunks. Cheap proxy for RAGAS faithfulness — swap for
        real RAGAS later without changing the storage schema.

        `llm_judge` must be a LangChain-compatible chat model (e.g. your
        existing llama-3.1-8b-instruct NIM client) exposing .invoke(str) -> response
        with a .content attribute, matching standard LangChain ChatModel interface.
        """
        context = "\n\n".join(
            f"[{c.chunk_id}] {c.text_preview}" for c in cited_chunks
        )
        claims = _split_into_claims(answer)

        detail = []
        supported_count = 0
        for claim in claims:
            prompt = (
                "You are a strict fact-checker. Given the CONTEXT and a CLAIM, "
                "answer only 'yes' or 'no': is the claim directly supported by "
                "the context?\n\n"
                f"CONTEXT:\n{context}\n\nCLAIM:\n{claim}\n\nAnswer (yes/no):"
            )
            response = llm_judge.invoke(prompt)
            verdict = str(getattr(response, "content", response)).strip().lower()
            supported = verdict.startswith("yes")
            if supported:
                supported_count += 1
            detail.append({"claim": claim, "supported": supported, "raw": verdict})

        score = supported_count / len(claims) if claims else 0.0

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE queries SET faithfulness_score = ?, faithfulness_detail = ? WHERE log_id = ?",
            (score, json.dumps(detail), log_id),
        )
        conn.commit()
        conn.close()
        return score

    def get_recent(self, limit: int = 50) -> list[dict]:
        """Fetch recent query logs for dashboard display."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM queries ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_faithfulness_trend(self) -> list[tuple[float, Optional[float]]]:
        """Returns list of (timestamp, faithfulness_score) for charting."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT timestamp, faithfulness_score FROM queries "
            "WHERE faithfulness_score IS NOT NULL ORDER BY timestamp ASC"
        ).fetchall()
        conn.close()
        return rows

    def clear_all(self):
        """Wipe all logged queries. Irreversible — use for resetting during dev/testing."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM queries")
        conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_to_dict(c: ChunkRecord) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "text_preview": c.text_preview,
        "source": c.source,
        "page": c.page,
        "bm25_score": c.bm25_score,
        "dense_score": c.dense_score,
        "rrf_score": c.rrf_score,
        "rerank_score": c.rerank_score,
    }


def _split_into_claims(answer: str) -> list[str]:
    """
    Naive sentence-level claim splitter. Good enough for a portfolio project;
    swap for a proper claim-extraction LLM call if you want higher fidelity.
    """
    import re
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]