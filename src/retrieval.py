# src/retrieval.py
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from langchain_cohere import CohereRerank
from src.config import (
    COHERE_API_KEY, RETRIEVE_K, RERANK_TOP_N
)

logger = logging.getLogger(__name__)

# Module-level cached reranker singleton with thread safety
_reranker_instance = None
_reranker_lock = threading.Lock()
MAX_RERANK_CANDIDATES = 15


def get_reranker():
    """Returns a thread-safe cached singleton CohereRerank client to reuse HTTP connections."""
    global _reranker_instance
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:
                _reranker_instance = CohereRerank(
                    cohere_api_key=COHERE_API_KEY,
                    model="rerank-english-v3.0",
                    top_n=RERANK_TOP_N,
                )
    return _reranker_instance


def _get_doc_key(doc) -> str:
    """
    Generates a reliable, unique key for a document chunk.
    Prefers explicit source + chunk_id metadata; falls back to SHA-256 of normalized text.
    """
    source = doc.metadata.get("source")
    chunk_id = doc.metadata.get("chunk_id")
    if source is not None and chunk_id is not None:
        return f"{source}:{chunk_id}"
    
    page = doc.metadata.get("page")
    if source is not None and page is not None:
        content_hash = hashlib.sha256(doc.page_content.strip().encode("utf-8")).hexdigest()[:16]
        return f"{source}:p{page}:{content_hash}"

    return hashlib.sha256(doc.page_content.strip().encode("utf-8")).hexdigest()


def build_hybrid_retriever(vectorstore, bm25_retriever):
    """
    Wraps vectorstore as a dense retriever with search depth k=RETRIEVE_K.
    Does NOT mutate shared state on bm25_retriever.
    """
    dense = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVE_K},
    )
    return dense, bm25_retriever


def reciprocal_rank_fusion(dense_docs, sparse_docs, k=60):
    """Merge dense + sparse results using RRF scoring with stable document identity."""
    scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense_docs):
        key = _get_doc_key(doc)
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_docs):
        key = _get_doc_key(doc)
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[k] for k in sorted_keys]


def retrieve(query, vectorstore, bm25_retriever, use_rerank=True):
    """
    Optimized hybrid retrieval pipeline:
    1. Parallel execution of Dense (vector) and Sparse (BM25/Search) search.
    2. Reciprocal Rank Fusion (RRF) with stable document deduplication.
    3. Candidate pool capping (top 15) to minimize WAN payload to Cohere.
    4. Reranking via thread-safe singleton Cohere client with explicit top_n capping.
    5. Graceful fallback to RRF candidates on network or API failures.
    """
    dense, sparse = build_hybrid_retriever(vectorstore, bm25_retriever)

    # 1. Fetch dense and sparse results concurrently in background threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_dense = executor.submit(dense.invoke, query)
        future_sparse = executor.submit(sparse.invoke, query)
        dense_docs = future_dense.result()
        sparse_docs = future_sparse.result()

    # 2. Merge with Reciprocal Rank Fusion
    merged = reciprocal_rank_fusion(dense_docs, sparse_docs)

    # 3. Optional or fallback reranking
    if not use_rerank or not merged:
        return merged[:RERANK_TOP_N]

    # Pre-filter to top candidate pool to reduce WAN payload & latency
    candidates = merged[:MAX_RERANK_CANDIDATES]

    try:
        reranker = get_reranker()
        reranked = reranker.compress_documents(candidates, query)
        # Explicitly enforce RERANK_TOP_N cap
        return reranked[:RERANK_TOP_N]
    except Exception as e:
        logger.warning("Cohere reranking failed (%s), falling back to top RRF results.", e)
        return candidates[:RERANK_TOP_N]
