# src/retrieval.py
import pickle
from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from src.config import (
    COHERE_API_KEY, RETRIEVE_K, RERANK_TOP_N
)

def build_hybrid_retriever(vectorstore, bm25_retriever):
    """BM25 + Chroma with manual merge (RRF-style)."""
    dense = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVE_K},
    )
    bm25_retriever.k = RETRIEVE_K
    return dense, bm25_retriever

def reciprocal_rank_fusion(dense_docs, sparse_docs, k=60):
    """Merge dense + sparse results using RRF scoring."""
    scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_docs):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[k] for k in sorted_keys]

def build_reranker():
    return CohereRerank(
        cohere_api_key=COHERE_API_KEY,
        model="rerank-english-v3.0",
        top_n=RERANK_TOP_N,
    )

def retrieve(query, vectorstore, bm25_retriever):
    """Full pipeline: hybrid → RRF → rerank."""
    dense, sparse = build_hybrid_retriever(vectorstore, bm25_retriever)

    # get results from both
    dense_docs  = dense.invoke(query)
    sparse_docs = sparse.invoke(query)

    # merge with RRF
    merged = reciprocal_rank_fusion(dense_docs, sparse_docs)

    # rerank top results
    reranker    = build_reranker()
    reranked    = reranker.compress_documents(merged, query)

    return reranked