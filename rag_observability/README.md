# RAG Observability

`rag_observability` is a lightweight, framework-agnostic drop-in logging and evaluation tool for RAG (Retrieval-Augmented Generation) pipelines. It tracks retrieval quality, latency, chunk movement (pre/post-reranking), and evaluates answer faithfulness using an LLM judge.

---

## Features

- **Pipeline Trace Logging**: Logs query details, latency, HyDE toggles, chunk details (retrieved, reranked, and final context chunks), and the generated answer to a SQLite database (`rag_logs.db`).
- **Groundedness / Faithfulness Evaluation**: Uses a LangChain-compatible LLM judge to grade how well the generated answer is supported by the context chunks (0.0 - 1.0 scale).
- **Streamlit Dashboard Component**: A pre-built Streamlit tab to monitor average faithfulness, query counts, historical trends, and detail views for each logged query.


