# Knowledge RAG

A production-grade RAG system for querying research papers using NVIDIA NIM models, hybrid search, Cohere reranking, and HyDE query expansion.

## Architecture

![RAG Architecture](assets/rag_architecture.jpg)

## Tech Stack

| Component | Tool |
|-----------|------|
| LLM | NVIDIA NIM — meta/llama-3.1-8b-instruct |
| Embeddings | NVIDIA NIM — nvidia/nv-embedqa-e5-v5 |
| Vector Store | Chroma (persistent) |
| Sparse Search | BM25 (rank-bm25) |
| Reranker | Cohere rerank-english-v3.0 |
| Query Expansion | HyDE |
| Framework | LangChain + Streamlit |
| Language | Python 3.12 |

## Features

- Hybrid search — BM25 + vector search merged with Reciprocal Rank Fusion
- Cohere reranking — retrieves 20 docs, reranks to top 5
- HyDE query expansion — generates hypothetical document for better retrieval
- Persistent index — index once, reloads automatically on every run
- Incremental indexing — only new PDFs get indexed, existing ones untouched
- Source citations — every answer cites the source PDF and page number
- No hallucination policy — answers strictly from provided documents

## Project Structure

```
RAG/
├── src/
│   ├── __init__.py
│   ├── config.py        # API keys, model names, chunking settings
│   ├── ingestion.py     # PDF loading, chunking, embedding, indexing
│   ├── retrieval.py     # Hybrid search, RRF merging, Cohere reranking
│   ├── hyde.py          # HyDE query expansion
│   └── generation.py    # RAG chain, prompt, answer generation
├── docs/                # PDF files (gitignored)
├── chroma_db/           # Persistent vector store (gitignored)
├── app.py               # Streamlit UI
├── bm25.pkl             # BM25 index (gitignored)
├── indexed_files.txt    # Tracks which PDFs have been indexed
├── requirements.txt
└── .env                 # API keys (gitignored)
```

## Setup

```bash
# Create virtual environment (used python 3.12)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the root:

```
NVIDIA_API_KEY=your_nvidia_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
```

Get your API keys:
- NVIDIA: https://build.nvidia.com (free credits)
- Cohere: https://cohere.com (free trial)

## Run

```bash
streamlit run app.py
```

1. Add PDF files to the `docs/` folder
2. Click **Index New Documents** in the sidebar
3. Ask questions in the chat

On every subsequent run, the existing index loads automatically — no need to re-index.

## Adding New Documents

Drop any new PDF into the `docs/` folder and click **Index New Documents**. Only the new files will be indexed and appended to the existing vector store.

## Test Questions

These questions were used to validate the system across multiple papers:

1. What are the key differences between RAG-Sequence and RAG-Token models?
2. How does Reflexion use verbal reinforcement learning to improve agent performance?
3. What datasets were used to evaluate the original RAG model and what were the results?
4. How does Self-RAG use critique tokens to assess relevance and support of retrieved passages?
5. What are the failure modes of the ReAct framework mentioned in the paper?

## Papers Indexed

| Paper | Authors | Year |
|-------|---------|------|
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Lewis et al. | 2020 |
| ReAct: Synergizing Reasoning and Acting in Language Models | Yao et al. | 2022 |
| Reflexion: Language Agents with Verbal Reinforcement Learning | Shinn et al. | 2023 |
| Self-RAG: Learning to Retrieve, Generate, and Critique | Asai et al. | 2023 |
