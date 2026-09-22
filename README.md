# Knowledge RAG

A fast, accurate RAG system for querying AI and machine learning research papers using NVIDIA NIM models, hybrid search (Chroma + BM25), Cohere reranking, HyDE expansion, LlamaParse markdown indexing, and live RAGAS evaluation.

---

## What's New

### 1. LlamaParse Markdown Pipeline & Separate Databases
- **High-Fidelity PDF Parsing**: Converts complex research PDFs (with tables, math, and code) into clean Markdown using LlamaParse.
- **Markdown-Aware Chunking**: Splits text hierarchically by headers (`#`, `##`, `###`) while preserving section context and metadata before applying recursive token splitting.
- **Dual Index Collections**:
  - **PyPDF (Original)**: Stored in `chroma_db/` & `bm25.pkl`.
  - **LlamaParse (Markdown)**: Stored in `chroma_db_llamaparse/` & `bm25_llamaparse.pkl`.
- **UI Switcher**: Switch between PyPDF and LlamaParse index collections directly from the sidebar.

### 2. Live RAGAS 0.4.3 Evaluation
- Evaluates queries live on 3 core metrics:
  - **Faithfulness**: Checks if claims in the answer are grounded in the retrieved context.
  - **Answer Relevancy**: Checks if the answer directly addresses the question.
  - **Context Utilization**: Measures how much of the retrieved context was effectively utilized.
- **Evaluation Dashboard Tab**: Displays real-time session averages, metric trend charts, and per-query logs with latency.

---

## Architecture & Pipeline

```
Query ──► [HyDE Expansion] ──► Hybrid Retrieval (Chroma Dense + BM25 Sparse)
                                             │
                                             ▼
                                  Reciprocal Rank Fusion (RRF)
                                             │
                                             ▼
                                  Cohere Rerank (Top 20 -> Top 5)
                                             │
                                             ▼
                                  NVIDIA LLM (Grounded Generation)
                                             │
                                             ▼
                                  RAGAS Evaluation & Streamlit UI
```

---

## Tech Stack

| Component | Tool / Model |
| :--- | :--- |
| **LLM** | NVIDIA NIM (`nvidia/nemotron-3-ultra-550b-a55b`) |
| **Embeddings** | NVIDIA NIM (`nvidia/llama-nemotron-embed-vl-1b-v2`) |
| **Vector DB** | ChromaDB (`chroma_db/` & `chroma_db_llamaparse/`) |
| **Sparse Search** | BM25 (`rank-bm25`) |
| **Reranker** | Cohere (`rerank-english-v3.0`) |
| **Query Expansion**| HyDE |
| **PDF Parser** | LlamaParse (Markdown result) & PyPDF |
| **Evaluation** | RAGAS 0.4.3 (Faithfulness, Relevancy, Context Utilization) |
| **UI** | Streamlit |

---

## Project Structure

```
RAG/
├── src/
│   ├── config.py                 # Paths, models, chunk sizes, and API config
│   ├── ingestion.py              # PyPDF loading, chunking, and Chroma/BM25 indexing
│   ├── markdown_ingestion.py     # LlamaParse markdown loading, header chunking, and indexing
│   ├── llamaparse_ingestion.py   # LlamaParse PDF-to-Markdown parser script
│   ├── retrieval.py              # Hybrid search, RRF merging, Cohere reranking
│   ├── hyde.py                   # Hypothetical Document Embeddings (HyDE) expansion
│   ├── generation.py             # Grounded prompt template, LLM generation, source citation
│   └── eval_ragas.py             # RAGAS 0.4.3 evaluation logic
├── docs/                         # Raw PDF documents
├── parsed_docs/llamaparse/       # Converted markdown files
├── chroma_db/                    # PyPDF Chroma database
├── chroma_db_llamaparse/         # LlamaParse Chroma database
├── bm25.pkl                      # PyPDF BM25 index
├── bm25_llamaparse.pkl           # LlamaParse BM25 index
├── indexed_files.txt             # PyPDF indexed file tracker
├── indexed_llamaparse_files.txt  # LlamaParse indexed file tracker
├── index_llamaparse.py           # CLI script to batch-index markdown documents
├── parse_all_pdfs.py             # CLI script to parse all PDFs to markdown via LlamaParse
├── app.py                         # Streamlit chat interface & evaluation dashboard
├── requirements.txt
└── .env                          # API keys
```

---

## Setup & Installation

### 1. Clone & create virtual environment
```bash
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment variables
Create a `.env` file in the root folder:
```env
NVIDIA_API_KEY=your_nvidia_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
```

---

## Usage

### Run the App
```bash
streamlit run app.py
```

### Ingestion Workflows
- **LlamaParse Markdown Mode**:
  - Run `python parse_all_pdfs.py` to parse PDFs in `docs/` to Markdown in `parsed_docs/llamaparse/`.
  - Run `python index_llamaparse.py` (or click **Index Markdown Documents** in the UI).
- **PyPDF Standard Mode**:
  - Upload PDFs in the sidebar and click **Index New PDFs**.

---

## Evaluation Metrics

When asking questions in the Chat tab, the app automatically evaluates answers:
- **Faithfulness**: Verifies zero hallucinations by checking claims against retrieved chunks.
- **Answer Relevancy**: Ensures response directly answers the user query.
- **Context Utilization**: Quantifies precision of retrieved context in the final answer.

Check aggregate statistics and trends anytime under the **RAGAS Evaluation** tab.