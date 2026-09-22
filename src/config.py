# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Models
EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
LLM_MODEL   = "nvidia/nemotron-3-ultra-550b-a55b"

# Chunking
CHUNK_SIZE    = 900
CHUNK_OVERLAP = 70 

# Retrieval
RETRIEVE_K   = 20   # fetch before rerank
RERANK_TOP_N = 5    # keep after rerank

# Vector Store (PyPDF Standard)
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "rag_papers"
BM25_PKL        = "bm25.pkl"
INDEXED_LOG     = "indexed_files.txt"

# Vector Store (LlamaParse Markdown)
CHROMA_DIR_LLAMAPARSE      = "./chroma_db_llamaparse"
COLLECTION_NAME_LLAMAPARSE = "rag_papers_llamaparse"
BM25_PKL_LLAMAPARSE        = "bm25_llamaparse.pkl"
INDEXED_LOG_LLAMAPARSE     = "indexed_llamaparse_files.txt"

# Docs folders
DOCS_DIR        = "./docs"
PARSED_DOCS_DIR = "./parsed_docs/llamaparse"

