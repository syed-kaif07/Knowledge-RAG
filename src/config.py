# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Models
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
LLM_MODEL   = "meta/llama-3.1-8b-instruct"

# Chunking
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 80

# Retrieval
RETRIEVE_K   = 20   # fetch before rerank
RERANK_TOP_N = 5    # keep after rerank

# Vector Store
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "rag_papers"

# Docs folder
DOCS_DIR = "./docs"