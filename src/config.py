# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Models
LLM_MODEL   = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
EMBED_MODEL = "models/gemini-embedding-001"

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