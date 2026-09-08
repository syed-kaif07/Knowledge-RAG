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

# Vector Store
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "rag_papers"

# Docs folder
DOCS_DIR = "./docs"
