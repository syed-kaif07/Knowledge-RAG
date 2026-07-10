# src/ingestion.py
import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from src.config import (
    NVIDIA_API_KEY, EMBED_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
    CHROMA_DIR, COLLECTION_NAME, DOCS_DIR
)

def load_documents():
    docs = []
    for file in os.listdir(DOCS_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(DOCS_DIR, file)
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
            print(f"  Loaded: {file}")
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    print(f"  Total chunks: {len(chunks)}")
    return chunks

def get_embeddings():
    return NVIDIAEmbeddings(
        model=EMBED_MODEL,
        api_key=NVIDIA_API_KEY,
        truncate="END",
    )

def build_vectorstore(chunks):
    print("  Building vector store...")
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    print("  Vector store ready")
    return vectorstore

def load_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

def build_bm25(chunks):
    print("  Building BM25 index...")
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 20
    with open("bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print("  BM25 ready")
    return bm25

def load_bm25():
    with open("bm25.pkl", "rb") as f:
        return pickle.load(f)

def ingest_pipeline():
    print("Starting ingestion...")
    docs   = load_documents()
    chunks = chunk_documents(docs)
    vs     = build_vectorstore(chunks)
    bm25   = build_bm25(chunks)
    print(f"Done. {len(chunks)} chunks indexed")
    return vs, bm25, len(chunks)