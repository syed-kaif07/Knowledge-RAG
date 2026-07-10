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

INDEXED_FILES_LOG = "indexed_files.txt"

def get_indexed_files():
    # track which files have already been indexed
    if not os.path.exists(INDEXED_FILES_LOG):
        return set()
    with open(INDEXED_FILES_LOG, "r") as f:
        return set(line.strip() for line in f.readlines())

def save_indexed_file(filename):
    with open(INDEXED_FILES_LOG, "a") as f:
        f.write(filename + "\n")

def get_new_files():
    indexed = get_indexed_files()
    all_pdfs = set(
        f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")
    )
    return all_pdfs - indexed

def load_documents(filenames):
    docs = []
    for file in filenames:
        path = os.path.join(DOCS_DIR, file)
        loader = PyPDFLoader(path)
        loaded = loader.load()
        docs.extend(loaded)
        print(f"  Loaded: {file} ({len(loaded)} pages)")
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

def append_to_vectorstore(chunks):
    # add new chunks to existing vector store without rebuilding
    print("  Appending to existing vector store...")
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    vectorstore.add_documents(chunks)
    print("  Appended successfully")
    return vectorstore

def load_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

def build_bm25_from_scratch():
    # build BM25 from all docs in docs/ folder
    all_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    docs      = load_documents(all_files)
    chunks    = chunk_documents(docs)
    bm25      = BM25Retriever.from_documents(chunks)
    bm25.k    = 20
    with open("bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    return bm25

def load_bm25():
    with open("bm25.pkl", "rb") as f:
        return pickle.load(f)

def ingest_pipeline():
    new_files = get_new_files()

    if not new_files:
        print("No new files to index.")
        return None, None, 0

    print(f"Found {len(new_files)} new file(s): {new_files}")
    docs   = load_documents(new_files)
    chunks = chunk_documents(docs)

    # append to vector store or build fresh
    if os.path.exists(CHROMA_DIR):
        vs = append_to_vectorstore(chunks)
    else:
        vs = build_vectorstore(chunks)

    # BM25 always rebuilt from all docs (it's fast and in-memory)
    bm25 = build_bm25_from_scratch()

    # mark new files as indexed
    for f in new_files:
        save_indexed_file(f)

    print(f"Done. {len(chunks)} new chunks indexed")
    return vs, bm25, len(chunks)