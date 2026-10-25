# src/ingestion.py
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_mongodb import MongoDBAtlasVectorSearch
from src.config import (
    NVIDIA_API_KEY, EMBED_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
    DOCS_DIR, MONGO_URI, MONGO_DB_NAME,
    MONGO_COLLECTION_PYPDF, MONGO_VECTOR_INDEX_NAME,
    BM25_PKL
)

INDEXED_FILES_LOG = "indexed_files.txt"

# Module-level cached singletons
_mongo_client = None
_embeddings = None


def get_mongo_client():
    global _mongo_client
    if _mongo_client is None:
        if not MONGO_URI:
            raise ValueError("MONGO_URI not found in environment or .env file.")
        _mongo_client = MongoClient(MONGO_URI)
    return _mongo_client


def get_mongo_collection():
    client = get_mongo_client()
    return client[MONGO_DB_NAME][MONGO_COLLECTION_PYPDF]


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = NVIDIAEmbeddings(
            model=EMBED_MODEL,
            api_key=NVIDIA_API_KEY,
            truncate="END",
        )
    return _embeddings


def get_indexed_files():
    # track which files have already been indexed
    if not os.path.exists(INDEXED_FILES_LOG):
        return set()
    with open(INDEXED_FILES_LOG, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines() if line.strip())


def save_indexed_file(filename):
    with open(INDEXED_FILES_LOG, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


def get_new_files():
    indexed = get_indexed_files()
    if not os.path.exists(DOCS_DIR):
        return set()
    all_pdfs = set(
        f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")
    )
    return all_pdfs - indexed


def _load_single_pdf(file):
    path = os.path.join(DOCS_DIR, file)
    try:
        loader = PyPDFLoader(path)
        loaded = loader.load()
        print(f"  Loaded: {file} ({len(loaded)} pages)")
        return loaded
    except Exception as e:
        print(f"  Failed loading {file}: {e}")
        return []


def load_documents(filenames):
    docs = []
    max_workers = min(4, len(filenames)) if filenames else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_load_single_pdf, filenames)
        for loaded_pages in results:
            docs.extend(loaded_pages)
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


def build_vectorstore(chunks, batch_size=100):
    """
    Builds or updates vector store in MongoDB Atlas with batched inserts.
    """
    print("  Connecting to MongoDB Atlas Vector Store...")
    collection = get_mongo_collection()
    embeddings = get_embeddings()

    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=MONGO_VECTOR_INDEX_NAME,
        relevance_score_fn="cosine",
    )

    if chunks:
        print(f"  Adding {len(chunks)} chunks to MongoDB Atlas in batches of {batch_size}...")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectorstore.add_documents(batch)
            print(f"    Indexed batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")

    try:
        # Create vector search index if it doesn't already exist
        vectorstore.create_vector_search_index(dimensions=2048)
        print("  Atlas vector search index configured.")
    except Exception as e:
        # If already exists or created via Atlas UI, log and proceed
        print(f"  Vector search index status: {e}")

    print("  MongoDB Atlas Vector Store ready.")
    return vectorstore


def append_to_vectorstore(chunks, batch_size=100):
    return build_vectorstore(chunks, batch_size=batch_size)


def load_vectorstore():
    collection = get_mongo_collection()
    embeddings = get_embeddings()
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=MONGO_VECTOR_INDEX_NAME,
        relevance_score_fn="cosine",
    )


def update_bm25_index(new_chunks=None):
    """
    Incrementally updates BM25 by reusing cached chunks without re-reading PDFs from disk.
    """
    all_chunks = []
    # If existing bm25.pkl exists, reuse existing chunks
    if os.path.exists(BM25_PKL):
        try:
            old_bm25 = load_bm25()
            if hasattr(old_bm25, "docs") and old_bm25.docs:
                all_chunks.extend(old_bm25.docs)
        except Exception as e:
            print(f"  Notice: could not load existing BM25 cache ({e}), rebuilding.")

    # If cache was empty or invalid, fallback to loading all PDFs once
    if not all_chunks:
        all_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
        docs = load_documents(all_files)
        all_chunks = chunk_documents(docs)
    elif new_chunks:
        all_chunks.extend(new_chunks)

    print(f"  Building in-memory BM25 index with {len(all_chunks)} total chunks...")
    bm25 = BM25Retriever.from_documents(all_chunks)
    bm25.k = 20
    with open(BM25_PKL, "wb") as f:
        pickle.dump(bm25, f)
    print("  BM25 index saved successfully.")
    return bm25


def build_bm25_from_scratch():
    return update_bm25_index(new_chunks=None)


def load_bm25():
    with open(BM25_PKL, "rb") as f:
        return pickle.load(f)


def ingest_pipeline():
    new_files = get_new_files()

    if not new_files:
        print("No new files to index.")
        return None, None, 0

    print(f"Found {len(new_files)} new file(s): {new_files}")
    docs = load_documents(new_files)
    chunks = chunk_documents(docs)

    # Ingest to MongoDB Atlas
    vs = append_to_vectorstore(chunks)

    # Incrementally update BM25 index
    bm25 = update_bm25_index(new_chunks=chunks)

    # Mark new files as indexed
    for f in new_files:
        save_indexed_file(f)

    print(f"Done. {len(chunks)} new chunks indexed into MongoDB Atlas.")
    return vs, bm25, len(chunks)