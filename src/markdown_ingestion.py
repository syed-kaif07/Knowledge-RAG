# src/markdown_ingestion.py
import os
import pickle
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from src.config import (
    NVIDIA_API_KEY, EMBED_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
    CHROMA_DIR_LLAMAPARSE, COLLECTION_NAME_LLAMAPARSE,
    BM25_PKL_LLAMAPARSE, INDEXED_LOG_LLAMAPARSE,
    PARSED_DOCS_DIR
)


def get_indexed_llamaparse_files():
    if not os.path.exists(INDEXED_LOG_LLAMAPARSE):
        return set()
    with open(INDEXED_LOG_LLAMAPARSE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines() if line.strip())


def save_indexed_llamaparse_file(filename):
    with open(INDEXED_LOG_LLAMAPARSE, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


def get_new_llamaparse_files():
    indexed = get_indexed_llamaparse_files()
    if not os.path.exists(PARSED_DOCS_DIR):
        return set()
    all_mds = set(
        f for f in os.listdir(PARSED_DOCS_DIR) if f.endswith(".md")
    )
    return all_mds - indexed


def load_markdown_documents(filenames):
    docs = []
    for file in filenames:
        path = os.path.join(PARSED_DOCS_DIR, file)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        doc = Document(
            page_content=text,
            metadata={"source": file, "doc_type": "markdown"}
        )
        docs.append(doc)
        print(f"  Loaded Markdown: {file} ({len(text)} chars)")
    return docs


def chunk_markdown_documents(docs):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    chunk_counter = 0

    for doc in docs:
        source_name = doc.metadata.get("source", "unknown")
        # Step 1: Split by markdown headers
        header_splits = header_splitter.split_text(doc.page_content)

        # Step 2: Split oversized sections using recursive character splitter
        for header_chunk in header_splits:
            # Preserve original doc metadata and merge header metadata
            meta = dict(doc.metadata)
            meta.update(header_chunk.metadata)
            meta["source"] = source_name

            sub_docs = text_splitter.create_documents(
                texts=[header_chunk.page_content],
                metadatas=[meta]
            )
            for sub_doc in sub_docs:
                sub_doc.metadata["chunk_id"] = chunk_counter
                chunk_counter += 1
                all_chunks.append(sub_doc)

    print(f"  Total Markdown chunks created: {len(all_chunks)}")
    return all_chunks


def get_embeddings():
    return NVIDIAEmbeddings(
        model=EMBED_MODEL,
        api_key=NVIDIA_API_KEY,
        truncate="END",
    )


def build_llamaparse_vectorstore(chunks):
    print("  Building LlamaParse vector store in ChromaDB...")
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR_LLAMAPARSE,
        collection_name=COLLECTION_NAME_LLAMAPARSE,
    )
    print("  LlamaParse vector store ready")
    return vectorstore


def append_to_llamaparse_vectorstore(chunks):
    print("  Appending to existing LlamaParse vector store...")
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR_LLAMAPARSE,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME_LLAMAPARSE,
    )
    vectorstore.add_documents(chunks)
    print("  Appended to LlamaParse vector store successfully")
    return vectorstore


def load_llamaparse_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR_LLAMAPARSE,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME_LLAMAPARSE,
    )


def build_llamaparse_bm25_from_scratch():
    if not os.path.exists(PARSED_DOCS_DIR):
        return None
    all_files = [f for f in os.listdir(PARSED_DOCS_DIR) if f.endswith(".md")]
    if not all_files:
        return None
    print(f"  Building LlamaParse BM25 index from {len(all_files)} markdown files...")
    docs = load_markdown_documents(all_files)
    chunks = chunk_markdown_documents(docs)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 20
    with open(BM25_PKL_LLAMAPARSE, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  Saved LlamaParse BM25 index to {BM25_PKL_LLAMAPARSE}")
    return bm25


def load_llamaparse_bm25():
    with open(BM25_PKL_LLAMAPARSE, "rb") as f:
        return pickle.load(f)


def ingest_llamaparse_pipeline():
    new_files = sorted(list(get_new_llamaparse_files()))

    if not new_files:
        print("No new Markdown files to index.")
        return None, None, 0

    print(f"Found {len(new_files)} new Markdown file(s): {new_files}")
    docs = load_markdown_documents(new_files)
    chunks = chunk_markdown_documents(docs)

    if os.path.exists(CHROMA_DIR_LLAMAPARSE) and os.listdir(CHROMA_DIR_LLAMAPARSE):
        vs = append_to_llamaparse_vectorstore(chunks)
    else:
        vs = build_llamaparse_vectorstore(chunks)

    bm25 = build_llamaparse_bm25_from_scratch()

    for f in new_files:
        save_indexed_llamaparse_file(f)

    print(f"Done. {len(chunks)} new Markdown chunks indexed.")
    return vs, bm25, len(chunks)
