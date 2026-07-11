# app.py
import os
import time
import streamlit as st
from src.ingestion import ingest_pipeline, load_vectorstore, load_bm25, get_new_files
from src.retrieval import retrieve
from src.generation import generate_answer, get_sources
from src.hyde import build_hyde_chain, expand_query
from src.config import CHROMA_DIR
from rag_observability.observability import RAGLogger, ChunkRecord
from rag_observability.dashboard import render_observability_tab

st.set_page_config(page_title="Research Paper RAG", layout="wide")
st.title("KNOWLEDGE RAG")


@st.cache_resource
def get_logger():
    return RAGLogger(db_path="rag_logs.db")


logger = get_logger()

chat_tab, obs_tab = st.tabs(["Chat", "Observability"])

with st.sidebar:
    st.header("Upload Papers")
    uploaded = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True,
        type=["pdf"],
    )

    if uploaded:
        # save with original filename, no temp files
        saved = []
        for f in uploaded:
            dest = os.path.join("./docs", f.name)
            if not os.path.exists(dest):
                with open(dest, "wb") as out:
                    out.write(f.read())
                saved.append(f.name)

        if saved:
            st.info(f"Saved {len(saved)} new file(s): {', '.join(saved)}")
        else:
            st.info("All uploaded files already exist in docs/")

    if st.button("Index New Documents"):
        new_files = get_new_files()
        if not new_files:
            st.success("Nothing new to index — all docs already indexed.")
        else:
            with st.spinner(f"Indexing {len(new_files)} new file(s)..."):
                vs, bm25, n = ingest_pipeline()
                if vs and bm25:
                    st.session_state["vs"]   = vs
                    st.session_state["bm25"] = bm25
                    st.success(f"Indexed {n} new chunks from {len(new_files)} file(s)")

    st.divider()
    use_hyde = st.toggle("HyDE query expansion", value=True)
    show_src = st.toggle("Show source chunks", value=True)

# auto-load existing index on every page load
if "vs" not in st.session_state and os.path.exists(CHROMA_DIR) and os.path.exists("bm25.pkl"):
    with st.spinner("Loading existing index..."):
        st.session_state["vs"]   = load_vectorstore()
        st.session_state["bm25"] = load_bm25()
    st.sidebar.success("Index loaded automatically")

if "messages" not in st.session_state:
    st.session_state.messages = []

with chat_tab:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask anything about your research papers..."):
        if "vs" not in st.session_state:
            st.error("No index found. Upload PDFs and click Index New Documents first.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            vs   = st.session_state["vs"]
            bm25 = st.session_state["bm25"]

            search_query = question
            hyde_doc     = None
            if use_hyde:
                hyde_chain   = build_hyde_chain()
                search_query = expand_query(question, hyde_chain)
                hyde_doc     = search_query

            start = time.time()

            with st.spinner("Searching papers..."):
                docs = retrieve(search_query, vs, bm25)

            with st.spinner("Generating answer..."):
                answer = generate_answer(question, docs)
                st.markdown(answer)

            latency_ms = (time.time() - start) * 1000

            if show_src and docs:
                sources = get_sources(docs)
                with st.expander(f"Sources ({len(sources)} chunks)"):
                    for s in sources:
                        label = s["source"]
                        if s["page"] != "":
                            label += f" - page {s['page']}"
                        st.markdown(f"**{label}**")
                        st.caption(s["snippet"])

            if use_hyde and hyde_doc:
                with st.expander("HyDE hypothetical document"):
                    st.caption(hyde_doc)

            # --- observability logging ---
            # docs are LangChain Documents from retrieve(); if your retrieve()
            # attaches score fields to metadata (e.g. "rrf_score", "rerank_score"),
            # ChunkRecord.from_langchain_doc will pick them up automatically.
            # Adjust the metadata keys below if your pipeline names them differently.
            chunk_records = [
                ChunkRecord.from_langchain_doc(
                    d,
                    rrf_score=d.metadata.get("rrf_score"),
                    rerank_score=d.metadata.get("rerank_score"),
                )
                for d in docs
            ]

            log_id = logger.log_query(
                query=question,
                retrieved_chunks=chunk_records,
                final_context_chunks=chunk_records,
                answer=answer,
                hyde_used=use_hyde,
                latency_ms=latency_ms,
            )

            # Faithfulness scoring reuses your generation LLM as judge.
            # generate_answer() must expose the underlying chat model for this
            # to work — swap `judge_llm` below for whatever object in
            # src/generation.py wraps your NIM llama-3.1-8b-instruct client.
            try:
                from src.generation import llm as judge_llm
                logger.score_faithfulness(
                    log_id=log_id,
                    answer=answer,
                    cited_chunks=chunk_records,
                    llm_judge=judge_llm,
                )
            except ImportError:
                pass  # skip scoring if no importable llm object found

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

with obs_tab:
    render_observability_tab(logger)