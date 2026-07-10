# app.py
# main streamlit app, ties all modules together

import os
import tempfile
import streamlit as st
from src.ingestion import ingest_pipeline, load_vectorstore, load_bm25
from src.retrieval import retrieve
from src.generation import generate_answer, get_sources, format_docs
from src.hyde import build_hyde_chain, expand_query
from src.config import CHROMA_DIR

st.set_page_config(page_title="RAG - Research Papers", layout="wide")
st.title("KNOWLEDGE RAG")

# sidebar for uploading and settings
with st.sidebar:
    st.header("Upload Papers")
    uploaded = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True,
        type=["pdf"],
    )

    if uploaded and st.button("Index Documents"):
        # save uploaded files to docs/ temporarily
        paths = []
        for f in uploaded:
            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
                dir="./docs"
            )
            tmp.write(f.read())
            tmp.close()
            paths.append(tmp.name)

        with st.spinner("Chunking and embedding papers..."):
            vs, bm25, n = ingest_pipeline()
            st.session_state["vs"]   = vs
            st.session_state["bm25"] = bm25
        st.success(f"Indexed {n} chunks from {len(paths)} papers")

    st.divider()
    use_hyde = st.toggle("HyDE query expansion", value=True)
    show_src = st.toggle("Show source chunks", value=True)

# load existing index if it exists
if "vs" not in st.session_state and os.path.exists(CHROMA_DIR) and os.path.exists("bm25.pkl"):
    with st.spinner("Loading existing index..."):
        st.session_state["vs"]   = load_vectorstore()
        st.session_state["bm25"] = load_bm25()

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# handle new question
if question := st.chat_input("Ask anything about your research papers..."):
    if "vs" not in st.session_state:
        st.error("Please upload and index papers first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        vs   = st.session_state["vs"]
        bm25 = st.session_state["bm25"]

        # expand query with HyDE if enabled
        search_query = question
        hyde_doc     = None
        if use_hyde:
            hyde_chain   = build_hyde_chain()
            search_query = expand_query(question, hyde_chain)
            hyde_doc     = search_query

        # retrieve relevant chunks
        with st.spinner("Searching papers..."):
            docs = retrieve(search_query, vs, bm25)

        # generate answer
        with st.spinner("Generating answer..."):
            answer = generate_answer(question, docs)
            st.markdown(answer)

        # show sources
        if show_src and docs:
            sources = get_sources(docs)
            with st.expander(f"Sources ({len(sources)} chunks)"):
                for s in sources:
                    label = s["source"]
                    if s["page"] != "":
                        label += f" - page {s['page']}"
                    st.markdown(f"**{label}**")
                    st.caption(s["snippet"])

        # show HyDE expansion
        if use_hyde and hyde_doc:
            with st.expander("HyDE hypothetical document"):
                st.caption(hyde_doc)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )