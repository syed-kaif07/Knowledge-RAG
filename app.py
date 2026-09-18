# app.py
import os
import time
import streamlit as st
import pandas as pd
from src.ingestion import ingest_pipeline, load_vectorstore, load_bm25, get_new_files
from src.retrieval import retrieve
from src.generation import generate_answer, get_sources
from src.hyde import build_hyde_chain, expand_query
from src.config import CHROMA_DIR
from src.eval_ragas import evaluate_query_ragas

st.set_page_config(page_title="Research Paper RAG", layout="wide")
st.title("KNOWLEDGE RAG")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ragas_history" not in st.session_state:
    st.session_state.ragas_history = []

chat_tab, obs_tab = st.tabs(["Chat", "RAGAS Evaluation"])

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
    enable_eval = st.toggle("Run live RAGAS evaluation", value=True)

# auto-load existing index on every page load
if "vs" not in st.session_state and os.path.exists(CHROMA_DIR) and os.path.exists("bm25.pkl"):
    with st.spinner("Loading existing index..."):
        st.session_state["vs"]   = load_vectorstore()
        st.session_state["bm25"] = load_bm25()
    st.sidebar.success("Index loaded automatically")

with chat_tab:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("ragas_scores"):
                scores = msg["ragas_scores"]
                with st.expander("Evaluation Metrics (RAGAS 0.4.3)"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Faithfulness", f"{scores['faithfulness']:.0%}")
                    c2.metric("Answer Relevancy", f"{scores['answer_relevancy']:.0%}")
                    c3.metric("Context Utilization", f"{scores['context_utilization']:.0%}")

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

            # --- RAGAS 0.4.3 Online Evaluation (Non-critical execution) ---
            ragas_scores = None
            if enable_eval and docs:
                try:
                    with st.spinner("Running RAGAS evaluation..."):
                        context_texts = [d.page_content for d in docs]
                        ragas_scores = evaluate_query_ragas(question, answer, context_texts)
                        if ragas_scores:
                            ragas_entry = {
                                "query": question,
                                "latency_ms": latency_ms,
                                **ragas_scores
                            }
                            st.session_state.ragas_history.append(ragas_entry)
                            with st.expander("Evaluation Metrics (RAGAS 0.4.3)"):
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Faithfulness", f"{ragas_scores['faithfulness']:.0%}")
                                c2.metric("Answer Relevancy", f"{ragas_scores['answer_relevancy']:.0%}")
                                c3.metric("Context Utilization", f"{ragas_scores['context_utilization']:.0%}")
                except Exception as e:
                    print(f"[RAGAS Warning] Evaluation skipped: {e}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "ragas_scores": ragas_scores}
        )

with obs_tab:
    st.subheader("RAGAS 0.4.3 Evaluation Dashboard")

    if not st.session_state.ragas_history:
        st.info("No queries evaluated in this session yet. Ask a question in the Chat tab to view real-time RAGAS metrics.")
    else:
        df_history = pd.DataFrame(st.session_state.ragas_history)

        avg_faith = df_history["faithfulness"].mean()
        avg_rel   = df_history["answer_relevancy"].mean()
        avg_util  = df_history["context_utilization"].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Faithfulness", f"{avg_faith:.0%}")
        m2.metric("Avg Answer Relevancy", f"{avg_rel:.0%}")
        m3.metric("Avg Context Utilization", f"{avg_util:.0%}")
        m4.metric("Total Evaluated", len(df_history))

        st.divider()
        st.subheader("Session Metric Trends")
        st.line_chart(df_history[["faithfulness", "answer_relevancy", "context_utilization"]])

        st.divider()
        st.subheader("Evaluated Queries Log")
        for i, row in df_history.iloc[::-1].iterrows():
            with st.expander(f"Query {i+1}: {row['query'][:80]}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Faithfulness", f"{row['faithfulness']:.0%}")
                c2.metric("Answer Relevancy", f"{row['answer_relevancy']:.0%}")
                c3.metric("Context Utilization", f"{row['context_utilization']:.0%}")
                if "latency_ms" in row and pd.notnull(row["latency_ms"]):
                    st.caption(f"Latency: {row['latency_ms']:.0f} ms")