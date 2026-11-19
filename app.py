# app.py
import html
import os
import time
import streamlit as st
import pandas as pd
from src.ingestion import ingest_pipeline, load_vectorstore, load_bm25, get_new_files
from src.markdown_ingestion import (
    ingest_llamaparse_pipeline,
    load_llamaparse_vectorstore,
    load_llamaparse_bm25,
    get_new_llamaparse_files
)
from src.retrieval import retrieve
from src.generation import generate_answer, get_sources
from src.config import (
    CHROMA_DIR, BM25_PKL,
    CHROMA_DIR_LLAMAPARSE, BM25_PKL_LLAMAPARSE
)
from src.eval_ragas import evaluate_query_ragas
from src.error_handler import setup_logging, log_api_error

logger = setup_logging()

st.set_page_config(page_title="Research Paper RAG", layout="wide")
st.title("KNOWLEDGE RAG")


def render_error_ui(error_info: dict) -> None:
    """Render a compact, clear UI card for request timeouts and API errors."""
    is_timeout = error_info.get("is_timeout", False)
    is_overloaded = error_info.get("is_overloaded", False)
    stage = html.escape(error_info.get("stage", "Request"))
    timestamp = html.escape(error_info.get("timestamp", ""))
    log_file = html.escape(error_info.get("log_file", "logs/app.log"))

    if is_overloaded:
        border_color = "#f97316"
        bg_color = "rgba(249, 115, 22, 0.10)"
        badge_bg = "rgba(249, 115, 22, 0.22)"
        badge_text = "#fb923c"
        title = "API Service Overloaded (503 / Rate Limit)"
        message = (
            f"The remote NVIDIA NIM service is temporarily overloaded during <strong>{stage}</strong>. "
            "Retries were attempted automatically; please wait a few seconds and try submitting your question again."
        )
    elif is_timeout:
        border_color = "#f59e0b"
        bg_color = "rgba(245, 158, 11, 0.10)"
        badge_bg = "rgba(245, 158, 11, 0.22)"
        badge_text = "#fbbf24"
        title = "Request Timeout Error"
        message = (
            f"The request timed out during <strong>{stage}</strong>. "
            "The remote API did not respond in time — please try again."
        )
    else:
        border_color = "#ef4444"
        bg_color = "rgba(239, 68, 68, 0.10)"
        badge_bg = "rgba(239, 68, 68, 0.22)"
        badge_text = "#f87171"
        title = "Request Failed"
        message = (
            f"An error occurred during <strong>{stage}</strong>. "
            "Please check your connection or API settings and try again."
        )

    meta_parts = []
    if error_info.get("host"):
        host_str = html.escape(error_info["host"])
        if error_info.get("timeout_sec"):
            host_str += f" ({html.escape(error_info['timeout_sec'])})"
        meta_parts.append(f"Endpoint: <code>{host_str}</code>")
    elif error_info.get("timeout_sec"):
        meta_parts.append(f"Timeout: <code>{html.escape(error_info['timeout_sec'])}</code>")

    if timestamp:
        meta_parts.append(f"Time: <code>{timestamp}</code>")
    meta_parts.append(f"Logged to: <code>{log_file}</code>")
    meta_html = " &nbsp;•&nbsp; ".join(meta_parts)

    st.markdown(
        f"""
        <div style="
            border: 1px solid {border_color};
            border-left: 5px solid {border_color};
            background-color: {bg_color};
            border-radius: 8px;
            padding: 12px 16px;
            margin: 6px 0 10px 0;
        ">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:8px; flex-wrap:wrap; margin-bottom:6px;">
                <span style="font-weight:700; font-size:0.98rem; color:{badge_text};">{title}</span>
                <span style="
                    font-size:0.75rem;
                    font-weight:600;
                    padding:2px 8px;
                    border-radius:999px;
                    background-color:{badge_bg};
                    color:{badge_text};
                ">{stage}</span>
            </div>
            <div style="font-size:0.9rem; line-height:1.45; margin-bottom:8px;">
                {message}
            </div>
            <div style="font-size:0.78rem; opacity:0.85;">
                {meta_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if error_info.get("details"):
        with st.expander("Technical details"):
            st.code(error_info["details"], language="text")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ragas_history" not in st.session_state:
    st.session_state.ragas_history = []

if "active_mode" not in st.session_state:
    st.session_state.active_mode = None

chat_tab, obs_tab = st.tabs(["Chat", "RAGAS Evaluation"])

with st.sidebar:
    st.header("Knowledge Base Settings")
    
    index_mode = st.radio(
        "Select Index Collection",
        ["LlamaParse (Markdown Docs)", "PyPDF (Standard Docs)"],
        index=0,
        help="Choose between structured LlamaParse markdown index or raw PyPDF chunk index."
    )
    
    # Reload store if switching modes
    mode_key = "llamaparse" if "LlamaParse" in index_mode else "pypdf"
    if st.session_state.active_mode != mode_key:
        st.session_state.active_mode = mode_key
        st.session_state.pop("vs", None)
        st.session_state.pop("bm25", None)

    st.divider()
    st.header("Upload & Ingestion")

    if mode_key == "pypdf":
        uploaded = st.file_uploader(
            "Upload PDF files for PyPDF",
            accept_multiple_files=True,
            type=["pdf"],
        )
        if uploaded:
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

        if st.button("Index New PDFs"):
            new_files = get_new_files()
            if not new_files:
                st.success("All PDFs are already indexed.")
            else:
                try:
                    with st.spinner(f"Indexing {len(new_files)} new file(s)..."):
                        vs, bm25, n = ingest_pipeline()
                        if vs and bm25:
                            st.session_state["vs"] = vs
                            st.session_state["bm25"] = bm25
                            st.success(f"Indexed {n} new chunks from {len(new_files)} file(s)")
                except Exception as e:
                    err_info = log_api_error("PyPDF Indexing", e)
                    render_error_ui(err_info)

    else:
        # LlamaParse Markdown mode
        unindexed_mds = get_new_llamaparse_files()
        if unindexed_mds:
            st.warning(f"Found {len(unindexed_mds)} unindexed markdown document(s) in parsed_docs/")
        else:
            st.success("All parsed markdown documents are indexed.")

        if st.button("Index Markdown Documents"):
            try:
                with st.spinner("Indexing markdown documents..."):
                    vs, bm25, n = ingest_llamaparse_pipeline()
                    if vs and bm25:
                        st.session_state["vs"] = vs
                        st.session_state["bm25"] = bm25
                        st.success(f"Indexed {n} new chunks from {len(unindexed_mds)} markdown file(s)")
            except Exception as e:
                err_info = log_api_error("Markdown Indexing", e)
                render_error_ui(err_info)

    st.divider()
    show_src = st.toggle("Show source chunks", value=True)
    enable_eval = st.toggle("Run live RAGAS evaluation", value=False)

# Auto-load existing index for active mode
if "vs" not in st.session_state:
    if mode_key == "pypdf" and os.path.exists(CHROMA_DIR) and os.path.exists(BM25_PKL):
        with st.spinner("Loading PyPDF index..."):
            st.session_state["vs"] = load_vectorstore()
            st.session_state["bm25"] = load_bm25()
        st.sidebar.success("PyPDF index loaded")
    elif mode_key == "llamaparse" and os.path.exists(CHROMA_DIR_LLAMAPARSE) and os.path.exists(BM25_PKL_LLAMAPARSE):
        with st.spinner("Loading LlamaParse index..."):
            st.session_state["vs"] = load_llamaparse_vectorstore()
            st.session_state["bm25"] = load_llamaparse_bm25()
        st.sidebar.success("LlamaParse index loaded")

with chat_tab:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("error"):
                render_error_ui(msg["error"])
            else:
                st.markdown(msg["content"])
                if msg.get("ragas_scores"):
                    scores = msg["ragas_scores"]
                    with st.expander("Evaluation Metrics (RAGAS 0.4.3)"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Faithfulness", f"{scores['faithfulness']:.0%}")
                        c2.metric("Answer Relevancy", f"{scores['answer_relevancy']:.0%}")
                        c3.metric("Context Utilization", f"{scores['context_utilization']:.0%}")

    if question := st.chat_input("Ask anything about your research papers..."):
        if "vs" not in st.session_state or st.session_state.get("vs") is None:
            st.error("No index found for the selected mode. Click index in the sidebar first.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            vs = st.session_state["vs"]
            bm25 = st.session_state["bm25"]

            start = time.time()
            error_info = None
            docs = None
            answer = None

            try:
                with st.spinner("Searching papers..."):
                    docs = retrieve(question, vs, bm25)
            except Exception as e:
                error_info = log_api_error("Retrieval (Embedding / Search)", e, query=question)
                render_error_ui(error_info)

            if error_info is None:
                try:
                    with st.spinner("Generating answer..."):
                        answer = generate_answer(question, docs)
                        st.markdown(answer)
                except Exception as e:
                    error_info = log_api_error("Answer Generation", e, query=question)
                    render_error_ui(error_info)

            if error_info is not None:
                st.session_state.messages.append(
                    {"role": "assistant", "content": "", "error": error_info}
                )
                st.stop()

            latency_ms = (time.time() - start) * 1000

            if show_src and docs:
                sources = get_sources(docs)
                with st.expander(f"Sources ({len(sources)} chunks)"):
                    for s in sources:
                        label = s["source"]
                        extras = []
                        if s.get("page"):
                            extras.append(f"page {s['page']}")
                        if s.get("header"):
                            extras.append(f"section: {s['header']}")
                        if extras:
                            label += f" ({', '.join(extras)})"
                        st.markdown(f"**{label}**")
                        st.caption(s["snippet"])


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
                    ragas_err = log_api_error("RAGAS Evaluation", e, query=question)
                    if ragas_err["is_timeout"]:
                        st.warning(f"RAGAS evaluation timed out (logged to `{ragas_err['log_file']}`).")
                    else:
                        logger.warning("RAGAS evaluation skipped: %s", e)

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
        avg_rel = df_history["answer_relevancy"].mean()
        avg_util = df_history["context_utilization"].mean()

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