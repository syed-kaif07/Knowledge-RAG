"""
dashboard_tab.py

Drop this into your existing Streamlit app as a new tab/section to visualize
retrieval observability. Example:

    import streamlit as st
    from rag_observability.dashboard_tab import render_observability_tab

    tab1, tab2 = st.tabs(["Chat", "Observability"])
    with tab1:
        ... your existing chat UI ...
    with tab2:
        render_observability_tab(logger)  # pass your RAGLogger instance
"""

import streamlit as st
import pandas as pd
import json


def render_observability_tab(logger):
    st.subheader("Retrieval Observability")

    trend = logger.get_faithfulness_trend()
    if trend:
        df = pd.DataFrame(trend, columns=["timestamp", "faithfulness_score"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("timestamp")

        avg_score = df["faithfulness_score"].mean()
        col1, col2 = st.columns(2)
        col1.metric("Avg Faithfulness", f"{avg_score:.0%}")
        col2.metric("Queries Scored", len(df))

        st.line_chart(df["faithfulness_score"])
    else:
        st.info("No scored queries yet — faithfulness scores populate after queries run.")

    st.divider()
    st.subheader("Recent Queries")

    recent = logger.get_recent(limit=20)
    if not recent:
        st.info("No queries logged yet.")
        return

    for row in recent:
        score = row.get("faithfulness_score")
        score_label = f"{score:.0%}" if score is not None else "not scored"
        with st.expander(f"[{score_label}] {row['query'][:80]}"):
            st.write("**Answer:**", row["answer"])
            st.write("**HyDE used:**", bool(row["hyde_used"]))
            if row.get("latency_ms"):
                st.write("**Latency:**", f"{row['latency_ms']:.0f} ms")

            final_chunks = json.loads(row["final_context_chunks"])
            st.write("**Chunks used:**")
            for c in final_chunks:
                st.caption(
                    f"`{c['source']}` p.{c.get('page', '?')} "
                    f"— rerank: {c.get('rerank_score', 'n/a')} "
                    f"— rrf: {c.get('rrf_score', 'n/a')}"
                )

            if row.get("faithfulness_detail"):
                detail = json.loads(row["faithfulness_detail"])
                st.write("**Per-claim support:**")
                for d in detail:
                    label = "supported" if d["supported"] else "not supported"
                    st.caption(f"[{label}] {d['claim']}")