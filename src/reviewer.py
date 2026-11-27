# src/reviewer.py
# Automated Reviewer Agent for Knowledge RAG
# Inspects telemetry logs, unindexed documents, fix logs, and RAGAS evaluation drift.

import os
import re
from datetime import datetime
from src.ingestion import get_new_files
from src.markdown_ingestion import get_new_llamaparse_files

LOG_PATH = "logs/app.log"
FIX_LOG_PATH = "Daily_Fix_log.md"


def scan_telemetry() -> dict:
    """Scans logs/app.log for errors, overloads, timeouts, and fallbacks."""
    data = {
        "overloads_503": 0,
        "timeouts": 0,
        "groq_fallbacks": 0,
        "bm25_fallbacks": 0,
        "last_error_time": None,
        "last_error_snippet": None,
    }

    if not os.path.exists(LOG_PATH):
        return data

    try:
        with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        content = "".join(lines)
        data["overloads_503"] = len(re.findall(r"503.*overloaded|overloaded.*503", content, re.IGNORECASE))
        data["timeouts"] = len(re.findall(r"read timed out|timeout", content, re.IGNORECASE))
        data["groq_fallbacks"] = len(re.findall(r"fallback to groq", content, re.IGNORECASE))
        data["bm25_fallbacks"] = len(re.findall(r"falling back to bm25", content, re.IGNORECASE))

        # Find the latest ERROR / WARNING line
        for line in reversed(lines):
            if "ERROR" in line or "WARNING" in line:
                match_time = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                if match_time and not data["last_error_time"]:
                    data["last_error_time"] = match_time.group(1)
                    data["last_error_snippet"] = line.strip()[:100]
                    break
    except Exception as e:
        print(f"[Reviewer Warning] Failed to scan telemetry: {e}")

    return data


def scan_corpus_health() -> dict:
    """Checks for unindexed documents across PyPDF and LlamaParse."""
    unindexed_pdfs = []
    unindexed_mds = []
    try:
        unindexed_pdfs = list(get_new_files())
    except Exception:
        pass

    try:
        unindexed_mds = list(get_new_llamaparse_files())
    except Exception:
        pass

    return {
        "unindexed_pdfs": unindexed_pdfs,
        "unindexed_mds": unindexed_mds,
        "total_unindexed": len(unindexed_pdfs) + len(unindexed_mds),
    }


def scan_dev_debt() -> list[str]:
    """Extracts recent notes or open issues from Daily_Fix_log.md."""
    if not os.path.exists(FIX_LOG_PATH):
        return []

    debt_items = []
    try:
        with open(FIX_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Look for drawbacks or open items
        for line in lines[-25:]:
            if any(k in line.lower() for k in ["drawback", "issue", "todo", "next", "latency", "observe"]):
                clean = line.lstrip("-*#0123456789. ")
                if len(clean) > 8 and clean not in debt_items:
                    debt_items.append(clean)
    except Exception:
        pass

    return debt_items[:3]


def generate_daily_briefing(session_ragas_history: list[dict] | None = None) -> str:
    """
    Generates the formatted daily briefing based on live system state.
    """
    telemetry = scan_telemetry()
    corpus = scan_corpus_health()
    debt = scan_dev_debt()

    # 1. Critical Alerts Today
    critical_alerts = []
    if telemetry["overloads_503"] > 0:
        critical_alerts.append(
            f"{telemetry['overloads_503']} NVIDIA 503 errors detected in {LOG_PATH}. "
            f"Hybrid retry/fallback handled these, but remote endpoint instability is elevated."
        )

    if telemetry["timeouts"] > 0:
        critical_alerts.append(
            f"{telemetry['timeouts']} request timeout events recorded in {LOG_PATH}."
        )

    if corpus["unindexed_mds"]:
        critical_alerts.append(
            f"{len(corpus['unindexed_mds'])} unindexed markdown document(s) detected in parsed_docs/llamaparse/."
        )

    if corpus["unindexed_pdfs"]:
        critical_alerts.append(
            f"{len(corpus['unindexed_pdfs'])} unindexed PDF file(s) detected in docs/."
        )

    if not critical_alerts:
        critical_alerts.append("No critical system faults or API outages detected in the last session.")

    # 2. Priority Fixes
    priority_fixes = []
    if corpus["unindexed_mds"]:
        priority_fixes.append("Run 'python index_llamaparse.py' to sync markdown documents into ChromaDB.")

    if corpus["unindexed_pdfs"]:
        priority_fixes.append("Click 'Index New PDFs' in the sidebar or run ingestion to index pending docs.")

    # Check for evaluation score drops
    low_score_queries = []
    if session_ragas_history:
        for item in session_ragas_history:
            if item.get("faithfulness", 1.0) < 0.70 or item.get("context_utilization", 1.0) < 0.70:
                low_score_queries.append(item.get("query", "Unknown query")[:50])

    if low_score_queries:
        priority_fixes.append(
            f"Review RAGAS scores for queries with low grounding: '{low_score_queries[0]}...'."
        )
    else:
        priority_fixes.append("Run benchmark evaluation ('python eval_workflow/run_baseline_eval.py') to track score drift.")

    # 3. Low-Priority Maintenance
    maintenance = []
    if debt:
        maintenance.append(f"Follow up on recent fix log note: {debt[0]}.")
    else:
        maintenance.append("Review logs/app.log and clear historical session traces if storage grows.")

    if telemetry["bm25_fallbacks"] > 0:
        maintenance.append(
            f"Tune dense embedding retry backoff; {telemetry['bm25_fallbacks']} queries relied entirely on BM25 fallback."
        )
    else:
        maintenance.append("Validate Cohere reranker candidate pool size (top 15) to maintain sub-2s latency.")

    # Construct briefing
    lines = [
        "=== DAILY RAG SYSTEM REVIEW BRIEFING ===",
        "",
        "1. Critical Alerts Today:",
    ]
    for alert in critical_alerts:
        lines.append(f"   - {alert}")

    lines.append("")
    lines.append("2. Priority Fixes:")
    for fix in priority_fixes:
        lines.append(f"   - {fix}")

    lines.append("")
    lines.append("3. Low-Priority Maintenance:")
    for item in maintenance:
        lines.append(f"   - {item}")

    return "\n".join(lines)
