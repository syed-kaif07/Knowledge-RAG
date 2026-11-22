# src/generation.py
# builds the final RAG chain that takes retrieved docs,
# formats them into a prompt, and generates a grounded answer

import logging
import random
import time
import warnings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from src.config import (
    NVIDIA_API_KEY, LLM_MODEL,
    GROQ_API_KEY, GROQ_LLM_MODEL
)

# Suppress benign model type classification warning from langchain_nvidia_ai_endpoints
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")

logger = logging.getLogger(__name__)

# Maximum total context character budget (~900-1000 tokens)
MAX_CONTEXT_CHARS = 4000

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research assistant. Answer the question using only "
     "the context provided below. If the answer is not in the context, "
     "say 'I could not find this in the provided papers.' "
     "At the end of your answer, cite the source like this: [Source: filename]\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs: list[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Attaches source info to each chunk and applies context window budgeting.
    Prioritizes top reranked chunks and caps total length to prevent prompt bloat.
    """
    parts = []
    accumulated_chars = 0

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        header = doc.metadata.get("Header 2") or doc.metadata.get("Header 1") or ""

        details = []
        if page != "":
            details.append(f"page {page}")
        if header != "":
            details.append(f"section: {header}")

        detail_str = f" ({', '.join(details)})" if details else ""
        label = f"{source}{detail_str}"
        chunk_entry = f"[{label}]\n{doc.page_content.strip()}"
        chunk_len = len(chunk_entry)

        # Budget enforcement: stop adding if budget exceeded
        if accumulated_chars + chunk_len > max_chars and parts:
            # If remaining allowance is significant (>250 chars), add a trimmed snippet
            remaining = max_chars - accumulated_chars
            if remaining > 250:
                trimmed_content = doc.page_content.strip()[: remaining - len(f"[{label}]\n...") ] + "..."
                parts.append(f"[{label}]\n{trimmed_content}")
            break

        parts.append(chunk_entry)
        accumulated_chars += chunk_len

    return "\n\n---\n\n".join(parts)


def get_sources(docs: list[Document]) -> list[dict]:
    # deduplicate sources for display in the UI
    seen = set()
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        header = doc.metadata.get("Header 2") or doc.metadata.get("Header 1") or ""
        key = f"{source}-{page}-{header}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": source,
                "page": page,
                "header": header,
                "snippet": doc.page_content[:200] + "...",
            })
    return sources


def build_llm():
    return ChatNVIDIA(
        model=LLM_MODEL,
        api_key=NVIDIA_API_KEY,
        max_tokens=4096,
        temperature=0.1,
    )


_llm_instance = None
_judge_llm_instance = None
_rag_chain_instance = None


def get_llm():
    """Returns cached ChatNVIDIA client singleton, initialized lazily."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = build_llm()
    return _llm_instance


def get_judge_llm():
    """Returns deterministic LLM instance for judging."""
    global _judge_llm_instance
    if _judge_llm_instance is None:
        _judge_llm_instance = ChatNVIDIA(
            model=LLM_MODEL,
            api_key=NVIDIA_API_KEY,
            max_tokens=100,
            temperature=0.0,
        )
    return _judge_llm_instance


def _parse_llm_response(message) -> str:
    """Extracts text content from message, with fallback for reasoning models."""
    if hasattr(message, "content") and message.content:
        return str(message.content)
    # Fallback if reasoning model output only populated reasoning_content
    if hasattr(message, "additional_kwargs") and "reasoning_content" in message.additional_kwargs:
        return str(message.additional_kwargs["reasoning_content"])
    return str(message)


def get_rag_chain():
    """Returns cached primary NVIDIA RAG chain."""
    global _rag_chain_instance
    if _rag_chain_instance is None:
        _rag_chain_instance = RAG_PROMPT | get_llm() | _parse_llm_response
    return _rag_chain_instance

# Groq fallback chain singleton
_groq_chain_instance = None


def get_groq_chain():
    """Builds and caches the secondary fallback Groq RAG chain."""
    global _groq_chain_instance
    if _groq_chain_instance is None and GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            groq_llm = ChatGroq(
                model=GROQ_LLM_MODEL,
                api_key=GROQ_API_KEY,
                max_tokens=4096,
                temperature=0.1,
            )
            _groq_chain_instance = RAG_PROMPT | groq_llm | _parse_llm_response
        except Exception as e:
            logger.warning("Failed to initialize Groq fallback client: %s", e)
    return _groq_chain_instance


def _invoke_groq_fallback(question: str, context: str) -> str | None:
    """Invokes Groq when NVIDIA fails on attempt 1."""
    groq_chain = get_groq_chain()
    if groq_chain is None:
        logger.info("Groq fallback skipped (GROQ_API_KEY not configured or client failed).")
        return None
    try:
        logger.info("Executing immediate fallback to Groq (%s)...", GROQ_LLM_MODEL)
        ans = groq_chain.invoke({
            "question": question,
            "context": context,
        })
        logger.info("Groq fallback generation succeeded.")
        return ans
    except Exception as ge:
        logger.warning("Groq fallback generation failed (%s). Continuing NVIDIA retries.", ge)
        return None


def generate_answer(question: str, docs: list[Document]) -> str:
    """
    Generates a grounded RAG answer.
    Resilience flow:
    1. Primary attempt with NVIDIA NIM.
    2. If NVIDIA NIM fails on attempt 1, immediately attempt failover to Groq.
    3. If Groq is unavailable or fails, continue NVIDIA retries with exponential backoff.
    """
    if not docs:
        return "I could not find this in the provided papers."

    context = format_docs(docs)

    for attempt in range(4):
        try:
            primary_chain = get_rag_chain()
            return primary_chain.invoke({
                "question": question,
                "context": context,
            })
        except Exception as e:
            err_str = str(e).lower()
            # Fast fail on non-retryable client errors
            if any(fatal in err_str for fatal in ["401", "403", "unauthorized", "invalid_api_key", "bad request"]):
                logger.error("Non-retryable generation error: %s", e)
                raise e

            # Immediate fallback to Groq on attempt 1 failure
            if attempt == 0:
                logger.warning("NVIDIA generation attempt 1 failed (%s). Attempting immediate fallback to Groq...", e)
                groq_answer = _invoke_groq_fallback(question, context)
                if groq_answer:
                    return groq_answer

            if attempt == 3:
                logger.error("Max generation retry attempts reached: %s", e)
                raise e

            # Exponential backoff with jitter (longer delay for transient server 503 / 429 overloads)
            is_overloaded = any(k in err_str for k in ["503", "overloaded", "429", "rate limit"])
            base = 1.5 if is_overloaded else 0.8
            backoff = min(base * (2 ** attempt), 6.0) + random.uniform(0.1, 0.4)
            logger.warning("NVIDIA generation attempt %d failed (%s). Retrying in %.2fs...", attempt + 1, e, backoff)
            time.sleep(backoff)