# src/generation.py
# builds the final RAG chain that takes retrieved docs,
# formats them into a prompt, and generates a grounded answer

import logging
import random
import time
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from src.config import NVIDIA_API_KEY, LLM_MODEL

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


# module-level instance — used for actual answer generation
llm = build_llm()

# separate deterministic instance for faithfulness judging.
# temperature=0.1 on the generation llm was causing the judge to flip
# verdicts on borderline claims between identical runs, which made the
# faithfulness score swing wildly (e.g. 30% vs 57% on the same question).
# temperature=0.0 makes the judge's yes/no calls consistent run to run.
judge_llm = ChatNVIDIA(
    model=LLM_MODEL,
    api_key=NVIDIA_API_KEY,
    max_tokens=100,
    temperature=0.0,
)


def _parse_llm_response(message) -> str:
    """Extracts text content from ChatNVIDIA message, with fallback for reasoning models."""
    if hasattr(message, "content") and message.content:
        return str(message.content)
    # Fallback if reasoning model output only populated reasoning_content
    if hasattr(message, "additional_kwargs") and "reasoning_content" in message.additional_kwargs:
        return str(message.additional_kwargs["reasoning_content"])
    return str(message)


def build_rag_chain():
    return RAG_PROMPT | llm | _parse_llm_response


# Pre-compiled module-level chain — eliminates recompilation per query
rag_chain = build_rag_chain()



def generate_answer(question: str, docs: list[Document]) -> str:
    """
    Generates a grounded RAG answer with exponential backoff retry and fast fatal-error failure.
    """
    context = format_docs(docs)

    for attempt in range(3):
        try:
            return rag_chain.invoke({
                "question": question,
                "context": context,
            })
        except Exception as e:
            err_str = str(e).lower()
            # Fast fail on non-retryable client errors
            if any(fatal in err_str for fatal in ["401", "403", "unauthorized", "invalid_api_key", "bad request"]):
                logger.error("Non-retryable generation error: %s", e)
                raise e

            if attempt == 2:
                logger.error("Max generation retry attempts reached: %s", e)
                raise e

            # Exponential backoff with jitter
            backoff = min(0.5 * (2 ** attempt), 2.0) + random.uniform(0.05, 0.2)
            logger.warning("Generation attempt %d failed (%s). Retrying in %.2fs...", attempt + 1, e, backoff)
            time.sleep(backoff)