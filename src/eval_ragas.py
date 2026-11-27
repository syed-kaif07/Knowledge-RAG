# src/eval_ragas.py
# RAGAS 0.4.3 Online Evaluation Module

import math
import sys
import types
import warnings

# --- Ragas 0.4.3 / LangChain Community Compatibility Patch ---
# Ragas 0.4.3 attempts to import langchain_community.chat_models.vertexai,
# which was sunset in modern langchain-community versions.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    vertex_patch = types.ModuleType("vertexai")
    vertex_patch.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = vertex_patch

# Suppress non-critical deprecation warnings during metric evaluations
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    LLMContextPrecisionWithoutReference,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from src.config import NVIDIA_API_KEY, LLM_MODEL, EMBED_MODEL


def get_evaluator_llm():
    """Deterministic LLM judge instance (temperature=0.0) for stable scoring."""
    return LangchainLLMWrapper(
        ChatNVIDIA(
            model=LLM_MODEL,
            api_key=NVIDIA_API_KEY,
            temperature=0.0,
        )
    )


def get_evaluator_embeddings():
    """Embeddings provider for vector-based evaluation metrics like answer_relevancy."""
    return LangchainEmbeddingsWrapper(
        NVIDIAEmbeddings(
            model=EMBED_MODEL,
            api_key=NVIDIA_API_KEY,
        )
    )


def _clean_metric_score(val) -> float:
    """
    Safely converts a Ragas metric score to a float clamped between 0.0 and 1.0.
    Handles NaN, None, inf, and conversion errors by returning 0.0 rather
    than erroneously converting NaN to 1.0 via min/max.
    """
    if val is None:
        return 0.0
    try:
        score = float(val)
        if math.isnan(score) or math.isinf(score):
            return 0.0
        return round(max(0.0, min(1.0, score)), 4)
    except (ValueError, TypeError):
        return 0.0


def evaluate_query_ragas(user_query: str, answer: str, contexts: list[str]) -> dict:
    """
    Evaluates a single RAG query response using Ragas 0.4.3 metrics:
    - Faithfulness (is the answer grounded in the context?)
    - Answer Relevancy (is the answer relevant to the query?)
    - Context Utilization (reference-free context precision score)

    Returns dict with keys: 'faithfulness', 'answer_relevancy', 'context_utilization'.
    """
    if not contexts:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_utilization": 0.0,
        }

    try:
        dataset = Dataset.from_dict({
            "question": [user_query],
            "answer": [answer],
            "contexts": [contexts],
        })

        llm = get_evaluator_llm()
        embeddings = get_evaluator_embeddings()
        context_util_metric = LLMContextPrecisionWithoutReference()

        eval_result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_util_metric],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )

        scores = eval_result.to_pandas().iloc[0]

        return {
            "faithfulness": _clean_metric_score(scores.get("faithfulness")),
            "answer_relevancy": _clean_metric_score(scores.get("answer_relevancy")),
            "context_utilization": _clean_metric_score(scores.get("llm_context_precision_without_reference")),
        }

    except Exception as e:
        print(f"[RAGAS Eval Warning] Evaluation failed: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_utilization": 0.0,
        }
