# eval_workflow/run_baseline_eval.py
# Baseline Evaluation Workflow Script for RAG Pipeline (Phase 5)

import os
import json
import time
import pandas as pd
from src.ingestion import load_vectorstore, load_bm25
from src.retrieval import retrieve
from src.generation import generate_answer
from src.hyde import build_hyde_chain, expand_query
from src.eval_ragas import evaluate_query_ragas
from src.config import CHROMA_DIR

# Curated 20 benchmark evaluation questions covering research papers
BENCHMARK_QUESTIONS = [
    "What are the core architectural differences between dense vector retrieval and sparse keyword search?",
    "How does Reciprocal Rank Fusion (RRF) combine ranked results from multiple retrieval strategies?",
    "What is Hypothetical Document Embeddings (HyDE) and how does it improve retrieval quality?",
    "What role does document reranking with Cohere play in a hybrid RAG pipeline?",
    "How does chunk size and chunk overlap affect vector search precision and recall?",
    "What is the difference between extractive RAG and generative RAG?",
    "How does self-attention mechanism work in transformer-based language models?",
    "What strategies are used to prevent hallucinations in RAG systems?",
    "How is faithfulness defined and calculated in RAG evaluation frameworks?",
    "What are the advantages of multi-vector retrieval strategies over single-vector retrieval?",
    "How does context precision measure the quality of retrieved document chunks?",
    "What is the impact of table structure degradation during PDF parsing on RAG accuracy?",
    "How do LLM agent architectures use tools to perform multi-step reasoning?",
    "What are the trade-offs between dense semantic retrieval and BM25 lexical matching?",
    "How does temperature setting in LLMs affect deterministic fact-checking evaluation?",
    "What is the role of metadata filtering in vector database indexing?",
    "How does query rewriting improve document retrieval for short or vague questions?",
    "What are the main causes of context loss during chunking in long-document RAG?",
    "How does Ragas measure answer relevancy using semantic embeddings?",
    "What is the benefit of combining dense embeddings with sparse BM25 scores in hybrid search?"
]


def run_baseline_evaluation():
    print("=" * 65)
    print("      STARTING RAGAS 0.4.3 BASELINE EVALUATION WORKFLOW      ")
    print("=" * 65)

    if not os.path.exists(CHROMA_DIR) or not os.path.exists("bm25.pkl"):
        print("[Error] Vector store or BM25 index not found. Please index documents first.")
        return

    print("\n[1/3] Loading VectorStore and BM25 Index...")
    vectorstore = load_vectorstore()
    bm25 = load_bm25()
    hyde_chain = build_hyde_chain()
    print("  Index loaded successfully.")

    results = []
    print(f"\n[2/3] Evaluating {len(BENCHMARK_QUESTIONS)} benchmark questions...\n")

    for idx, question in enumerate(BENCHMARK_QUESTIONS, start=1):
        print(f"--- Question [{idx}/{len(BENCHMARK_QUESTIONS)}]: '{question[:60]}...' ---")
        start_time = time.time()

        # Step 1: HyDE query expansion
        search_query = expand_query(question, hyde_chain)

        # Step 2: Hybrid retrieval + RRF + Cohere rerank
        retrieved_docs = retrieve(search_query, vectorstore, bm25)
        contexts = [doc.page_content for doc in retrieved_docs]

        # Step 3: NVIDIA Answer generation
        answer = generate_answer(question, retrieved_docs)

        latency_sec = time.time() - start_time

        # Step 4: RAGAS 0.4.3 Evaluation
        ragas_scores = evaluate_query_ragas(question, answer, contexts)

        record = {
            "id": idx,
            "question": question,
            "answer": answer,
            "num_contexts": len(contexts),
            "latency_sec": round(latency_sec, 2),
            "faithfulness": ragas_scores.get("faithfulness", 0.0),
            "answer_relevancy": ragas_scores.get("answer_relevancy", 0.0),
            "context_utilization": ragas_scores.get("context_utilization", 0.0),
        }
        results.append(record)

        print(
            f"  Faithfulness: {record['faithfulness']:.0%} | "
            f"Relevancy: {record['answer_relevancy']:.0%} | "
            f"Utilization: {record['context_utilization']:.0%} | "
            f"Latency: {record['latency_sec']}s\n"
        )

    # Step 5: Save & Aggregate
    df = pd.DataFrame(results)
    avg_faithfulness = df["faithfulness"].mean()
    avg_relevancy = df["answer_relevancy"].mean()
    avg_utilization = df["context_utilization"].mean()
    avg_latency = df["latency_sec"].mean()

    output_dir = os.path.dirname(__file__)
    json_path = os.path.join(output_dir, "baseline_results.json")

    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": "PyPDFLoader + Chroma + BM25 + HyDE + Cohere Rerank + NVIDIA LLM",
        "total_questions": len(BENCHMARK_QUESTIONS),
        "averages": {
            "faithfulness": round(avg_faithfulness, 4),
            "answer_relevancy": round(avg_relevancy, 4),
            "context_utilization": round(avg_utilization, 4),
            "avg_latency_sec": round(avg_latency, 2),
        },
        "details": results,
    }

    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 65)
    print("             PYPDF BASELINE EVALUATION SUMMARY             ")
    print("=" * 65)
    print(f" Total Evaluated Questions: {len(BENCHMARK_QUESTIONS)}")
    print(f" Baseline Faithfulness   : {avg_faithfulness:.2%}")
    print(f" Baseline Answer Relevancy : {avg_relevancy:.2%}")
    print(f" Baseline Context Util    : {avg_utilization:.2%}")
    print(f" Average Latency          : {avg_latency:.2f} seconds")
    print("=" * 65)
    print(f"\nSaved detailed JSON results to: {json_path}")


if __name__ == "__main__":
    run_baseline_evaluation()
