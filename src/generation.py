# src/generation.py
# builds the final RAG chain that takes retrieved docs,
# formats them into a prompt, and generates a grounded answer

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from src.config import NVIDIA_API_KEY, LLM_MODEL

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research assistant. Answer the question using only "
     "the context provided below. If the answer is not in the context, "
     "say 'I could not find this in the provided papers.' "
     "At the end of your answer, cite the source like this: [Source: filename]\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

def format_docs(docs: list[Document]) -> str:
    # attach source info to each chunk so the model can cite it
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "")
        label  = source + (f" page {page}" if page != "" else "")
        parts.append(f"[{label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

def get_sources(docs: list[Document]) -> list[dict]:
    # deduplicate sources for display in the UI
    seen    = set()
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "")
        key    = f"{source}-{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source":  source,
                "page":    page,
                "snippet": doc.page_content[:200] + "...",
            })
    return sources

def build_llm():
    return ChatNVIDIA(
        model=LLM_MODEL,
        api_key=NVIDIA_API_KEY,
        max_tokens=1024,
        temperature=0.1,
    )

# module-level instance — reused by app.py as the faithfulness judge
# so we don't spin up a second NIM client just for scoring
llm = build_llm()

def build_rag_chain():
    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain

def generate_answer(question, docs):
    context = format_docs(docs)
    chain   = build_rag_chain()
    return chain.invoke({
        "question": question,
        "context":  context,
    })