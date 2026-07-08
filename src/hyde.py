# src/hyde.py
# HyDE generates a fake answer to the query, then uses that
# as the search query instead of the raw user question.
# This works because a fake answer is closer to real document
# chunks than a short question is.

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import NVIDIA_API_KEY, LLM_MODEL

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a research assistant. Write a 3-4 sentence passage "
     "that directly answers the question below, as if it were an "
     "excerpt from an academic paper. Do not mention the question itself."),
    ("human", "{question}"),
])

def build_hyde_chain():
    llm = ChatNVIDIA(
        model=LLM_MODEL,
        api_key=NVIDIA_API_KEY,
        max_tokens=200,
        temperature=0.3,
    )
    return HYDE_PROMPT | llm | StrOutputParser()

def expand_query(question, hyde_chain):
    # if hyde fails for any reason, fall back to original question
    try:
        hypothetical_doc = hyde_chain.invoke({"question": question})
        return hypothetical_doc
    except Exception as e:
        print(f"HyDE failed, using original query. Reason: {e}")
        return question