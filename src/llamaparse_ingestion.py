import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()


def parse_pdf(pdf_path: str) -> str:
    """
    Parses a PDF document into markdown using LlamaParse.
    """
    api_key = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMA_PARSE_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not found. Please set LLAMA_CLOUD_API_KEY or LLAMA_PARSE_API_KEY in your .env file."
        )

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        verbose=False,
    )

    documents = parser.load_data(pdf_path)
    return "\n\n".join([doc.text for doc in documents])