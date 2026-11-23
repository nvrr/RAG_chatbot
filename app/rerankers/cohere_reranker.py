
import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_cohere import CohereRerank
from langchain_core.documents import Document


logger = logging.getLogger(__name__)

load_dotenv()


@lru_cache(maxsize=1)
def get_reranker() -> CohereRerank:

    api_key = os.getenv("CO_API_KEY")

    if not api_key:
        raise ValueError("CO_API_KEY is not configured")

    return CohereRerank(
        cohere_api_key=api_key,
        model="rerank-english-v3.0",
        top_n=10
    )


def rerank_documents(
    documents: list[Document],
    query: str
) -> list[Document]:

    if not documents:
        return []

    try:
        reranker = get_reranker()

        reranked_docs = reranker.compress_documents(
            documents=documents,
            query=query
        )

        logger.info(
            "Cohere reranked %d documents and returned %d",
            len(documents),
            len(reranked_docs)
        )

        return reranked_docs

    except Exception:
        logger.exception("Cohere reranking failed")
        raise
