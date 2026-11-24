
import logging
from collections import defaultdict
from typing import List

from app.vectorstore import vectorstore, COLLECTION_NAME
from app.database import engine
from app.retrievers.keyword_retriever import PostgresFullTextRetriever

logger = logging.getLogger(__name__)

# Hybrid retrieval configuration
VECTOR_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

RRF_CONSTANT = 60


# --------------------------------
# 1. Configure retrievers
# --------------------------------

vector_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 5,
        "lambda_mult": 0.5
    }
)

keyword_retriever = PostgresFullTextRetriever(
    engine=engine,
    collection_name=COLLECTION_NAME,
    k=3
)


# --------------------------------
# 2. Hybrid retrieval
# --------------------------------

def retrieve_documents(question: str):
    try:
        vector_documents = vector_retriever.invoke(question)
        keyword_documents = keyword_retriever.invoke(question)

        logger.info(
            "Hybrid retrieval: vector=%d, keyword=%d",
            len(vector_documents),
            len(keyword_documents)
        )

        return [
            (vector_documents, VECTOR_WEIGHT),
            (keyword_documents, KEYWORD_WEIGHT)
        ]

    except Exception as exc:
        logger.exception("Hybrid retrieval failed")
        raise RuntimeError(
            "Failed to retrieve documents"
        ) from exc


# --------------------------------
# 3. Multi-query retrieval
# --------------------------------

def search_with_query_variations(
    query_variations: List[str]
):
    all_retrieval_results = []
    failed_queries = 0

    for i, query in enumerate(query_variations, 1):
        try:
            logger.info(
                "Retrieving query %d: %s",
                i,
                query
            )

            retrieval_results = retrieve_documents(query)
            all_retrieval_results.extend(retrieval_results)

        except Exception:
            failed_queries += 1
            logger.exception(
                "Retrieval failed for query %d",
                i
            )

    if failed_queries == len(query_variations):
        raise RuntimeError(
            "All document retrieval attempts failed"
        )

    return all_retrieval_results


# --------------------------------
# 4. Weighted global RRF
# --------------------------------

def reciprocal_rank_fusion(
    chunk_lists,
    rrf_constant=RRF_CONSTANT
):
    rrf_scores = defaultdict(float)
    unique_chunks = {}

    for chunks, weight in chunk_lists:

        for position, document in enumerate(chunks, 1):

            content = document.page_content

            if content not in unique_chunks:
                unique_chunks[content] = document

            score = weight / (rrf_constant + position)

            rrf_scores[content] += score

    sorted_chunks = sorted(
        [
            (unique_chunks[content], score)
            for content, score in rrf_scores.items()
        ],
        key=lambda item: item[1],
        reverse=True
    )

    logger.info(
        "Global RRF completed: %d unique documents",
        len(sorted_chunks)
    )

    return sorted_chunks


# --------------------------------
# 5. Complete hybrid search
# --------------------------------

def hybrid_search(query_variations: List[str]):
    retrieval_results = search_with_query_variations(
        query_variations
    )

    if not any(
        documents
        for documents, weight in retrieval_results
    ):
        return []

    return reciprocal_rank_fusion(retrieval_results)
