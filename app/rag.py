import time
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import LLM_MODEL, OLLAMA_BASE_URL
from app.vectorstore import vectorstore, COLLECTION_NAME
from app.database import engine
from app.retrievers.keyword_retriever import PostgresFullTextRetriever
from app.rerankers.cohere_reranker import rerank_documents
from app.retrievers.hybrid_retriever import hybrid_search

from typing import Literal, Optional
from pydantic import BaseModel, Field
from typing import List

# --------------------------------
# 1. logging and custom exceptions
# --------------------------------
logger = logging.getLogger(__name__)

class RAGError(Exception):
    """Base exception for RAG pipeline errors."""


class LLMServiceError(RAGError):
    """Raised when an LLM operation fails."""


class RetrievalError(RAGError):
    """Raised when document retrieval fails."""


class AnswerGenerationError(RAGError):
    """Raised when final answer generation fails."""


class RewriteDecision(BaseModel):
    action: Literal["ready", "clarify"]

    rewritten_question: Optional[str] = Field(
        default=None,
        description="A complete question ready for retrieval"
    )

    clarification_question: Optional[str] = Field(
        default=None,
        description="A question asking the user for missing information"
    )

# --------------------------------
# 1. Configure LLM
# --------------------------------

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0
)

# --------------------------------
#  Store conversation histories
# --------------------------------

# Store our conversation as messages
chat_history = []

rewrite_prompt = ChatPromptTemplate.from_template(""" 
    Based on the user’s previous history query about {chat_history} rewrite their new query: {question} into a standalone search query.

""")

# --------------------------------
#  Answer generation prompt
# --------------------------------

answer_prompt = ChatPromptTemplate.from_template("""
Based on the following context, please answer this question: {question}
    Context:{context}
Please provide a clear, helpful answer using only the information from these documents. 
If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents

""")



# ──────────────────────────────────────────────────────────────────
# Step : Generate Multiple Query Variations
# ──────────────────────────────────────────────────────────────────
# Pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str] = Field(min_length=1, max_length=3)

def generate_multi_query_variations(original_query: str) -> List[str]:
    try:
        llm_with_structured_output = llm.with_structured_output(
            QueryVariations
        )

        prompt = f"""
        Generate 3 different variations of this query using synonyms and related terms,
        that would help retrieve relevant documents.

        Original query: {original_query}

        Return alternative queries that rephrase or
        approach the same question from different angles.
        """

        response = llm_with_structured_output.invoke(prompt)

        query_variations = [
            query.strip()
            for query in response.queries
            if query and query.strip()
        ]

        # Remove duplicate queries
        query_variations = list(dict.fromkeys(query_variations))
        print(f"original Question: {original_query}, Query variations: {query_variations}")

        return query_variations

    except Exception as exc:

        logger.exception("Multi-query generation failed")

        # Let the caller decide whether to fall back
        raise LLMServiceError(
            "Failed to generate query variations"
        ) from exc

# ──────────────────────────────────────────────────────────────────
    # Step : Rewrite the user's question
# ──────────────────────────────────────────────────────────────────    
def rewrite_question(question: str, recent_history:List ) -> str:
 # Convert history into readable text
    history_text = "\n".join(
    f"{message['role']}: {message['content']}"
    for message in recent_history )

    try:
        rewrite_chain = rewrite_prompt | llm

        response = rewrite_chain.invoke({
            "chat_history": history_text,
            "question": question
        })

        rewritten_question = response.content.strip()
        print(f"orgQUESTION: {question}, MODIFIED QUESTION IS: {rewritten_question}")

        if not rewritten_question:
            logger.warning(
                "Question rewriting returned an empty response. "
                "Using original question."
            )
            return question

        return rewritten_question

    except Exception:
        logger.exception(
            "Question rewriting failed. Using original question."
        )

        return question


# --------------------------------
# 6. Generate answer with chat history
# --------------------------------

def generate_answer(
    question: str,
    # chat_history: list = None,
):

    start = time.perf_counter()

    def log(message):
        elapsed = time.perf_counter() - start
        print(f"[{elapsed:.2f}s] {message}", flush=True)

    log("Starting generate_answer")

  # --------------------------------
    # 1. Prepare conversation history
  # -------------------------------- 
      
    recent_history = chat_history[-10:]

     # --------------------------------
    # 2. Rewrite question
    # --------------------------------
   

    print("Rewrite QUESTION:")
    print("======================================")

    if chat_history:

        log("Starting question rewriting")

        search_question = rewrite_question(
            question,
            recent_history
        )

        log("Question rewriting completed")

    else:
        search_question = question

    print(f"Original question: {question} \n SEARCH Question:: {search_question}")
# ──────────────────────────────────────────────────────────────────
# 3. Generate Multiple Query Variations
# ──────────────────────────────────────────────────────────────────

    log("Starting multi-query generation")

    try:
        multi_query_variations = generate_multi_query_variations(
            search_question
        )

    except LLMServiceError:

        logger.warning(
            "Falling back to original question"
        )

        multi_query_variations = []


#-------queries_to_search
    queries_to_search = list(dict.fromkeys(
        [search_question] + multi_query_variations
    ))

    log(f"Generated {len(queries_to_search)} search queries")

# ──────────────────────────────────────────────────────────────────
    # 4. Hybrid retrieval + global RRF
# ──────────────────────────────────────────────────────────────────    
    reciprocal_rank_fusion_results =  hybrid_search(queries_to_search)

    if not reciprocal_rank_fusion_results:

        answer = "I could not find relevant information."

        chat_history.extend([
            {
            "role": "user",
            "content": question,
            "standalone_question": search_question
            },
             {
            "role": "assistant",
            "content": answer
             }
        ])

        return {
        "answer": answer,
        "sources": []
        }
# --------------------------------
#  7.Apply Cohere Reranking
# --------------------------------
   
    try:
    # RRF returns (Document, score) tuples.
    # Cohere expects a list of Document objects.
        rrf_documents = [
        document
        for document, score in reciprocal_rank_fusion_results
        ]
        print(f"Reranker starting with {len(rrf_documents)} RRF documents")

        reranked_docs = rerank_documents(
        documents=rrf_documents,
        query=search_question
        )
        
    except Exception as exc:
        logger.exception("Cohere reranking failed. Falling back to RRF results.")
        reranked_docs = rrf_documents[:8]

        raise RetrievalError(
        "Failed to rerank retrieved documents"
        ) from exc
    print(f"reranker completed")

    top_chunks = reranked_docs


    if not top_chunks:

        answer = "I could not find relevant information."

        chat_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])

        return {
            "answer": answer,
            "sources": []
        }

    
# ──────────────────────────────────────────────────────────────────
    #  8. Prepare context
# ──────────────────────────────────────────────────────────────────    
    context = "\n\n".join(
        document.page_content
        for document in top_chunks
    )
    log("Context prepared")

    print(f"documents sending to llm are: {len(top_chunks)}")
# ──────────────────────────────────────────────────────────────────
    # 9. Generate answer
# ──────────────────────────────────────────────────────────────────    
    log("Starting answer generation")

    try:

        answer_chain = answer_prompt | llm

        response = answer_chain.invoke({
            "context": context,
            "question": search_question
        })

        answer = response.content

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("LLM returned an empty answer")

        log("Answer generation completed")

    except Exception as exc:

        logger.exception("Answer generation failed")

        raise AnswerGenerationError(
            "Failed to generate the final answer"
        ) from exc

    
# ──────────────────────────────────────────────────────────────────
    # 10. Save conversation history
# ──────────────────────────────────────────────────────────────────    
    chat_history.extend([
        {"role": "user", "content": question,"standalone_question": search_question},
        {"role": "assistant", "content": answer}
    ])

   
#---------------------------------------------------------------------
    # log(f"Conversation history saved: {chat_history}")
    # log(f"Conversation history updated. Total messages: {len(chat_history)}")

  # --------------------------------
    # 11. Return response
  # --------------------------------
    # print(f"answer is: {answer} , \n sources: {[ document.metadata 
    #             for document in top_chunks]}")

    return {
        "answer": answer,
        "sources": [ 
            document.metadata 
            for document in top_chunks
        ],
        
            }






#ollama pull qwen3:8b
#ollama list
#ollama run qwen3:8b
#/bye