import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import LLM_MODEL, OLLAMA_BASE_URL
from app.vectorstore import vectorstore


# --------------------------------
# 1. Configure LLM
# --------------------------------

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0
)


# --------------------------------
# 2. Store conversation histories
# --------------------------------

# Store our conversation as messages
chat_history = []


# --------------------------------
# 3. Question rewriting prompt
# --------------------------------

rewrite_prompt = ChatPromptTemplate.from_template("""
You are a question rewriting assistant.

Your task is to rewrite the user's latest question
as a standalone question using the conversation history.

Rules:
- Understand references from previous messages.
- Replace unclear words like "it", "that", "those",
  and "the first one" with their actual meaning.
- Preserve the user's original intention.
- If the question is already standalone, return it unchanged.
- Return only the rewritten question.

Conversation history:
{chat_history}

Latest question:
{question}

Standalone question:
""")


# --------------------------------
# 4. Answer generation prompt
# --------------------------------

answer_prompt = ChatPromptTemplate.from_template("""
You are a helpful university information assistant.

Answer the user's question using only the provided context.

If the answer is not available in the context,
say that the information is not available.

Do not invent information.

Context:
{context}

Question:
{question}

Answer:
""")


# --------------------------------
# 5. Retrieve relevant documents
# --------------------------------


# def retrieve_documents(question: str, k: int = 5):
    
#     documents = vectorstore.similarity_search(
#         question,
#         k=k
#     )

#     return documents


# def retrieve_documents(question: str, k: int = 5):

#     retriever = vectorstore.as_retriever(
#         search_type="similarity",
#         search_kwargs={
#             "k": k
#         }
#     )

#     documents = retriever.invoke(question)

#     return documents


# retriever = vectorstore.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": k,
#         "score_threshold": 0.3
#     }
# )

# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": k,
#         "fetch_k": 10,
#         "lambda_mult": 0.5
#     }
# )

def retrieve_documents(question: str, k: int = 5):
    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": k,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
    
    documents = retriever.invoke(
        question
    )

    return documents


# --------------------------------
# 6. Generate answer with chat history
# --------------------------------

def generate_answer(
    question: str,
    k: int = 5
):

    start = time.perf_counter()

    def log(message):
        elapsed = time.perf_counter() - start
        print(f"[{elapsed:.2f}s] {message}", flush=True)

    log("Starting generate_answer")

#     # Get history for this conversation
#     chat_history = conversation_histories.get(
#         conversation_id,
#         []
#     )

    # Convert history into readable text
    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in chat_history
    )

    # Step 1: Rewrite the user's question
    if chat_history:
        log("Starting question rewriting")
        rewrite_chain = rewrite_prompt | llm

        rewritten_response = rewrite_chain.invoke({
            "chat_history": history_text,
            "question": question
        })

        search_question = rewritten_response.content.strip()
        log("Question rewriting completed")

    else:
        search_question = question

    print(f"Original question: {question}")
    print(f"Search question: {search_question}")

    # Step 2: Retrieve relevant documents
    log("Starting PostgreSQL similarity search")
    documents = retrieve_documents(
        search_question,
        k=k
    )

    log(f"Document retrieval completed: {len(documents)} documents")

    # Step 3: Handle no relevant documents
    if not documents:

        answer = "I could not find relevant information."

        chat_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
        
        # conversation_histories[conversation_id] = chat_history[-10:]

        return {
            "answer": answer,
            "sources": [],
        #     "conversation_id": conversation_id
        }

    # Step 4: Combine document contents
    context = "\n\n".join(
        document.page_content
        for document in documents
    )
    log("Context prepared")

    # Step 5: Generate answer
    log("Starting Ollama answer generation")
    answer_chain = answer_prompt | llm

    response = answer_chain.invoke({
        "context": context,
        "question": search_question
    })

    log("Ollama answer generation completed")

    answer = response.content

    # Step 6: Save conversation history
    chat_history.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ])
    log(f"Conversation history saved: {chat_history}",)

    # Keep the latest 10 messages
#     conversation_histories[conversation_id] = chat_history[-10:]

    # Step 7: Return response
    return {
        "answer": answer,
        "sources": [
            document.metadata for document in documents
        ],
        # "conversation_id": conversation_id
    }






#ollama pull qwen3:8b
#ollama list
#ollama run qwen3:8b
#/bye