from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag import (
    generate_answer,
    LLMServiceError,
    RetrievalError,
    AnswerGenerationError,
    RAGError
)
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(
    title="Basic RAG Api",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):

    try:
        result = generate_answer(request.question)

        return result

    except RetrievalError:
        raise HTTPException(
            status_code=503,
            detail="Document retrieval service is unavailable."
        )

    except LLMServiceError:
        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable."
        )

    except AnswerGenerationError:
        raise HTTPException(
            status_code=503,
            detail="Unable to generate an answer right now."
        )

    except RAGError:
        raise HTTPException(
            status_code=500,
            detail="An error occurred in the RAG pipeline."
        )
