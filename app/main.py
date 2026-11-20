from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import generate_answer
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

    result = generate_answer(request.question)

    return result
