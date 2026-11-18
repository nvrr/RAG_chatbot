import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/BGE-M3"
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)


# LLM_MODEL = os.getenv(
#     "LLM_MODEL",
#     "qwen3:8b"
# )

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen3:0.6b"
)
