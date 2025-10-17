from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl
from typing import Optional

class Settings(BaseSettings):
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_TOP_K: int = 3
    MAX_TOOL_TIMEOUT_SEC: int = 8
    MODEL_BACKEND: str = "llama"  or "hf"
    HF_API_URL: Optional[AnyHttpUrl] = None
    HF_API_TOKEN: Optional[str] = None
    PROJECT_NAME: str = "Minimal RAG Agent"
    CONTAINER_NAME: Optional[str] = None
    VERSION: str = "0.1.0"

    class Config:
        env_file = ".env"

settings = Settings()
