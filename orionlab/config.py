from pydantic import BaseModel
from typing import Optional

class Settings(BaseModel):
    qdrant_url: str = "http://localhost:6333"
    collection: str = "orionlab"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

settings = Settings()
