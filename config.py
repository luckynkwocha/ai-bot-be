
import os


config = {
    "qdrant_url": os.getenv("QDRANT_URL", "test"),
    "qdrant_api_key": os.getenv("QDRANT_API_KEY", ""),
    "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", "")
}