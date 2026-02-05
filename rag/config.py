"""Configuration from environment and defaults."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Paths (needed before load_dotenv to resolve .env location)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root so MISTRAL_API_KEY, QDRANT_*, etc. are set
load_dotenv(PROJECT_ROOT / ".env")

EMAILS_DIR = Path(os.environ.get("EMAILS_DIR", str(PROJECT_ROOT / "emails")))

# Qdrant
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "email_chunks")
QDRANT_VECTOR_SIZE = int(os.environ.get("QDRANT_VECTOR_SIZE", "1536"))
QDRANT_TIMEOUT = int(os.environ.get("QDRANT_TIMEOUT", "120"))  # seconds for large upserts
QDRANT_UPSERT_BATCH_SIZE = int(os.environ.get("QDRANT_UPSERT_BATCH_SIZE", "50"))

# Embedding
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-mpnet-base-v2")

# Retrieval
TOP_K = int(os.environ.get("TOP_K", "5"))

# Mistral
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
