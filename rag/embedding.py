"""Embedding via sentence-transformers; pad to Qdrant vector size (1536)."""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from rag.config import EMBEDDING_MODEL, QDRANT_VECTOR_SIZE

logger = logging.getLogger(__name__)

# Lazy singleton to avoid loading model multiple times
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-transformers model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _pad_vector(vector: list[float], target_size: int) -> list[float]:
    """Pad vector with zeros to target_size (for Qdrant 1536-dim)."""
    if len(vector) >= target_size:
        return vector[:target_size]
    return vector + [0.0] * (target_size - len(vector))


def embed_texts(texts: list[str]) -> List[List[float]]:
    """Embed texts and pad each vector to QDRANT_VECTOR_SIZE (1536)."""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    raw = embeddings.tolist()
    return [_pad_vector(v, QDRANT_VECTOR_SIZE) for v in raw]


def embed_query(query: str) -> list[float]:
    """Embed a single query and pad to QDRANT_VECTOR_SIZE."""
    return embed_texts([query])[0]


def embedding_dimension() -> int:
    """Dimension of vectors after padding (for Qdrant collection)."""
    return QDRANT_VECTOR_SIZE
