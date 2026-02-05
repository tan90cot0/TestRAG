"""Qdrant vector store: add chunks, search with optional payload filters."""

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag.config import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_TIMEOUT,
    QDRANT_UPSERT_BATCH_SIZE,
    QDRANT_URL,
    QDRANT_VECTOR_SIZE,
)
from rag.embedding import embed_texts, embedding_dimension
from rag.models import Chunk

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """Return Qdrant client from env (url + optional api_key + timeout)."""
    kwargs: dict[str, Any] = {"url": QDRANT_URL, "timeout": QDRANT_TIMEOUT}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    return QdrantClient(**kwargs)


def build_store_from_chunks(
    chunks: list[Chunk],
    collection_name: str | None = None,
    *,
    _persist: bool = True,
) -> None:
    """
    Create or recreate collection, embed chunks (padded to 1536), upsert to Qdrant.
    Uses sentence-transformers for embedding; vectors are padded to QDRANT_VECTOR_SIZE.
    """
    client = get_qdrant_client()
    name = collection_name or QDRANT_COLLECTION_NAME
    size = embedding_dimension()

    # Delete existing collection for idempotent re-index
    try:
        client.delete_collection(name)
        logger.info("Deleted existing collection %s", name)
    except Exception:  # noqa: S110
        pass

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
    )

    # Payload indexes required for filtering by subject, from, to, source_file
    for field in ("subject", "from", "to", "source_file"):
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    # Qdrant expects int or UUID; use stable UUID from chunk_id
    point_ids = [uuid.uuid5(uuid.NAMESPACE_DNS, c.chunk_id) for c in chunks]
    payloads = [
        {
            "text": c.text,
            "source_file": c.source_file,
            "subject": c.subject,
            "from": c.from_,
            "to": c.to,
        }
        for c in chunks
    ]
    embeddings = embed_texts([c.text for c in chunks])

    batch_size = QDRANT_UPSERT_BATCH_SIZE
    for i in range(0, len(chunks), batch_size):
        batch_ids = point_ids[i : i + batch_size]
        batch_vectors = embeddings[i : i + batch_size]
        batch_payloads = payloads[i : i + batch_size]
        points = [
            models.PointStruct(id=pid, vector=vec, payload=payload)
            for pid, vec, payload in zip(batch_ids, batch_vectors, batch_payloads)
        ]
        client.upsert(collection_name=name, points=points)
        logger.debug("Upserted batch %d–%d", i + 1, min(i + batch_size, len(chunks)))
    logger.info("Indexed %d chunks into Qdrant collection %s", len(chunks), name)


def get_collection_name(collection_name: str | None = None) -> str:
    """Return collection name for retrieve."""
    return collection_name or QDRANT_COLLECTION_NAME
