"""Retrieval: embed query (padded), search Qdrant with optional payload filters."""

import logging
from typing import Any

from qdrant_client.http import models

from rag.config import QDRANT_COLLECTION_NAME, TOP_K
from rag.embedding import embed_query
from rag.models import RetrieveResult
from rag.store import get_qdrant_client

logger = logging.getLogger(__name__)


def _where_to_qdrant_filter(where: dict[str, Any]) -> models.Filter | None:
    """Convert simple where dict to Qdrant Filter. Supports $eq and $and."""
    if not where:
        return None
    if "$and" in where:
        must = []
        for cond in where["$and"]:
            sub = _where_to_qdrant_filter(cond)
            if sub and sub.must:
                must.extend(sub.must)
        return models.Filter(must=must) if must else None
    # Single key: {"subject": "Meeting Request"} or {"subject": {"$eq": "..."}}
    must = []
    for key, val in where.items():
        if isinstance(val, dict) and "$eq" in val:
            val = val["$eq"]
        must.append(models.FieldCondition(key=key, match=models.MatchValue(value=val)))
    return models.Filter(must=must)


def retrieve(
    query: str,
    top_k: int | None = None,
    *,
    where: dict[str, Any] | None = None,
    collection_name: str | None = None,
) -> list[RetrieveResult]:
    """
    Embed query (padded to 1536), search Qdrant, return top-k results.
    where: e.g. {"subject": "Meeting Request"} or {"subject": {"$eq": "..."}}
    """
    k = top_k if top_k is not None else TOP_K
    query_vector = embed_query(query)
    client = get_qdrant_client()
    name = collection_name or QDRANT_COLLECTION_NAME

    query_filter = _where_to_qdrant_filter(where) if where else None

    response = client.query_points(
        collection_name=name,
        query=query_vector,
        limit=k,
        query_filter=query_filter,
        with_payload=True,
    )

    out: list[RetrieveResult] = []
    for hit in response.points:
        payload = hit.payload or {}
        text = payload.get("text", "")
        metadata = {
            "source_file": payload.get("source_file", ""),
            "subject": payload.get("subject", ""),
            "from": payload.get("from", ""),
            "to": payload.get("to", ""),
        }
        out.append(
            RetrieveResult(
                text=text,
                metadata=metadata,
                distance=hit.score,
            )
        )
    logger.debug("Retrieved %d results for query (k=%d)", len(out), k)
    return out
