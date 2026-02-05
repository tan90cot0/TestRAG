"""End-to-end RAG pipeline: index and query."""

import logging
from pathlib import Path
from typing import Any

from rag.chunking import chunk_emails
from rag.config import EMAILS_DIR, TOP_K
from rag.generate import generate
from rag.ingest import load_all_emails
from rag.models import Chunk, ParsedEmail, RetrieveResult
from rag.query_plan import plan_queries
from rag.retrieve import retrieve
from rag.store import build_store_from_chunks

logger = logging.getLogger(__name__)


def _merge_and_dedupe_results(
    results_list: list[list[RetrieveResult]],
    top_k: int,
) -> list[RetrieveResult]:
    """Merge results from multiple queries, dedupe by (source_file, text), sort by score, take top_k."""
    seen: set[tuple[str, str]] = set()
    merged: list[RetrieveResult] = []
    for results in results_list:
        for r in results:
            key = (r.source_file, r.text)
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
    # Higher distance = better match (cosine similarity)
    merged.sort(key=lambda r: (r.distance or 0.0), reverse=True)
    return merged[:top_k]


class RAGPipeline:
    """
    Single entry point: build index from emails, then answer questions
    with optional metadata filters (Qdrant payload filters).
    """

    def __init__(
        self,
        emails_dir: Path | None = None,
        collection_name: str | None = None,
    ):
        self.emails_dir = emails_dir or EMAILS_DIR
        self.collection_name = collection_name

    def index(self) -> None:
        """Load emails, chunk, embed (padded to 1536), and store in Qdrant."""
        emails = load_all_emails(self.emails_dir)
        if not emails:
            raise ValueError(f"No emails loaded from {self.emails_dir}")
        chunks = chunk_emails(emails)
        build_store_from_chunks(chunks, collection_name=self.collection_name)
        logger.info("Indexing complete: %d chunks", len(chunks))

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        *,
        where: dict[str, Any] | None = None,
    ) -> tuple[str, list[RetrieveResult]]:
        """
        Plan search queries via Mistral (structured JSON), retrieve for each query,
        merge and dedupe results, then generate answer from context.
        Returns (answer, list of retrieved results).
        """
        k = top_k if top_k is not None else TOP_K
        planned = plan_queries(query)
        per_query_k = max(2, (k + len(planned) - 1) // len(planned))
        results_list: list[list[RetrieveResult]] = []
        for q in planned:
            results_list.append(
                retrieve(
                    q,
                    top_k=per_query_k,
                    where=where,
                    collection_name=self.collection_name,
                )
            )
        results = _merge_and_dedupe_results(results_list, k)
        if not results:
            return "I have no relevant emails in the context to answer this question.", []
        answer = generate(query, results)
        return answer, results
