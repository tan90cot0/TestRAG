"""End-to-end RAG pipeline: index and query."""

import logging
from pathlib import Path
from typing import Any

from rag.chunking import chunk_emails
from rag.config import EMAILS_DIR
from rag.generate import generate
from rag.ingest import load_all_emails
from rag.models import Chunk, ParsedEmail, RetrieveResult
from rag.retrieve import retrieve
from rag.store import build_store_from_chunks

logger = logging.getLogger(__name__)


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
        Retrieve top-k chunks (optionally filtered by where), then generate answer.
        Returns (answer, list of retrieved results).
        """
        results = retrieve(
            query,
            top_k=top_k,
            where=where,
            collection_name=self.collection_name,
        )
        if not results:
            return "I have no relevant emails in the context to answer this question.", []
        answer = generate(query, results)
        return answer, results
