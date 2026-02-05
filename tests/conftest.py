"""Pytest fixtures: pipeline and pre-built Qdrant index (uses env QDRANT_*)."""

import os
from pathlib import Path

import pytest

# Set test env before importing rag modules that read config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("EMAILS_DIR", str(PROJECT_ROOT / "emails"))
os.environ.setdefault("QDRANT_COLLECTION_NAME", "test_email_chunks")


@pytest.fixture(scope="session")
def pipeline_with_index():
    """Build index once per session; return pipeline that uses it. Requires QDRANT_URL (and QDRANT_API_KEY if needed)."""
    if not os.environ.get("QDRANT_URL"):
        pytest.skip("QDRANT_URL not set; set it in .env for e2e tests")
    from rag.chunking import chunk_emails
    from rag.ingest import load_all_emails
    from rag.pipeline import RAGPipeline
    from rag.store import build_store_from_chunks

    emails_dir = Path(os.environ["EMAILS_DIR"])
    if not emails_dir.exists():
        pytest.skip("emails/ directory not found")
    emails = load_all_emails(emails_dir)
    if not emails:
        pytest.skip("No emails loaded")
    chunks = chunk_emails(emails)
    build_store_from_chunks(
        chunks,
        collection_name=os.environ["QDRANT_COLLECTION_NAME"],
    )
    pipe = RAGPipeline(
        emails_dir=emails_dir,
        collection_name=os.environ["QDRANT_COLLECTION_NAME"],
    )
    yield pipe
