"""Chunking strategy: paragraph-based with email context and metadata."""

import logging
from rag.models import Chunk, ParsedEmail

logger = logging.getLogger(__name__)


def chunk_email(email: ParsedEmail) -> list[Chunk]:
    """
    Split email body by paragraphs; each chunk gets subject/from/to prepended
    and the same metadata for filtering.
    """
    chunks: list[Chunk] = []
    # Normalize: split by blank lines, keep non-empty blocks
    raw_paragraphs = [p.strip() for p in email.body.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        # Single block or no body: treat whole body as one chunk
        raw_paragraphs = [email.body] if email.body.strip() else []

    header = f"Subject: {email.subject}\nFrom: {email.from_display()}\nTo: {email.to_display()}\n\n"
    for i, para in enumerate(raw_paragraphs):
        chunk_text = header + para
        chunk_id = f"{email.source_file}_{i}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source_file=email.source_file,
                subject=email.subject,
                from_=email.from_display(),
                to=email.to_display(),
                paragraph_index=i,
            )
        )

    return chunks


def chunk_emails(emails: list[ParsedEmail]) -> list[Chunk]:
    """Chunk all emails; returns flat list of chunks with metadata."""
    all_chunks: list[Chunk] = []
    for email in emails:
        all_chunks.extend(chunk_email(email))
    logger.info("Produced %d chunks from %d emails", len(all_chunks), len(emails))
    return all_chunks
