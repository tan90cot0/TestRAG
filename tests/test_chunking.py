"""Unit tests for chunking strategy."""

from rag.chunking import chunk_email, chunk_emails
from rag.models import ParsedEmail


def test_chunk_email_single_paragraph():
    email = ParsedEmail(
        source_file="email_001.txt",
        subject="Test",
        from_name="A",
        from_email="a@x.com",
        to_name="B",
        to_email="b@x.com",
        body="One paragraph only.",
    )
    chunks = chunk_email(email)
    assert len(chunks) == 1
    assert "Test" in chunks[0].text
    assert "A" in chunks[0].from_
    assert chunks[0].chunk_id == "email_001.txt_0"
    assert chunks[0].to_metadata()["subject"] == "Test"


def test_chunk_email_multiple_paragraphs():
    email = ParsedEmail(
        source_file="e.txt",
        subject="S",
        from_name="X",
        from_email="x@y.com",
        to_name="Y",
        to_email="y@y.com",
        body="First para.\n\nSecond para.\n\nThird.",
    )
    chunks = chunk_email(email)
    assert len(chunks) >= 2
    assert chunks[0].paragraph_index == 0
    assert "First para" in chunks[0].text
    assert "Second para" in chunks[1].text
    assert all(c.source_file == "e.txt" for c in chunks)
