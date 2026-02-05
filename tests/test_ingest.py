"""Unit tests for email loading and parsing."""

from pathlib import Path

import pytest

from rag.ingest import load_email_file, parse_email_content


def test_parse_email_content_valid():
    content = """Subject: Meeting Request

From: Helen Powell <helen.powell@tech.io>
To: Nico Clark <nico.clark@corp.org>

Hello Nico,

I would like to schedule a meeting.
Thanks,
Helen Powell"""
    parsed = parse_email_content(content, "email_050.txt")
    assert parsed is not None
    assert parsed.subject == "Meeting Request"
    assert "Helen" in parsed.from_name
    assert "helen.powell" in parsed.from_email
    assert "Nico" in parsed.to_name
    assert "meeting" in parsed.body.lower()


def test_parse_email_content_missing_headers():
    parsed = parse_email_content("Just some text\nNo headers", "x.txt")
    assert parsed is None


def test_load_email_file_real(emails_dir=None):
    emails_dir = emails_dir or Path(__file__).resolve().parent.parent / "emails"
    if not emails_dir.exists():
        pytest.skip("emails/ not found")
    path = next(emails_dir.glob("email_*.txt"), None)
    if not path:
        pytest.skip("No email_*.txt files")
    parsed = load_email_file(path)
    assert parsed is not None
    assert parsed.source_file == path.name
    assert parsed.subject
    assert parsed.body
