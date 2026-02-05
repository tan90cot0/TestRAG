"""Typed data structures for the RAG pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedEmail:
    """A single email parsed from disk."""

    source_file: str
    subject: str
    from_name: str
    from_email: str
    to_name: str
    to_email: str
    body: str

    def from_display(self) -> str:
        """Display string for 'from' (e.g. 'Name <email>')."""
        return f"{self.from_name} <{self.from_email}>"

    def to_display(self) -> str:
        """Display string for 'to'."""
        return f"{self.to_name} <{self.to_email}>"


@dataclass
class Chunk:
    """A single chunk with text and metadata for embedding and storage."""

    chunk_id: str
    text: str
    source_file: str
    subject: str
    from_: str  # sender display (name or email)
    to: str  # receiver display
    paragraph_index: int = 0

    def to_metadata(self) -> dict[str, str]:
        """Payload metadata dict (string values only) for Qdrant."""
        return {
            "source_file": self.source_file,
            "subject": self.subject,
            "from": self.from_,
            "to": self.to,
        }


@dataclass
class RetrieveResult:
    """One retrieved chunk with metadata and optional distance."""

    text: str
    metadata: dict[str, Any]
    distance: float | None = None

    @property
    def source_file(self) -> str:
        return self.metadata.get("source_file", "")

    @property
    def subject(self) -> str:
        return self.metadata.get("subject", "")

    @property
    def from_(self) -> str:
        return self.metadata.get("from", "")

    @property
    def to(self) -> str:
        return self.metadata.get("to", "")
