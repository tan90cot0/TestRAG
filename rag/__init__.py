"""
Mini RAG pipeline: ingest emails → chunk → embed → Qdrant → retrieve → Mistral.
No LangChain/LlamaIndex; sentence-transformers + Qdrant + Mistral only.
"""

from rag.models import ParsedEmail, Chunk, RetrieveResult

# Lazy import to avoid loading embedding/Qdrant/Mistral when only using ingest/chunking
def __getattr__(name: str):
    if name == "RAGPipeline":
        from rag.pipeline import RAGPipeline
        return RAGPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ParsedEmail",
    "Chunk",
    "RetrieveResult",
    "RAGPipeline",
]
