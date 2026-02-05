"""Generation via Mistral API."""

import logging
from typing import Any

from mistralai import Mistral

from rag.config import MISTRAL_API_KEY, MISTRAL_MODEL
from rag.models import RetrieveResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise assistant that answers questions using ONLY the provided context from company emails.
Rules:
- Base your answer strictly on the given context. Do not use external knowledge.
- If the context does not contain enough information to answer, say so clearly.
- When possible, cite the source (e.g. "According to email from ..." or "In the email about ...").
- Keep answers concise and factual. Do not hallucinate or invent details."""


def _format_context(results: list[RetrieveResult]) -> str:
    """Format retrieved chunks for the prompt."""
    parts = []
    for i, r in enumerate(results, 1):
        source = r.metadata.get("source_file", "unknown")
        subject = r.metadata.get("subject", "")
        parts.append(f"[Source {i} — {source} | Subject: {subject}]\n{r.text}")
    return "\n\n---\n\n".join(parts)


def build_messages(query: str, context_results: list[RetrieveResult]) -> list[dict[str, str]]:
    """Build Mistral messages: system + user with context and question."""
    context = _format_context(context_results)
    user_content = f"""Context from company emails:\n\n{context}\n\nQuestion: {query}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate(
    query: str,
    context_results: list[RetrieveResult],
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Call Mistral chat completion with context and query.
    Raises if MISTRAL_API_KEY is missing or API call fails.
    """
    key = api_key or MISTRAL_API_KEY
    if not key:
        raise ValueError("MISTRAL_API_KEY is not set. Set it in the environment or pass api_key=.")

    client = Mistral(api_key=key)
    model_name = model or MISTRAL_MODEL
    messages = build_messages(query, context_results)

    try:
        response = client.chat.complete(
            model=model_name,
            messages=messages,
        )
    except Exception as e:
        logger.exception("Mistral API error: %s", e)
        raise

    choice = response.choices[0] if response.choices else None
    if not choice or not choice.message:
        raise RuntimeError("Mistral returned no message content")
    return (choice.message.content or "").strip()
