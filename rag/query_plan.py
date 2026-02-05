"""Query planning: use Mistral to turn a user question into structured search queries for RAG."""

import json
import logging
import re
from typing import Any

from mistralai import Mistral

from rag.config import MISTRAL_API_KEY, MISTRAL_MODEL

logger = logging.getLogger(__name__)

PLAN_SYSTEM = """You are a search query planner for a company email corpus.
Given a user question, output 1 or more search queries that will be run against an email search index (semantic search).
- Each query should be a short phrase or question that would retrieve relevant emails (e.g. "budget approval request", "meeting schedule Q3").
- You may output one query that rephrases the user question, or multiple queries if the question touches several topics (e.g. budget AND training).
- Output only valid JSON, no other text. Use this exact schema: {"queries": ["query1", "query2", ...]}"""


def _extract_json(raw: str) -> str:
    """Extract JSON from response, stripping markdown code blocks if present."""
    raw = raw.strip()
    # ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if match:
        return match.group(1).strip()
    return raw


def plan_queries(
    user_question: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> list[str]:
    """
    Ask Mistral to produce a list of search queries from the user question.
    Returns a list of query strings to run against the RAG index.
    On API or parse failure, returns [user_question] as fallback.
    """
    key = api_key or MISTRAL_API_KEY
    if not key:
        raise ValueError("MISTRAL_API_KEY is not set. Set it in the environment or pass api_key=.")

    client = Mistral(api_key=key)
    model_name = model or MISTRAL_MODEL
    messages = [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": user_question},
    ]

    try:
        response = client.chat.complete(
            model=model_name,
            messages=messages,
        )
    except Exception as e:
        logger.warning("Query plan API error, using original question: %s", e)
        return [user_question]

    choice = response.choices[0] if response.choices else None
    if not choice or not choice.message or not choice.message.content:
        logger.warning("Query plan returned no content, using original question")
        return [user_question]

    raw = (choice.message.content or "").strip()
    json_str = _extract_json(raw)

    try:
        data: dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning("Query plan invalid JSON, using original question: %s", e)
        return [user_question]

    queries = data.get("queries")
    if not isinstance(queries, list):
        logger.warning("Query plan missing 'queries' list, using original question")
        return [user_question]

    out = [str(q).strip() for q in queries if q]
    if not out:
        return [user_question]
    logger.info("Planned %d search queries: %s", len(out), out)
    return out
