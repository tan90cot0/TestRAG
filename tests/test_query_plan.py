"""Unit tests for query planning (Mistral → structured JSON)."""

from unittest.mock import MagicMock, patch

import pytest

from rag.query_plan import plan_queries


def test_plan_queries_returns_list_from_valid_json():
    """Valid JSON with 'queries' array is parsed and returned."""
    with patch("rag.query_plan.Mistral") as MockMistral:
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"queries": ["budget approval", "training workshop"]}'
                    )
                )
            ]
        )
        MockMistral.return_value = mock_client
        result = plan_queries("What about budget and training?")
    assert result == ["budget approval", "training workshop"]


def test_plan_queries_accepts_markdown_code_block():
    """JSON inside ```json ... ``` is extracted and parsed."""
    with patch("rag.query_plan.Mistral") as MockMistral:
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='```json\n{"queries": ["meeting schedule"]}\n```'
                    )
                )
            ]
        )
        MockMistral.return_value = mock_client
        result = plan_queries("When is the meeting?")
    assert result == ["meeting schedule"]


def test_plan_queries_fallback_on_invalid_json():
    """Invalid JSON falls back to original question as single query."""
    with patch("rag.query_plan.Mistral") as MockMistral:
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="not json at all"))]
        )
        MockMistral.return_value = mock_client
        result = plan_queries("What was decided?")
    assert result == ["What was decided?"]


def test_plan_queries_fallback_on_empty_queries():
    """Empty queries array falls back to original question."""
    with patch("rag.query_plan.Mistral") as MockMistral:
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"queries": []}'))]
        )
        MockMistral.return_value = mock_client
        result = plan_queries("Something?")
    assert result == ["Something?"]


def test_plan_queries_requires_api_key():
    """Raises when MISTRAL_API_KEY is not set and no key passed."""
    with patch("rag.query_plan.MISTRAL_API_KEY", ""):
        with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
            plan_queries("test")
