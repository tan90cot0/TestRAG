"""
End-to-end quality tests: retrieval relevance and generation correctness.
Uses real emails and real embedding/Qdrant; Mistral is optional (skip if no API key).
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from rag.models import RetrieveResult


class TestRetrievalQuality:
    """Retrieval should return relevant chunks for semantic queries."""

    def test_index_has_chunks(self, pipeline_with_index):
        """After indexing, querying returns results."""
        from rag.retrieve import retrieve

        results = retrieve("meeting schedule", top_k=3)
        assert len(results) > 0
        assert all(hasattr(r, "text") and hasattr(r, "metadata") for r in results)
        assert any("subject" in r.metadata for r in results)

    def test_retrieval_by_topic_meeting(self, pipeline_with_index):
        """Query about meetings should retrieve Meeting Request emails in top-k."""
        from rag.retrieve import retrieve

        results = retrieve("schedule a meeting to discuss strategy", top_k=5)
        subjects = [r.metadata.get("subject", "") for r in results]
        assert any("Meeting" in s or "meeting" in s.lower() for s in subjects), (
            f"Expected at least one meeting-related subject in top-5, got: {subjects}"
        )

    def test_retrieval_by_topic_budget(self, pipeline_with_index):
        """Query about budget should retrieve Budget Approval emails in top-k."""
        from rag.retrieve import retrieve

        results = retrieve("budget approval and fiscal year", top_k=5)
        texts = [r.text.lower() for r in results]
        assert any("budget" in t for t in texts), (
            "Expected at least one chunk mentioning budget in top-5"
        )

    def test_retrieval_by_topic_training(self, pipeline_with_index):
        """Query about training should retrieve Training Opportunity emails."""
        from rag.retrieve import retrieve

        results = retrieve("training workshop professional development", top_k=5)
        subjects = [r.metadata.get("subject", "") for r in results]
        assert any("Training" in s for s in subjects), (
            f"Expected Training in subjects, got: {subjects}"
        )

    def test_retrieval_metadata_filter_subject(self, pipeline_with_index):
        """Filtering by subject should restrict results to that subject."""
        from rag.retrieve import retrieve

        # First find a subject that exists
        all_results = retrieve("training", top_k=20)
        subjects = {r.metadata.get("subject") for r in all_results if r.metadata.get("subject")}
        if not subjects:
            pytest.skip("No subjects in index")
        subject = next(iter(subjects))
        filtered = retrieve("anything", top_k=10, where={"subject": subject})
        assert len(filtered) > 0
        for r in filtered:
            assert r.metadata.get("subject") == subject


def _plan_response():
    """Response for query-planning call: structured JSON with search queries."""
    return MagicMock(
        choices=[
            MagicMock(message=MagicMock(content='{"queries": ["What is this email about?"]}'))
        ]
    )


def _gen_response():
    """Response for generation call."""
    return MagicMock(
        choices=[
            MagicMock(message=MagicMock(content="The email discusses scheduling a meeting to review strategy and performance metrics."))
        ]
    )


class TestGenerationQuality:
    """Generation should use context and not hallucinate."""

    @pytest.fixture
    def mock_mistral(self):
        """Patch Mistral: first call = query plan (JSON), second call = generate answer."""
        with patch("rag.generate.Mistral") as MockGen, patch("rag.query_plan.Mistral") as MockPlan:
            mock_client = MagicMock()
            mock_client.chat.complete.side_effect = [_plan_response(), _gen_response()]
            MockGen.return_value = mock_client
            MockPlan.return_value = mock_client
            yield mock_client

    def test_generate_receives_context_and_query(self, pipeline_with_index, mock_mistral):
        """Mistral is called for plan then for generate; generate receives context and query."""
        answer, results = pipeline_with_index.ask("What is this email about?", top_k=2)
        assert mock_mistral.chat.complete.call_count >= 2
        # Second call is generate (context + question)
        call_kwargs = mock_mistral.chat.complete.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert messages
        user_content = next((m["content"] for m in messages if m.get("role") == "user"), "")
        assert "Question:" in user_content
        assert "What is this email about?" in user_content
        assert "Context" in user_content or "Source" in user_content

    def test_generate_returns_string(self, pipeline_with_index, mock_mistral):
        """Answer is a non-empty string."""
        answer, _ = pipeline_with_index.ask("What is the main topic?", top_k=2)
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY"),
        reason="MISTRAL_API_KEY not set; skipping live API test",
    )
    def test_generation_live_answer_grounded(self, pipeline_with_index):
        """With real API key, answer should be grounded in context (no 'I don't have access')."""
        answer, results = pipeline_with_index.ask(
            "What did the sender want to discuss in this email?",
            top_k=3,
        )
        assert isinstance(answer, str)
        assert len(answer) > 20
        # Should not be a generic "I cannot" response when we have context
        assert "I don't have" not in answer or "context" in answer.lower()
        assert results


class TestPipelineIntegration:
    """Full pipeline: index and ask."""

    def test_ask_without_results_returns_fallback(self, pipeline_with_index):
        """When retrieval returns no results, we return a fallback message (no generate call)."""
        with patch("rag.query_plan.Mistral") as MockMistral:
            mock_client = MagicMock()
            mock_client.chat.complete.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content='{"queries": ["What happened?"]}'))]
            )
            MockMistral.return_value = mock_client
            answer, results = pipeline_with_index.ask(
                "What happened?",
                top_k=2,
                where={"subject": {"$eq": "Nonexistent Subject 12345"}},
            )
        assert len(results) == 0
        assert "no relevant" in answer.lower() or "don't have" in answer.lower()