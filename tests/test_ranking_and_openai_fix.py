"""Tests for two fixes: the explicit-ranking feature ('which cause is most
dominant' should name a winner, not just list numbers) and the OpenAI
max_completion_tokens parameter fix."""
import inspect

from backend.agent import _wants_ranking, _synthesize_with_openai


def test_ranking_language_detected():
    assert _wants_ranking("Which churn cause is most dominant?")
    assert _wants_ranking("What is the biggest driver of churn?")
    assert _wants_ranking("Rank the churn causes by risk")


def test_non_ranking_questions_not_flagged():
    assert not _wants_ranking("What is the price sensitivity churn rate?")
    assert not _wants_ranking("What should we do about price-sensitive churners?")


def test_openai_call_uses_max_completion_tokens_not_max_tokens():
    """Regression test for a real error: gpt-5.4-mini (and other newer
    OpenAI models) reject the older `max_tokens` parameter with a 400 error
    asking for `max_completion_tokens` instead. This doesn't call the real
    API (no network in tests) -- it inspects the source to confirm the
    correct parameter name is used, so this can't silently regress."""
    source = inspect.getsource(_synthesize_with_openai)
    assert "max_completion_tokens" in source
    assert "max_tokens=" not in source  # the OLD, now-wrong parameter name
