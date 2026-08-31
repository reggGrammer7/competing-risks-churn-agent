"""
Regression tests for real bugs found and fixed during development -- see
README.md's "Fixed since the first pass" section for the full write-up of
each. These tests exist so the same bug can't silently come back.
"""
from backend.agent import _wants_recommendation


def test_action_keyword_does_not_match_dissatisfaction():
    """The original bug: `"action" in question.lower()` matched inside the
    word "dissatisfaction" (...dissatisf-ACTION), so a purely descriptive
    question about dissatisfaction was incorrectly treated as a request for
    a recommendation. Fixed with word-boundary regex."""
    assert _wants_recommendation("How does tech support affect dissatisfaction churn?") is False
    assert _wants_recommendation("Why are customers dissatisfied with our service?") is False


def test_action_keyword_still_matches_as_a_real_word():
    """The fix must not overcorrect -- 'action' as an actual standalone word
    should still trigger the recommendation branch."""
    assert _wants_recommendation("What action should we take for competitive loss?") is True


def test_recommendation_phrases_still_detected():
    assert _wants_recommendation("What should we do about price-sensitive churners?") is True
    assert _wants_recommendation("Can you recommend a retention strategy?") is True
    assert _wants_recommendation("What should we do about churn?") is True


def test_purely_descriptive_questions_do_not_trigger_policy_lookup():
    assert _wants_recommendation("What is the cumulative incidence for price sensitivity?") is False
    assert _wants_recommendation("How does contract type affect price churn?") is False
