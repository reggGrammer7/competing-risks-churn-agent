"""Tests for multi-cause comparison (backend/query_parser.py: parse_causes)
and the 'customers' token-collision bug found and fixed in this session."""
from backend.query_parser import parse_causes, parse_cause


def test_direct_comparison_extracts_both_causes():
    result = parse_causes("How does dissatisfaction churn compare to price sensitivity?")
    assert result is not None
    assert set(result) == {"Dissatisfaction", "Price sensitivity"}


def test_named_pair_comparison():
    result = parse_causes("Compare price sensitivity and competitive loss for premium customers")
    assert result is not None
    assert set(result) == {"Price sensitivity", "Competitive loss"}


def test_explicit_all_causes_phrase_returns_none_not_a_list():
    """'Rank all causes' means all four -- handled via cause=None downstream,
    not as a 'these specific 2+ causes' list."""
    assert parse_causes("Rank all churn causes by risk within 12 months") is None
    assert parse_causes("Compare all churn causes for premium customers") is None


def test_single_cause_question_returns_none():
    """A single-cause question should NOT trigger multi-cause mode -- that's
    parse_cause()'s job."""
    assert parse_causes("What is the price sensitivity churn rate?") is None


def test_no_cause_mentioned_returns_none():
    assert parse_causes("What is the overall churn rate?") is None


def test_generic_word_customers_does_not_falsely_match_any_cause():
    """Regression test: 'customers' was accidentally left in only the Price
    sensitivity doc after an earlier fix, so ANY question mentioning
    'customers' (nearly all of them) got a small false match to Price
    sensitivity. Same bug class as the earlier 'rate' collision."""
    result = parse_cause("What is the short-term churn risk for premium customers")
    assert result["cause"] is None
    result2 = parse_cause("How does churn behave long term for standard customers")
    assert result2["cause"] is None
