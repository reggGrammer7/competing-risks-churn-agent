"""Tests for backend/query_parser.py -- covers every example phrase from
the original spec this module was built against, plus the two real bugs
found and fixed while building it (a generic-word collision on 'rate', and
two cause docs missing their own exact cause name as a token)."""
from backend.query_parser import parse_query, parse_time_horizon, parse_segment, parse_cause


# ---------------------------------------------------------------------
# Time horizon
# ---------------------------------------------------------------------
def test_explicit_months():
    assert parse_time_horizon("What is churn risk within 3 months?")["time_horizon_months"] == 3
    assert parse_time_horizon("Churn risk in 8 months")["time_horizon_months"] == 8


def test_explicit_years():
    assert parse_time_horizon("Churn over 2 years")["time_horizon_months"] == 24


def test_named_horizons():
    cases = [
        ("What does early churn look like?", 6),
        ("Give me short-term churn risk", 6),
        ("What about long-term churn?", 24),
        ("Churn risk in the first year", 12),
        ("Churn risk in the first quarter", 3),
    ]
    for q, expected in cases:
        result = parse_time_horizon(q)
        assert result["time_horizon_months"] == expected, q


def test_full_curve_phrases_set_the_flag_not_just_a_number():
    for q in ["What is lifetime churn risk?", "How does churn change over time?"]:
        result = parse_time_horizon(q)
        assert result["wants_full_curve"] is True, q


def test_no_horizon_mentioned_defaults_to_12_months():
    result = parse_time_horizon("What is the overall churn rate?")
    assert result["time_horizon_months"] == 12
    assert result["wants_full_curve"] is False


# ---------------------------------------------------------------------
# Cause
# ---------------------------------------------------------------------
def test_cause_synonyms():
    cases = [
        ("They left because of cost", "Price sensitivity"),
        ("Customers reporting service issues", "Dissatisfaction"),
        ("They switched to a competitor", "Competitive loss"),
        ("Customers who moved away", "Non-behavioral"),
        ("A customer who is now deceased", "Non-behavioral"),
    ]
    for q, expected in cases:
        assert parse_cause(q)["cause"] == expected, q


def test_exact_cause_names_match_their_own_cause():
    """Regression test: 'Competitive loss' and 'Price sensitivity' originally
    didn't contain their own exact name as a token in cause_docs.md, so
    asking about a cause by its literal name under-matched."""
    assert parse_cause("What is price sensitivity churn?")["cause"] == "Price sensitivity"
    assert parse_cause("Tell me about competitive loss")["cause"] == "Competitive loss"


def test_generic_churn_rate_does_not_falsely_match_price():
    """Regression test: the bare word 'rate' was in the Price sensitivity
    doc (for 'rate increase'), which falsely matched the generic phrase
    'churn rate' to price sensitivity even with no pricing language."""
    result = parse_cause("What is the overall churn rate?")
    assert result["cause"] is None


def test_no_cause_mentioned_returns_none_meaning_all_causes():
    assert parse_cause("Give me general churn statistics")["cause"] is None


# ---------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------
def test_segment_examples():
    assert parse_segment("churn risk for new customers")["type"] == "tenure_max"
    assert parse_segment("churn risk for fiber customers")["column"] == "InternetService"
    assert parse_segment("churn risk for all customers")["type"] == "all"


def test_unsupported_segment_terms_flagged_not_silently_ignored():
    """'premium' and 'student' aren't attributes in this dataset's schema --
    the parser should say so, not silently default to 'all customers'
    without explanation."""
    result = parse_segment("churn risk for premium customers")
    assert result["type"] == "unsupported"
    assert "premium" in result["label"]


# ---------------------------------------------------------------------
# Full parse_query -- ambiguity and defaults
# ---------------------------------------------------------------------
def test_vague_question_triggers_clarification():
    result = parse_query("why")
    assert result["ambiguous"] is True
    assert result["task_type"] == "clarification_needed"
    assert result["clarifying_question"] is not None


def test_specific_question_does_not_trigger_clarification():
    result = parse_query("What is the price sensitivity churn rate for new customers in 3 months?")
    assert result["ambiguous"] is False
    assert result["cause"] == "Price sensitivity"
    assert result["time_horizon_months"] == 3
    assert result["segment"]["type"] == "tenure_max"


def test_defaults_when_nothing_specified():
    result = parse_query("What is the overall churn rate?")
    assert result["time_horizon_months"] == 12  # default horizon
    assert result["cause"] is None  # -> all causes
    assert result["segment"]["type"] == "all"  # -> all customers
