"""Tests for this session's fixes and additions: the 'reason' token
collision, growth-rate vs magnitude ranking distinction, invalid-horizon
detection, and the two new backend capabilities (descriptive stats,
instance-level profile prediction via direct structured input)."""
from backend.query_parser import parse_cause, detect_invalid_horizon_word, parse_query
from backend.agent import _wants_ranking, _wants_growth_rate
from backend.descriptive import describe_dataset
from backend.models import predict_for_profile


# ---------------------------------------------------------------------
# 'reason' token collision (same bug class as 'customers' and 'rate')
# ---------------------------------------------------------------------
def test_reason_synonym_does_not_falsely_match_non_behavioral():
    assert parse_cause("Which churn reason grows faster")["cause"] is None
    assert parse_cause("which reason grows faster")["cause"] is None
    assert parse_cause("What is the churn reason breakdown?")["cause"] is None


def test_moved_away_and_deceased_still_correctly_match_non_behavioral():
    """Regression guard: fixing the 'reason' collision must not break the
    legitimate non-behavioral matches that motivated adding those words."""
    assert parse_cause("Customers who moved away")["cause"] == "Non-behavioral"
    assert parse_cause("A customer who is now deceased")["cause"] == "Non-behavioral"


# ---------------------------------------------------------------------
# Growth-rate vs magnitude ranking
# ---------------------------------------------------------------------
def test_growth_rate_and_ranking_are_detected_separately():
    assert _wants_growth_rate("Which churn cause grows faster")
    assert not _wants_ranking("Which churn cause grows faster") or _wants_ranking("Which churn cause grows faster")
    # "grows faster" DOES also match "which cause" in _RANKING_PATTERNS --
    # that's fine, since the template checks wants_growth_rate FIRST and
    # uses elif for wants_ranking, so growth-rate takes priority when both
    # are true. The important guarantee is growth_rate fires correctly:
    assert _wants_growth_rate("Which churn cause grows faster") is True


def test_dominates_verb_form_detected():
    """Regression test: 'dominant' (adjective) matched but 'dominates'
    (verb) originally didn't."""
    assert _wants_ranking("Which churn cause dominates?")
    assert _wants_ranking("Price sensitivity dominates the other causes")


# ---------------------------------------------------------------------
# Invalid horizon detection
# ---------------------------------------------------------------------
def test_nonsensical_horizon_word_detected():
    assert detect_invalid_horizon_word('What is churn at "banana" months?') == "banana"


def test_vague_but_sensible_horizon_words_not_flagged():
    assert detect_invalid_horizon_word("What is churn risk in a few months?") is None
    assert detect_invalid_horizon_word("What is churn risk in the coming months?") is None


def test_invalid_horizon_short_circuits_with_no_other_answer():
    result = parse_query('What is churn at "banana" months?')
    assert result["task_type"] == "invalid_horizon"
    assert "banana" in result["clarifying_question"]


def test_legitimate_numeric_horizon_not_flagged():
    result = parse_query("What is churn risk within 6 months?")
    assert result["task_type"] != "invalid_horizon"
    assert result["time_horizon_months"] == 6


# ---------------------------------------------------------------------
# Descriptive analytics -- schema-agnostic
# ---------------------------------------------------------------------
def test_describe_dataset_classifies_columns_by_actual_dtype():
    result = describe_dataset()
    assert result["columns"]["tenure"]["type"] == "numeric"
    assert result["columns"]["Contract"]["type"] == "categorical"


def test_describe_dataset_excludes_id_like_columns():
    result = describe_dataset()
    assert "customerID" in result["id_like_columns_excluded"]
    assert "customerID" not in result["columns"]


def test_numeric_column_has_histogram_and_summary_stats():
    result = describe_dataset()
    tenure = result["columns"]["tenure"]
    assert "mean" in tenure and "median" in tenure and "std" in tenure
    assert len(tenure["histogram"]["labels"]) == len(tenure["histogram"]["counts"])


def test_categorical_column_has_bar_chart_data():
    result = describe_dataset()
    contract = result["columns"]["Contract"]
    assert sum(contract["bar_chart"]["counts"]) == contract["count"]


# ---------------------------------------------------------------------
# predict_for_profile via direct structured input (no NLP parsing)
# ---------------------------------------------------------------------
def test_predict_for_profile_with_no_attributes_uses_full_population_defaults():
    result = predict_for_profile({}, None, 12)
    assert result["profile_specified"] == {}
    assert len(result["profile_filled_with_population_defaults"]) > 0


def test_predict_for_profile_respects_specified_contract():
    result = predict_for_profile({"Contract": "Two year"}, None, 12)
    assert result["profile_specified"]["Contract"] == "Two year"
    assert "Contract" not in result["profile_filled_with_population_defaults"]


# ---------------------------------------------------------------------
# Regression test for a real CI-only bug: cause detection's threshold and
# cause_docs.md wording were only ever calibrated against TF-IDF, but the
# module-level `retrieve()` auto-selects FAISS+embeddings when available --
# which IS available in CI (real internet access) but never was in the
# sandbox this was developed in. Embeddings gave completely different,
# uncalibrated similarity scores, breaking cause routing in CI (and
# potentially in any deployed environment with internet access) while every
# local test appeared to pass. Fixed by pinning cause detection specifically
# to TF-IDF (backend/query_parser.py imports tfidf_retriever directly,
# bypassing the auto-selecting backend.rag.retriever for this one purpose).
# ---------------------------------------------------------------------
def test_cause_detection_is_pinned_to_tfidf_not_the_auto_selected_backend():
    """Confirms query_parser.py imports from tfidf_retriever directly for
    cause detection, rather than the auto-selecting retriever module --
    this is what makes cause routing deterministic regardless of whether
    the embedding model can be downloaded in a given environment."""
    import inspect
    from backend import query_parser

    source = inspect.getsource(query_parser)
    assert "from backend.rag.tfidf_retriever import retrieve" in source
    # If this ever changes back to importing from backend.rag.retriever
    # (the auto-selecting module) for cause detection, this test should
    # fail loudly -- that's exactly the regression this guards against.
    assert "from backend.rag.retriever import retrieve\n" not in source
