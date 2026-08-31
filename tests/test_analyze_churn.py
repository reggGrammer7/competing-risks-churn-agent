"""Tests for backend/models.py's analyze_churn() -- the function that
replaced always returning the whole-dataset summary regardless of what was
asked. Confirms it actually respects the horizon and segment, and fails
safely on a segment too small to fit."""
from backend.models import analyze_churn
from backend.data_utils import CAUSES


def test_horizon_changes_the_reported_value():
    """The bug being fixed: a 3-month question and a 24-month question must
    NOT return the same number."""
    r3 = analyze_churn("Price sensitivity", 3, None)
    r24 = analyze_churn("Price sensitivity", 24, None)
    cif_3 = r3["by_cause"]["Price sensitivity"]["cif_at_horizon"]
    cif_24 = r24["by_cause"]["Price sensitivity"]["cif_at_horizon"]
    assert cif_3 != cif_24
    assert cif_3 < cif_24  # cumulative incidence is non-decreasing over time


def test_segment_changes_the_result():
    """A segment must be a genuinely different fit, not the whole-dataset
    curve with a label slapped on it."""
    whole = analyze_churn("Price sensitivity", 12, None)
    segment = analyze_churn("Price sensitivity", 12,
                             {"type": "category", "column": "Contract", "value": "Month-to-month"})
    assert segment["n_customers_in_segment"] < whole["n_customers_in_segment"]
    assert segment["by_cause"]["Price sensitivity"]["cif_at_horizon"] != \
        whole["by_cause"]["Price sensitivity"]["cif_at_horizon"]


def test_no_cause_returns_all_four():
    result = analyze_churn(None, 12, None)
    assert set(result["by_cause"].keys()) == set(CAUSES)


def test_tiny_segment_fails_safely_instead_of_returning_a_misleading_number():
    """A segment too small to fit reliably should say so, not silently
    return an unreliable point estimate."""
    tiny_segment = {"type": "category", "column": "SeniorCitizen", "value": 1}
    # force an artificially strict scenario isn't needed -- pick a
    # cause/segment combo genuinely small in the synthetic data, or assert
    # the mechanism directly:
    result = analyze_churn("Non-behavioral", 12, tiny_segment)
    r = result["by_cause"]["Non-behavioral"]
    # Either it ran (enough events) or it correctly flagged insufficient data --
    # what it must NOT do is silently return a number computed from too few events.
    if r["insufficient_data"]:
        assert "message" in r
    else:
        assert r["n_events"] >= 5


def test_cif_at_horizon_never_exceeds_full_horizon_value():
    result = analyze_churn("Dissatisfaction", 6, None)
    r = result["by_cause"]["Dissatisfaction"]
    assert r["cif_at_horizon"] <= r["cif_full_horizon"]
