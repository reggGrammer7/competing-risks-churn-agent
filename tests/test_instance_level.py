"""Tests for instance-level prediction: detection/routing in
backend/query_parser.py and the actual prediction in backend/models.py."""
from backend.query_parser import is_instance_level_query, parse_instance_attributes, parse_query
from backend.models import predict_for_profile


# ---------------------------------------------------------------------
# Detection: singular "customer" vs plural "customers"
# ---------------------------------------------------------------------
def test_singular_customer_framing_is_instance_level():
    assert is_instance_level_query("Predict churn risk for a premium customer on a one year contract.")
    assert is_instance_level_query("Give me the survival curve for a standard customer.")
    assert is_instance_level_query("How long is a month to month customer expected to stay?")
    assert is_instance_level_query("Predict churn risk for this customer.")


def test_plural_customers_is_not_instance_level():
    assert not is_instance_level_query("What is churn for fiber customers?")
    assert not is_instance_level_query("Compare churn causes for standard vs premium customers.")


# ---------------------------------------------------------------------
# Attribute extraction
# ---------------------------------------------------------------------
def test_contract_attribute_extracted():
    result = parse_instance_attributes("How long is a month to month customer expected to stay?")
    assert result["attributes"]["Contract"] == "Month-to-month"


def test_unsupported_tier_flagged():
    result = parse_instance_attributes("Predict churn risk for a premium customer on a one year contract.")
    assert result["unsupported_term"] == "premium customers"
    assert result["attributes"]["Contract"] == "One year"


def test_standard_profile_signal_detected():
    result = parse_instance_attributes("Give me the survival curve for a standard customer.")
    assert result["wants_standard_profile"] is True
    assert result["attributes"] == {}


# ---------------------------------------------------------------------
# Full parse_query routing -- the three-way split that matters
# ---------------------------------------------------------------------
def test_vague_instance_question_asks_clarifying_question():
    """Zero attributes AND no 'standard profile' signal -> genuinely
    ambiguous, matches the spec's exact expected behavior."""
    result = parse_query("Predict churn risk for this customer.")
    assert result["task_type"] == "clarification_needed"
    assert result["clarifying_question"] == "What segment and contract?"


def test_standard_profile_question_does_not_ask_clarifying_question():
    """Zero attributes but explicit 'standard/average/typical' signal means
    proceed with population defaults, NOT ask for more detail."""
    result = parse_query("Give me the survival curve for a standard customer.")
    assert result["task_type"] == "instance_level"
    assert result["ambiguous"] is False


def test_unsupported_attribute_short_circuits():
    result = parse_query("Predict churn risk for a premium customer on a one year contract.")
    assert result["task_type"] == "instance_unsupported_attribute"


def test_specific_attribute_routes_to_instance_level():
    result = parse_query("How long is a month to month customer expected to stay?")
    assert result["task_type"] == "instance_level"
    assert result["instance_attributes"]["attributes"]["Contract"] == "Month-to-month"


# ---------------------------------------------------------------------
# predict_for_profile -- the actual prediction
# ---------------------------------------------------------------------
def test_different_profiles_give_different_predictions():
    """The core thing this feature has to prove: a stated attribute must
    actually change the prediction, not just get echoed back unused."""
    mtm = predict_for_profile({"Contract": "Month-to-month"}, None, 12)
    two_yr = predict_for_profile({"Contract": "Two year"}, None, 12)
    assert mtm["churn_probability_at_horizon"] != two_yr["churn_probability_at_horizon"]


def test_unstated_attributes_are_reported_as_filled_defaults():
    result = predict_for_profile({"Contract": "One year"}, None, 12)
    assert "Contract" in result["profile_specified"]
    assert "Contract" not in result["profile_filled_with_population_defaults"]
    assert "InternetService" in result["profile_filled_with_population_defaults"]


def test_cause_specific_risk_included_when_cause_given():
    result = predict_for_profile({}, "Price sensitivity", 12)
    assert result["cause"] == "Price sensitivity"
    assert "cause_specific_relative_risk" in result


def test_no_cause_means_no_cause_specific_fields():
    result = predict_for_profile({}, None, 12)
    assert "cause" not in result


def test_churn_and_survival_probabilities_sum_to_one():
    result = predict_for_profile({}, None, 12)
    assert abs(result["churn_probability_at_horizon"] + result["survival_probability_at_horizon"] - 1.0) < 1e-6
