"""Tests for backend/data_validation.py -- confirms clean data passes and
each class of bad data is actually caught, not just that the function runs."""
import pandas as pd
import pytest

from backend.data_utils import load_raw
from backend.data_validation import validate_raw


@pytest.fixture
def valid_df():
    from backend.data_utils import DATA_PATH
    return pd.read_csv(DATA_PATH)


def test_clean_data_passes(valid_df):
    report = validate_raw(valid_df, raise_on_error=False)
    assert report.passed
    assert report.errors == []


def test_missing_required_column_is_caught(valid_df):
    bad = valid_df.drop(columns=["tenure"])
    report = validate_raw(bad, raise_on_error=False)
    assert not report.passed
    assert any(c.name == "required_columns_present" for c in report.errors)


def test_duplicate_customer_id_is_caught(valid_df):
    bad = valid_df.copy()
    bad.loc[1, "customerID"] = bad.loc[0, "customerID"]
    report = validate_raw(bad, raise_on_error=False)
    assert not report.passed
    assert any(c.name == "no_duplicate_customer_ids" for c in report.errors)


def test_negative_tenure_is_caught(valid_df):
    bad = valid_df.copy()
    bad.loc[0, "tenure"] = -3
    report = validate_raw(bad, raise_on_error=False)
    assert not report.passed
    assert any(c.name == "tenure_non_negative" for c in report.errors)


def test_inconsistent_label_reason_is_caught(valid_df):
    """A churned customer (Churn Label == Yes) with no Churn Reason, or a
    non-churned customer with one set, is an internally impossible
    combination -- the direct equivalent of an impossible censoring date."""
    bad = valid_df.copy()
    bad.loc[0, "Churn Label"] = "Yes"
    bad.loc[0, "Churn Reason"] = None
    report = validate_raw(bad, raise_on_error=False)
    assert not report.passed
    assert any(c.name == "churn_label_reason_consistency" for c in report.errors)


def test_raise_on_error_true_raises(valid_df):
    bad = valid_df.copy()
    bad.loc[0, "tenure"] = -1
    with pytest.raises(ValueError):
        validate_raw(bad, raise_on_error=True)


def test_load_raw_validates_real_data_successfully():
    """End-to-end: the actual pipeline entry point should load without
    raising, since the shipped synthetic data is clean."""
    df = load_raw()
    assert len(df) > 0
