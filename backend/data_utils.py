"""
Loads the churn dataset and prepares it for competing-risks modeling.

Uses the REAL IBM Telco Customer Churn dataset (7,043 customers, the
33-column version that includes "Churn Label" and "Churn Reason" -- see
data/README.md for provenance and the exact transformation applied). This
module doesn't care whether the file at DATA_PATH is this real dataset or
the synthetic generator's output, as long as the column names match.
"""
import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco.csv")

# Collapses the real dataset's 20 granular Churn Reason categories into the
# 4 competing-risk buckets this project models. Every one of the real
# dataset's actual category strings is mapped explicitly below (verified
# against the live value_counts() of the dataset in use) -- nothing here is
# guessed. Anything genuinely unmapped falls through to "Other" rather than
# silently miscategorized.
REASON_MAP = {
    "Price sensitivity": "Price sensitivity",
    "Dissatisfaction": "Dissatisfaction",
    "Competitive loss": "Competitive loss",
    "Non-behavioral": "Non-behavioral",

    # --- Price sensitivity: cost-driven departures ---
    "Price too high": "Price sensitivity",
    "Extra data charges": "Price sensitivity",
    "Long distance charges": "Price sensitivity",
    "Lack of affordable download/upload speed": "Price sensitivity",

    # --- Dissatisfaction: service/support quality issues ---
    "Attitude of support person": "Dissatisfaction",
    "Attitude of service provider": "Dissatisfaction",
    "Poor expertise of online support": "Dissatisfaction",
    "Poor expertise of phone support": "Dissatisfaction",
    "Network reliability": "Dissatisfaction",
    "Product dissatisfaction": "Dissatisfaction",
    "Service dissatisfaction": "Dissatisfaction",
    "Lack of self-service on Website": "Dissatisfaction",
    "Limited range of services": "Dissatisfaction",

    # --- Competitive loss: a competitor's offer ---
    "Competitor offered more data": "Competitive loss",
    "Competitor offered higher download speeds": "Competitive loss",
    "Competitor made better offer": "Competitive loss",
    "Competitor had better devices": "Competitive loss",

    # --- Non-behavioral: outside the company's control ---
    "Moved": "Non-behavioral",
    "Deceased": "Non-behavioral",
    "Don't know": "Non-behavioral",
}

CAUSES = ["Price sensitivity", "Dissatisfaction", "Competitive loss", "Non-behavioral"]

COVARIATES_CATEGORICAL = [
    "Contract", "InternetService", "TechSupport", "PaymentMethod", "Dependents", "PaperlessBilling",
    "PhoneService", "MultipleLines",
]
COVARIATES_NUMERIC = ["MonthlyCharges", "SeniorCitizen"]

# Gender and Partner are intentionally NOT covariates. Both were checked
# empirically before this decision, not assumed:
#   - Gender: ~0.8-point churn-rate spread between categories -- a
#     demographic attribute with essentially no predictive signal, so
#     excluding it costs nothing.
#   - Partner: a genuinely different case -- ~13-point spread (unpartnered
#     customers churn at nearly double the rate of partnered ones), a real,
#     measurable predictive signal. Excluded anyway: marital/partnership
#     status is a family-status attribute, and using it to drive retention
#     targeting raises the same category of fairness concern as other
#     protected/sensitive attributes, even outside a strict legal
#     requirement (unlike lending, ECOA doesn't directly govern a churn
#     model). This is a disclosed, deliberate accuracy-for-fairness
#     tradeoff, not a costless exclusion -- see data/README.md for the full
#     reasoning and the exact numbers behind it.


def load_raw() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"No dataset found at {DATA_PATH}. Run `python data/generate_synthetic_data.py` "
            "to create a synthetic one, or place the real Telco CSV there."
        )
    df = pd.read_csv(DATA_PATH)
    # Validate BEFORE anything downstream touches this data -- raises with a
    # full list of failures if any hard check fails, so a bad row never
    # silently reaches a model fit. See backend/data_validation.py.
    from backend.data_validation import validate_raw
    validate_raw(df, raise_on_error=True)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Adds standardized event_type / event_observed columns used by every model."""
    df = df.copy()
    df["event_observed"] = (df["Churn Label"] == "Yes").astype(int)
    df["event_type_raw"] = df["Churn Reason"].fillna("Censored")
    df["event_type"] = df["event_type_raw"].map(lambda r: REASON_MAP.get(r, r if r == "Censored" else "Other"))
    # numeric code per cause for lifelines' cause-specific fitting (0 = censored)
    cause_code = {c: i + 1 for i, c in enumerate(CAUSES)}
    cause_code["Other"] = len(CAUSES) + 1
    cause_code["Censored"] = 0
    df["event_type_code"] = df["event_type"].map(cause_code)
    return df


def get_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encodes covariates for modeling."""
    X = pd.get_dummies(df[COVARIATES_CATEGORICAL], drop_first=True)
    X[COVARIATES_NUMERIC] = df[COVARIATES_NUMERIC]
    return X.astype(float)


def load_prepared() -> pd.DataFrame:
    return prepare(load_raw())
