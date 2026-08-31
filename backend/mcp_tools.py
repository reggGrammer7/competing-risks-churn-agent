"""
Tool layer, structured the way MCP tools are structured (typed input/output,
one job per tool) so wiring this into an actual MCP server later is a thin
wrapper, not a rewrite.

NOTE ON SCOPE: this scaffold exposes these as plain Python functions called
directly by the FastAPI routes (see main.py), which is enough to run and demo
end-to-end. To turn this into a REAL MCP server: install the `mcp` Python SDK,
create a Server instance, and register each function below as a tool with
the same Pydantic schemas -- the business logic doesn't change at all.
"""
from pydantic import BaseModel, Field
from typing import Literal
from backend import models

Cause = Literal["Price sensitivity", "Dissatisfaction", "Competitive loss", "Non-behavioral"]


class CustomerProfile(BaseModel):
    contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    monthly_charges: float = Field(70.0, ge=0)
    internet_service: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    tech_support: Literal["Yes", "No", "No internet service"] = "No"
    payment_method: str = "Electronic check"
    senior_citizen: int = Field(0, ge=0, le=1)
    dependents: Literal["Yes", "No"] = "No"
    paperless_billing: Literal["Yes", "No"] = "Yes"


def get_cif(cause: Cause) -> dict:
    """MCP tool: cumulative incidence function for one competing-risk cause."""
    aj = models.aalen_johansen_cif()
    return {"cause": cause, **aj["by_cause"][cause]}


def compare_causes() -> dict:
    """MCP tool: CIFs for all causes side by side."""
    return models.aalen_johansen_cif()["by_cause"]


def predict_churn_binary() -> dict:
    """MCP tool: the deliberate strawman classifier's output, for contrast."""
    return models.fit_binary_strawman()


def fit_summary(model_name: Literal["cause_specific_cox", "random_survival_forest"]) -> dict:
    """MCP tool: coefficient/importance summary for a fitted model."""
    if model_name == "cause_specific_cox":
        return models.cause_specific_cox()
    return models.random_survival_forest()


def explain_prediction(cause: Cause) -> dict:
    """MCP tool: which covariates drive risk for a given cause (from Cox hazard ratios)."""
    cox = models.cause_specific_cox()
    return {"cause": cause, "drivers": cox["top_covariates_by_cause"].get(cause, [])}


def get_cif_with_ci(cause: Cause, n_boot: int = 200) -> dict:
    """MCP tool: cumulative incidence function with a bootstrapped confidence
    band, computed via a multiprocessing pool (see backend/models.py for why
    multiprocessing rather than asyncio is the right tool for this one)."""
    return models.bootstrap_ci_for_cause(cause, n_boot=n_boot)


def analyze_churn(cause, time_horizon_months: int, segment: dict | None) -> dict:
    """MCP tool: segment- and horizon-aware churn analysis -- the tool the
    agent actually calls now. Takes ONLY structured parameters (cause may be
    None, a single cause, or a list of specific causes to compare); never
    touches raw question text -- see backend/query_parser.py for where a
    question becomes these parameters."""
    return models.analyze_churn(cause, time_horizon_months, segment)


def predict_customer_profile(attributes: dict, cause, time_horizon_months: int) -> dict:
    """MCP tool: instance-level prediction for ONE constructed customer
    profile, as opposed to analyze_churn()'s population/segment view. Takes
    only the attributes the question actually specified -- everything else
    is filled with population defaults inside models.py, and which fields
    were filled vs. stated is returned explicitly in the payload."""
    return models.predict_for_profile(attributes, cause, time_horizon_months)
