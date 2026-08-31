from dotenv import load_dotenv
load_dotenv()  # reads .env into the environment BEFORE anything below (agent.py's
                # os.environ.get("OPENAI_API_KEY") etc.) looks for a key -- must run first

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend import models, mcp_tools
from backend.agent import answer_question

app = FastAPI(title="Competing-Risks Churn Agent API")

# CORS: wide open by default (fine for local dev, and harmless even in
# production since every route here is read-only analytics with no auth --
# there's nothing a malicious origin could do that a curious one couldn't
# also do). Still made configurable via ALLOWED_ORIGINS for anyone who wants
# to lock it to their actual deployed frontend domain: set it in Render's
# environment variables as a comma-separated list, e.g.
# "https://your-app.vercel.app,http://localhost:5500".
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS")
_allowed_origins = [o.strip() for o in _allowed_origins_env.split(",")] if _allowed_origins_env else ["*"]
app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models/strawman")
def strawman():
    return models.fit_binary_strawman()


@app.get("/models/kaplan-meier")
def km():
    return models.naive_km_curve()


@app.get("/models/aalen-johansen")
def aj():
    return models.aalen_johansen_cif()


@app.get("/models/cox")
def cox():
    return models.cause_specific_cox()


@app.get("/models/rsf")
def rsf():
    return models.random_survival_forest()


@app.get("/models/all")
def all_models():
    return models.run_all()


@app.get("/tools/cif/{cause}")
def tool_cif(cause: str):
    try:
        return mcp_tools.get_cif(cause)  # type: ignore
    except KeyError:
        raise HTTPException(400, f"Unknown cause: {cause}")


@app.get("/tools/compare-causes")
def tool_compare():
    return mcp_tools.compare_causes()


@app.get("/tools/explain/{cause}")
def tool_explain(cause: str):
    return mcp_tools.explain_prediction(cause)  # type: ignore


@app.get("/tools/cif-ci/{cause}")
def tool_cif_ci(cause: str, n_boot: int = 200):
    try:
        return mcp_tools.get_cif_with_ci(cause, n_boot=n_boot)  # type: ignore
    except KeyError:
        raise HTTPException(400, f"Unknown cause: {cause}")


class AgentQuery(BaseModel):
    question: str


@app.post("/agent/ask")
async def agent_ask(q: AgentQuery):
    return await answer_question(q.question)


@app.get("/evaluation")
def evaluation(n_bootstrap: int = 100):
    # Lighter default than scripts/run_evaluation.py's 300 -- this is a
    # live, synchronous endpoint someone might hit from the frontend, so it
    # trades a bit of bootstrap precision for a faster response. Use the
    # script directly (or pass a higher n_bootstrap here) for the numbers
    # that actually go in a README.
    from backend.evaluation import evaluate_all
    return evaluate_all(n_bootstrap=n_bootstrap)


@app.get("/validation")
def validation():
    import pandas as pd
    from backend.data_utils import DATA_PATH
    from backend.data_validation import validate_raw

    # Reads the CSV directly (not via load_raw(), which raises on failure) --
    # this endpoint's whole purpose is to SHOW the report even when it fails,
    # not to throw an exception.
    df = pd.read_csv(DATA_PATH)
    report = validate_raw(df, raise_on_error=False)
    return {
        "passed": report.passed,
        "errors": [{"name": c.name, "message": c.message} for c in report.errors],
        "warnings": [{"name": c.name, "message": c.message} for c in report.warnings],
    }


@app.get("/descriptive")
def descriptive():
    # Schema-agnostic (see backend/descriptive.py) -- this endpoint doesn't
    # hardcode a single column name, so it keeps working if the dataset at
    # data/telco.csv is swapped for a different one that follows the same
    # loading convention (data_utils.load_raw()).
    from backend.descriptive import describe_dataset
    return describe_dataset()


class ProfilePredictRequest(BaseModel):
    # All fields optional -- unset ones are filled with population defaults
    # inside models.predict_for_profile(). This is the direct, structured
    # counterpart to the agent's NLP-driven instance-level path: a form
    # submission skips text parsing entirely and hands these fields
    # straight to the same underlying function.
    contract: str | None = None
    internet_service: str | None = None
    tech_support: str | None = None
    payment_method: str | None = None
    senior_citizen: bool | None = None
    dependents: str | None = None
    paperless_billing: str | None = None
    phone_service: str | None = None
    multiple_lines: str | None = None
    monthly_charges: float | None = None
    cause: str | None = None
    time_horizon_months: int = 12


@app.post("/predict-profile")
def predict_profile(req: ProfilePredictRequest):
    # Map the request's readable field names onto the dataset's actual
    # column names (see backend/data_utils.py: COVARIATES_CATEGORICAL/
    # COVARIATES_NUMERIC) -- only include a key when the field was actually
    # provided, so unset fields fall through to population defaults inside
    # predict_for_profile() rather than being overridden with None.
    field_to_column = {
        "contract": "Contract", "internet_service": "InternetService",
        "tech_support": "TechSupport", "payment_method": "PaymentMethod",
        "dependents": "Dependents", "paperless_billing": "PaperlessBilling",
        "phone_service": "PhoneService", "multiple_lines": "MultipleLines",
        "monthly_charges": "MonthlyCharges",
    }
    attributes = {}
    for field, column in field_to_column.items():
        value = getattr(req, field)
        if value is not None:
            attributes[column] = value
    if req.senior_citizen is not None:
        attributes["SeniorCitizen"] = int(req.senior_citizen)

    try:
        return mcp_tools.predict_customer_profile(attributes, req.cause, req.time_horizon_months)
    except KeyError:
        raise HTTPException(400, f"Unknown cause: {req.cause}")
