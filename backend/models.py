"""
Model roster, in the order they should be understood/presented:

  1. XGBoost binary classifier (+ SHAP)   -- deliberate strawman
  2. Kaplan-Meier (all-cause)             -- naive survival baseline
  3. Aalen-Johansen CIF per cause         -- correct nonparametric baseline
  4. Cause-specific Cox PH                -- first correct parametric model
  5. Random Survival Forest               -- ML nonlinearity

(Fine-Gray and DeepHit are noted in the README as the natural next additions --
Fine-Gray needs a package with subdistribution-hazard support and DeepHit needs
`pycox`/torch, both left out here to keep this scaffold's install footprint
light. See README "Extending this" section.)

Every function returns plain dicts/lists (not fitted-object internals) so the
FastAPI layer can serialize them directly and the MCP tool layer can wrap them
with typed schemas without reaching into library internals.
"""
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

from backend.data_utils import load_prepared, get_design_matrix, CAUSES, COVARIATES_CATEGORICAL, COVARIATES_NUMERIC

_cache = {}


def _get_data():
    if "df" not in _cache:
        df = load_prepared()
        X = get_design_matrix(df)
        _cache["df"] = df
        _cache["X"] = X
    return _cache["df"], _cache["X"]


# ---------- 1. Strawman: binary classifier ----------
def fit_binary_strawman():
    df, X = _get_data()
    y = df["event_observed"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    model = xgb.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05, eval_metric="logloss")
    model.fit(X_train, y_train)
    auc = float(model.score(X_test, y_test))  # accuracy; README notes AUC would be reported properly too
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.iloc[:200])
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = sorted(zip(X.columns, mean_abs_shap.tolist()), key=lambda t: -t[1])[:8]
    _cache["binary_model"] = model
    _cache["binary_X_columns"] = list(X.columns)
    return {
        "model": "XGBoost binary churn classifier (strawman)",
        "test_accuracy": round(auc, 4),
        "top_shap_features": [{"feature": f, "mean_abs_shap": round(v, 4)} for f, v in importance],
        "caveat": (
            "Ignores censoring (treats all non-churners as a hard negative regardless of how "
            "long they've been observed) and cannot distinguish WHICH competing cause or WHEN."
        ),
    }


# ---------- 2 & 3. KM and Aalen-Johansen ----------
def naive_km_curve():
    df, _ = _get_data()
    kmf = KaplanMeierFitter()
    kmf.fit(df["tenure"], event_observed=df["event_observed"])
    sf = kmf.survival_function_.reset_index()
    return {
        "model": "Kaplan-Meier (naive all-cause)",
        "months": sf["timeline"].round(1).tolist()[::3],
        "survival_prob": sf.iloc[:, 1].round(4).tolist()[::3],
        "note": "Treats ALL churn causes as one event -- does not distinguish why.",
    }


def aalen_johansen_cif():
    import warnings
    df, _ = _get_data()
    results = {}
    for i, cause in enumerate(CAUSES, start=1):
        # seed=0: lifelines auto-jitters tied event times (inevitable here --
        # tenure is measured in whole months, so many customers share the
        # exact same duration) to satisfy the estimator's continuous-time
        # assumption. Unseeded, that jitter is a different tiny random
        # perturbation on every run; seeding it makes the fitted curve
        # bit-for-bit reproducible without changing what it estimates.
        # The warning is suppressed HERE ONLY (not globally) because we've
        # already deliberately accounted for it -- it's not being hidden,
        # it's being handled.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Tied event times were detected*")
            ajf = AalenJohansenFitter(calculate_variance=False, seed=0)
            ajf.fit(df["tenure"], df["event_type_code"], event_of_interest=i)
        cif = ajf.cumulative_density_.reset_index()
        cif.columns = ["t", "cif"]
        cif["month"] = cif["t"].round().astype(int)
        by_month = cif.groupby("month")["cif"].last().reindex(range(0, 67), method="ffill").fillna(0)
        results[cause] = {
            "months": by_month.index.tolist()[::3],
            "cif": by_month.round(4).tolist()[::3],
        }
    return {
        "model": "Aalen-Johansen cumulative incidence function (correct competing-risks baseline)",
        "by_cause": results,
    }


# ---------- 4. Cause-specific Cox ----------
def cause_specific_cox():
    df, X = _get_data()
    models = {}
    for i, cause in enumerate(CAUSES, start=1):
        d = X.copy()
        d["duration"] = df["tenure"]
        # cause-specific hazard: events of OTHER causes are treated as censored
        d["event"] = (df["event_type_code"] == i).astype(int)
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(d, duration_col="duration", event_col="event")
        summary = cph.summary.reset_index()
        top = summary.reindex(summary["coef"].abs().sort_values(ascending=False).index).head(5)
        models[cause] = [
            {
                "covariate": row["covariate"],
                "hazard_ratio": round(float(np.exp(row["coef"])), 3),
                "p_value": round(float(row["p"]), 4),
            }
            for _, row in top.iterrows()
        ]
        _cache[f"cox_{cause}"] = cph
    return {"model": "Cause-specific Cox proportional hazards", "top_covariates_by_cause": models}


# ---------- 5. Random Survival Forest ----------
def random_survival_forest():
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    df, X = _get_data()
    # RSF here modeled on the "any churn" event for simplicity in this scaffold;
    # a full build fits one RSF per cause the same way cause_specific_cox does.
    y = Surv.from_arrays(event=df["event_observed"].astype(bool), time=df["tenure"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=10, n_jobs=-1, random_state=0)
    rsf.fit(X_train, y_train)
    c_index = rsf.score(X_test, y_test)
    _cache["rsf_model"] = rsf
    _cache["rsf_X_columns"] = list(X.columns)
    return {
        "model": "Random Survival Forest (all-cause, extend to per-cause for full build)",
        "concordance_index": round(float(c_index), 4),
        "n_trees": 100,
    }


def run_all():
    return {
        "strawman": fit_binary_strawman(),
        "kaplan_meier": naive_km_curve(),
        "aalen_johansen": aalen_johansen_cif(),
        "cause_specific_cox": cause_specific_cox(),
        "random_survival_forest": random_survival_forest(),
    }


# ---------- Segment- and horizon-aware analysis (structured params only) ----------
MIN_SEGMENT_ROWS = 30
MIN_SEGMENT_EVENTS = 5


def analyze_churn(cause: str | list | None, time_horizon_months: int, segment: dict | None = None) -> dict:
    """
    The one function the agent's tool layer calls for a parsed query -- takes
    ONLY structured parameters (never raw question text; see
    backend/query_parser.py for where text becomes these parameters).
    Filters the dataset to the requested segment, fits Aalen-Johansen (for
    the CIF) and Cox (for risk factors) on JUST that filtered subset, and
    reports the CIF at the requested horizon alongside the full-horizon
    value -- this is what fixes the "always returns the whole-dataset
    summary regardless of what was asked" behavior.

    WHY REFIT PER SEGMENT INSTEAD OF REUSING THE CACHED WHOLE-DATASET FIT: a
    segment's hazard structure can genuinely differ from the population's --
    "new customers" or "fiber customers" isn't just a filtered VIEW of the
    same curve, it can BE a different curve. Reusing the cached whole-
    dataset model would silently ignore the segment entirely, which is
    exactly the bug this function exists to fix. The honest cost: this is
    slower than the cached path (no caching across different segment
    queries) -- acceptable for a project this size; a production version
    would cache per-segment fits or precompute a set of common segments.

    FAILS SAFELY on a segment too small to fit reliably (see
    MIN_SEGMENT_ROWS/MIN_SEGMENT_EVENTS) instead of returning a misleading
    number from too little data.
    """
    import warnings
    df, X = _get_data()

    if segment and segment.get("type") not in (None, "all", "unsupported"):
        if segment["type"] == "tenure_max":
            mask = df[segment["column"]] < segment["value"]
        elif segment["type"] == "tenure_min":
            mask = df[segment["column"]] >= segment["value"]
        elif segment["type"] == "category":
            mask = df[segment["column"]] == segment["value"]
        else:
            mask = pd.Series(True, index=df.index)
        df = df[mask].reset_index(drop=True)
        X = X[mask.values].reset_index(drop=True)

    n_customers = len(df)
    # cause can be: a single cause name, a LIST of specific cause names
    # (explicit multi-cause comparison, e.g. "compare price sensitivity and
    # competitive loss"), or None (report all four).
    if isinstance(cause, list):
        causes_to_run = cause
    else:
        causes_to_run = [cause] if cause else CAUSES
    by_cause = {}

    for c in causes_to_run:
        cause_index = CAUSES.index(c) + 1
        n_events = int((df["event_type_code"] == cause_index).sum())

        if n_customers < MIN_SEGMENT_ROWS or n_events < MIN_SEGMENT_EVENTS:
            by_cause[c] = {
                "insufficient_data": True,
                "n_customers": n_customers,
                "n_events": n_events,
                "message": (
                    f"Only {n_events} {c.lower()} event(s) among {n_customers} customers in this "
                    f"segment -- too few to fit a reliable model (minimum {MIN_SEGMENT_EVENTS} events, "
                    f"{MIN_SEGMENT_ROWS} customers)."
                ),
            }
            continue

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Tied event times were detected*")
            ajf = AalenJohansenFitter(calculate_variance=False, seed=0)
            ajf.fit(df["tenure"], df["event_type_code"], event_of_interest=cause_index)
        cif = ajf.cumulative_density_.reset_index()
        cif.columns = ["t", "cif"]
        cif["month"] = cif["t"].round().astype(int)
        max_month = max(1, int(df["tenure"].max()))
        by_month = cif.groupby("month")["cif"].last().reindex(range(0, max_month + 1), method="ffill").fillna(0)
        months_arr = by_month.index.to_numpy()
        cif_arr = by_month.to_numpy()

        horizon = min(time_horizon_months, max_month)
        horizon_was_capped = time_horizon_months > max_month
        cif_at_horizon = float(np.interp(horizon, months_arr, cif_arr))
        cif_full_horizon = float(cif_arr[-1])

        drivers = []
        try:
            d = X.copy()
            d["duration"] = df["tenure"].values
            d["event"] = (df["event_type_code"] == cause_index).astype(int).values
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(d, duration_col="duration", event_col="event")
            summary = cph.summary.reset_index()
            top = summary.reindex(summary["coef"].abs().sort_values(ascending=False).index).head(3)
            drivers = [
                {"covariate": row["covariate"], "hazard_ratio": round(float(np.exp(row["coef"])), 3)}
                for _, row in top.iterrows()
            ]
        except Exception:
            drivers = []  # Cox can fail to converge on a small/skewed segment; the CIF result still stands

        by_cause[c] = {
            "insufficient_data": False,
            "n_customers": n_customers,
            "n_events": n_events,
            "cif_at_horizon": round(cif_at_horizon, 4),
            "horizon_months_used": horizon,
            "requested_horizon_months": time_horizon_months,
            "horizon_was_capped": horizon_was_capped,
            "cif_full_horizon": round(cif_full_horizon, 4),
            "full_horizon_months": max_month,
            "months_curve": months_arr.tolist()[::3] or months_arr.tolist(),
            "cif_curve": (np.round(cif_arr, 4).tolist()[::3] or np.round(cif_arr, 4).tolist()),
            "top_drivers": drivers,
            "model_used": "Aalen-Johansen + Cox (cause-specific, refit on this segment)",
        }

    return {"n_customers_in_segment": n_customers, "by_cause": by_cause}


# ---------- 6. Bootstrapped confidence intervals (multiprocessing) ----------
#
# WHY MULTIPROCESSING HERE, NOT ASYNCIO: a bootstrap CI means refitting the
# SAME estimator on N resampled datasets and looking at the spread of curves
# you get back. Each resample is real CPU work (no waiting on a network or a
# disk) -- so the only way to actually speed it up is to run resamples on
# separate CPU cores at the same time. asyncio's cooperative multitasking
# only helps when a task is *waiting* on something external; there's no wait
# to overlap here, so asyncio would not make this faster. That's the
# dividing line: CPU-bound, repeated, independent work -> multiprocessing.
#
# This has to be a MODULE-LEVEL function (not a closure or method) because
# multiprocessing.Pool pickles the function and its arguments to send them
# to each worker process -- closures and bound methods generally can't be
# pickled.
def _fit_one_bootstrap(payload: tuple) -> list:
    """One bootstrap resample: refit Aalen-Johansen on a resampled dataset
    and return its CIF curve. Runs inside a worker process."""
    df_records, cause_index, seed = payload
    import pandas as pd
    from lifelines import AalenJohansenFitter

    df = pd.DataFrame.from_records(df_records)
    # reset_index after sampling with replacement: duplicate index values
    # (inevitable when the same row gets drawn twice) break lifelines'
    # internal jitter step, which reindexes on the duration index.
    resampled = df.sample(n=len(df), replace=True, random_state=seed).reset_index(drop=True)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tied event times were detected*")
        ajf = AalenJohansenFitter(calculate_variance=False, seed=seed)
        ajf.fit(resampled["tenure"], resampled["event_type_code"], event_of_interest=cause_index)
    cif = ajf.cumulative_density_.reset_index()
    cif.columns = ["t", "cif"]
    cif["month"] = cif["t"].round().astype(int)
    by_month = cif.groupby("month")["cif"].last().reindex(range(0, 67), method="ffill").fillna(0)
    return by_month.tolist()[::3]


def bootstrap_ci_for_cause(cause: str, n_boot: int = 200, alpha: float = 0.05, n_jobs: int | None = None) -> dict:
    """Bootstrapped confidence band around the Aalen-Johansen CIF for one
    cause, computed by refitting on N resampled datasets in parallel across
    CPU cores. Returns the point estimate plus lower/upper CI bounds at each
    timepoint, and the wall-clock time actually spent in the parallel pool
    (so you have a real number to quote, not a guess)."""
    import time
    from multiprocessing import Pool, cpu_count

    if cause not in CAUSES:
        raise KeyError(f"Unknown cause: {cause}")

    df, _ = _get_data()
    cause_index = CAUSES.index(cause) + 1
    records = df.to_dict("records")
    payloads = [(records, cause_index, seed) for seed in range(n_boot)]
    n_jobs = n_jobs or cpu_count()

    t0 = time.perf_counter()
    with Pool(processes=n_jobs) as pool:
        curves = pool.map(_fit_one_bootstrap, payloads)
    parallel_seconds = time.perf_counter() - t0

    arr = np.array(curves)  # shape: (n_boot, n_timepoints)
    lower = np.percentile(arr, 100 * alpha / 2, axis=0)
    upper = np.percentile(arr, 100 * (1 - alpha / 2), axis=0)
    point = aalen_johansen_cif()["by_cause"][cause]

    return {
        "cause": cause,
        "months": point["months"],
        "cif": point["cif"],
        "ci_lower": np.round(lower, 4).tolist(),
        "ci_upper": np.round(upper, 4).tolist(),
        "n_bootstrap": n_boot,
        "n_processes": n_jobs,
        "parallel_seconds": round(parallel_seconds, 3),
    }


# ---------- Instance-level prediction (one constructed profile, not a population) ----------
def predict_for_profile(attributes: dict, cause: str | None, time_horizon_months: int) -> dict:
    """
    Instance-level prediction for ONE constructed customer profile -- the
    RSF/Cox equivalent of analyze_churn(), applied to a single row instead
    of a population/segment. This is the extension flagged throughout this
    project's README as "the models are technically capable of this,
    nothing currently exposed it" -- this function is that extension.

    `attributes` holds ONLY the covariates the question actually specified
    (e.g. {"Contract": "One year"}). Every other covariate is filled with
    the population's mean (numeric) or mode (categorical) -- this builds a
    REPRESENTATIVE profile, not a real individual customer, and that
    distinction is returned explicitly in the payload (which fields were
    stated vs. filled in) rather than left implicit. Silently presenting a
    population-filled profile as "this specific customer's prediction"
    would overclaim precision the inputs don't support.
    """
    df, X = _get_data()

    # Ensure the whole-population RSF (and the relevant cause-specific Cox,
    # if a cause was requested) are fitted and cached. Cheap on repeat
    # calls within a process since _get_data() is already cached; the
    # model fits themselves aren't cached across separate analyze_churn()
    # segment queries by design (segments differ), but the WHOLE-population
    # RSF/Cox used here is the same one models.py's own tabs already use,
    # so reusing it (rather than refitting per instance query) is both
    # correct and free after the first call.
    if "rsf_model" not in _cache:
        random_survival_forest()
    if cause and f"cox_{cause}" not in _cache:
        cause_specific_cox()

    profile_raw = {}
    filled_defaults = {}
    for col in COVARIATES_CATEGORICAL:
        if col in attributes:
            profile_raw[col] = attributes[col]
        else:
            profile_raw[col] = df[col].mode().iloc[0]
            filled_defaults[col] = profile_raw[col]
    for col in COVARIATES_NUMERIC:
        if col in attributes:
            profile_raw[col] = attributes[col]
        else:
            profile_raw[col] = float(df[col].mean())
            filled_defaults[col] = round(profile_raw[col], 2)

    profile_df = pd.DataFrame([profile_raw])
    profile_X = pd.get_dummies(profile_df[COVARIATES_CATEGORICAL])
    profile_X[COVARIATES_NUMERIC] = profile_df[COVARIATES_NUMERIC]

    rsf = _cache["rsf_model"]
    rsf_cols = _cache["rsf_X_columns"]
    # One-hot columns present in the full training data but not triggered
    # by this single row (e.g. a Contract category this profile doesn't
    # have) need to exist as an explicit 0 column, not be silently
    # dropped -- reindex does that.
    profile_X_rsf = profile_X.reindex(columns=rsf_cols, fill_value=0.0).astype(float)

    surv_fn = rsf.predict_survival_function(profile_X_rsf, return_array=True)[0]
    times = rsf.unique_times_
    max_month = int(times.max())
    horizon = min(time_horizon_months, max_month)
    survival_prob_at_horizon = float(np.interp(horizon, times, surv_fn))
    below_half = times[surv_fn <= 0.5]
    median_expected_tenure_months = float(below_half[0]) if len(below_half) else None

    result = {
        "profile_specified": attributes,
        "profile_filled_with_population_defaults": filled_defaults,
        "horizon_months_used": horizon,
        "full_horizon_months": max_month,
        "survival_probability_at_horizon": round(survival_prob_at_horizon, 4),
        "churn_probability_at_horizon": round(1 - survival_prob_at_horizon, 4),
        "median_expected_tenure_months": median_expected_tenure_months,
        "model_used": "Random Survival Forest (population-fitted, applied to one constructed profile)",
    }
    if median_expected_tenure_months is None:
        result["median_expected_tenure_note"] = (
            f"This profile's predicted survival probability never drops to 50% within the "
            f"{max_month}-month observation window -- most customers with this profile are "
            f"expected to still be active at month {max_month}, so no median tenure can be reported."
        )

    if cause:
        cph = _cache.get(f"cox_{cause}")
        if cph is not None:
            cox_cols = list(cph.params_.index)
            profile_X_cox = profile_X.reindex(columns=cox_cols, fill_value=0.0).astype(float)
            try:
                risk = float(cph.predict_partial_hazard(profile_X_cox).iloc[0])
                result["cause"] = cause
                result["cause_specific_relative_risk"] = round(risk, 3)
                result["cause_specific_relative_risk_note"] = (
                    "Relative to the average customer in the training population (partial hazard = 1.0 "
                    "would mean average risk; this profile's value shows how much higher/lower)."
                )
            except Exception:
                pass  # Cox can fail on a profile combination it never saw fit converge for; RSF result still stands

    return result
