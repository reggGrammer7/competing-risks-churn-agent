"""
Proper held-out evaluation for the competing-risks models -- the single
biggest credibility gap this project had before this file existed: a
C-index computed on the SAME data a model was fit on tells you almost
nothing about whether it generalizes.

Evaluated PER CAUSE, not as one generic "did they churn" event -- the same
cause-specific framing used everywhere else in this project (an event of
a DIFFERENT cause is treated as censored for this cause's evaluation,
matching cause_specific_cox() in models.py).

Metrics reported per cause, per model:
  - C-index via IPCW (concordance_index_ipcw) -- like the plain C-index,
    measures ranking quality (do higher-risk-scored customers actually
    churn sooner), but corrects for bias from the censoring distribution
    instead of assuming it away.
  - Integrated Brier score (IBS) -- a metric C-index CAN'T give you:
    C-index only cares about relative ranking, so a model that ranks
    customers correctly but is badly CALIBRATED (systematically over- or
    under-confident about actual risk) still scores well on C-index. IBS
    penalizes that.
  - A naive baseline (identical risk score for every test customer) is
    evaluated alongside every real model, for the same reason a strawman
    classifier exists elsewhere in this project: it's the floor every real
    model should clearly beat. It should land at (or very near) 0.5 by
    construction, since a constant score can't discriminate between any
    two customers.
  - A bootstrap confidence interval around each model's C-index, computed
    by RESAMPLING THE TEST SET'S ALREADY-COMPUTED PREDICTIONS -- not by
    refitting the model. That's a deliberate difference from
    bootstrap_ci_for_cause() in models.py, which DOES refit the estimator
    on every resample: refitting is only necessary when you're
    bootstrapping an ESTIMATOR itself (a fitted curve). Here we're
    bootstrapping a fixed model's performance on a fixed test set, which
    is cheap enough to run as a plain sequential loop -- no
    multiprocessing needed, on purpose, to keep this lean.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from sksurv.ensemble import RandomSurvivalForest
from lifelines import CoxPHFitter

from backend.data_utils import load_prepared, get_design_matrix, CAUSES


def _cause_specific_arrays(df, cause_index):
    event = (df["event_type_code"] == cause_index).values
    time = df["tenure"].values.astype(float)
    return event, time


def _bootstrap_c_index_ci(y_train, event_test, time_test, risk_scores, tau, n_bootstrap, seed, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(event_test)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_test_boot = Surv.from_arrays(event=event_test[idx], time=time_test[idx])
        try:
            c, *_ = concordance_index_ipcw(y_train, y_test_boot, risk_scores[idx], tau=tau)
            scores.append(c)
        except Exception:
            continue  # an occasional resample can lack events entirely for IPCW; skip it
    if not scores:
        return [None, None]
    return [round(float(np.percentile(scores, 100 * alpha / 2)), 4),
            round(float(np.percentile(scores, 100 * (1 - alpha / 2))), 4)]


def evaluate_cause(cause: str, X_train, X_test, df_train, df_test, n_bootstrap: int = 300, seed: int = 0) -> dict:
    cause_index = CAUSES.index(cause) + 1
    event_train, time_train = _cause_specific_arrays(df_train, cause_index)
    event_test, time_test = _cause_specific_arrays(df_test, cause_index)
    y_train = Surv.from_arrays(event=event_train, time=time_train)
    y_test = Surv.from_arrays(event=event_test, time=time_test)

    # concordance_index_ipcw requires every test time to fall within the
    # training time range (it estimates the censoring distribution from
    # training data); tau caps evaluation at just under the training max so
    # a test customer with a longer tenure than anyone in training doesn't
    # raise a ValueError.
    tau = float(time_train.max()) * 0.999
    t_min = max(1.0, float(time_test.min()))
    t_max = min(float(time_test.max()) - 1, tau)
    brier_times = np.linspace(t_min, t_max, 15)

    results = {"cause": cause, "n_test": int(len(df_test)), "n_events_in_test": int(event_test.sum())}

    # --- naive baseline ---
    baseline_estimate = np.zeros(len(X_test))
    baseline_c, *_ = concordance_index_ipcw(y_train, y_test, baseline_estimate, tau=tau)
    results["baseline_c_index"] = round(float(baseline_c), 4)

    # --- Cox (cause-specific), same fitting logic as models.py's cause_specific_cox() ---
    d_train = X_train.copy()
    d_train["duration"] = time_train
    d_train["event"] = event_train.astype(int)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(d_train, duration_col="duration", event_col="event")
    cox_risk = cph.predict_partial_hazard(X_test).values

    cox_c, *_ = concordance_index_ipcw(y_train, y_test, cox_risk, tau=tau)
    cox_surv_at_times = cph.predict_survival_function(X_test, times=brier_times).values.T
    cox_ibs = integrated_brier_score(y_train, y_test, cox_surv_at_times, brier_times)

    results["cox_c_index"] = round(float(cox_c), 4)
    results["cox_c_index_ci"] = _bootstrap_c_index_ci(y_train, event_test, time_test, cox_risk, tau, n_bootstrap, seed)
    results["cox_integrated_brier_score"] = round(float(cox_ibs), 4)

    # --- Random Survival Forest (cause-specific) ---
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=10, n_jobs=-1, random_state=seed)
    rsf.fit(X_train, y_train)
    rsf_risk = rsf.predict(X_test)

    rsf_c, *_ = concordance_index_ipcw(y_train, y_test, rsf_risk, tau=tau)
    rsf_surv_fns = rsf.predict_survival_function(X_test, return_array=True)
    rsf_surv_at_times = np.array([np.interp(brier_times, rsf.unique_times_, row) for row in rsf_surv_fns])
    rsf_ibs = integrated_brier_score(y_train, y_test, rsf_surv_at_times, brier_times)

    results["rsf_c_index"] = round(float(rsf_c), 4)
    results["rsf_c_index_ci"] = _bootstrap_c_index_ci(y_train, event_test, time_test, rsf_risk, tau, n_bootstrap, seed)
    results["rsf_integrated_brier_score"] = round(float(rsf_ibs), 4)

    return results


def evaluate_all(test_size: float = 0.25, n_bootstrap: int = 300, seed: int = 0) -> dict:
    """Full held-out evaluation across all 4 causes -- entry point for the
    /evaluation API route and scripts/run_evaluation.py."""
    df = load_prepared()
    X = get_design_matrix(df)

    # Stratified by event_type_code (not a plain random split): with 4
    # causes plus censoring, a plain random split on a smaller cause could
    # leave a test fold with too few events of that cause to evaluate
    # meaningfully. Stratifying preserves each cause's proportion in both
    # the train and test folds.
    df_train, df_test, X_train, X_test = train_test_split(
        df, X, test_size=test_size, random_state=seed, stratify=df["event_type_code"]
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    by_cause = {
        cause: evaluate_cause(cause, X_train, X_test, df_train, df_test, n_bootstrap, seed)
        for cause in CAUSES
    }
    return {
        "test_size": test_size,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "n_bootstrap": n_bootstrap,
        "by_cause": by_cause,
    }
