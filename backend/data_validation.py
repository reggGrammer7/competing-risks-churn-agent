"""
Lightweight data-quality layer -- plain assertions, no Pandera/Great
Expectations. The point isn't the tooling, it's the mindset: don't trust
incoming data, and fail loudly with a specific reason instead of letting bad
rows silently corrupt a model fit somewhere downstream.

Deliberately NOT a validation framework -- just a list of (name, check)
pairs and a report. That's enough to demonstrate the engineering habit
without adding a new dependency for a project this size.
"""
from dataclasses import dataclass, field

REQUIRED_COLUMNS = [
    "customerID", "tenure", "Contract", "MonthlyCharges", "InternetService",
    "TechSupport", "PaymentMethod", "SeniorCitizen", "Dependents",
    "PaperlessBilling", "Churn Label", "Churn Reason",
]
VALID_CONTRACT = {"Month-to-month", "One year", "Two year"}
VALID_INTERNET = {"DSL", "Fiber optic", "No"}
VALID_TECH_SUPPORT = {"Yes", "No", "No internet service"}
VALID_PAYMENT = {"Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"}
VALID_CHURN_LABEL = {"Yes", "No"}
MAX_PLAUSIBLE_TENURE_MONTHS = 600  # 50 years -- generous, just catches obvious data-entry garbage


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: str  # "error" (blocks the pipeline) or "warning" (flagged, doesn't block)
    message: str


@dataclass
class ValidationReport:
    checks: list = field(default_factory=list)

    @property
    def errors(self):
        return [c for c in self.checks if c.severity == "error" and not c.passed]

    @property
    def warnings(self):
        return [c for c in self.checks if c.severity == "warning" and not c.passed]

    @property
    def passed(self):
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [f"Validation: {'PASSED' if self.passed else 'FAILED'} "
                 f"({len(self.errors)} error(s), {len(self.warnings)} warning(s))"]
        for c in self.checks:
            if not c.passed:
                lines.append(f"  [{c.severity.upper()}] {c.name}: {c.message}")
        return "\n".join(lines)


def _check(name, condition, severity, message):
    return CheckResult(name=name, passed=bool(condition), severity=severity, message=message)


def validate_raw(df, raise_on_error: bool = True) -> ValidationReport:
    """Runs every check against a raw (unprepared) dataframe -- call this
    BEFORE prepare()/get_design_matrix(), so bad data never reaches a model
    fit. Errors block the pipeline (raise by default); warnings are flagged
    but don't stop anything, since they're the kind of thing worth a human
    glance rather than an automatic failure."""
    checks = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    checks.append(_check(
        "required_columns_present", not missing_cols, "error",
        f"Missing required column(s): {missing_cols}" if missing_cols else "all present",
    ))
    if missing_cols:
        # Nothing else below can be checked meaningfully without these columns.
        report = ValidationReport(checks=checks)
        if raise_on_error and not report.passed:
            raise ValueError(report.summary())
        return report

    dup_ids = df["customerID"].duplicated().sum()
    checks.append(_check(
        "no_duplicate_customer_ids", dup_ids == 0, "error",
        f"{dup_ids} duplicate customerID value(s) found" if dup_ids else "no duplicates",
    ))

    negative_tenure = (df["tenure"] < 0).sum()
    checks.append(_check(
        "tenure_non_negative", negative_tenure == 0, "error",
        f"{negative_tenure} row(s) with negative tenure" if negative_tenure else "all non-negative",
    ))

    huge_tenure = (df["tenure"] > MAX_PLAUSIBLE_TENURE_MONTHS).sum()
    checks.append(_check(
        "tenure_plausible_range", huge_tenure == 0, "warning",
        f"{huge_tenure} row(s) with tenure over {MAX_PLAUSIBLE_TENURE_MONTHS} months -- "
        "likely a data-entry error, not necessarily wrong" if huge_tenure else "within range",
    ))

    bad_label = ~df["Churn Label"].isin(VALID_CHURN_LABEL)
    checks.append(_check(
        "churn_label_valid_values", bad_label.sum() == 0, "error",
        f"{bad_label.sum()} row(s) with Churn Label outside {VALID_CHURN_LABEL}" if bad_label.sum() else "valid",
    ))

    # The invariant that actually matters for competing-risks modeling: a
    # censored customer (Churn Label == No) must NOT have a reason, and a
    # churned customer (Churn Label == Yes) MUST have one -- this is the
    # direct equivalent of "is the churn date before the signup date": an
    # internally inconsistent label/reason pair would silently corrupt
    # which customers count as events vs. censored observations.
    yes_missing_reason = ((df["Churn Label"] == "Yes") & df["Churn Reason"].isna()).sum()
    no_has_reason = ((df["Churn Label"] == "No") & df["Churn Reason"].notna()).sum()
    checks.append(_check(
        "churn_label_reason_consistency",
        yes_missing_reason == 0 and no_has_reason == 0, "error",
        f"{yes_missing_reason} churned row(s) missing a reason, {no_has_reason} non-churned "
        f"row(s) with a reason set" if (yes_missing_reason or no_has_reason) else "consistent",
    ))

    null_required = df[REQUIRED_COLUMNS].drop(columns=["Churn Reason"]).isna().sum().sum()
    checks.append(_check(
        "no_unexpected_nulls", null_required == 0, "error",
        f"{null_required} unexpected null value(s) in required columns "
        "(Churn Reason is allowed to be null for non-churners)" if null_required else "none",
    ))

    for col, valid_set in [
        ("Contract", VALID_CONTRACT), ("InternetService", VALID_INTERNET),
        ("TechSupport", VALID_TECH_SUPPORT), ("PaymentMethod", VALID_PAYMENT),
    ]:
        bad = ~df[col].isin(valid_set)
        checks.append(_check(
            f"{col}_valid_categories", bad.sum() == 0, "warning",
            f"{bad.sum()} row(s) with an unrecognized {col} value: "
            f"{sorted(df.loc[bad, col].unique().tolist())[:5]}" if bad.sum() else "all recognized",
        ))

    non_positive_charges = (df["MonthlyCharges"] <= 0).sum()
    checks.append(_check(
        "monthly_charges_positive", non_positive_charges == 0, "warning",
        f"{non_positive_charges} row(s) with non-positive MonthlyCharges" if non_positive_charges else "all positive",
    ))

    report = ValidationReport(checks=checks)
    if raise_on_error and not report.passed:
        raise ValueError(report.summary())
    return report
