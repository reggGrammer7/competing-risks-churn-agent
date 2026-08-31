"""
Generates a synthetic dataset that mirrors the schema of the IBM Watson
Analytics "Telco Customer Churn" dataset (extended version with Churn Reason).

WHY SYNTHETIC: the real dataset isn't bundled here for licensing/access reasons.
Download the real one from Kaggle ("Telco Customer Churn IBM Watson Analytics
Churn Reason") and drop it at data/telco.csv with the same column names below —
everything downstream (backend/data_utils.py) reads from that path and doesn't
care whether the file is real or synthetic, as long as the columns match.

This generator builds in REAL competing-risks structure on purpose:
- tenure (duration)
- Churn Label (censoring indicator)
- Churn Reason (event type, only populated for churners) with a genuine
  cause-specific relationship to covariates, so the downstream models have
  something real to detect (not just noise).
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 4000

def generate(n=N):
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    monthly_charges = np.round(np.random.normal(65, 25, n).clip(18, 120), 2)
    internet = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
    tech_support = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.3, 0.5, 0.2])
    payment = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
    )
    senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
    dependents = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
    paperless = np.random.choice(["Yes", "No"], n, p=[0.6, 0.4])

    # --- cause-specific hazard construction (this is the "real" signal) ---
    # baseline monthly hazard for each competing cause, modulated by covariates
    h_price = 0.0012 + 0.0028 * (contract == "Month-to-month") + 0.00006 * (monthly_charges - 65).clip(min=0)
    h_service = 0.0008 + 0.0022 * (tech_support == "No") + 0.0010 * (internet == "Fiber optic")
    h_competitor = 0.0006 + 0.0012 * (payment == "Electronic check") + 0.0006 * (contract == "Month-to-month")
    h_other = np.full(n, 0.0004)  # moved / deceased / non-behavioral, roughly covariate-independent

    max_t = 72  # months, matches Telco's tenure range
    event_time = np.full(n, max_t, dtype=float)
    event_type = np.array(["Censored"] * n, dtype=object)

    for i in range(n):
        t = 0
        hazards = np.array([h_price[i], h_service[i], h_competitor[i], h_other[i]])
        causes = ["Price sensitivity", "Dissatisfaction", "Competitive loss", "Non-behavioral"]
        while t < max_t:
            t += 1
            # at each month, small chance of each competing event
            u = np.random.random(4)
            fired = u < hazards
            if fired.any():
                # if multiple "fire" same month, pick the one with smallest u/hazard ratio (closest to triggering)
                idx = np.argmin(u / hazards)
                event_time[i] = t
                event_type[i] = causes[idx]
                break
        # else stays censored at max_t

    churn_label = np.where(event_type == "Censored", "No", "Yes")

    df = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "tenure": event_time.astype(int),
        "Contract": contract,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": np.round(monthly_charges * event_time, 2),
        "InternetService": internet,
        "TechSupport": tech_support,
        "PaymentMethod": payment,
        "SeniorCitizen": senior,
        "Dependents": dependents,
        "PaperlessBilling": paperless,
        "Churn Label": churn_label,
        "Churn Reason": np.where(event_type == "Censored", np.nan, event_type),
    })
    return df

if __name__ == "__main__":
    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telco.csv")
    df = generate()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["Churn Label"].value_counts())
    print(df["Churn Reason"].value_counts(dropna=False))
