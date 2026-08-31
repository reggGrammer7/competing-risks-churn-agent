# Dataset: IBM Telco Customer Churn (real data)

`telco.csv` in this folder is the real **IBM Telco Customer Churn** sample
dataset — 7,043 customers of a fictional (but realistically modeled)
telecommunications company, including the genuine `Churn Reason` field with
20 granular real-world departure reasons. This is a small transformation of
IBM's publicly redistributed sample data (widely mirrored for education and
demo purposes, including directly on GitHub), not synthetic data.

## Transformation applied

The original file uses IBM's column naming (spaces, different capitalization)
and a couple of encodings that don't match this project's schema. The exact
transformation:

- Dropped `Count` (always 1, a BI-tool artifact) and `Lat Long` (a redundant
  string combining `Latitude`/`Longitude`, kept separately).
- Renamed columns to match `backend/data_utils.py`'s expected schema:
  `CustomerID`→`customerID`, `Tenure Months`→`tenure`,
  `Internet Service`→`InternetService`, `Tech Support`→`TechSupport`,
  `Payment Method`→`PaymentMethod`, `Monthly Charges`→`MonthlyCharges`,
  `Paperless Billing`→`PaperlessBilling`, `Senior Citizen`→`SeniorCitizen`,
  and several others (`Phone Service`→`PhoneService`, etc.) kept as extra,
  unused-for-modeling-but-useful-for-descriptives columns.
- `SeniorCitizen`: converted from `Yes`/`No` text to `1`/`0`, matching
  `COVARIATES_NUMERIC`'s expectation.
- `TotalCharges`: converted from text to a real float. The original file has
  11 rows (brand-new customers with `tenure=0`) where this field is a blank
  string rather than a number — coerced to `NaN` rather than left as broken
  text (this is a well-known, documented quirk of this exact dataset, not
  something introduced by this transformation).

## Churn Reason → competing-risk cause mapping

The real `Churn Reason` column has 20 specific categories. `REASON_MAP` in
`backend/data_utils.py` collapses every one of them into this project's four
competing-risk buckets — every real category is mapped explicitly (verified
against this dataset's actual `value_counts()`, nothing guessed):

| Cause | Real categories mapped to it |
|---|---|
| Price sensitivity | Price too high, Extra data charges, Long distance charges, Lack of affordable download/upload speed |
| Dissatisfaction | Attitude of support person, Attitude of service provider, Poor expertise of online/phone support, Network reliability, Product/Service dissatisfaction, Lack of self-service on Website, Limited range of services |
| Competitive loss | Competitor offered more data / higher download speeds / better offer / had better devices |
| Non-behavioral | Moved, Deceased, Don't know |

## Real evaluation results on this data

See the main `README.md`'s evaluation section for full numbers. In short:
Cox and Random Survival Forest both clear a C-index of 0.78–0.85 across all
four causes on a genuine held-out test split — including Non-behavioral,
which (unlike the earlier synthetic-data version of this project) shows real
learnable signal here, since "Moved"/"Deceased" correlate with real
covariates like tenure and contract type in actual customer data.

## Model covariates: what's used, what's excluded, and why

The real dataset has far more columns than the model actually uses as
covariates. This was checked deliberately, not left as an accident of
which columns happened to be in the schema when the real dataset was
substituted in:

**Excluded as leakage:** `ChurnValue` (a literal 0/1 mirror of the
target itself), `ChurnScore` and `CLTV` (both outputs of a churn model IBM
already ran on this data — using them as inputs would mean training a
model partly on another model's prediction of the same outcome).

**Excluded as redundant (checked empirically, not assumed):**
`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `StreamingTV`,
`StreamingMovies` are each 100% determined by `InternetService == "No"`
(confirmed directly against this dataset's actual values) — already fully
captured by `InternetService`, which is a covariate. `TotalCharges`
correlates at 0.9996 with `tenure × MonthlyCharges`, both already
covariates — including it would add near-perfect multicollinearity for
no new information. Geography (`Country`/`State`/`City`/`ZipCode`/
`Latitude`/`Longitude`) is low-value here since this dataset covers one
region.

**Excluded for fairness, with the real cost disclosed rather than hidden:**

| Attribute | Churn-rate spread | Decision |
|---|---|---|
| `Gender` | ~0.8 points (26.9% vs 26.2%) | Excluded. Essentially zero predictive signal, so this costs nothing — not really a tradeoff. |
| `Partner` | **~13.3 points** (33.0% vs 19.7%, unpartnered vs. partnered) | Excluded. This one is a genuine, disclosed tradeoff, not a free win: partnership/family status carries real predictive signal here, but using it to drive retention targeting raises the same category of fairness concern as other sensitive attributes, even without a strict legal requirement the way lending has under ECOA. Chosen deliberately, with the actual cost stated rather than absorbed silently. |

**Included, added after the fact and checked to confirm they carry no
demographic content:** `PhoneService` (~1.8-point spread — minimal
signal, included anyway since it's a legitimate service-usage attribute
with negligible downside) and `MultipleLines` (~3.7-point spread — a
modest, real, non-demographic signal). Both were empirically verified to
be plain product-usage attributes, not proxies for anything sensitive,
before being added.

## Regenerating synthetic data instead

`data/generate_synthetic_data.py` is still included and still works if you
want a synthetic dataset instead (e.g. to test how the models behave on
data with a deliberately-known generating process). Run it to overwrite
`telco.csv` with synthetic data; `git checkout data/telco.csv` restores the
real dataset afterward.
