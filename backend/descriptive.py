"""
Schema-agnostic descriptive statistics over whatever dataset is currently
loaded -- this does NOT hardcode column names (Contract, InternetService,
etc.), on purpose, so it keeps working if you swap in a different dataset
later that follows the same loading convention (a CSV at data/telco.csv,
read via data_utils.load_raw()). It classifies every column as numeric or
categorical from its actual dtype and computes the appropriate summary for
each, rather than assuming this project's specific schema.

Deliberately reads the RAW (unprepared) data, not load_prepared()'s
encoded/collapsed version -- descriptive stats should describe what's
actually in the source file, not the modeling-ready design matrix.
"""
import numpy as np
import pandas as pd

from backend.data_utils import load_raw

MAX_CATEGORIES_SHOWN = 12  # beyond this, group the long tail into "Other"
NUMERIC_HISTOGRAM_BINS = 12


def _describe_numeric(series: pd.Series) -> dict:
    clean = series.dropna()
    counts, bin_edges = np.histogram(clean, bins=NUMERIC_HISTOGRAM_BINS)
    bin_labels = [f"{bin_edges[i]:.1f}\u2013{bin_edges[i+1]:.1f}" for i in range(len(bin_edges) - 1)]
    return {
        "type": "numeric",
        "count": int(clean.count()),
        "missing": int(series.isna().sum()),
        "mean": round(float(clean.mean()), 3),
        "median": round(float(clean.median()), 3),
        "std": round(float(clean.std()), 3) if len(clean) > 1 else 0.0,
        "min": round(float(clean.min()), 3),
        "max": round(float(clean.max()), 3),
        "histogram": {"labels": bin_labels, "counts": [int(c) for c in counts]},
    }


def _describe_categorical(series: pd.Series) -> dict:
    clean = series.dropna().astype(str)
    value_counts = clean.value_counts()
    top = value_counts.head(MAX_CATEGORIES_SHOWN)
    other_count = int(value_counts.iloc[MAX_CATEGORIES_SHOWN:].sum()) if len(value_counts) > MAX_CATEGORIES_SHOWN else 0
    labels = top.index.tolist()
    counts = top.values.tolist()
    if other_count > 0:
        labels.append(f"Other ({len(value_counts) - MAX_CATEGORIES_SHOWN} more categories)")
        counts.append(other_count)
    return {
        "type": "categorical",
        "count": int(clean.count()),
        "missing": int(series.isna().sum()),
        "n_unique": int(series.nunique(dropna=True)),
        "bar_chart": {"labels": labels, "counts": [int(c) for c in counts]},
    }


def describe_dataset() -> dict:
    """Full schema-agnostic description of the currently loaded dataset --
    every column classified by its actual dtype, not a hardcoded list."""
    df = load_raw()
    id_like_cols = [c for c in df.columns if df[c].nunique() == len(df) and df[c].dtype == object]

    columns = {}
    for col in df.columns:
        if col in id_like_cols:
            # An ID-like column (every value unique -- e.g. customerID)
            # isn't a "variable" in the descriptive-stats sense; skip it
            # rather than showing a meaningless 4000-category bar chart.
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            columns[col] = _describe_numeric(df[col])
        else:
            columns[col] = _describe_categorical(df[col])

    return {
        "n_rows": int(len(df)),
        "n_columns_described": len(columns),
        "id_like_columns_excluded": id_like_cols,
        "columns": columns,
    }
