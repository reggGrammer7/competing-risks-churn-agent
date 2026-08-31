"""
Regression test for the bootstrap resampling bug: df.sample(replace=True)
inevitably creates duplicate pandas index values, which broke lifelines'
internal tie-jitter step with a duplicate-label reindex error. Fixed with
.reset_index(drop=True) after resampling. See README.md.
"""
from backend.models import bootstrap_ci_for_cause
from backend.data_utils import CAUSES


def test_bootstrap_ci_runs_without_duplicate_index_error():
    """Small n_boot/n_jobs purely to keep this test fast -- the point is
    that it completes at all, not the specific CI values."""
    result = bootstrap_ci_for_cause(CAUSES[0], n_boot=10, n_jobs=2)
    assert result["cause"] == CAUSES[0]
    assert len(result["cif"]) == len(result["ci_lower"]) == len(result["ci_upper"])
    assert all(lo <= point <= hi for lo, point, hi in zip(result["ci_lower"], result["cif"], result["ci_upper"]))


def test_bootstrap_ci_bounds_are_ordered():
    """Lower bound should never exceed the upper bound at any timepoint --
    would have failed loudly under the original duplicate-index bug, since
    the fit wouldn't have completed at all."""
    result = bootstrap_ci_for_cause(CAUSES[1], n_boot=10, n_jobs=2)
    assert all(lo <= hi for lo, hi in zip(result["ci_lower"], result["ci_upper"]))
