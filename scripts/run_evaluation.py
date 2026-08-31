"""
Run this to get real, held-out evaluation numbers for your README's results
table -- not guesses, not training-set performance.

    python scripts/run_evaluation.py

Uses more bootstrap resamples than the live /evaluation API endpoint
(300 vs. 100 by default) since this runs offline and the numbers here are
the ones that should actually go in a README or resume bullet.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.evaluation import evaluate_all


def main():
    result = evaluate_all(n_bootstrap=300)
    print(f"\nHeld-out evaluation -- {result['n_train']} train / {result['n_test']} test "
          f"({result['test_size']*100:.0f}% held out), {result['n_bootstrap']} bootstrap resamples per CI\n")

    header = f"{'Cause':<20} {'Model':<10} {'C-index':>9} {'95% CI':>16} {'Int. Brier':>11} {'N events (test)':>16}"
    print(header)
    print("-" * len(header))
    for cause, r in result["by_cause"].items():
        print(f"{cause:<20} {'baseline':<10} {r['baseline_c_index']:>9.3f} {'':>16} {'':>11} {r['n_events_in_test']:>16}")
        print(f"{'':<20} {'cox':<10} {r['cox_c_index']:>9.3f} "
              f"{'[' + str(r['cox_c_index_ci'][0]) + ', ' + str(r['cox_c_index_ci'][1]) + ']':>16} "
              f"{r['cox_integrated_brier_score']:>11.4f} {'':>16}")
        print(f"{'':<20} {'rsf':<10} {r['rsf_c_index']:>9.3f} "
              f"{'[' + str(r['rsf_c_index_ci'][0]) + ', ' + str(r['rsf_c_index_ci'][1]) + ']':>16} "
              f"{r['rsf_integrated_brier_score']:>11.4f} {'':>16}")
        print()

    print(
        "Reading this: baseline should sit at ~0.5 (a constant risk score can't rank anyone).\n"
        "Cox/RSF meaningfully above baseline = real, learnable signal for that cause.\n"
        "Cox/RSF at or below baseline = no real signal beyond noise for that cause -- check\n"
        "whether that matches what you'd expect (e.g. a cause that's genuinely close to random\n"
        "with respect to your covariates SHOULD score near baseline; that's the framework\n"
        "working correctly, not a bug)."
    )
    print("\nCopy the numbers above into your README's results table -- these are real, not illustrative.")


if __name__ == "__main__":
    main()
