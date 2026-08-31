"""
Run this to get REAL timing numbers for your README, resume, and interview
answers -- not guesses.

    python scripts/benchmark_concurrency.py

It measures two genuinely different things, on purpose, because they use
two different concurrency tools for two different reasons:

  1. Bootstrapped CI (multiprocessing): CPU-bound work -- refitting the same
     statistical estimator on N resampled datasets. Speedup comes from
     spreading that work across CPU cores. This part always runs.

  2. Agent LLM synthesis (asyncio): I/O-bound work -- waiting on a network
     round-trip to the Anthropic API. Speedup comes from overlapping the
     WAITING, not from more CPU. This part only runs if ANTHROPIC_API_KEY
     is set, since it needs a real API key to make real calls.

If you can't explain why these two benchmarks use two different tools,
that's the thing to go back and understand before an interview -- not just
that the numbers came out favorably.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.data_utils import CAUSES, load_prepared
from backend.models import _fit_one_bootstrap, bootstrap_ci_for_cause


def bench_multiprocessing(n_boot: int = 200):
    print(f"\n--- Multiprocessing bootstrap CI ({n_boot} resamples, cause='{CAUSES[0]}') ---")

    result = bootstrap_ci_for_cause(CAUSES[0], n_boot=n_boot)
    parallel_seconds = result["parallel_seconds"]
    print(f"Pooled   ({result['n_processes']} processes): {parallel_seconds}s")

    # Same work, one process at a time, for a fair comparison.
    df = load_prepared()
    records = df.to_dict("records")
    cause_index = CAUSES.index(CAUSES[0]) + 1
    t0 = time.perf_counter()
    for seed in range(n_boot):
        _fit_one_bootstrap((records, cause_index, seed))
    sequential_seconds = time.perf_counter() - t0
    print(f"Sequential (1 process): {round(sequential_seconds, 3)}s")

    speedup = sequential_seconds / parallel_seconds if parallel_seconds else float("inf")
    print(f"Speedup: {round(speedup, 2)}x")
    return sequential_seconds, parallel_seconds, speedup


async def bench_async():
    import asyncio
    from backend.agent import answer_question

    print("\n--- Async LLM synthesis (asyncio.gather vs sequential awaits) ---")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Skipped: set ANTHROPIC_API_KEY to run this part with real API calls.")
        return

    questions = [
        "Why are customers on month-to-month contracts churning?",
        "What should we do about price-sensitive churners?",
        "How does tech support affect dissatisfaction churn?",
    ]

    t0 = time.perf_counter()
    for q in questions:
        await answer_question(q)
    sequential_seconds = time.perf_counter() - t0
    print(f"Sequential awaits:            {round(sequential_seconds, 3)}s")

    t0 = time.perf_counter()
    await asyncio.gather(*(answer_question(q) for q in questions))
    concurrent_seconds = time.perf_counter() - t0
    print(f"Concurrent (asyncio.gather):  {round(concurrent_seconds, 3)}s")

    speedup = sequential_seconds / concurrent_seconds if concurrent_seconds else float("inf")
    print(f"Speedup: {round(speedup, 2)}x")


if __name__ == "__main__":
    import asyncio

    bench_multiprocessing()
    asyncio.run(bench_async())
    print(
        "\nCopy the numbers above into your README / resume bullet -- e.g. "
        "'parallelized bootstrap CI computation across N cores, cutting "
        "runtime by Xx' -- with your own measured X, not this comment's."
    )
