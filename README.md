# Competing-Risks Churn Decision-Support Agent

![CI](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-teal)
![LangGraph](https://img.shields.io/badge/LangGraph-agent-6d6bf5)

> Replace `YOUR-USERNAME/YOUR-REPO` above with your actual GitHub path once
> pushed, so the CI badge renders correctly.

A production-shaped system that models **why and when** customers churn as
a *competing-risks survival problem* — not a generic yes/no classifier —
and wraps that analysis in a **LangGraph agent** that answers plain-English
questions, grounded strictly in real model output.

Runs on the real **IBM Telco Customer Churn** dataset (7,043 customers,
including the genuine `Churn Reason` field with 20 real-world departure
categories) — not synthetic data. See [`data/README.md`](data/README.md)
for the dataset's provenance and the exact transformation applied.

**🔗 Live demo:** _add your Render/Vercel URLs here once deployed_
**📄 Full engineering history:** [`ENGINEERING_LOG.md`](ENGINEERING_LOG.md) — every real bug found and fixed, chronologically

---

## The problem this solves

Customer churn isn't one event — a customer who leaves because of price is
a fundamentally different problem than one who leaves for a competitor or
stops responding because they moved. A plain binary classifier collapses
all of this into "churned: yes/no" and mishandles customers who simply
haven't churned *yet* (censoring). This project treats churn as a
**competing-risks survival problem**: one customer, several possible
"causes of exit" racing against each other, with the added complication
that not knowing an outcome yet is itself information.

## What it actually does

Ask it a question in plain English:

> *"What's the price-sensitivity churn rate for fiber customers within the
> first year?"*

The agent extracts the **cause**, **time horizon**, and **customer
segment** from the question (not just the cause), refits the relevant
model on exactly that slice of the population, and answers with the real
number — not a canned summary. Ask something structurally different —
*"which cause grows fastest?"*, *"compare price sensitivity and
competitive loss"*, *"predict churn for a customer on a one-year
contract"* — and it correctly routes to a genuinely different computation,
not the same one dressed up.

## Architecture

```
Browser (frontend/)
    |  POST /agent/ask { "question": "..." }
    v
FastAPI (backend/main.py)
    v
LangGraph StateGraph (backend/agent.py)
    |
    +-- parse_query (backend/query_parser.py)
    |     English -> {cause, time_horizon_months, segment, task_type}
    |     NO modeling happens here -- extraction only
    |
    +-- route_after_parse (conditional edge)
    |     clarify | unsupported_segment | instance_level | population
    |
    +-- [parallel] call_analyze  +  retrieve_methodology
    |     backend/models.py: analyze_churn()          backend/rag/retriever.py
    |     filters the dataset to the segment,          FAISS+embeddings,
    |     refits Aalen-Johansen + Cox on JUST           falls back to TF-IDF
    |     that subset
    |
    +-- route_policy (conditional edge: fetch a policy doc only if asked)
    |
    +-- synthesize (the ONLY async node -- the one real network call)
          LLM-written (Anthropic/OpenAI) if a key is set,
          deterministic template otherwise -- either way,
          strictly grounded in the tool output above
```

Full file-by-file walkthrough, including why each design choice was made:
see the docstrings in each module, or the extended write-up referenced at
the bottom of this README.

## Features

| # | Tab | What it shows |
|---|---|---|
| 1 | Strawman classifier | Deliberate naive baseline (XGBoost + SHAP), included to show what's wrong with the obvious approach |
| 2 | Kaplan-Meier | All-cause survival curve — can't distinguish *why* |
| 3 | Cumulative incidence | The correct competing-risks estimator, per cause, with an overlay + auto-computed "leading cause by time range" |
| 4 | Cox hazard ratios | Cause-specific risk factors (explicitly labeled associational, not causal) |
| 5 | Random Survival Forest | Nonlinear comparison model |
| 6 | Ask the agent | Natural-language Q&A, structured-query-aware |
| 7 | Evaluation | Real held-out C-index (IPCW) + integrated Brier score, with bootstrap CIs |
| 8 | Data quality | Live validation report |
| 9 | Predict a customer | Instance-level prediction for one constructed profile |
| 10 | Descriptive analytics | Schema-agnostic dataset summary — works on any dataset following the same format |

## Tech stack

**Statistics / ML:** lifelines (Kaplan-Meier, Cox, Aalen-Johansen), scikit-survival (Random Survival Forest, IPCW C-index, integrated Brier score), XGBoost + SHAP
**Agent / retrieval:** LangGraph (real parallel fan-out + conditional branches), FAISS + sentence-transformers (with automatic TF-IDF fallback), Anthropic/OpenAI (optional, with a deterministic template fallback)
**Backend:** FastAPI, Pydantic, python-dotenv
**Concurrency:** `multiprocessing` for CPU-bound bootstrap resampling, `asyncio` for the one I/O-bound LLM call — deliberately different tools for genuinely different problems (see `ENGINEERING_LOG.md`)
**Frontend:** vanilla HTML/CSS/JS, no framework or build step
**Testing/CI:** pytest (69 tests), GitHub Actions (code tests + a model-performance regression gate + a Docker build check)

## Quickstart (local)

```bash
git clone <this-repo>
cd survival_churn_agent
pip install -r requirements.txt        # or: pip install -r requirements.txt --break-system-packages

uvicorn backend.main:app --reload
# -> http://127.0.0.1:8000/docs
```

Then open `frontend/index.html` (e.g. via VS Code's "Live Server"
extension). It already points at `http://127.0.0.1:8000` via
`frontend/config.js` — edit that one file if your backend runs elsewhere.

**Optional — LLM-written answers instead of the template fallback:** copy
`.env.example` to `.env` and fill in **one** of `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY`. Never paste a key directly into a `.py` file — `.env`
is already gitignored. Without either key, the agent uses a deterministic
template built from the same underlying numbers, so the whole system runs
end-to-end with zero external API dependency.

**First run needs internet once** (not after): the vector retriever
downloads a small embedding model on first use, then caches it. If that
fails for any reason (offline, firewalled), it automatically falls back to
TF-IDF retrieval — check `retriever.BACKEND` to see which one is active.

## Docker

```bash
docker compose up --build
# -> http://localhost:8000
```

## Deploying for real: Render (backend) + Vercel (frontend)

### Backend on Render

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, point it at the repo. `render.yaml` at
   the repo root configures everything (build command, start command,
   health check) automatically.
3. Set **one** environment variable in Render's dashboard —
   `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` — under the service's
   Environment tab. Neither is required.
4. Deploy. Render's free tier spins down when idle, so the first request
   after inactivity will be slower (cold start + possibly re-downloading
   the embedding model) — this is expected, not a bug.
5. Copy the deployed URL (`https://your-service.onrender.com`).

### Frontend on Vercel

1. In Vercel: **New Project**, import the same repo, set **Root
   Directory** to `frontend/`. `frontend/vercel.json` handles the rest —
   no build step needed (it's static HTML/JS/CSS).
2. **Before or after deploying**, edit `frontend/config.js`:
   ```js
   const API_BASE = "https://your-service.onrender.com";
   ```
   This is the *only* file that needs editing to point the frontend at a
   deployed backend — `app.js` itself never needs to change.
3. Deploy. Vercel gives you a URL immediately.

### Locking down CORS (optional, safe to skip)

The backend's CORS is wide open by default (`ALLOWED_ORIGINS` unset) —
fine for this project, since every route is read-only analytics with no
authentication. To restrict it anyway, set `ALLOWED_ORIGINS` in Render's
environment variables to a comma-separated list, e.g.:
```
https://your-app.vercel.app,http://localhost:5500
```

## Real dataset

This project uses the real **IBM Telco Customer Churn** dataset — 7,043
customers, with the genuine `Churn Reason` field (20 granular real-world
categories, collapsed into this project's four competing-risk buckets).
Full provenance, the exact column transformation applied, and the
reason→cause mapping: [`data/README.md`](data/README.md).

## Evaluation results

Real, held-out numbers (25% test split, 300 bootstrap resamples per CI) —
not illustrative placeholders. Reproduce with `python scripts/run_evaluation.py`.
(Reflects the current covariate set — see
[`data/README.md`](data/README.md#model-covariates-whats-used-whats-excluded-and-why)
for exactly which columns are used and why, including the deliberate,
disclosed exclusion of `Gender`/`Partner` on fairness grounds.)

| Cause | Model | C-index | 95% CI | Integrated Brier | Events (test) |
|---|---|---|---|---|---|
| Price sensitivity | baseline | 0.500 | — | — | 61 |
| | Cox | **0.785** | [0.727, 0.853] | 0.033 | |
| | RSF | 0.779 | [0.710, 0.840] | 0.031 | |
| Dissatisfaction | baseline | 0.500 | — | — | 198 |
| | Cox | **0.824** | [0.789, 0.860] | 0.078 | |
| | RSF | 0.816 | [0.775, 0.857] | 0.072 | |
| Competitive loss | baseline | 0.500 | — | — | 155 |
| | Cox | **0.847** | [0.819, 0.877] | 0.069 | |
| | RSF | 0.839 | [0.806, 0.867] | 0.063 | |
| Non-behavioral | baseline | 0.500 | — | — | 53 |
| | Cox | **0.863** | [0.828, 0.897] | 0.031 | |
| | RSF | 0.852 | [0.821, 0.884] | 0.028 | |

**Read this honestly, not optimistically.** Baseline sits at 0.5 by
construction (a constant risk score can't rank anyone). Every cause here
clears it decisively (0.78–0.86), including Non-behavioral — a real,
interesting finding: unlike an earlier synthetic-data version of this
project (where Non-behavioral was deliberately built covariate-independent
and correctly scored near baseline), *real* "Moved"/"Deceased" departures
do correlate with real covariates like tenure and contract type. The
evaluation framework reporting a strong number here is trustworthy
precisely because it also knows how to correctly report a weak one when
the data doesn't support a strong one — see `ENGINEERING_LOG.md` for that
story on the synthetic-data version.

## Testing & CI

```bash
python -m pytest tests/ -v
```

69 tests, each tied to a real bug found or a capability that needed proof
it worked — not coverage for its own sake. CI (`.github/workflows/ci.yml`)
runs three things on every push:

1. **The full test suite** — ordinary regression protection.
2. **A model-performance gate** — re-runs held-out evaluation and fails
   the build if a cause with real signal regresses toward the naive
   baseline. This uses survival-appropriate metrics (C-index via IPCW),
   not classification accuracy — the same *concept* as CI for any ML
   model, just the right metric for a survival problem.
3. **A Docker build check** — confirms the image still builds cleanly.

## Data quality

`backend/data_validation.py` — plain Python assertions (not a framework
like Great Expectations), run automatically before any model touches the
data. Checks required columns, duplicate IDs, negative tenure, and — the
one that matters most for this exact modeling problem — that a churned
customer has a stated reason and a non-churned customer doesn't. Inspect
the live report via `GET /validation` or the "Data quality" tab.

## Project structure

```
backend/
  data_utils.py        # load + validate + prepare data, design matrix
  data_validation.py   # plain-assertion data-quality checks
  query_parser.py       # English -> {cause, horizon, segment, task_type} -- no modeling
  models.py             # 5-model roster + analyze_churn() + predict_for_profile()
  evaluation.py          # held-out C-index (IPCW) + integrated Brier + bootstrap CI
  descriptive.py         # schema-agnostic dataset summary
  mcp_tools.py           # typed, single-purpose tool wrappers
  agent.py               # LangGraph StateGraph -- the whole agent
  main.py                # FastAPI routes, .env loading, CORS config
  rag/                   # FAISS+embeddings retriever with automatic TF-IDF fallback
frontend/
  index.html, app.js, style.css, config.js
data/
  telco.csv             # the real IBM dataset (see data/README.md)
  generate_synthetic_data.py   # optional synthetic alternative
scripts/
  benchmark_concurrency.py     # real multiprocessing/asyncio timing numbers
  run_evaluation.py             # real held-out evaluation numbers
tests/                   # 69 tests
.github/workflows/ci.yml # tests + model-quality gate + Docker build check
render.yaml               # backend deployment blueprint
```

## Honest limitations

- **Population/segment-level by default**, with a genuine (but separate)
  instance-level path for one constructed customer profile — see tab 9.
- **Stateless clarification, not multi-turn.** A too-vague question's
  *answer* is a clarifying question; the caller asks again with more
  detail. Real back-and-forth would need conversation state this
  architecture doesn't have.
- **No real MCP server yet** — `backend/mcp_tools.py` is structured
  exactly like one (typed, single-purpose functions) so wiring in the
  actual protocol later is a thin registration layer, not a rewrite.
- **No live cloud deployment baked into this repo** — `render.yaml` and
  `frontend/vercel.json` make it a few clicks away, but you'll need to
  actually click them.

Full chronological detail on every one of the above (and every bug that
led to a design decision) is in [`ENGINEERING_LOG.md`](ENGINEERING_LOG.md).
