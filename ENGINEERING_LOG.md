# Engineering Log: Every Real Bug Found and Fixed

This is the honest, chronological history of this project's development --
not a cleaned-up narrative. Every entry below documents a real problem that
was actually encountered, how it was diagnosed, exactly which file(s)
changed, and what the fix was. Kept separate from the main `README.md` so
that document stays scannable for a first-time visitor, while this history
stays available for anyone (including an interviewer) who wants the full
story.

**Why this file exists at all:** most of the debugging insight in this
project came from real usage, not upfront design -- and that's a better,
more specific interview story than "I built X" ever is. The recurring
"generic-word-collision" bug family in particular (the same underlying
mistake made three separate times, with three different words) is one of
the more valuable patterns documented here.

---

## Fixed: cause detection broke in CI (and possibly in production) because of an environment difference nothing local could catch

**Symptom:** 11 tests failed in GitHub Actions CI that passed locally --
generic questions like "What is the overall churn rate?" started matching
a specific cause instead of correctly returning "no cause mentioned," and
"They left because of cost" started matching "Competitive loss" instead of
"Price sensitivity."

**Root cause, and why local testing could never have caught it:**
`backend/query_parser.py` detects a question's cause via
`backend/rag/retriever.py`, which auto-selects FAISS+embeddings when
available and falls back to TF-IDF otherwise. `CAUSE_MATCH_THRESHOLD` and
every "generic word collision" fix in `cause_docs.md` (see below) were
tuned and tested exclusively against TF-IDF's cosine-similarity scoring --
because the environment this project was developed in could never reach
`huggingface.co` to download the embedding model, so cause detection was
*always* silently running on the TF-IDF fallback during development, no
matter which backend "should" have been active. GitHub Actions runners
have normal internet access, so in CI the embedding model downloaded
successfully for the first time -- and dense-embedding similarity scores
behave completely differently from TF-IDF's exact-word-overlap scoring,
especially against `cause_docs.md`, which is a keyword LIST, not natural
prose (exactly the content TF-IDF suits and embeddings don't reliably).
The uncalibrated threshold broke cause routing the moment a real
embedding-capable environment actually ran the code.

**Why this could have been a live production bug, not just a CI
annoyance:** any deployment environment with normal internet access
(Render included) can also download the embedding model -- meaning the
live agent could have been silently misrouting cause detection for real
users, with every local test appearing to pass the whole time.

**The fix:** `backend/query_parser.py` now imports `tfidf_retriever`
directly for cause detection, bypassing the auto-selecting `retriever`
module entirely for this one purpose. Document retrieval (methodology/
policy docs) is unaffected and still free to use FAISS+embeddings when
available, since that path has no hardcoded numeric thresholds to miscalibrate.
Regression test added (`test_cause_detection_is_pinned_to_tfidf_not_the_auto_selected_backend`
in `tests/test_session_fixes.py`) that inspects the actual import statement,
so this can't silently regress back to the auto-selecting backend.

**The broader lesson:** a test suite that always runs in the same
environment can hide a bug that only a *different* environment exposes --
here, specifically, "my sandbox has no internet access" was doing
invisible, unintentional work keeping cause detection on a code path that
happened to be well-calibrated. The fix isn't "add internet access
everywhere" -- it's recognizing that FAISS+embeddings and TF-IDF are
genuinely different tools suited to genuinely different kinds of content
(natural prose vs. keyword lists), and pinning each retrieval task to the
tool actually validated for it, rather than letting an auto-selecting
fallback silently determine which one runs.

**Fixed since the first pass** (worth knowing the "why" cold, since these are
natural interview follow-ups):
- **The agent always summarized the WHOLE dataset regardless of what was
  asked** — "what's the churn rate in 3 months" and "what's the overall
  churn rate" produced the same answer, because nothing ever extracted the
  "3 months," and no segment mention (e.g. "new customers," "fiber
  customers") changed which rows the models were fit on. This was the
  single biggest gap: an agent that looks like it's answering the specific
  question but is actually answering a generic one every time. Fixed with
  a real extract-then-compute split — see "Structured query parsing"
  below, the largest architectural change in this project since the
  LangGraph rewrite.
- **Cause routing was a hardcoded keyword list that silently defaulted to
  "Non-behavioral" for any question it didn't recognize** — every
  unrecognized phrasing got the same answer. Replaced with the same TF-IDF
  retrieval mechanism already used for doc lookups (`backend/agent.py:
  _detect_cause`, corpus in `backend/rag/cause_docs.md`), so cause-detection
  and doc-retrieval are one general-purpose mechanism instead of two
  different ad-hoc ones. Low-confidence matches are now surfaced honestly in
  the answer and the trace instead of hidden.
- **Two cause docs were missing their own exact cause name as a token** —
  `cause_docs.md`'s "Price sensitivity" and "Competitive loss" entries
  never literally contained the words "sensitivity" or "competitive"/"loss,"
  so asking about a cause by its exact name could under-match. Fixed by
  adding the literal cause-name tokens to each doc.
- **A generic word ("rate") falsely matched "churn rate" to Price
  sensitivity** — the word was in the pricing doc for "rate increase," but
  matched any mention of a generic "churn rate" too. Removed the bare word,
  kept the specific phrase.
- **Bootstrap resampling hit a duplicate-index bug** in lifelines' internal
  jitter step (`df.sample(replace=True)` produces duplicate index values by
  construction) — fixed with `.reset_index(drop=True)` after resampling.
- **`_wants_recommendation()` used naive substring matching** (`"action" in
  question`), which false-positived on any question containing the word
  "dissatisf**action**" — same class of bug as the cause-router issue above,
  just in a different function. Fixed with word-boundary regex (`\baction\b`).
- **Tied event times in Aalen-Johansen** (`tenure` is measured in whole
  months, so many customers share the exact same duration) triggered
  lifelines' auto-jitter warning on every run, and that jitter was unseeded
  so results weren't bit-for-bit reproducible run to run. Seeded it
  (`AalenJohansenFitter(seed=0)` / `seed=<bootstrap resample's own seed>`)
  and suppressed the now-handled warning at the fit site only.
- **No held-out evaluation existed** — every metric reported was on the
  same data the model was fit on, which tells you almost nothing about
  generalization. Added `backend/evaluation.py`: a real train/test split,
  cause-specific C-index (IPCW-corrected) and integrated Brier score for
  Cox and RSF, a naive constant-risk baseline every model should clearly
  beat, and a bootstrap CI around each C-index. See "Evaluation results"
  below.
- **No data-quality layer existed** — a malformed row (negative tenure, a
  duplicate customer ID, a churned customer with no reason given) could
  have silently reached a model fit. Added `backend/data_validation.py`,
  wired into `data_utils.load_raw()` so it runs before anything else
  touches the data.
- **All of the above now have regression tests** (`tests/`, 32 total) so
  none of these bugs can silently come back.

## New: instance-level prediction tab, descriptive analytics tab, and the "3%" fixes

**Two new frontend tabs, backed by new endpoints:**
- **"Predict a customer"** (tab 9) — a form exposing `models.predict_for_profile()`
  directly via `POST /predict-profile`, bypassing NLP parsing entirely (the
  agent's instance-level path still works too, for natural-language
  questions). Unset fields fall through to population defaults, and the
  result explicitly shows which fields were stated vs. filled.
- **"Descriptive analytics"** (tab 10) — `backend/descriptive.py` +
  `GET /descriptive`. Deliberately schema-agnostic: every column is
  classified as numeric or categorical from its actual pandas dtype, not a
  hardcoded list of this project's column names, so it keeps describing
  correctly if `data/telco.csv` is swapped for a different dataset that
  follows the same loading convention. ID-like columns (every value
  unique, e.g. `customerID`) are automatically excluded rather than shown
  as a meaningless 4000-category bar chart.

**Bugs found from real usage, fixed:**
- **"Which churn cause dominates?" sometimes answered Price sensitivity,
  sometimes Competitive loss.** Root cause: the word "reason" — used
  constantly as a plain synonym for "cause" — was living in only the
  Non-behavioral doc (leftover from "no clear reason", "other reasons"
  phrasing), so any question using "reason" instead of "cause" got a false
  pull toward Non-behavioral, silently changing which single cause got
  compared against what. Exact same bug class as the earlier "customers"
  and "rate" collisions. Fixed by rewording the Non-behavioral doc to avoid
  the bare word entirely; "cause" and "reason" phrasings now agree.
- **"Which cause is most dominant?" and "which cause grows faster?" were
  answered with the same computation** (highest cumulative incidence),
  even though they're genuinely different questions — magnitude vs. rate
  of increase. A cause can have a lower absolute CIF while still growing
  faster. Added a real, separate growth-rate calculation (average CIF
  increase per month over the observed window); "dominant" and "grows
  faster" can now legitimately point to different causes when the data
  says so, instead of one masquerading as the other.
- **"Dominates" (verb) didn't match** the ranking-question regex, which
  only had the adjective "dominant." Broadened.
- **"What is churn at 'banana' months?" silently defaulted to 12 months**
  and answered as if nothing was wrong. Now detected explicitly as an
  unparseable horizon attempt and answered with a direct "I couldn't
  understand 'banana' as a time horizon" message instead of guessing.
  Vague-but-sensible phrasing ("in a few months," "in the coming months")
  is deliberately NOT flagged — imprecise isn't the same as nonsensical.
- **Empty query** was already handled correctly (confirmed, not a bug) —
  worth knowing it doesn't need fixing.
- **Markdown asterisks showing up literally** (`**Price sensitivity**`)
  instead of rendering as bold. The agent's answers use `**bold**` for
  emphasis (both the template's ranking feature and an LLM's own writing
  style), but the frontend was inserting that text as-is. Added
  `renderAnswerText()` in `frontend/app.js`: escapes HTML first, then
  converts `**text**` into real `<strong>` tags.
- **The agent question box didn't wrap or grow** — a single-line `<input>`
  scrolled horizontally on a long question, hiding earlier text. Replaced
  with an auto-growing `<textarea>` (grows taller as you type; Enter
  submits, Shift+Enter for a literal newline).
- **14 new regression tests** (`tests/test_session_fixes.py`) — 69 total,
  up from 55.

**A question worth a direct answer rather than a feature:** "plot churn
risk at 3, 6, 12, and 24 months" doesn't render a chart — it returns the
value at each requested month as text. That's a deliberate choice, not a
gap: the agent's interface is a text answer, and a clear list of values
at each month is a complete, honest response to what's actually being
asked. Building a dynamic chart-from-chat-answer pipeline for one rare
phrasing would be a lot of engineering for little payoff, especially since
the CIF tab (tab 3) already gives a real, properly-labeled plot for any
single cause or all four overlaid — if a visual is wanted for a specific
multi-month question, that tab is the more direct path to it today.

## Fixed: OpenAI parameter error + ranking questions (earlier round)

- **`OpenAI synthesis failed: Unsupported parameter 'max_tokens'`** —
  `gpt-5.4-mini` (and OpenAI's other newer-generation models) reject the
  older `max_tokens` parameter outright, requiring `max_completion_tokens`
  instead. This only surfaced once a real `OPENAI_API_KEY` was actually in
  use — the template fallback caught the failure gracefully and showed the
  error inline exactly as designed, which is how it got noticed. Fixed in
  `_synthesize_with_openai()`; regression test confirms the correct
  parameter name via source inspection (no live API call needed to catch
  this if it ever comes back). `max_tokens` is still correct for the
  Anthropic path — this was an OpenAI-specific API change, not a rename.
- **Superlative/ranking questions** ("which cause is most dominant," "what's
  the biggest driver") used to just list all four causes' numbers and leave
  the reader to compare them manually — technically correct data, but not
  actually answering the question asked. Added `_wants_ranking()`: when
  detected, the answer now explicitly names the highest-scoring cause up
  front, in both the template and LLM paths. Single-cause questions with
  ranking language ("is X the highest cause?") correctly skip the extra
  line, since there's nothing to rank against.
- **Added an explicit LLM prompt scope-lock** for single-cause questions:
  the prompt now names exactly which cause is in scope and directly
  forbids the model from mentioning the other three, rather than relying
  on it to infer that boundary just from which data it was handed.

## Fixed after real-world test questions (earlier round)

I ran roughly 60 test questions against the agent myself, covering
implicit horizons, cause synonyms, segment extraction, and ambiguity
handling. Fixes below, in the order I found them:

- **"customers" was a silent false-positive trigger.** An earlier fix added
  "price sensitive customers" to the Price-sensitivity doc so the exact
  cause name would match itself -- but "customers" is a generic word that
  now appeared in ONLY that one doc, so any question containing the word
  "customers" (nearly all of them) got a small false similarity to Price
  sensitivity. Same bug class as an earlier "rate" collision: a generic
  word left in only one doc masquerades as a distinguishing signal. Fixed
  by removing it; regression test in `tests/test_multi_cause.py`.
- **Multi-cause comparison** ("compare price sensitivity and competitive
  loss", "how does dissatisfaction compare to price sensitivity") wasn't
  supported at all -- `parse_cause()` only ever returns one best match.
  Added `parse_causes()` (checks all 4 causes independently, returns every
  one that clears a threshold) and extended `models.analyze_churn()` to
  accept a list of causes directly, not just one-or-all.
- **Instance-level questions** ("predict churn for a customer on a one-year
  contract," "how long is a month-to-month customer expected to stay")
  were being treated as population/segment questions with a filter, which
  is the wrong question entirely -- singular "a customer" wants ONE
  prediction, not a filtered summary. Added a whole new path: detection
  (`is_instance_level_query`), attribute extraction reusing the existing
  segment patterns, and `models.predict_for_profile()`, which applies the
  already-fitted population RSF (and cause-specific Cox, if a cause was
  named) to one constructed row -- filling anything unstated with
  population defaults, and reporting explicitly which fields were stated
  vs. filled, so nothing is silently presented as more precise than it is.
  Ambiguity is handled per the original spec exactly: "a STANDARD/average
  customer" with no other attributes proceeds with population defaults;
  "THIS customer" with nothing else stated asks "What segment and
  contract?" instead of guessing.
- **Unsupported segments/attributes ("premium," "VIP," "student") used to
  silently fall back to running the full population analysis** with a
  caveat buried in the trace nobody would read. Now short-circuits straight
  to a direct answer ("this dataset doesn't have a 'premium customers'
  attribute...") with no model calls at all -- both for population
  questions and instance-level ones.
- **Implicit horizon phrases** ("early in the lifecycle," not just "early
  churn") were missing from the horizon regex table; broadened.
- **"At month N" and multi-month phrasing** ("CIF at month 24," "at 3, 6,
  12, and 24 months") weren't recognized at all; added.
- **An out-of-range horizon** ("what is churn risk at month 200?") used to
  either error or silently clamp with no explanation. Now caps at the
  dataset's actual max and says so explicitly in the answer.
- **Every answer unconditionally appended a truncated, mid-sentence-cut doc
  snippet** ("...(what-the-collapsed-causes-mean: ...departures, ...")
  regardless of whether the question asked for one. Added
  `_wants_explanation()` (mirrors `_wants_recommendation()`) so the doc
  text only appears when actually requested, and removed the hard
  180-character truncation entirely (the docs are already short).
- **20 new regression tests** (`tests/test_multi_cause.py`,
  `tests/test_instance_level.py`) covering every fix above — 52 total, up
  from 32.

**Consciously deferred, not silently skipped:** explicit "use FAISS" /
"use TF-IDF" backend-selection commands (would need a per-request override
of the retriever, a real but separate feature), and dedicated horizon/
segment input controls on the KM/RSF frontend tabs specifically (the "Ask
the agent" tab is the actual interface for horizon/segment/cause today,
and it got the deeper work this round).

## Structured query parsing (extract meaning, then compute)

The agent's tool layer used to receive only a cause. Now it receives a full
structured query, extracted BEFORE any model runs:

```
"What's the price sensitivity churn rate for new customers in the first 3 months?"
                                    |
                         backend/query_parser.py
                                    |
                                    v
{
  "time_horizon_months": 3,
  "cause": "Price sensitivity",
  "segment": {"type": "tenure_max", "column": "tenure", "value": 3, ...},
  "task_type": "population_churn"
}
                                    |
                      backend/models.py: analyze_churn()
                    (filters the dataset to the segment,
                     refits Aalen-Johansen + Cox on JUST
                     that subset, reports CIF at month 3
                     specifically, not just the full curve)
```

**`backend/query_parser.py` does NO survival analysis** — it only extracts
meaning, via regex/keyword tables for time horizon and segment, and TF-IDF
retrieval for cause (pinned specifically to TF-IDF, not the auto-selecting
FAISS/TF-IDF backend used elsewhere — see the entry above this one for why
that distinction turned out to matter a lot). Rule-based rather than an LLM
call, deliberately: the phrases this needs to recognize ("first quarter,"
"short-term," "new customers")
are a closed, enumerable set for this project's scope, which makes a regex
table faster, free, fully offline, and exactly testable (see
`tests/test_query_parser.py` — every example phrase from the original spec
this was built against has a test).

**Defaults, matching the original spec exactly:** no time horizon
mentioned → 12 months. No cause mentioned → all four causes reported. No
segment mentioned → all customers. Question too vague to extract ANY signal
→ the response IS a clarifying question (see caveat below).

**`backend/models.py: analyze_churn()`** is the one function the agent's
tool layer calls now — takes ONLY structured parameters, filters the
dataset to the requested segment, and refits (not just re-reads a cached
whole-dataset curve) on that subset. **Fails safely** on a segment too
small to fit reliably (`MIN_SEGMENT_ROWS`/`MIN_SEGMENT_EVENTS`) instead of
returning a misleading number from too little data — verified with a
dedicated test.

**Honest limitations, worth having ready:**
- **Stateless clarification, not real multi-turn.** `/agent/ask` has no
  memory between calls, so "ask a clarifying question, wait for the reply"
  isn't something this architecture supports today. What's implemented: a
  too-vague question's ANSWER is the clarifying question itself; the caller
  has to ask again with more detail in a new request. A real back-and-forth
  needs conversation state this project doesn't have.
- **The "new customers" segment (tenure < 3 months) has a real statistical
  subtlety**, documented directly in `query_parser.py`: filtering on tenure
  mixes customers currently in their first 3 months with customers who
  churned that early, since both have a recorded tenure under 3 — a form of
  conditioning on the same axis the outcome is measured against, which
  inflates the apparent near-term risk for this specific segment
  definition. The other segments (contract type, internet service, etc.)
  don't have this problem. A production version would define "new
  customer" cohorts by signup-date snapshot instead.
- **"Premium customers" and "students" are recognized but unsupported** —
  this dataset's schema has no matching attribute, so the parser says so
  explicitly rather than silently mapping them to something incorrect or
  ignoring them without explanation.

## Evaluation results

Real, held-out numbers from `scripts/run_evaluation.py` (25% test split,
300 bootstrap resamples per CI) — not illustrative placeholders:

| Cause | Model | C-index | 95% CI | Integrated Brier | Events (test) |
|---|---|---|---|---|---|
| Price sensitivity | baseline | 0.500 | — | — | 186 |
| | Cox | 0.622 | [0.575, 0.660] | 0.0941 | |
| | RSF | 0.604 | [0.559, 0.645] | 0.0947 | |
| Dissatisfaction | baseline | 0.500 | — | — | 125 |
| | Cox | 0.668 | [0.624, 0.711] | 0.0667 | |
| | RSF | 0.658 | [0.608, 0.706] | 0.0660 | |
| Competitive loss | baseline | 0.500 | — | — | 68 |
| | Cox | 0.618 | [0.561, 0.681] | 0.0410 | |
| | RSF | 0.600 | [0.531, 0.673] | 0.0412 | |
| Non-behavioral | baseline | 0.500 | — | — | 23 |
| | Cox | 0.491 | [0.352, 0.636] | 0.0161 | |
| | RSF | 0.396 | [0.286, 0.507] | 0.0164 | |

**Read this honestly, not optimistically.** Baseline sits at 0.5 by
construction (a constant risk score can't rank anyone). Cox and RSF clear
that baseline meaningfully for Price sensitivity, Dissatisfaction, and
Competitive loss — real, learnable signal for those causes. For
Non-behavioral, both models land AT OR BELOW baseline. That's not a bug —
it's the framework correctly detecting that there's no real signal to find:
`data/generate_synthetic_data.py` deliberately makes that cause's hazard
roughly covariate-independent (`h_other = np.full(n, 0.0004)`, no covariate
terms at all). A model scoring near-baseline on a cause with no true signal
is the evaluation working as intended, not evidence of a broken model — and
being able to explain that distinction clearly is worth more in an
interview than the raw numbers themselves.

*(Note: the numbers above are from this project's earlier synthetic-data
phase, kept here as history. The current version runs on the real IBM
Telco dataset — see the main `README.md` for up-to-date evaluation results
and `data/README.md` for the real dataset's provenance and the model
covariate decisions, including the Gender/Partner fairness tradeoff.)*
