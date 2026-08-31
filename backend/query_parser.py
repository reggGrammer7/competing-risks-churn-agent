"""
Turns a free-text question into STRUCTURED parameters -- this module does
NO survival analysis and touches NO models. Its only job is: given text,
extract {time_horizon_months, cause, segment, task_type}, with defaults for
anything not mentioned. Everything downstream (models.py's analyze_churn())
only ever sees these structured fields, never the raw question.

This is a genuinely different responsibility from the old _detect_cause() in
agent.py, which only ever extracted the cause. This module extracts cause
AND time horizon AND segment in one pass, which is why it lives separately
rather than being bolted onto the old function.

WHY RULE-BASED (regex/keyword) RATHER THAN AN LLM CALL: the phrases this
needs to recognize ("first quarter", "short-term", "new customers") are a
closed, enumerable set for this project's scope -- a regex table is fast,
free, fully offline, and testable with exact input/output pairs (see
tests/test_query_parser.py), none of which is true of an LLM call. Cause
detection specifically still goes through the existing retrieval mechanism
(backend/rag/retriever.py against cause_docs.md) rather than its own keyword
list, for the same "one general-purpose mechanism, not several ad-hoc ones"
reasoning as before.
"""
import re

from backend.data_utils import CAUSES
# Cause detection deliberately uses TF-IDF directly, NOT backend.rag.retriever
# (which prefers FAISS+embeddings when available). This isn't an oversight --
# it's a real bug found in CI: CAUSE_MATCH_THRESHOLD and every generic-word
# fix in cause_docs.md were tuned and tested against TF-IDF's cosine-
# similarity scoring, which behaves very differently from dense-embedding
# similarity. cause_docs.md is a keyword LIST, not natural prose -- exactly
# the content TF-IDF suits and embeddings don't reliably. This surfaced only
# in CI (which has real internet access to download the embedding model)
# since the sandbox this was developed in could never reach huggingface.co,
# so cause detection was always silently tested against TF-IDF regardless of
# which backend "should" have loaded. Document retrieval (methodology/policy)
# has no hardcoded thresholds, so it's unaffected and still free to use
# FAISS+embeddings when available -- only cause detection is pinned.
from backend.rag.tfidf_retriever import retrieve as retrieve_causes

DEFAULT_HORIZON_MONTHS = 12
CAUSE_MATCH_THRESHOLD = 0.08  # below this, treat as "no specific cause mentioned" -> all causes

# ---------------------------------------------------------------------
# Time horizon
# ---------------------------------------------------------------------
_EXPLICIT_MONTHS = re.compile(r"\b(?:within|in|over|next)?\s*(\d+)\s*[- ]?months?\b")
# Catches the number AFTER the word "month" -- "at month 3", "CIF at month 24".
# _EXPLICIT_MONTHS above only catches the number BEFORE ("3 months"), so
# without this, "give me the CIF at month 24" silently fell through to the
# default (12) instead of the 24 that was actually asked for.
_MONTH_N = re.compile(r"\bmonth\s+(\d+)\b")
_EXPLICIT_YEARS = re.compile(r"\b(\d+)\s*[- ]?years?\b")

_NAMED_HORIZONS = [
    # (pattern, months, wants_full_curve)
    (re.compile(r"\bfirst quarter\b"), 3, False),
    (re.compile(r"\bfirst year\b"), 12, False),
    (re.compile(r"\bshort[- ]term\b"), 6, False),
    # Bare "\bearly\b", not just the exact phrase "early churn" -- the
    # original pattern missed "early in the lifecycle", "early on", "early
    # days", anything that wasn't the literal two-word phrase it was
    # written for.
    (re.compile(r"\bearly\b"), 6, False),
    (re.compile(r"\blong[- ]term\b"), 24, False),
    (re.compile(r"\blifetime\b"), None, True),
    (re.compile(r"\bover time\b"), None, True),
    (re.compile(r"\bfull (?:curve|horizon)\b"), None, True),
]

# Multiple explicit horizons in one question -- "at 3, 6, 12, and 24 months"
# or a range like "from month 1 to month 12". Doesn't need any model
# changes: analyze_churn() already computes the full curve internally for
# every request, so extra requested points are just interpolated from
# data that was already returned, not refit.
_MULTI_MONTHS_LIST = re.compile(r"\bat\s+((?:\d+\s*,\s*)+(?:and\s+)?\d+)\s*months?\b")
_MONTH_RANGE = re.compile(r"\bfrom\s+(?:month\s+)?(\d+)\s*(?:months?)?\s+to\s+(?:month\s+)?(\d+)\s*(?:months?)?\b")


def parse_multiple_horizons(question: str) -> list | None:
    q = question.lower()
    m = _MULTI_MONTHS_LIST.search(q)
    if m:
        return sorted(set(int(n) for n in re.findall(r"\d+", m.group(1))))
    m = _MONTH_RANGE.search(q)
    if m:
        return sorted(set([int(m.group(1)), int(m.group(2))]))
    return None


# A horizon was clearly ATTEMPTED (a preposition + word + "month(s)") but
# the word isn't a number and isn't a vague-but-sensible quantity word --
# "at banana months" should say plainly that it doesn't understand, not
# silently default to 12 months and answer as if nothing was wrong. Vague
# quantity words ("a few months", "several months") are deliberately NOT
# flagged -- they're imprecise but not nonsensical, and defaulting
# gracefully for those is the right call.
_HORIZON_ATTEMPT_PATTERN = re.compile(r"\b(?:at|in|within|after|before|for)\s+['\"]?([a-zA-Z]+)['\"]?\s+months?\b")
_VAGUE_BUT_VALID_QUANTITY_WORDS = {
    "a", "few", "several", "some", "many", "couple", "the", "next", "coming", "past", "recent",
}


def detect_invalid_horizon_word(question: str) -> str | None:
    """Returns the offending word if the question clearly attempted a
    numeric horizon but used something unparseable, else None."""
    q = question.lower()
    m = _HORIZON_ATTEMPT_PATTERN.search(q)
    if not m:
        return None
    word = m.group(1)
    if word in _VAGUE_BUT_VALID_QUANTITY_WORDS:
        return None
    return word


def parse_time_horizon(question: str) -> dict:
    q = question.lower()

    # Explicit numbers win over named phrases -- "in 8 months" should give 8,
    # not fall through to a named-phrase default.
    m = _EXPLICIT_MONTHS.search(q)
    if m:
        return {"time_horizon_months": int(m.group(1)), "wants_full_curve": False, "matched": "explicit_months"}
    m = _MONTH_N.search(q)
    if m:
        return {"time_horizon_months": int(m.group(1)), "wants_full_curve": False, "matched": "month_n"}
    m = _EXPLICIT_YEARS.search(q)
    if m:
        return {"time_horizon_months": int(m.group(1)) * 12, "wants_full_curve": False, "matched": "explicit_years"}

    for pattern, months, full_curve in _NAMED_HORIZONS:
        if pattern.search(q):
            return {
                "time_horizon_months": months if months is not None else DEFAULT_HORIZON_MONTHS,
                "wants_full_curve": full_curve,
                "matched": pattern.pattern,
            }

    return {"time_horizon_months": DEFAULT_HORIZON_MONTHS, "wants_full_curve": False, "matched": "default"}


# ---------------------------------------------------------------------
# Segment -- only covers attributes that actually exist in this dataset's
# schema (see data/generate_synthetic_data.py). "premium"/"student" are
# recognized as MENTIONED but flagged unsupported rather than silently
# mapped to a real column that doesn't represent what the person meant.
# ---------------------------------------------------------------------
_SEGMENT_PATTERNS = [
    # CAVEAT worth knowing cold: filtering to "tenure < 3" mixes customers
    # currently in their first 3 months with customers who churned THAT
    # early -- both have a recorded tenure under 3. That inflates the
    # apparent near-term CIF for this segment, because you're conditioning
    # on the same time axis the survival curve is measured against (a form
    # of conditioning on the outcome). A production version would instead
    # define "new customer" cohorts by signup-date snapshot, not by
    # filtering on tenure itself. The other segments below (contract type,
    # internet service, etc.) don't have this problem -- they're not
    # measured on the same axis as the outcome.
    (re.compile(r"\bnew customers?\b|\brecently joined\b|\brecent signups?\b"),
     {"type": "tenure_max", "column": "tenure", "value": 3, "label": "new customers (tenure under 3 months)"}),
    (re.compile(r"\bfiber\b"),
     {"type": "category", "column": "InternetService", "value": "Fiber optic", "label": "fiber-internet customers"}),
    (re.compile(r"\bsenior(?:s| citizens?)?\b"),
     {"type": "category", "column": "SeniorCitizen", "value": 1, "label": "senior citizens"}),
    (re.compile(r"\bmonth[- ]to[- ]month\b"),
     {"type": "category", "column": "Contract", "value": "Month-to-month", "label": "month-to-month customers"}),
    (re.compile(r"\btwo[- ]year\b"),
     {"type": "category", "column": "Contract", "value": "Two year", "label": "two-year-contract customers"}),
    (re.compile(r"\bone[- ]year\b"),
     {"type": "category", "column": "Contract", "value": "One year", "label": "one-year-contract customers"}),
    (re.compile(r"\bno(?:t| )?\s*tech support\b|\bwithout tech support\b"),
     {"type": "category", "column": "TechSupport", "value": "No", "label": "customers without tech support"}),
    (re.compile(r"\bpaperless\b"),
     {"type": "category", "column": "PaperlessBilling", "value": "Yes", "label": "paperless-billing customers"}),
    (re.compile(r"\blong[- ]tenure\b|\blong[- ]time customers?\b|\blong[- ]standing customers?\b"),
     {"type": "tenure_min", "column": "tenure", "value": 24, "label": "long-tenure customers (tenure of 24+ months)"}),
]
_UNSUPPORTED_SEGMENT_TERMS = [
    (re.compile(r"\bpremium\b"), "premium customers"),
    (re.compile(r"\bstudents?\b"), "student customers"),
    (re.compile(r"\bvip\b"), "VIP customers"),
    (re.compile(r"\bgold\b"), "gold-tier customers"),
    (re.compile(r"\bplatinum\b"), "platinum-tier customers"),
    (re.compile(r"\benterprise\b"), "enterprise customers"),
    (re.compile(r"\bbusiness\b"), "business customers"),
    (re.compile(r"\bresidential\b"), "residential customers"),
]


def parse_segment(question: str) -> dict:
    q = question.lower()
    for pattern, spec in _SEGMENT_PATTERNS:
        if pattern.search(q):
            return spec
    for pattern, label in _UNSUPPORTED_SEGMENT_TERMS:
        if pattern.search(q):
            return {
                "type": "unsupported", "column": None, "value": None, "term": label,
                "label": f"all customers (\"{label}\" was mentioned, but this dataset has no matching "
                         f"attribute -- see README's segment limitations note)",
            }
    return {"type": "all", "column": None, "value": None, "label": "all customers"}


# ---------------------------------------------------------------------
# Instance-level detection -- "for a customer on a one-year contract" is a
# fundamentally different question from "for month-to-month customers":
# singular framing asks for ONE constructed profile's prediction (routes to
# RSF/Cox applied to one row), not a population/segment summary. Reuses
# _SEGMENT_PATTERNS/_UNSUPPORTED_SEGMENT_TERMS above since a stated
# attribute means the same real covariate either way -- "for a fiber
# customer" and "for fiber customers" both mean InternetService ==
# "Fiber optic", just applied to one row instead of a filter.
# ---------------------------------------------------------------------
_INSTANCE_LEVEL_PATTERN = re.compile(
    r"\bthis customer\b|\bfor (?:a|an|this|that) .*?\bcustomer\b|"
    r"\bhow long is a .*?\bcustomer\b|\bsurvival curve for a .*?\bcustomer\b|"
    r"\bexpected to stay\b"
)
# "a STANDARD/average/typical customer" is a deliberate request for the
# population-typical profile (zero attributes to fill in IS the answer,
# not a gap) -- different from "THIS customer" with nothing else stated,
# which has no signal at all about what's being asked. Only the latter
# should trigger a clarifying question.
_STANDARD_PROFILE_TERMS = re.compile(r"\bstandard\b|\baverage\b|\btypical\b|\bbaseline\b|\bregular\b")


def is_instance_level_query(question: str) -> bool:
    return bool(_INSTANCE_LEVEL_PATTERN.search(question.lower()))


def parse_instance_attributes(question: str) -> dict:
    """Extracts whichever customer attributes were actually stated in an
    instance-level question, plus whether an unsupported tier term
    ("premium", "VIP", etc.) was mentioned -- the same honesty check
    applies here as for a population segment: silently substituting a
    population default for a stated-but-unavailable attribute would be
    worse than saying plainly that the data doesn't have it."""
    q = question.lower()
    attributes = {}
    for pattern, spec in _SEGMENT_PATTERNS:
        if not pattern.search(q):
            continue
        if spec["type"] == "category":
            attributes[spec["column"]] = spec["value"]
        # tenure_max/tenure_min segments ("new customers", "long-tenure")
        # describe a RANGE, not a single value a row can hold -- not
        # applicable to a single constructed profile, so intentionally
        # not translated into an attribute override here.
    unsupported_term = None
    for pattern, label in _UNSUPPORTED_SEGMENT_TERMS:
        if pattern.search(q):
            unsupported_term = label
            break
    return {
        "attributes": attributes,
        "unsupported_term": unsupported_term,
        "wants_standard_profile": bool(_STANDARD_PROFILE_TERMS.search(q)),
    }


# ---------------------------------------------------------------------
# Cause -- reuses the existing retrieval mechanism (TF-IDF or
# FAISS+embeddings) against backend/rag/cause_docs.md
# ---------------------------------------------------------------------
def parse_cause(question: str) -> dict:
    match = retrieve_causes(question, corpus="causes", top_k=1)[0]
    if match["score"] < CAUSE_MATCH_THRESHOLD:
        return {"cause": None, "cause_confidence": match["score"]}  # "no cause" -> all causes
    return {"cause": match["title"], "cause_confidence": match["score"]}


_COMPARE_ALL_PATTERN = re.compile(r"\ball (?:churn )?causes\b|\brank all\b|\bevery cause\b")


def parse_causes(question: str) -> list | None:
    """Like parse_cause(), but for questions naming MULTIPLE specific causes
    at once -- "compare price sensitivity and competitive loss for premium
    customers" should analyze exactly those two, not fall back to either
    "just the single best match" (parse_cause's behavior) or silently drop
    to all four. Scores every cause independently (parse_cause only looks
    at the single best match) and returns every cause that clears the
    threshold, in descending score order.

    Returns:
      - None if 0 or 1 causes clear the threshold (the single-cause/no-cause
        path in parse_cause already handles those correctly) -- OR if the
        question explicitly asks to compare/rank ALL causes, since that
        means "all four," not "whichever few happened to score above 0."
      - A list of 2+ cause names if multiple are genuinely mentioned.
    """
    if _COMPARE_ALL_PATTERN.search(question.lower()):
        return None  # explicit "all"/"rank all" -> every cause, handled as cause=None downstream
    doc_matches = retrieve_causes(question, corpus="causes", top_k=len(CAUSES))
    # A lower bar than CAUSE_MATCH_THRESHOLD, deliberately: this function
    # already requires 2+ independent matches to return anything, which is
    # its own guard against false positives -- so it can afford to be more
    # permissive per-cause than parse_cause(), which has to pick a single
    # best answer and can't rely on a "does a second one also match" check.
    mentioned = [m["title"] for m in doc_matches if m["score"] >= CAUSE_MATCH_THRESHOLD / 2]
    if len(mentioned) >= 2:
        return mentioned
    return None


# ---------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------
def parse_query(question: str) -> dict:
    """Returns the full structured payload the agent graph needs. Ambiguity
    handling is intentionally narrow: this is a STATELESS endpoint (each
    /agent/ask call has no memory of previous ones), so a genuine multi-turn
    "ask a clarifying question, wait for the reply" loop isn't something
    this architecture supports today -- that would need conversation state.
    What IS implemented: if the question is too short/vague to extract ANY
    signal at all, the response IS the clarifying question, returned as the
    answer text, with task_type set to "clarification_needed" so a caller
    can detect this and prompt the user to ask again with more detail."""
    q = question.strip()
    horizon = parse_time_horizon(q)
    segment = parse_segment(q)
    cause_info = parse_cause(q)
    multi_causes = parse_causes(q)
    multiple_horizons = parse_multiple_horizons(q)
    instance_level = is_instance_level_query(q)
    invalid_horizon_word = detect_invalid_horizon_word(q)

    is_too_vague = len(q.split()) <= 3 and cause_info["cause_confidence"] < CAUSE_MATCH_THRESHOLD

    result = {
        "time_horizon_months": horizon["time_horizon_months"],
        "wants_full_curve": horizon["wants_full_curve"],
        "multiple_horizons_months": multiple_horizons,
        # cause: a single name, None (all four), or -- when multi_causes is
        # set -- ignored in favor of "causes" below, which takes priority.
        "cause": cause_info["cause"],
        "cause_confidence": cause_info["cause_confidence"],
        "causes": multi_causes,  # None unless 2+ specific causes were both mentioned
        "segment": segment,
        "task_type": "population_churn",
        "ambiguous": is_too_vague,
        "clarifying_question": None,
        "instance_attributes": None,
    }

    # Checked FIRST, before every other routing decision: a clearly
    # attempted-but-unparseable horizon ("at banana months") should say so
    # plainly rather than silently defaulting to 12 months and answering as
    # if nothing was wrong -- regardless of what else the question asks.
    if invalid_horizon_word:
        result["task_type"] = "invalid_horizon"
        result["clarifying_question"] = (
            f"I couldn't understand \"{invalid_horizon_word}\" as a time horizon. Please use a number of "
            f"months (e.g. \"6 months\", \"within 3 months\") or a phrase like \"first year\" or \"long-term\"."
        )
        return result

    # Instance-level routing takes priority over the generic vague-question
    # check below -- "for this customer" is 3 words and would otherwise trip
    # the generic "too short to have any signal" rule, but it has a VERY
    # specific signal (singular customer framing) the generic check doesn't
    # know about.
    if instance_level:
        parsed_instance = parse_instance_attributes(q)
        if parsed_instance["unsupported_term"]:
            result["task_type"] = "instance_unsupported_attribute"
            result["instance_attributes"] = parsed_instance
            return result
        if not parsed_instance["attributes"] and not parsed_instance["wants_standard_profile"]:
            result["task_type"] = "clarification_needed"
            result["ambiguous"] = True
            result["clarifying_question"] = "What segment and contract?"
            return result
        result["task_type"] = "instance_level"
        result["instance_attributes"] = parsed_instance
        return result

    if is_too_vague:
        result["task_type"] = "clarification_needed"
        result["clarifying_question"] = (
            "Could you say a bit more about what you'd like to know? For example: which churn cause "
            "(price, support/dissatisfaction, a competitor, or other reasons), what time horizon "
            "(e.g. \"in the first 3 months\", \"over the first year\"), and which customer segment, "
            "if any (e.g. \"new customers\", \"fiber customers\")."
        )
    return result
