"""
Agent layer -- a LangGraph StateGraph that PARSES a question into structured
parameters before touching any model, instead of always summarizing the
whole dataset regardless of what was actually asked.

THE BUG THIS ARCHITECTURE FIXES: earlier, every question funneled down to
"detect a cause, show its whole-dataset CIF" -- so "what's the churn rate in
3 months" and "what's the overall churn rate" produced the same answer,
because nothing ever extracted the "3 months" or noticed a segment mention.
Now: backend/query_parser.py extracts {time_horizon_months, cause, segment,
task_type} from the question FIRST, with defaults for anything unstated,
and backend/models.py's analyze_churn() refits on the requested segment and
reports the CIF at the requested horizon specifically -- not just whatever
the full curve happens to show. The agent's tool layer never sees raw
question text past the parse_query node; every tool call downstream takes
only structured parameters. That separation (extract meaning -> query
structured backend -> synthesize prose) is deliberate, not incidental.

GRAPH SHAPE: parse_query has a real conditional fan-out. If the question is
too vague to extract any signal at all, the graph short-circuits straight
to a clarifying question (see "STATELESS CAVEAT" below) and skips every
tool call. Otherwise it fans out to call_analyze and retrieve_methodology in
parallel (same fan-out/join/conditional-policy-branch pattern as before),
then synthesizes.

STATELESS CAVEAT, worth having ready for an interview: `/agent/ask` has no
memory between calls, so a genuine multi-turn "ask a clarifying question,
wait for the reply, continue" loop isn't something this architecture
supports today -- that needs conversation state this project doesn't have.
What's actually implemented: if a question is too vague, the ANSWER IS the
clarifying question itself, and the caller has to ask again with more
detail in a new request. That's an honest, narrower thing than a real
back-and-forth, and it's worth saying so directly rather than implying more.

LLM SYNTHESIS: if an ANTHROPIC_API_KEY environment variable is set, the
final answer is written by Claude; if OPENAI_API_KEY is set instead (and no
Anthropic key), it's written by GPT. Either way it's grounded strictly in
the tool output and retrieved doc text (never inventing numbers or
recommendations). If neither key is set, a clear template-based fallback
answer is used instead, so the whole system still runs end-to-end with
zero external API dependency.
"""
import os
import re
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from backend import mcp_tools
from backend.rag.retriever import retrieve
from backend.query_parser import parse_query, CAUSE_MATCH_THRESHOLD

_RECOMMENDATION_PATTERNS = [r"\bwhat should\b", r"\brecommend\b", r"\bdo about\b", r"\baction\b", r"\bretain\b"]
_RANKING_PATTERNS = [
    r"\bmost dominant\b", r"\bdominant\b", r"\bdominates?\b", r"\bhighest\b", r"\bbiggest\b", r"\blargest\b",
    r"\bleading cause\b", r"\bwhich cause\b", r"\brank\b", r"\btop cause\b", r"\bworst\b",
]
# Deliberately SEPARATE from _RANKING_PATTERNS: "dominant"/"biggest" ask
# about MAGNITUDE (which cause has the highest CIF), while "grows faster"
# asks about RATE OF INCREASE -- a cause can have a lower absolute CIF but
# still be increasing faster than a bigger one. Conflating these two into
# one "pick the winner" computation gives the right answer only by
# coincidence; they need their own metric.
_GROWTH_RATE_PATTERNS = [
    r"\bgrows? faster\b", r"\bgrowing faster\b", r"\bfastest.?growing\b", r"\bgrows? fastest\b",
    r"\bgrowing fastest\b", r"\brate of growth\b", r"\bincreasing fastest\b", r"\bgrows? quick(er|est)\b",
    r"\baccelerating\b",
]
_EXPLANATION_PATTERNS = [
    r"\bexplain\b", r"\bwhat does .* mean\b", r"\bwhy (is|are|do|does)\b",
    r"\bhow (is|are|does) .* (calculated|computed|work|estimated|defined)\b",
    r"\bwhat is the difference between\b", r"\bdefine\b",
]


def _wants_recommendation(question: str) -> bool:
    # WORD-BOUNDARY regex, not `word in question` substring checks: plain
    # substring matching has the same failure mode as an earlier keyword-
    # based router did -- "action" as a bare substring matches inside
    # "dissatisfaction" (...dissatisf-ACTION), which silently pulled in a
    # policy doc for questions that never asked for one. See
    # tests/test_agent_routing.py for the regression test.
    q = question.lower()
    return any(re.search(p, q) for p in _RECOMMENDATION_PATTERNS)


def _wants_ranking(question: str) -> bool:
    """Detects MAGNITUDE ranking language ('which cause is most dominant',
    'what's the biggest driver') -- these questions deserve an explicit
    named answer ('X is the dominant cause'), not just a dump of all four
    numbers left for the reader to compare themselves."""
    q = question.lower()
    return any(re.search(p, q) for p in _RANKING_PATTERNS)


def _wants_growth_rate(question: str) -> bool:
    """Detects RATE-OF-INCREASE language ('which cause grows faster') --
    checked separately from and BEFORE _wants_ranking, since a growth-rate
    question needs a genuinely different computation (average rate of
    increase, not absolute magnitude at a point)."""
    q = question.lower()
    return any(re.search(p, q) for p in _GROWTH_RATE_PATTERNS)


def _wants_explanation(question: str) -> bool:
    # Gates whether the methodology doc snippet appears in the visible
    # answer at all. Before this existed, the doc was appended to EVERY
    # answer regardless of relevance -- a question like "what is the
    # churn risk for X" was getting a random, often unrelated methodology
    # paragraph bolted onto the end, hard-truncated mid-sentence. Only
    # genuinely explanatory questions ("explain X", "why does Y happen",
    # "how is Z calculated") should surface it; a plain "what is the X
    # risk" is asking for a NUMBER, not a methodology lecture, so it's
    # deliberately excluded even though it also starts with "what is".
    q = question.lower()
    return any(re.search(p, q) for p in _EXPLANATION_PATTERNS)


# ---------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    question: str
    time_horizon_months: int
    wants_full_curve: bool
    multiple_horizons_months: Optional[list]
    cause: Optional[str]
    cause_confidence: float
    causes: Optional[list]
    segment: dict
    task_type: str
    ambiguous: bool
    clarifying_question: Optional[str]
    instance_attributes: Optional[dict]
    instance_analysis: Optional[dict]
    wants_recommendation: bool
    wants_explanation: bool
    wants_ranking: bool
    wants_growth_rate: bool
    analysis: dict
    method_doc: dict
    policy_doc: Optional[dict]
    answer: str
    trace: dict


def node_parse_query(state: AgentState) -> dict:
    parsed = parse_query(state["question"])
    parsed["wants_recommendation"] = _wants_recommendation(state["question"])
    parsed["wants_explanation"] = _wants_explanation(state["question"])
    parsed["wants_ranking"] = _wants_ranking(state["question"])
    parsed["wants_growth_rate"] = _wants_growth_rate(state["question"])
    return parsed


def route_after_parse(state: AgentState):
    """Real conditional fan-out: a too-vague question routes to a single
    'clarify' node and skips every tool call; a segment this dataset
    genuinely doesn't have (e.g. "premium customers") routes to a direct,
    short answer instead of silently running a full population-wide
    analysis dressed up with a caveat nobody asked to read past. An
    instance-level question ("for a customer on a one-year contract")
    routes to a completely different tool (predict_customer_profile
    instead of analyze_churn) since it's a different KIND of question, not
    just a differently-filtered version of the population one. Anything
    else with real extractable signal fans out to BOTH call_analyze and
    retrieve_methodology at once, since neither depends on the other's
    output."""
    if state["ambiguous"] or state["task_type"] == "invalid_horizon":
        return "clarify"
    if state["task_type"] == "instance_unsupported_attribute":
        return "unsupported_segment"
    if state["task_type"] == "instance_level":
        return ["predict_instance", "retrieve_methodology"]
    if state["segment"].get("type") == "unsupported":
        return "unsupported_segment"
    return ["call_analyze", "retrieve_methodology"]


def node_clarify(state: AgentState) -> dict:
    return {
        "answer": state["clarifying_question"],
        "trace": {"task_type": state["task_type"]},
    }


def node_unsupported_segment(state: AgentState) -> dict:
    """Short-circuits straight to a direct answer -- no model calls, no
    doc retrieval -- for an attribute this dataset has no column for.
    Handles both the population-segment case ("premium customers") and the
    instance-level case ("a premium customer on a one-year contract") with
    the same honest message, since both are the same underlying problem:
    an attribute was stated that this dataset simply doesn't have. Before
    this existed, an unsupported segment silently fell back to running the
    full analysis on ALL customers with a caveat buried in the trace, which
    is a much longer and less honest answer than just saying plainly that
    the data doesn't support the request."""
    term = state["segment"].get("term") or (state.get("instance_attributes") or {}).get("unsupported_term") or "that attribute"
    return {
        "answer": (
            f"This dataset doesn't have a \"{term}\" attribute to filter on -- the available "
            f"customer attributes are contract type, internet service, tech support, payment "
            f"method, senior citizen status, dependents, paperless billing, and tenure. "
            f"Ask again using one of those (e.g. \"fiber customers\", \"new customers\", "
            f"\"month-to-month customers\") or ask about all customers."
        ),
        "trace": {"task_type": "unsupported_segment", "requested_segment": term},
    }


def node_call_analyze(state: AgentState) -> dict:
    # An explicit multi-cause comparison ("compare X and Y") takes priority
    # over the single-cause result -- see query_parser.parse_causes() for
    # why these are extracted separately rather than parse_cause() just
    # returning a list always (a single best match and "which of these N
    # are actually mentioned" are different enough questions to warrant
    # different thresholds).
    cause_param = state["causes"] if state.get("causes") else state["cause"]
    result = mcp_tools.analyze_churn(cause_param, state["time_horizon_months"], state["segment"])
    return {"analysis": result}


def node_predict_instance(state: AgentState) -> dict:
    """Instance-level counterpart to node_call_analyze -- calls a
    DIFFERENT tool (predict_customer_profile, not analyze_churn) because
    this is a different KIND of question (one constructed row, not a
    population/segment), not just a narrower filter on the same question."""
    attrs = (state.get("instance_attributes") or {}).get("attributes", {})
    result = mcp_tools.predict_customer_profile(attrs, state["cause"], state["time_horizon_months"])
    return {"instance_analysis": result}


def node_retrieve_methodology(state: AgentState) -> dict:
    doc = retrieve(state["question"], corpus="methodology", top_k=1)[0]
    return {"method_doc": doc}


def node_join(state: AgentState) -> dict:
    """No-op fan-in point: LangGraph only runs a node once every node with
    an edge INTO it has finished, so routing call_analyze and
    retrieve_methodology through here before the conditional edge
    guarantees both are present before deciding on a policy doc."""
    return {}


def route_policy(state: AgentState) -> str:
    return "retrieve_policy" if state["wants_recommendation"] else "skip_policy"


def node_retrieve_policy(state: AgentState) -> dict:
    # cause can be None (all causes) -- fall back to a generic retention
    # query so policy retrieval still returns something sensible rather
    # than erroring on a None argument.
    query = state["cause"] or "reduce overall churn retention strategy"
    doc = retrieve(query, corpus="policy", top_k=1)[0]
    return {"policy_doc": doc}


def node_skip_policy(state: AgentState) -> dict:
    return {"policy_doc": None}


async def node_synthesize(state: AgentState) -> dict:
    """The only async node -- see _synthesize() below: it's the one node
    with a real network call in it."""
    answer = await _synthesize(state)
    is_instance = state["task_type"] == "instance_level"
    trace = {
        "task_type": state["task_type"],
        "detected_cause": state["causes"] if state.get("causes") else state["cause"],
        "detected_cause_confidence": state["cause_confidence"],
        "time_horizon_months": state["time_horizon_months"],
        "wants_full_curve": state["wants_full_curve"],
        "segment": "one constructed customer profile" if is_instance else state["segment"]["label"],
        "n_customers_in_segment": 1 if is_instance else state["analysis"]["n_customers_in_segment"],
        "wants_explanation": state.get("wants_explanation", False),
        "tool_calls": ["predict_customer_profile"] if is_instance else ["analyze_churn"],
        "methodology_doc_used": state["method_doc"]["title"],
    }
    if state.get("policy_doc"):
        trace["policy_doc_used"] = state["policy_doc"]["title"]
    return {"answer": answer, "trace": trace}


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("parse_query", node_parse_query)
    g.add_node("clarify", node_clarify)
    g.add_node("unsupported_segment", node_unsupported_segment)
    g.add_node("call_analyze", node_call_analyze)
    g.add_node("predict_instance", node_predict_instance)
    g.add_node("retrieve_methodology", node_retrieve_methodology)
    g.add_node("join", node_join)
    g.add_node("retrieve_policy", node_retrieve_policy)
    g.add_node("skip_policy", node_skip_policy)
    g.add_node("synthesize", node_synthesize)

    g.add_edge(START, "parse_query")

    g.add_conditional_edges("parse_query", route_after_parse, {
        "clarify": "clarify",
        "unsupported_segment": "unsupported_segment",
        "call_analyze": "call_analyze",
        "predict_instance": "predict_instance",
        "retrieve_methodology": "retrieve_methodology",
    })
    g.add_edge("clarify", END)
    g.add_edge("unsupported_segment", END)

    g.add_edge("call_analyze", "join")
    g.add_edge("predict_instance", "join")
    g.add_edge("retrieve_methodology", "join")

    g.add_conditional_edges("join", route_policy, {
        "retrieve_policy": "retrieve_policy",
        "skip_policy": "skip_policy",
    })
    g.add_edge("retrieve_policy", "synthesize")
    g.add_edge("skip_policy", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


# Built once at import time, reused across requests.
_GRAPH = _build_graph()

# Task types that short-circuit straight to an answer with no model/doc
# calls -- their "grounding" is intentionally empty, since there's nothing
# to ground (nothing was computed).
_SHORT_CIRCUIT_TASK_TYPES = {"clarification_needed", "unsupported_segment", "invalid_horizon"}


async def answer_question(question: str) -> dict:
    """Runs the graph and reshapes the final state into the API response
    shape main.py expects. WHY async: node_synthesize is async (it awaits
    the LLM network call), and LangGraph's ainvoke is what runs a graph
    containing an async node."""
    state: AgentState = await _GRAPH.ainvoke({"question": question})

    if state["trace"].get("task_type") in _SHORT_CIRCUIT_TASK_TYPES:
        return {"answer": state["answer"], "trace": state["trace"], "grounding": {}}

    if state["trace"].get("task_type") == "instance_level":
        return {
            "answer": state["answer"],
            "trace": state["trace"],
            "grounding": {
                "instance_analysis": state["instance_analysis"],
                "methodology_doc": state["method_doc"],
                "policy_doc": state.get("policy_doc"),
            },
        }

    return {
        "answer": state["answer"],
        "trace": state["trace"],
        "grounding": {
            "analysis": state["analysis"],
            "methodology_doc": state["method_doc"],
            "policy_doc": state.get("policy_doc"),
        },
    }


# ---------------------------------------------------------------------
# Synthesis -- shared across both LLM providers and the template fallback
# ---------------------------------------------------------------------
def _format_cause_result(cause_name: str, r: dict, horizon: int, wants_full_curve: bool,
                          multiple_horizons: Optional[list] = None) -> str:
    if r.get("insufficient_data"):
        return f"{cause_name}: {r['message']}"

    # Multiple explicit horizons requested ("at 3, 6, 12, and 24 months" or
    # a range) -- interpolate each point from the curve arrays analyze_churn
    # already returned, rather than needing a separate model call per point.
    if multiple_horizons:
        import numpy as np
        months_arr = r["months_curve"]
        cif_arr = r["cif_curve"]
        max_month = r["full_horizon_months"]
        points = []
        for m in multiple_horizons:
            capped = min(m, max_month)
            note = f" (capped at {max_month}, asked for {m})" if m > max_month else ""
            val = float(np.interp(capped, months_arr, cif_arr))
            points.append(f"{val*100:.1f}% by month {capped}{note}")
        line = f"{cause_name}: " + ", ".join(points) + "."
        if r.get("top_drivers"):
            drivers = ", ".join(f"{d['covariate']} (HR={d['hazard_ratio']})" for d in r["top_drivers"])
            line += f" Risk factors associated with this cause: {drivers}."
        return line

    cap_note = ""
    if r.get("horizon_was_capped"):
        cap_note = (
            f" (you asked for month {r['requested_horizon_months']}, but this dataset only follows "
            f"customers to month {r['full_horizon_months']} -- showing the value at month {r['full_horizon_months']} instead)"
        )
    if wants_full_curve:
        line = (
            f"{cause_name}: cumulative incidence rises to {r['cif_full_horizon']*100:.1f}% "
            f"by month {r['full_horizon_months']} (full curve)."
        )
    else:
        full_horizon_note = "" if r.get("horizon_was_capped") else (
            f" (full-horizon value: {r['cif_full_horizon']*100:.1f}% by month {r['full_horizon_months']})"
        )
        line = (
            f"{cause_name}: {r['cif_at_horizon']*100:.1f}% cumulative incidence by month "
            f"{r['horizon_months_used']}{cap_note}{full_horizon_note}."
        )
    if r.get("top_drivers"):
        drivers = ", ".join(f"{d['covariate']} (HR={d['hazard_ratio']})" for d in r["top_drivers"])
        line += f" Risk factors associated with this cause: {drivers}."
    return line


def _format_instance_result(state: AgentState) -> str:
    r = state["instance_analysis"]
    lines = ["Prediction for one constructed customer profile (not a real individual -- a representative "
             "row built from the attributes you specified, with population defaults for the rest):"]
    if r["profile_specified"]:
        stated = ", ".join(f"{k}={v}" for k, v in r["profile_specified"].items())
        lines.append(f"Specified: {stated}.")
    if r["profile_filled_with_population_defaults"]:
        filled = ", ".join(f"{k}={v}" for k, v in r["profile_filled_with_population_defaults"].items())
        lines.append(f"Filled with population defaults: {filled}.")
    lines.append(
        f"Predicted churn probability by month {r['horizon_months_used']}: "
        f"{r['churn_probability_at_horizon']*100:.1f}% (survival probability: "
        f"{r['survival_probability_at_horizon']*100:.1f}%)."
    )
    if r.get("median_expected_tenure_months") is not None:
        lines.append(f"Median expected tenure: {r['median_expected_tenure_months']:.0f} months.")
    elif r.get("median_expected_tenure_note"):
        lines.append(r["median_expected_tenure_note"])
    if r.get("cause"):
        lines.append(
            f"Relative risk for {r['cause']} specifically: {r['cause_specific_relative_risk']}x the average "
            f"customer in the training population."
        )
    return "\n".join(lines)


def _synthesize_template(state: AgentState) -> str:
    if state["task_type"] == "instance_level":
        text = _format_instance_result(state)
        if state.get("wants_explanation"):
            text += f"\n\n{state['method_doc']['title']}: {state['method_doc']['text']}"
        return text

    analysis = state["analysis"]
    segment_label = state["segment"]["label"]
    horizon = state["time_horizon_months"]
    wants_full_curve = state["wants_full_curve"]

    lines = []
    if state.get("causes"):
        lines.append(f"Comparing {len(state['causes'])} churn causes you mentioned: {', '.join(state['causes'])}.")
    elif state["cause"] is None:
        lines.append("No specific churn cause was mentioned, so here's the breakdown across all four causes.")
    lines.append(f"Segment: {segment_label} (n={analysis['n_customers_in_segment']}).")

    # Growth-rate questions ("which cause grows faster") need a DIFFERENT
    # metric than magnitude ranking below: average rate of increase
    # (full-horizon CIF / months observed), not which cause has the
    # highest absolute value. Checked first and independently -- a cause
    # can have a lower CIF but still be increasing faster.
    if state.get("wants_growth_rate") and len(analysis["by_cause"]) > 1:
        rankable = {c: r for c, r in analysis["by_cause"].items() if not r.get("insufficient_data")}
        if rankable:
            rates = {c: r["cif_full_horizon"] / max(1, r["full_horizon_months"]) for c, r in rankable.items()}
            winner = max(rates, key=lambda c: rates[c])
            lines.append(
                f"**{winner} is growing fastest**, at an average of {rates[winner]*100:.3f}% cumulative "
                f"incidence per month over the full {rankable[winner]['full_horizon_months']}-month "
                f"observation window -- the steepest average trajectory of the four causes."
            )
    # Magnitude/ranking questions ("which cause is most dominant") get an
    # explicit named answer up front, not just four numbers left for the
    # reader to compare -- only meaningful when there's more than one cause
    # to rank in the first place (a single-cause question can't be "most
    # dominant" relative to anything). Skipped if a growth-rate answer was
    # already given above, since answering both would be redundant/
    # confusing for a question that only asked about one.
    elif state.get("wants_ranking") and len(analysis["by_cause"]) > 1:
        ranking_field = "cif_full_horizon" if wants_full_curve else "cif_at_horizon"
        rankable = {c: r for c, r in analysis["by_cause"].items() if not r.get("insufficient_data")}
        if rankable:
            winner = max(rankable, key=lambda c: rankable[c][ranking_field])
            winner_val = rankable[winner][ranking_field]
            winner_month = rankable[winner]["full_horizon_months"] if wants_full_curve else rankable[winner]["horizon_months_used"]
            lines.append(
                f"**{winner} is the dominant cause**, at {winner_val*100:.1f}% cumulative incidence by "
                f"month {winner_month} -- higher than any other cause in this segment/horizon."
            )

    for cause_name, r in analysis["by_cause"].items():
        lines.append(_format_cause_result(cause_name, r, horizon, wants_full_curve, state.get("multiple_horizons_months")))

    text = "\n".join(lines)
    # Only surface the methodology doc's text in the visible answer when the
    # question actually asked for an explanation (see _wants_explanation).
    # Before this check existed, whatever doc the retriever happened to
    # match was appended to EVERY answer, hard-truncated at 180 characters --
    # which is why answers used to end mid-sentence with "...". The doc is
    # always short (a few hundred characters, see backend/rag/*_docs.md), so
    # once it's actually relevant there's no need to truncate it at all.
    if state.get("wants_explanation"):
        text += f"\n\n{state['method_doc']['title']}: {state['method_doc']['text']}"
    if state.get("policy_doc"):
        text += f"\n\nRecommended action ({state['policy_doc']['title']}): {state['policy_doc']['text']}"
    return text


def _build_llm_context(state: AgentState) -> str:
    """Shared prompt-building for both LLM providers -- same grounding
    rules apply regardless of which model writes the final sentence."""
    if state["task_type"] == "instance_level":
        r = state["instance_analysis"]
        context = f"""Question: {state['question']}
This is an INSTANCE-LEVEL prediction for one constructed customer profile, not a population summary.
Attributes specified in the question: {r['profile_specified']}
Attributes filled with population defaults (make this distinction clear in the answer): {r['profile_filled_with_population_defaults']}
Predicted churn probability by month {r['horizon_months_used']}: {r['churn_probability_at_horizon']}
Predicted survival probability by month {r['horizon_months_used']}: {r['survival_probability_at_horizon']}
Median expected tenure (months, null if not reached within the observed window): {r.get('median_expected_tenure_months')}
"""
        if r.get("cause"):
            context += f"Cause-specific relative risk for {r['cause']}: {r['cause_specific_relative_risk']}x average\n"
        if state.get("wants_explanation"):
            context += f"Methodology doc ({state['method_doc']['title']}): {state['method_doc']['text']}\n"
        context += (
            "\nAnswer in 3-5 sentences. Make clear this is a REPRESENTATIVE profile built from stated "
            "attributes plus population defaults, not a real individual customer. Use ONLY the numbers "
            "given above -- never invent a statistic."
        )
        return context

    analysis = state["analysis"]
    context = f"""Question: {state['question']}
Detected cause: {state['cause'] or 'none specified -- report on all 4 causes'} (match confidence: {state['cause_confidence']:.3f})
Time horizon requested: {state['time_horizon_months']} months (full curve requested: {state['wants_full_curve']})
Segment: {state['segment']['label']} ({analysis['n_customers_in_segment']} customers)
Analysis results by cause: {analysis['by_cause']}
"""
    # Only include the methodology doc's text when the question actually
    # asked for an explanation -- otherwise it's irrelevant context that
    # risks the model volunteering an unrequested methodology tangent
    # instead of just answering the question asked.
    if state.get("wants_explanation"):
        context += f"Methodology doc ({state['method_doc']['title']}): {state['method_doc']['text']}\n"
    if state.get("policy_doc"):
        context += f"Policy doc ({state['policy_doc']['title']}): {state['policy_doc']['text']}\n"

    # Explicit scope lock: naming exactly which cause(s) are in scope, and
    # directly forbidding the other three, is a stronger constraint than
    # just handing over correctly-scoped data and hoping the model infers
    # the boundary on its own. This matters specifically for the LLM path --
    # the template path can't drift off-scope since it only ever loops over
    # analysis['by_cause'], which already contains nothing else.
    causes_in_scope = list(analysis["by_cause"].keys())
    if len(causes_in_scope) == 1:
        all_causes = ["Price sensitivity", "Dissatisfaction", "Competitive loss", "Non-behavioral"]
        other_causes = [c for c in all_causes if c not in causes_in_scope]
        context += (
            f"\nSCOPE: the question asked about {causes_in_scope[0]} ONLY. Discuss ONLY {causes_in_scope[0]} "
            f"in your answer. Do NOT mention, summarize, compare against, or speculate about "
            f"{', '.join(other_causes)} -- no data for those causes was retrieved for this question, "
            f"so anything you said about them would be invented."
        )

    if state.get("wants_growth_rate") and len(causes_in_scope) > 1:
        rankable = {c: r for c, r in analysis["by_cause"].items() if not r.get("insufficient_data")}
        if rankable:
            rates = {c: r["cif_full_horizon"] / max(1, r["full_horizon_months"]) for c, r in rankable.items()}
            fastest = max(rates, key=lambda c: rates[c])
            context += (
                f"\nGROWTH RATE (precomputed -- use this exact answer, don't estimate your own): the question "
                f"asks which cause is growing FASTEST (rate of increase, not absolute size). {fastest} has the "
                f"highest average rate of increase ({rates[fastest]*100:.3f}% per month over the full "
                f"observation window) -- state this explicitly as the answer."
            )
    elif state.get("wants_ranking") and len(causes_in_scope) > 1:
        context += (
            f"\nRANKING: the question asks WHICH cause is dominant/highest/biggest (absolute magnitude, not "
            f"growth rate) -- explicitly name the single cause with the highest cumulative incidence value "
            f"above as the answer, don't just list all {len(causes_in_scope)} numbers and leave it to the "
            f"reader to compare."
        )

    context += (
        "\nAnswer the question in 3-5 sentences. Use ONLY the numbers and recommendations given above -- "
        "never invent a statistic or action not present in the context. If a cause's result says "
        "insufficient data, say so plainly instead of guessing a number. Do not explain methodology or "
        "define terms unless the question explicitly asked for an explanation."
    )
    return context


async def _synthesize(state: AgentState) -> str:
    # Anthropic checked first (this project's "native" provider), OpenAI as
    # a real alternative for anyone without an Anthropic key. Whichever key
    # is present wins; if neither is set, the template fallback below still
    # gives the full experience with zero API dependency either way.
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if anthropic_key:
        try:
            return await _synthesize_with_anthropic(state, anthropic_key)
        except Exception as e:
            return _synthesize_template(state) + f"\n\n[Anthropic synthesis failed ({e}), showing template answer instead.]"
    if openai_key:
        try:
            return await _synthesize_with_openai(state, openai_key)
        except Exception as e:
            return _synthesize_template(state) + f"\n\n[OpenAI synthesis failed ({e}), showing template answer instead.]"
    return _synthesize_template(state)


async def _synthesize_with_anthropic(state: AgentState, api_key: str) -> str:
    import anthropic
    # AsyncAnthropic, not the sync client: lets `await client.messages.create(...)`
    # yield control back to FastAPI's event loop instead of blocking the
    # whole server on this one network round-trip.
    client = anthropic.AsyncAnthropic(api_key=api_key)
    context = _build_llm_context(state)
    resp = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": context}],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text"))


async def _synthesize_with_openai(state: AgentState, api_key: str) -> str:
    import openai
    # AsyncOpenAI for the same reason as AsyncAnthropic above. Model choice:
    # gpt-5.4-mini -- cheap enough for a portfolio project's usage volume,
    # current as of mid-2026. Swap the `model=` string if OpenAI has
    # released something newer/cheaper by the time you're reading this.
    #
    # max_completion_tokens, NOT max_tokens: OpenAI's newer-generation
    # models (the gpt-5.x and o-series reasoning models) reject the older
    # `max_tokens` parameter outright with a 400 error asking for this one
    # instead. `max_tokens` is still correct for Anthropic's API above --
    # this is an OpenAI-specific API change, not a general rename.
    client = openai.AsyncOpenAI(api_key=api_key)
    context = _build_llm_context(state)
    resp = await client.chat.completions.create(
        model="gpt-5.4-mini",
        max_completion_tokens=400,
        messages=[{"role": "user", "content": context}],
    )
    return resp.choices[0].message.content
