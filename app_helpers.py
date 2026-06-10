"""
Pure helpers for the web UI (no Streamlit imports here on purpose, so this
stays unit-testable and the package logic is reused verbatim).

Everything runs on the deterministic stub backend (force_stub=True): no API
key, no network, no real Claude calls. This is a friendly front end over the
exact same agents, guardrails, and state the CLI uses; it does not change any
of them.
"""
from __future__ import annotations

import contextlib
import io

from rebooking_agent import guardrails as g
from rebooking_agent.llm import get_llm
from rebooking_agent.models import Status
from rebooking_agent.observability import Level, configure_tracer, get_tracer
from rebooking_agent.orchestrator import Orchestrator
from rebooking_agent.run import SCENARIOS
from rebooking_agent.tools import Toolbox


# Human-friendly scenario blurbs for the UI.
SCENARIO_INFO = {
    "happy":      ("Happy path", "Platinum flyer, carrier at fault. Should rebook into the owed Business cabin and add a hotel."),
    "overbudget": ("Over budget", "Basic-economy flyer; the only option costs more than the cap. Should escalate, not over-book."),
    "visa":       ("Visa / connection", "Both options break a rule (a visa-restricted transit, and a too-short connection). Should reject both."),
    "apifail":    ("Search fails", "The flight search times out, retries, and comes back empty. Should escalate without inventing a flight."),
    "hitl":       ("Needs sign-off", "A high-value booking that pauses for a human to approve before anything is booked."),
}


def engine_cls(name: str):
    if name == "langgraph":
        from rebooking_agent.graph import GraphOrchestrator
        return GraphOrchestrator
    return Orchestrator


def _capture(fn):
    """Run fn() while collecting the verbose trace into the tracer's buffer.
    Uses in-memory capture (not stdout redirection) so it's safe under a server."""
    t = get_tracer()
    prev_level, prev_cap, prev_sink = t.level, t.capture, t.sink
    t.level = Level.VERBOSE
    t.capture = True
    t.sink = []
    try:
        result = fn()
        trace = "\n".join(t.sink)
    finally:
        t.level, t.capture, t.sink = prev_level, prev_cap, prev_sink
    return result, trace


def start_case(scenario_key: str, engine: str):
    """Build and run a case up to its first stopping point (done or paused).

    Returns (orchestrator, state, trace_text). Always uses the stub backend.
    """
    state, world, threshold, _resume = SCENARIOS[scenario_key]()
    toolbox = Toolbox(world, state)
    orch = engine_cls(engine)(state, llm=get_llm(force_stub=True), hitl_threshold=threshold)
    state, trace = _capture(lambda: orch.run(toolbox))
    return orch, state, trace


def resume_case(orch, approved: bool):
    """Resume a paused (awaiting-approval) case with the human's decision."""
    state, trace = _capture(lambda: orch.resume_with_human_decision(approved))
    return state, trace


def derive_stages(state) -> list[tuple[str, str]]:
    """The pipeline path this case actually took, as (label, status) pairs.

    status is one of: done, escalate, await.
    """
    routes = " ".join(d.decision for d in state.decisions_log)
    stages: list[tuple[str, str]] = [("Triage", "done")]
    if state.search_attempts > 0 or "route -> search" in routes:
        stages.append(("Search", "done"))
    if state.policy_verdicts:
        stages.append(("Policy", "done"))
    if state.status == Status.RESOLVED:
        stages.append(("Book & notify", "done"))
    elif state.status == Status.ESCALATED:
        stages.append(("Escalate", "escalate"))
    elif state.status == Status.AWAITING_HUMAN_APPROVAL:
        stages.append(("Await approval", "await"))
    return stages


def build_check_rows(state) -> list[dict]:
    """Per-option breakdown of the four deterministic checks, for the table.

    Recomputed with the same guardrail functions the Policy gate uses, so the
    table is faithful to the actual decision, not a paraphrase of it.
    """
    if not state.assessment:
        return []
    ent = state.assessment.entitlement
    profile = state.passenger_profile
    totals = {v.flight_id: v.total_cost for v in state.policy_verdicts}
    verdicts = {v.flight_id: v.verdict.value for v in state.policy_verdicts}

    rows = []
    for f in state.flight_candidates:
        total = totals.get(f.flight_id, f.price)
        checks = {
            "Cabin": g.cabin_ok(f, ent),
            "Budget": g.budget_ok(total, ent),
            "Visa": g.visa_ok(f, profile),
            "Connection": g.mct_ok(f),
        }
        rows.append({
            "flight": f.flight_id,
            "cabin": f.cabin.label,
            "price": f.price,
            "total": total,
            "checks": checks,                       # name -> (ok, reason)
            "verdict": verdicts.get(f.flight_id, "-"),
        })
    return rows


def outcome_summary(state) -> dict:
    """Compact outcome for the result card."""
    opt = state.approved_option
    return {
        "status": state.status.value,
        "booking_pnr": state.booking_pnr,
        "escalation_reason": state.escalation_reason,
        "flight": (opt.flight.flight_id if opt else None),
        "cabin": (opt.flight.cabin.label if opt else None),
        "total": (opt.total_cost if opt else None),
        "hotel": (opt.hotel.name if (opt and opt.hotel) else None),
        "voucher": (opt.meal_voucher_amount if opt else 0.0),
        "message": state.final_message,
        "hops": state.hops,
        "llm_calls": state.llm_calls,
        "search_attempts": state.search_attempts,
    }


def disruption_dict(state) -> dict:
    d = state.disruption
    return {"reason": d.reason.value, "jurisdiction": d.jurisdiction,
            "flight_number": d.flight_number,
            "route": f"{d.original_itinerary.legs[0].origin}\u2192{d.original_itinerary.legs[-1].destination}"}


def snapshot_dict(state, trace: str = "") -> dict:
    """The full payload the frontend needs for one render. Shared by the live
    server and the offline demo builder so they can't drift."""
    return {
        "stages": derive_stages(state),
        "rows": [
            {"flight": r["flight"], "cabin": r["cabin"], "price": r["price"],
             "total": r["total"], "verdict": r["verdict"],
             "checks": {k: {"ok": v[0], "reason": v[1]} for k, v in r["checks"].items()}}
            for r in build_check_rows(state)
        ],
        "outcome": outcome_summary(state),
        "passenger": {"name": state.passenger_profile.name,
                      "tier": state.passenger_profile.loyalty_tier.label},
        "disruption": disruption_dict(state),
        "trace": trace,
        "awaiting": state.status.value == "AWAITING_HUMAN_APPROVAL",
    }


SCENARIO_ORDER = ["happy", "overbudget", "visa", "apifail", "hitl"]
