"""
Demo driver. Runs the five graded scenarios end-to-end and prints a trace.

    python -m rebooking_agent.run                       # all scenarios, plain engine
    python -m rebooking_agent.run happy                  # one scenario by key
    python -m rebooking_agent.run --engine langgraph     # run them on the LangGraph engine
    python -m rebooking_agent.run happy --engine langgraph

Runs on the deterministic StubLLM by default (no API key needed). Set
ANTHROPIC_API_KEY to route the reasoning steps through real Claude instead.
The two engines (`orchestrator.Orchestrator` and `graph.GraphOrchestrator`)
produce identical outcomes; --engine just selects which one drives the case.
"""

from __future__ import annotations

import sys

from .guardrails import DEFAULT_HITL_THRESHOLD
from .llm import get_llm
from .mock_data import PASSENGERS, MockWorld, FlightFault, cancelled_event
from .models import DisruptionReason, State, Status
from .orchestrator import Orchestrator
from .tools import Toolbox


def _engine_cls(name: str):
    if name == "langgraph":
        from .graph import GraphOrchestrator
        return GraphOrchestrator
    return Orchestrator


def _make_state(case_id: str, pax_key: str, origin: str, dest: str,
                reason: DisruptionReason, jurisdiction: str, transit: str | None) -> State:
    profile = PASSENGERS[pax_key]
    disruption = cancelled_event(f"{case_id}-FL", origin, dest, reason, jurisdiction, transit)
    return State(case_id=case_id, passenger_profile=profile, disruption=disruption)


# --------------------------------------------------------------------------- #
# Scenario builders -> (State, MockWorld, hitl_threshold, resume_human?)
# --------------------------------------------------------------------------- #
def scenario_happy():
    s = _make_state("HAPPY", "PAX-PLAT", "JFK", "LHR",
                    DisruptionReason.TECHNICAL, "EU", "GB")
    return s, MockWorld(), DEFAULT_HITL_THRESHOLD, None


def scenario_overbudget():
    s = _make_state("OVERBUDGET", "PAX-BASIC", "BOS", "LHR",
                    DisruptionReason.WEATHER, "US", "GB")
    return s, MockWorld(), DEFAULT_HITL_THRESHOLD, None


def scenario_visa():
    s = _make_state("VISA", "PAX-VISA", "LAX", "NRT",
                    DisruptionReason.TECHNICAL, "US", "JP")
    return s, MockWorld(), DEFAULT_HITL_THRESHOLD, None


def scenario_api_failure():
    s = _make_state("APIFAIL", "PAX-PLAT", "JFK", "LHR",
                    DisruptionReason.IT_OUTAGE, "EU", "GB")
    world = MockWorld()
    world.inject_flight_fault("JFK", "LHR", FlightFault(timeouts_remaining=1, then_empty=True))
    return s, world, DEFAULT_HITL_THRESHOLD, None


def scenario_hitl():
    s = _make_state("HITL", "PAX-PLAT", "JFK", "LHR",
                    DisruptionReason.TECHNICAL, "EU", "GB")
    # Lower threshold so the (otherwise fine) ~$2,980 resolution must be approved.
    return s, MockWorld(), 2500.0, True


SCENARIOS = {
    "happy": scenario_happy,
    "overbudget": scenario_overbudget,
    "visa": scenario_visa,
    "apifail": scenario_api_failure,
    "hitl": scenario_hitl,
}


# --------------------------------------------------------------------------- #
# Trace printing
# --------------------------------------------------------------------------- #
def print_trace(state: State) -> None:
    print(f"\n{'='*72}\nCASE {state.case_id}  |  passenger {state.passenger_profile.name} "
          f"({state.passenger_profile.loyalty_tier.label})")
    if state.assessment:
        a = state.assessment
        print(f"  triage: severity={a.severity.value} overnight={a.needs_overnight} "
              f"cap=${a.entitlement.budget_cap:.0f} ceiling={a.entitlement.cabin_ceiling.label}")
    print("  --- supervisor route + decisions ---")
    for d in state.decisions_log:
        print(f"    [{d.actor:10}] {d.decision}" + (f"  ({d.why})" if d.why else ""))
    print("  --- tool calls (audit) ---")
    for t in state.audit_trail:
        flag = "ok" if t.ok else f"ERR:{t.error}"
        print(f"    {t.tool:24} {flag:14} {t.result_summary}")
    if state.policy_verdicts:
        print("  --- policy verdicts ---")
        for v in state.policy_verdicts:
            print(f"    {v.flight_id:8} {v.verdict.value:9} ${v.total_cost:>7.0f}  "
                  f"{'; '.join(v.reasons)}")
    _print_outcome(state)


def _print_outcome(state: State) -> None:
    print(f"  --- OUTCOME: {state.status.value} ---")
    if state.escalation_reason:
        print(f"    escalation_reason: {state.escalation_reason}")
    if state.booking_pnr:
        print(f"    booking: {state.booking_pnr}")
    if state.final_message:
        print("    traveler message:")
        for line in state.final_message.splitlines():
            print(f"      | {line}")
    print(f"  stats: hops={state.hops} llm_calls={state.llm_calls} "
          f"search_attempts={state.search_attempts}")


def run_one(key: str, engine: str = "python", live: bool = False) -> State:
    state, world, threshold, resume = SCENARIOS[key]()
    toolbox = Toolbox(world, state)
    orch = _engine_cls(engine)(state, llm=get_llm(), hitl_threshold=threshold)

    if live:
        # Live mode: events stream from inside the run; print a header first and
        # only a compact outcome at the end (no re-dump of the logs).
        print(f"\n{'='*72}\nCASE {state.case_id}  |  passenger {state.passenger_profile.name} "
              f"({state.passenger_profile.loyalty_tier.label})  [streaming]")
        state = orch.run(toolbox)
        if resume and state.status == Status.AWAITING_HUMAN_APPROVAL:
            _print_outcome(state)
            print("\n  >>> human reviews and APPROVES the booking; resuming...\n")
            state = orch.resume_with_human_decision(approved=True)
        _print_outcome(state)
        return state

    state = orch.run(toolbox)
    if resume and state.status == Status.AWAITING_HUMAN_APPROVAL:
        print_trace(state)
        print("\n  >>> human reviews and APPROVES the booking; resuming...\n")
        state = orch.resume_with_human_decision(approved=True)
    print_trace(state)
    return state


def main() -> None:
    args = sys.argv[1:]
    engine = "python"
    live = False
    from .observability import Level, configure_tracer
    cleaned: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--engine":
            engine = args[i + 1] if i + 1 < len(args) else "langgraph"
            i += 2
            continue
        if a.startswith("--engine="):
            engine = a.split("=", 1)[1]
            i += 1
            continue
        if a in ("--stream", "-s"):
            configure_tracer(Level.STREAM); live = True; i += 1; continue
        if a in ("--verbose", "-v"):
            configure_tracer(Level.VERBOSE); live = True; i += 1; continue
        cleaned.append(a)
        i += 1
    llm_name = get_llm().name
    mode = "verbose" if live and get_tracer_level() == 2 else ("stream" if live else "summary")
    print(f"LLM backend: {llm_name} | engine: {engine} | output: {mode}")
    keys = cleaned or list(SCENARIOS)
    for k in keys:
        if k not in SCENARIOS:
            print(f"unknown scenario '{k}'. options: {', '.join(SCENARIOS)}")
            continue
        run_one(k, engine=engine, live=live)


def get_tracer_level() -> int:
    from .observability import get_tracer
    return int(get_tracer().level)


if __name__ == "__main__":
    main()
