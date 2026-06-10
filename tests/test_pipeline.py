"""Integration tests: run each held-out scenario end-to-end through the
orchestrator using the deterministic StubLLM (no live model, fully repeatable).

These assert the *outcomes the brief grades on*: the happy path resolves with
the owed Business cabin; every failure mode escalates honestly instead of
over-booking or hallucinating; and the high-value case pauses for a human.
"""
from rebooking_agent.llm import get_llm
from rebooking_agent.models import CabinClass, Status
from rebooking_agent.orchestrator import Orchestrator
from rebooking_agent.tools import Toolbox
from rebooking_agent.run import (
    scenario_happy,
    scenario_overbudget,
    scenario_visa,
    scenario_api_failure,
    scenario_hitl,
)


def _run(builder):
    state, world, threshold, _resume = builder()
    toolbox = Toolbox(world, state)
    orch = Orchestrator(state, llm=get_llm(force_stub=True), hitl_threshold=threshold)
    return orch, orch.run(toolbox)


# --------------------------------------------------------------------------- #
# (a) Happy path: Platinum owed Business + overnight hotel.
# --------------------------------------------------------------------------- #
def test_happy_path_books_business_with_hotel():
    _orch, state = _run(scenario_happy)
    assert state.status == Status.RESOLVED
    assert state.booking_pnr is not None
    assert state.approved_option is not None
    # Must be the Business reroute (VA-201), NOT the cheaper Economy downgrade.
    assert state.approved_option.flight.flight_id == "VA-201"
    assert state.approved_option.flight.cabin == CabinClass.BUSINESS
    # Overnight care was arranged.
    assert state.approved_option.hotel is not None
    # Exactly one irreversible booking happened.
    booked = [t for t in state.audit_trail if t.tool == "book_flight" and t.ok]
    assert len(booked) == 1


# --------------------------------------------------------------------------- #
# (b) Basic Economy, only flight over budget -> MUST escalate, MUST NOT book.
# --------------------------------------------------------------------------- #
def test_overbudget_escalates_without_booking():
    _orch, state = _run(scenario_overbudget)
    assert state.status == Status.ESCALATED
    assert state.booking_pnr is None
    assert all(not (t.tool == "book_flight" and t.ok) for t in state.audit_trail)
    assert state.escalation_reason
    # The only candidate was rejected on budget.
    assert any(v.verdict.value == "rejected" and "exceeds cap" in " ".join(v.reasons)
               for v in state.policy_verdicts)


# --------------------------------------------------------------------------- #
# (c) Only reroute transits a visa-restricted country -> MUST reject.
# --------------------------------------------------------------------------- #
def test_visa_restricted_route_rejected_and_escalates():
    _orch, state = _run(scenario_visa)
    assert state.status == Status.ESCALATED
    assert state.booking_pnr is None
    reasons = " ".join(r for v in state.policy_verdicts for r in v.reasons)
    # One option blocked on visa/transit, the other on MCT -- both compliance.
    assert "permitted" in reasons.lower() or "CN" in reasons
    assert "MCT" in reasons


# --------------------------------------------------------------------------- #
# (d) Search times out, then returns empty -> degrade gracefully, no hallucination.
# --------------------------------------------------------------------------- #
def test_api_failure_degrades_and_escalates():
    _orch, state = _run(scenario_api_failure)
    assert state.status == Status.ESCALATED
    assert state.booking_pnr is None
    assert state.approved_option is None
    # It retried (one timeout + one empty) and stopped at the cap, not forever.
    assert state.search_attempts >= 2
    # A timeout was recorded as a structured failure, not an unhandled crash.
    assert any(t.tool == "search_flights" and not t.ok for t in state.audit_trail)
    # No flight was invented.
    assert not state.flight_candidates


# --------------------------------------------------------------------------- #
# (e) Booking above HITL threshold -> pause, then resolve only after approval.
# --------------------------------------------------------------------------- #
def test_hitl_pauses_then_resolves_on_approval():
    orch, state = _run(scenario_hitl)
    # First it must PAUSE, with a prepared recommendation but no booking.
    assert state.status == Status.AWAITING_HUMAN_APPROVAL
    assert state.approved_option is not None
    assert state.booking_pnr is None
    assert all(not (t.tool == "book_flight" and t.ok) for t in state.audit_trail)

    # Human approves -> now it books and resolves.
    resumed = orch.resume_with_human_decision(approved=True)
    assert resumed.status == Status.RESOLVED
    assert resumed.booking_pnr is not None


def test_hitl_human_rejection_does_not_book():
    orch, state = _run(scenario_hitl)
    assert state.status == Status.AWAITING_HUMAN_APPROVAL
    resumed = orch.resume_with_human_decision(approved=False)
    assert resumed.status == Status.ESCALATED
    assert resumed.booking_pnr is None


# --------------------------------------------------------------------------- #
# Cross-cutting: loop/cost ceilings respected on every scenario.
# --------------------------------------------------------------------------- #
def test_loop_and_cost_ceilings_respected():
    from rebooking_agent.orchestrator import MAX_HOPS, MAX_LLM_CALLS
    for builder in (scenario_happy, scenario_overbudget, scenario_visa,
                    scenario_api_failure, scenario_hitl):
        _orch, state = _run(builder)
        assert state.hops <= MAX_HOPS
        assert state.llm_calls <= MAX_LLM_CALLS


# --------------------------------------------------------------------------- #
# Cross-cutting: PII never leaks into the public snapshot.
# --------------------------------------------------------------------------- #
def test_public_snapshot_redacts_passport():
    _orch, state = _run(scenario_happy)
    snap = repr(state.public_snapshot())
    assert state.passenger_profile.passport_number not in snap
