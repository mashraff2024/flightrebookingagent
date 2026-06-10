"""Parity tests for the LangGraph engine.

These run the same five held-out scenarios through `graph.GraphOrchestrator`
and assert the same outcomes the plain engine produces in test_pipeline.py.
Passing both suites is the evidence that the framework swap changed only the
orchestration layer, not the behavior: same agents, same guardrails, same
results. Skipped automatically if langgraph isn't installed.
"""
import pytest

pytest.importorskip("langgraph")

from rebooking_agent.graph import GraphOrchestrator
from rebooking_agent.llm import get_llm
from rebooking_agent.models import CabinClass, Status
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
    orch = GraphOrchestrator(state, llm=get_llm(force_stub=True), hitl_threshold=threshold)
    return orch, orch.run(toolbox)


def test_graph_happy_books_business_with_hotel():
    _orch, state = _run(scenario_happy)
    assert state.status == Status.RESOLVED
    assert state.booking_pnr is not None
    assert state.approved_option.flight.flight_id == "VA-201"
    assert state.approved_option.flight.cabin == CabinClass.BUSINESS
    assert state.approved_option.hotel is not None


def test_graph_overbudget_escalates_without_booking():
    _orch, state = _run(scenario_overbudget)
    assert state.status == Status.ESCALATED
    assert state.booking_pnr is None
    assert all(not (t.tool == "book_flight" and t.ok) for t in state.audit_trail)


def test_graph_visa_rejected_and_escalates():
    _orch, state = _run(scenario_visa)
    assert state.status == Status.ESCALATED
    assert state.booking_pnr is None
    reasons = " ".join(r for v in state.policy_verdicts for r in v.reasons)
    assert ("permitted" in reasons.lower() or "CN" in reasons)
    assert "MCT" in reasons


def test_graph_api_failure_degrades_and_escalates():
    _orch, state = _run(scenario_api_failure)
    assert state.status == Status.ESCALATED
    assert state.booking_pnr is None
    assert state.search_attempts >= 2
    assert not state.flight_candidates


def test_graph_hitl_pauses_then_resolves_on_approval():
    orch, state = _run(scenario_hitl)
    assert state.status == Status.AWAITING_HUMAN_APPROVAL
    assert state.approved_option is not None
    assert state.booking_pnr is None
    resumed = orch.resume_with_human_decision(approved=True)
    assert resumed.status == Status.RESOLVED
    assert resumed.booking_pnr is not None


def test_graph_hitl_rejection_does_not_book():
    orch, state = _run(scenario_hitl)
    assert state.status == Status.AWAITING_HUMAN_APPROVAL
    resumed = orch.resume_with_human_decision(approved=False)
    assert resumed.status == Status.ESCALATED
    assert resumed.booking_pnr is None
