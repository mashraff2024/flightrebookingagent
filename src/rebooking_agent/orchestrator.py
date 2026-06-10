"""
Orchestrator / Supervisor -- the router.

This is a supervisor-worker graph over a shared State blackboard, written
explicitly so the control flow is inspectable. Control returns here after every
worker agent; the supervisor re-reads State and decides the next hop. It owns:

  * termination (hop + LLM-call ceilings),
  * the mandatory Policy gate (nothing reaches Comms unapproved),
  * search retry-on-failure (bounded),
  * one relax-and-retry loop back to Search,
  * the human-in-the-loop pause for high-value/irreversible actions,
  * graceful escalation when no compliant resolution exists.

LangGraph mapping: each `_run_*` is a node; `_decide_next` is the conditional
edge function; State is the graph state; the ceilings are recursion limits.
"""

from __future__ import annotations

from enum import Enum

from .agents import CommunicationAgent, PolicyAgent, SearchAgent, TriageAgent
from .agents.search import SearchOutcome
from .guardrails import DEFAULT_HITL_THRESHOLD, InputValidationError, relax_might_help, validate_case
from .llm import LLMClient, get_llm
from .models import State, Status
from .tools import Toolbox

MAX_HOPS = 12
MAX_LLM_CALLS = 8
MAX_SEARCH_ATTEMPTS = 2  # initial + one retry on upstream failure


class Route(str, Enum):
    TRIAGE = "triage"
    SEARCH = "search"
    POLICY = "policy"
    COMMS = "comms"
    FINALIZE = "finalize"
    ESCALATE = "escalate"
    HALT = "halt"


class Orchestrator:
    def __init__(self, state: State, llm: LLMClient | None = None,
                 hitl_threshold: float = DEFAULT_HITL_THRESHOLD):
        self.state = state
        self.hitl_threshold = hitl_threshold
        self.llm = llm or get_llm()
        self.toolbox: Toolbox | None = None  # bound in run()

    # ----- public API --------------------------------------------------------
    def run(self, toolbox: Toolbox) -> State:
        self.toolbox = toolbox
        self.triage = TriageAgent(self.llm, toolbox)
        self.search = SearchAgent(self.llm, toolbox)
        self.policy = PolicyAgent(self.llm, toolbox)
        self.comms = CommunicationAgent(self.llm, toolbox)

        try:
            validate_case(self.state.passenger_profile, self.state.disruption)
        except InputValidationError as e:
            return self._escalate(f"invalid input: {e}")

        while self.state.status == Status.IN_PROGRESS:
            if self.state.hops >= MAX_HOPS:
                return self._escalate("hop ceiling reached")
            if self.state.llm_calls >= MAX_LLM_CALLS:
                return self._escalate("LLM-call ceiling reached")

            self.state.hops += 1
            route = self._decide_next()
            self._log_route(route)

            if route == Route.TRIAGE:
                self.triage.run(self.state)
            elif route == Route.SEARCH:
                self._run_search()
            elif route == Route.POLICY:
                self.policy.run(self.state, self.hitl_threshold)
            elif route == Route.FINALIZE:
                self._finalize()
            elif route == Route.ESCALATE:
                return self._escalate(self.state.escalation_reason or "no compliant resolution")
            elif route == Route.HALT:
                break
        return self.state

    def resume_with_human_decision(self, approved: bool) -> State:
        """Called after an AWAITING_HUMAN_APPROVAL pause."""
        self.state.human_approved = approved
        if not approved:
            return self._escalate("human declined the high-value booking")
        self.state.status = Status.IN_PROGRESS
        self.state.log_decision("human", "approved high-value booking")
        self._finalize()
        return self.state

    # ----- routing -----------------------------------------------------------
    def _decide_next(self) -> Route:
        s = self.state
        if s.assessment is None:
            return Route.TRIAGE
        if not s.assessment.needs_rebooking:
            return Route.FINALIZE  # nothing to rebook (minor delay)
        if not s.flight_candidates and s.search_attempts < MAX_SEARCH_ATTEMPTS:
            return Route.SEARCH
        if not s.flight_candidates:
            return Route.ESCALATE
        if not s.policy_verdicts:
            return Route.POLICY
        if s.approved_option is not None:
            return Route.FINALIZE
        # Policy ran, approved nothing. Try one relaxed re-search if it might help.
        if self._relax_might_help() and not s.search_relaxed:
            s.search_relaxed = True
            s.policy_verdicts = []
            s.flight_candidates = []
            s.log_decision("supervisor", "relaxing constraints, re-routing to Search",
                           why="policy approved nothing; widening search once")
            return Route.SEARCH
        return Route.ESCALATE

    def _relax_might_help(self) -> bool:
        """Relaxing helps for availability/MCT issues, not for hard legality (visa)
        or budget caps -- those won't change by searching again."""
        reasons = " ".join(r for v in self.state.policy_verdicts for r in v.reasons)
        return relax_might_help(reasons)

    # ----- nodes -------------------------------------------------------------
    def _run_search(self) -> None:
        self.state.search_attempts += 1
        outcome = self.search.run(self.state)
        if outcome == SearchOutcome.TOOL_FAILURE and self.state.search_attempts < MAX_SEARCH_ATTEMPTS:
            self.state.log_decision("supervisor", "search failed; will retry",
                                    why=f"attempt {self.state.search_attempts}")
        elif outcome in (SearchOutcome.TOOL_FAILURE, SearchOutcome.NO_AVAILABILITY):
            self.state.escalation_reason = (
                "Flight search unavailable after retries." if outcome == SearchOutcome.TOOL_FAILURE
                else "No flights available for this route.")

    def _finalize(self) -> None:
        s = self.state
        opt = s.approved_option
        if opt is None:
            self._escalate("finalize called with no approved option")
            return
        # HITL gate: high-value / irreversible action pauses for a human.
        if opt.requires_human_approval and not s.human_approved:
            s.status = Status.AWAITING_HUMAN_APPROVAL
            s.log_decision("supervisor", "pausing for human approval",
                           why=f"total ${opt.total_cost:.0f} >= HITL threshold ${self.hitl_threshold:.0f}")
            return
        # Irreversible booking happens here, only after the gate is cleared.
        booking = self.toolbox.book_flight(opt.flight.flight_id, s.passenger_profile.passenger_id)
        s.booking_pnr = booking.data.pnr if booking.ok else None
        if opt.meal_voucher_amount:
            self.toolbox.issue_voucher(s.passenger_profile.passenger_id, "meal", opt.meal_voucher_amount)
        self.comms.run(s)
        s.status = Status.RESOLVED
        s.log_decision("supervisor", "case resolved", why=f"PNR {s.booking_pnr}")

    def _escalate(self, reason: str) -> State:
        self.state.status = Status.ESCALATED
        self.state.escalation_reason = reason
        self.state.log_decision("supervisor", "ESCALATED to human", why=reason)
        return self.state

    def _log_route(self, route: Route) -> None:
        self.state.log_decision("supervisor", f"route -> {route.value}",
                                why=f"hop {self.state.hops}")
