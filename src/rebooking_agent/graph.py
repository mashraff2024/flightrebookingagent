"""
LangGraph implementation of the same supervisor-worker system.

This is the framework version the brief recommends. It deliberately reuses
*every* piece of domain logic from the plain-Python build unchanged -- the
agents (`agents/`), the deterministic tools (`tools.py`), the authoritative
guardrails (`guardrails.py`), the pluggable LLM (`llm.py`), and the shared
`State` blackboard (`models.py`). Only the orchestration layer differs:

  plain `orchestrator.py`            LangGraph (`graph.py`)
  ---------------------------------  -----------------------------------------
  `_decide_next()` if/elif ladder    conditional edges (router functions)
  the explicit `while` loop          the compiled `StateGraph` runtime
  `MAX_HOPS` guard                    `recursion_limit` in the run config
  `AWAITING_HUMAN_APPROVAL` + a       `interrupt()` + a checkpointer; resume
    separate `resume_*` method          with `Command(resume=...)`
  in-place State mutation            nodes mutate State and return {"state": ...}

Because the nodes are thin wrappers over the existing agents, the two engines
produce identical outcomes on all five scenarios (asserted in
tests/test_graph_pipeline.py). That parity is the point: the design is the
asset; the framework is swappable.
"""
from __future__ import annotations

import logging
from typing import Optional, TypedDict

# The in-memory checkpointer round-trips our Pydantic State through msgpack and
# logs a deprecation note for each custom enum/model type. The round-trip is
# safe here (trusted, in-process), so we quiet that specific cosmetic logger.
logging.getLogger("langgraph_checkpoint.serde.jsonplus").setLevel(logging.ERROR)
logging.getLogger("langgraph.checkpoint.serde.jsonplus").setLevel(logging.ERROR)

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from .agents import CommunicationAgent, PolicyAgent, SearchAgent, TriageAgent
from .agents.search import SearchOutcome
from .guardrails import (
    DEFAULT_HITL_THRESHOLD,
    InputValidationError,
    relax_might_help,
    validate_case,
)
from .llm import LLMClient, get_llm
from .models import State, Status
from .tools import Toolbox

# Same ceilings as the plain orchestrator. MAX_HOPS maps onto LangGraph's
# recursion_limit; the LLM-call ceiling is enforced inside the routers.
MAX_HOPS = 12
MAX_LLM_CALLS = 8
MAX_SEARCH_ATTEMPTS = 2


# --------------------------------------------------------------------------- #
# Graph state: a thin envelope carrying the shared blackboard. We keep the whole
# `State` object as a single channel and let nodes mutate it in place, which is
# what lets us reuse the existing agents verbatim.
# --------------------------------------------------------------------------- #
class GraphState(TypedDict):
    state: State


# Per-run dependencies are injected through config["configurable"], so the graph
# topology is built once and reused across cases.
class Deps:
    def __init__(self, toolbox: Toolbox, llm: LLMClient, hitl_threshold: float):
        self.toolbox = toolbox
        self.hitl_threshold = hitl_threshold
        self.triage = TriageAgent(llm, toolbox)
        self.search = SearchAgent(llm, toolbox)
        self.policy = PolicyAgent(llm, toolbox)
        self.comms = CommunicationAgent(llm, toolbox)


def _deps(config) -> Deps:
    return config["configurable"]["deps"]


# --------------------------------------------------------------------------- #
# Nodes -- each is a thin wrapper over an existing agent / the existing tools.
# --------------------------------------------------------------------------- #
def triage_node(gs: GraphState, config) -> GraphState:
    s = gs["state"]
    deps = _deps(config)
    deps.toolbox.state = s  # tools record onto the live, checkpointed state
    s.hops += 1
    s.log_decision("supervisor", "route -> triage", why=f"hop {s.hops}")
    deps.triage.run(s)
    return {"state": s}


def search_node(gs: GraphState, config) -> GraphState:
    s = gs["state"]
    deps = _deps(config)
    deps.toolbox.state = s
    s.hops += 1
    s.log_decision("supervisor", "route -> search", why=f"hop {s.hops}")
    s.search_attempts += 1
    outcome = deps.search.run(s)
    if outcome == SearchOutcome.TOOL_FAILURE and s.search_attempts < MAX_SEARCH_ATTEMPTS:
        s.log_decision("supervisor", "search failed; will retry",
                       why=f"attempt {s.search_attempts}")
    elif outcome in (SearchOutcome.TOOL_FAILURE, SearchOutcome.NO_AVAILABILITY):
        s.escalation_reason = (
            "Flight search unavailable after retries."
            if outcome == SearchOutcome.TOOL_FAILURE
            else "No flights available for this route.")
    return {"state": s}


def policy_node(gs: GraphState, config) -> GraphState:
    s = gs["state"]
    deps = _deps(config)
    deps.toolbox.state = s
    s.hops += 1
    s.log_decision("supervisor", "route -> policy", why=f"hop {s.hops}")
    deps.policy.run(s, deps.hitl_threshold)
    return {"state": s}


def finalize_node(gs: GraphState, config) -> GraphState:
    """Books, vouchers, and drafts the message -- but only after the HITL gate.

    HITL uses LangGraph's `interrupt()`: when human sign-off is required the node
    suspends and the graph returns control to the caller. On resume via
    `Command(resume=<bool>)`, `interrupt()` returns that decision and the node
    continues. The irreversible booking is placed *after* the interrupt, so a
    pause/replay can never double-book.
    """
    s = gs["state"]
    deps = _deps(config)
    deps.toolbox.state = s
    s.hops += 1
    s.log_decision("supervisor", "route -> finalize", why=f"hop {s.hops}")
    opt = s.approved_option
    if opt is None:
        s.status = Status.ESCALATED
        s.escalation_reason = "finalize reached with no approved option"
        return {"state": s}

    if opt.requires_human_approval and not s.human_approved:
        decision = interrupt({
            "kind": "approval_required",
            "flight_id": opt.flight.flight_id,
            "total_cost": opt.total_cost,
            "threshold": deps.hitl_threshold,
            "reason": f"total ${opt.total_cost:.0f} >= HITL threshold ${deps.hitl_threshold:.0f}",
        })
        approved = decision.get("approved") if isinstance(decision, dict) else bool(decision)
        s.human_approved = bool(approved)
        if not s.human_approved:
            s.status = Status.ESCALATED
            s.escalation_reason = "human declined the high-value booking"
            s.log_decision("human", "declined high-value booking")
            return {"state": s}
        s.log_decision("human", "approved high-value booking")

    booking = deps.toolbox.book_flight(opt.flight.flight_id, s.passenger_profile.passenger_id)
    s.booking_pnr = booking.data.pnr if booking.ok else None
    if opt.meal_voucher_amount:
        deps.toolbox.issue_voucher(s.passenger_profile.passenger_id, "meal", opt.meal_voucher_amount)
    deps.comms.run(s)
    s.status = Status.RESOLVED
    s.log_decision("supervisor", "case resolved", why=f"PNR {s.booking_pnr}")
    return {"state": s}


def escalate_node(gs: GraphState, config) -> GraphState:
    s = gs["state"]
    s.status = Status.ESCALATED
    if not s.escalation_reason:
        s.escalation_reason = "no compliant resolution"
    s.log_decision("supervisor", "ESCALATED to human", why=s.escalation_reason)
    return {"state": s}


# --------------------------------------------------------------------------- #
# Conditional edges -- these mirror `_decide_next()` exactly.
# --------------------------------------------------------------------------- #
def _ceiling_hit(s: State) -> Optional[str]:
    if s.llm_calls >= MAX_LLM_CALLS:
        s.escalation_reason = "LLM-call ceiling reached"
        return "escalate"
    return None


def route_after_triage(gs: GraphState) -> str:
    s = gs["state"]
    if (hit := _ceiling_hit(s)):
        return hit
    if not s.assessment or not s.assessment.needs_rebooking:
        return "finalize"  # nothing to rebook
    return "search"


def route_after_search(gs: GraphState) -> str:
    s = gs["state"]
    if (hit := _ceiling_hit(s)):
        return hit
    if s.flight_candidates:
        return "policy"
    if s.search_attempts < MAX_SEARCH_ATTEMPTS:
        return "search"  # bounded retry on failure
    return "escalate"


def route_after_policy(gs: GraphState) -> str:
    s = gs["state"]
    if (hit := _ceiling_hit(s)):
        return hit
    if s.approved_option is not None:
        return "finalize"
    reasons = " ".join(r for v in s.policy_verdicts for r in v.reasons)
    if relax_might_help(reasons) and not s.search_relaxed:
        s.search_relaxed = True
        s.policy_verdicts = []
        s.flight_candidates = []
        s.log_decision("supervisor", "relaxing constraints, re-routing to Search",
                       why="policy approved nothing; widening search once")
        return "search"
    return "escalate"


def route_after_finalize(gs: GraphState) -> str:
    # If finalize set an escalation (no option / human declined), record it.
    s = gs["state"]
    if s.status == Status.ESCALATED:
        return "escalate"
    return END


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("triage", triage_node)
    g.add_node("search", search_node)
    g.add_node("policy", policy_node)
    g.add_node("finalize", finalize_node)
    g.add_node("escalate", escalate_node)

    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", route_after_triage,
                            {"search": "search", "finalize": "finalize", "escalate": "escalate"})
    g.add_conditional_edges("search", route_after_search,
                            {"policy": "policy", "search": "search", "escalate": "escalate"})
    g.add_conditional_edges("policy", route_after_policy,
                            {"finalize": "finalize", "search": "search", "escalate": "escalate"})
    g.add_conditional_edges("finalize", route_after_finalize,
                            {"escalate": "escalate", END: END})
    g.add_edge("escalate", END)

    return g.compile(checkpointer=MemorySaver())


# --------------------------------------------------------------------------- #
# Driver: same surface as the plain Orchestrator, so run.py can use either.
# --------------------------------------------------------------------------- #
class GraphOrchestrator:
    """Adapter giving the LangGraph engine the same run()/resume() API as the
    plain Orchestrator, so the demo and tests are engine-agnostic."""

    _GRAPH = None  # compiled once, topology is stateless

    def __init__(self, state: State, llm: LLMClient | None = None,
                 hitl_threshold: float = DEFAULT_HITL_THRESHOLD):
        self.state = state
        self.hitl_threshold = hitl_threshold
        self.llm = llm or get_llm()
        if GraphOrchestrator._GRAPH is None:
            GraphOrchestrator._GRAPH = build_graph()
        self.graph = GraphOrchestrator._GRAPH
        self._thread = {"configurable": {"thread_id": state.case_id}}

    def run(self, toolbox: Toolbox) -> State:
        try:
            validate_case(self.state.passenger_profile, self.state.disruption)
        except InputValidationError as e:
            self.state.status = Status.ESCALATED
            self.state.escalation_reason = f"invalid input: {e}"
            self.state.log_decision("supervisor", "ESCALATED to human",
                                    why=self.state.escalation_reason)
            return self.state

        deps = Deps(toolbox, self.llm, self.hitl_threshold)
        config = {
            "configurable": {"thread_id": self.state.case_id, "deps": deps},
            "recursion_limit": MAX_HOPS * 3,  # generous; ceilings escalate first
        }
        self._thread = config
        self.graph.invoke({"state": self.state}, config=config)
        snap = self.graph.get_state(config)
        # Under checkpointing the graph operates on serialized copies, so read
        # the authoritative state back from the snapshot rather than the
        # original object.
        self.state = snap.values["state"]
        if snap.next:  # pending work remains -> we are paused on the HITL interrupt
            self.state.status = Status.AWAITING_HUMAN_APPROVAL
            opt = self.state.approved_option
            if opt is not None:
                self.state.log_decision(
                    "supervisor", "pausing for human approval",
                    why=f"total ${opt.total_cost:.0f} >= HITL threshold ${self.hitl_threshold:.0f}")
        return self.state

    def resume_with_human_decision(self, approved: bool) -> State:
        from langgraph.types import Command
        self.graph.invoke(Command(resume={"approved": approved}), config=self._thread)
        self.state = self.graph.get_state(self._thread).values["state"]
        return self.state
