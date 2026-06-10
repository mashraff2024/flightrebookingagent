"""
Typed domain model for the IRROPS rebooking agent.

Everything that crosses an agent boundary, a tool boundary, or lands in the
shared State is a Pydantic model defined here. This is the single place to look
to understand the data contracts of the whole system.

Design notes
------------
* Cabin classes are an *ordered* enum so guardrails can compare them numerically
  ("is BUSINESS <= the cabin ceiling?") without scattering magic strings.
* `State` is the blackboard / single source of truth. Agents read the fields
  they need and write only the fields they own (enforced by convention + the
  orchestrator, mirroring LangGraph state reducers).
* PII (passport number) lives in the profile but is *redacted* whenever the
  state is serialized for logs/handoff -- see `State.public_snapshot()`.
"""

from __future__ import annotations

import datetime as dt
from enum import IntEnum, Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class CabinClass(IntEnum):
    """Ordered so that `CabinClass.BUSINESS > CabinClass.ECONOMY` is meaningful."""

    BASIC_ECONOMY = 0
    ECONOMY = 1
    PREMIUM_ECONOMY = 2
    BUSINESS = 3
    FIRST = 4

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


class LoyaltyTier(IntEnum):
    NONE = 0
    SILVER = 1
    GOLD = 2
    PLATINUM = 3

    @property
    def label(self) -> str:
        return self.name.title()


class DisruptionReason(str, Enum):
    WEATHER = "weather"                 # carrier usually NOT liable (force majeure)
    TECHNICAL = "technical"             # carrier liable -> EU261 / DOT care duties
    CREW = "crew"                       # carrier liable
    IT_OUTAGE = "it_outage"             # carrier liable
    AIR_TRAFFIC_CONTROL = "atc"         # generally not carrier liable


class Severity(str, Enum):
    MINOR = "minor"        # short delay, no rebooking needed
    MODERATE = "moderate"  # rebooking needed, same day
    SEVERE = "severe"      # rebooking + overnight


class Status(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"
    AWAITING_HUMAN_APPROVAL = "AWAITING_HUMAN_APPROVAL"


class Verdict(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


# --------------------------------------------------------------------------- #
# Static inputs: passenger + disruption
# --------------------------------------------------------------------------- #
class PassengerProfile(BaseModel):
    passenger_id: str
    name: str
    loyalty_tier: LoyaltyTier = LoyaltyTier.NONE
    fare_class: CabinClass = CabinClass.ECONOMY
    home_airport: str
    passport_country: str                       # ISO-2, e.g. "US"
    passport_number: str = Field(repr=False)    # PII: never repr'd, redacted in logs
    # Countries the passenger may legally enter/transit (visa-free OR holds a visa).
    permitted_countries: list[str] = Field(default_factory=list)
    pax_count: int = 1

    @field_validator("passport_country")
    @classmethod
    def _country_code(cls, v: str) -> str:
        return v.upper().strip()


class Itinerary(BaseModel):
    """The passenger's original booked journey (may be multi-leg)."""

    legs: list["FlightLeg"]
    must_arrive_by: Optional[dt.datetime] = None  # e.g. cruise departure, wedding


class DisruptionEvent(BaseModel):
    flight_number: str
    date: dt.date
    reason: DisruptionReason
    cancelled: bool = True
    delay_minutes: int = 0
    original_itinerary: Itinerary
    jurisdiction: str = "EU"  # "EU" -> EU261, "US" -> DOT; drives entitlement rules


# --------------------------------------------------------------------------- #
# Flights & hotels (shared by tools and candidates)
# --------------------------------------------------------------------------- #
class FlightLeg(BaseModel):
    origin: str
    destination: str
    depart: dt.datetime
    arrive: dt.datetime
    transit_country: Optional[str] = None  # country this leg lands/transits in (ISO-2)


class FlightOption(BaseModel):
    flight_id: str
    carrier: str
    cabin: CabinClass
    price: float
    legs: list[FlightLeg]
    stops: int = 0
    # Minimum connection time *available* at each connection, in minutes.
    # len == len(legs) - 1. Compared against required MCT in guardrails.
    connection_minutes: list[int] = Field(default_factory=list)

    @property
    def origin(self) -> str:
        return self.legs[0].origin

    @property
    def destination(self) -> str:
        return self.legs[-1].destination

    @property
    def depart(self) -> dt.datetime:
        return self.legs[0].depart

    @property
    def arrive(self) -> dt.datetime:
        return self.legs[-1].arrive

    @property
    def transit_countries(self) -> list[str]:
        out: list[str] = []
        for leg in self.legs:
            if leg.transit_country:
                out.append(leg.transit_country.upper())
        return out


class HotelOption(BaseModel):
    hotel_id: str
    name: str
    nightly_rate: float
    distance_km: float
    available: bool = True
    loyalty_partner: bool = False


# --------------------------------------------------------------------------- #
# Agent outputs (structured, schema-validated)
# --------------------------------------------------------------------------- #
class Entitlement(BaseModel):
    """What the airline OWES under policy -- computed, then used as a hard ceiling."""

    budget_cap: float
    cabin_ceiling: CabinClass
    cabin_floor: CabinClass = CabinClass.ECONOMY
    hotel_eligible: bool
    max_nightly_rate: float
    meal_voucher_eligible: bool
    meal_voucher_amount: float = 0.0
    rationale: str = ""


class Assessment(BaseModel):
    """Triage output: frames the problem for everyone downstream."""

    severity: Severity
    needs_rebooking: bool
    needs_overnight: bool
    needs_meal_voucher: bool
    entitlement: Entitlement
    must_arrive_by: Optional[dt.datetime] = None
    notes: str = ""


class PolicyVerdict(BaseModel):
    """One verdict per candidate flight option."""

    flight_id: str
    verdict: Verdict
    reasons: list[str] = Field(default_factory=list)  # human-readable, code-derived
    total_cost: float  # flight (+ hotel if attached), computed in Python


class ApprovedOption(BaseModel):
    flight: FlightOption
    hotel: Optional[HotelOption] = None
    total_cost: float
    requires_human_approval: bool = False
    meal_voucher_amount: float = 0.0


# ----- structured outputs for individual LLM "reasoning" steps -------------- #
class RankedShortlist(BaseModel):
    """Search Agent's LLM step: orders candidate flight_ids best-first + trims."""

    order: list[str]
    rationale: str = ""


class DraftMessage(BaseModel):
    """Communication Agent's LLM step: the traveler-facing prose."""

    message: str


# --------------------------------------------------------------------------- #
# Audit + decision logging
# --------------------------------------------------------------------------- #
class DecisionLogEntry(BaseModel):
    ts: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    actor: str                  # which agent / the supervisor
    decision: str               # what it decided
    why: str = ""               # one-line reason


class ToolCallRecord(BaseModel):
    ts: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    tool: str
    args: dict = Field(default_factory=dict)
    ok: bool = True
    error: Optional[str] = None
    result_summary: str = ""    # summary, not the raw blob (context discipline)


# --------------------------------------------------------------------------- #
# THE shared State / blackboard
# --------------------------------------------------------------------------- #
class State(BaseModel):
    """Single source of truth. Threaded through the supervisor->worker graph."""

    case_id: str
    passenger_profile: PassengerProfile
    disruption: DisruptionEvent

    assessment: Optional[Assessment] = None
    flight_candidates: list[FlightOption] = Field(default_factory=list)
    hotel_candidates: list[HotelOption] = Field(default_factory=list)
    policy_verdicts: list[PolicyVerdict] = Field(default_factory=list)
    approved_option: Optional[ApprovedOption] = None
    final_message: Optional[str] = None

    decisions_log: list[DecisionLogEntry] = Field(default_factory=list)
    audit_trail: list[ToolCallRecord] = Field(default_factory=list)

    status: Status = Status.IN_PROGRESS
    hops: int = 0               # supervisor hops, capped to prevent runaway
    llm_calls: int = 0          # token/cost ceiling proxy
    search_attempts: int = 0    # flight-search attempts (retry-on-failure limit)
    escalation_reason: Optional[str] = None
    search_relaxed: bool = False  # set when supervisor loops back to Search
    human_approved: Optional[bool] = None  # HITL decision, once made
    booking_pnr: Optional[str] = None      # set after the irreversible book_flight

    # ----- helpers -----------------------------------------------------------
    def log_decision(self, actor: str, decision: str, why: str = "") -> None:
        self.decisions_log.append(DecisionLogEntry(actor=actor, decision=decision, why=why))
        from .observability import TRACER
        TRACER.decision(actor, decision, why)

    def log_tool(self, rec: ToolCallRecord) -> None:
        self.audit_trail.append(rec)
        from .observability import TRACER
        TRACER.tool(rec.tool, rec.ok, getattr(rec, "result_summary", "") or "", rec.error or "")

    def approved_flight_ids(self) -> list[str]:
        return [v.flight_id for v in self.policy_verdicts if v.verdict == Verdict.APPROVED]

    def public_snapshot(self) -> dict:
        """Serializable view with PII redacted -- safe for logs/handoff to humans."""
        data = self.model_dump(mode="json")
        try:
            data["passenger_profile"]["passport_number"] = "REDACTED"
        except (KeyError, TypeError):
            pass
        return data


# resolve forward refs
Itinerary.model_rebuild()
