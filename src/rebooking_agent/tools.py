"""
Tool layer == the mock API surface the agents call via function-calling.

Every tool:
  * has an explicit Pydantic input schema (so the model gets typed args),
  * returns a `ToolResult[T]` envelope -- never raises into the agent loop,
  * appends a *summary* (not the raw payload) to `state.audit_trail`.

`book_flight` is the irreversible one and is gated by the HITL guardrail before
the orchestrator ever lets it be called.
"""

from __future__ import annotations

import datetime as dt
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from . import guardrails as gr
from .models import (
    CabinClass,
    Entitlement,
    FlightOption,
    HotelOption,
    LoyaltyTier,
    State,
    ToolCallRecord,
)
from .mock_data import MockWorld

T = TypeVar("T")


class ToolResult(BaseModel, Generic[T]):
    ok: bool
    data: Optional[T] = None
    error: Optional[str] = None


# ----- tool input schemas (drive function-calling JSON schemas) ------------- #
class SearchFlightsInput(BaseModel):
    origin: str
    destination: str
    depart_after: dt.datetime
    cabin_max: CabinClass
    pax_count: int = 1


class SearchHotelsInput(BaseModel):
    location: str
    checkin: dt.date
    checkout: dt.date
    max_nightly_rate: float
    loyalty_tier: LoyaltyTier = LoyaltyTier.NONE


class FlightStatus(BaseModel):
    flight_number: str
    cancelled: bool
    delay_minutes: int
    reason: Optional[str] = None


class Booking(BaseModel):
    pnr: str
    flight_id: str
    passenger_id: str
    confirmed_at: dt.datetime


class Voucher(BaseModel):
    code: str
    type: str
    amount: float


class Toolbox:
    """Bound to a world + a state, so calls are recorded centrally."""

    def __init__(self, world: MockWorld, state: State):
        self.world = world
        self.state = state

    def _record(self, tool: str, args: dict, ok: bool,
                error: str | None = None, summary: str = "") -> None:
        self.state.log_tool(ToolCallRecord(
            tool=tool, args=args, ok=ok, error=error, result_summary=summary))

    # --- get_flight_status ---------------------------------------------------
    def get_flight_status(self, flight_number: str, date: dt.date) -> ToolResult[FlightStatus]:
        d = self.state.disruption
        status = FlightStatus(
            flight_number=flight_number,
            cancelled=d.cancelled,
            delay_minutes=d.delay_minutes,
            reason=d.reason.value,
        )
        self._record("get_flight_status", {"flight_number": flight_number, "date": str(date)},
                     ok=True, summary=f"cancelled={status.cancelled}")
        return ToolResult[FlightStatus](ok=True, data=status)

    # --- search_flights (with fault injection) -------------------------------
    def search_flights(self, **kwargs) -> ToolResult[list[FlightOption]]:
        try:
            inp = SearchFlightsInput(**kwargs)
        except ValidationError as e:
            self._record("search_flights", kwargs, ok=False, error="validation")
            return ToolResult[list[FlightOption]](ok=False, error=f"invalid args: {e.error_count()} errors")

        key = (inp.origin.upper(), inp.destination.upper())
        fault = self.world.faults.get(key)
        if fault and fault.timeouts_remaining > 0:
            fault.timeouts_remaining -= 1
            self._record("search_flights", inp.model_dump(mode="json"),
                         ok=False, error="timeout")
            return ToolResult[list[FlightOption]](ok=False, error="upstream timeout")

        results = [
            f for f in self.world.flights
            if f.origin == inp.origin.upper()
            and f.destination == inp.destination.upper()
            and f.depart >= inp.depart_after
            and f.cabin <= inp.cabin_max
        ]
        if fault and fault.then_empty:
            results = []

        self._record("search_flights", inp.model_dump(mode="json"),
                     ok=True, summary=f"{len(results)} flights")
        return ToolResult[list[FlightOption]](ok=True, data=results)

    # --- search_hotels -------------------------------------------------------
    def search_hotels(self, **kwargs) -> ToolResult[list[HotelOption]]:
        try:
            inp = SearchHotelsInput(**kwargs)
        except ValidationError as e:
            self._record("search_hotels", kwargs, ok=False, error="validation")
            return ToolResult[list[HotelOption]](ok=False, error=f"invalid args: {e.error_count()} errors")

        props = self.world.hotels.get(inp.location.upper(), [])
        results = [h for h in props if h.available and h.nightly_rate <= inp.max_nightly_rate]
        # prefer loyalty partners, then cheapest
        results.sort(key=lambda h: (not h.loyalty_partner, h.nightly_rate))
        self._record("search_hotels", inp.model_dump(mode="json"),
                     ok=True, summary=f"{len(results)} hotels")
        return ToolResult[list[HotelOption]](ok=True, data=results)

    # --- get_passenger_entitlement (deterministic policy math) ---------------
    def get_passenger_entitlement(self) -> ToolResult[Entitlement]:
        ent = gr.compute_entitlement(self.state.passenger_profile, self.state.disruption)
        self._record("get_passenger_entitlement", {"jurisdiction": self.state.disruption.jurisdiction},
                     ok=True, summary=f"cap=${ent.budget_cap:.0f} ceiling={ent.cabin_ceiling.label}")
        return ToolResult[Entitlement](ok=True, data=ent)

    # --- book_flight (IRREVERSIBLE -- gated upstream by HITL) -----------------
    def book_flight(self, flight_id: str, passenger_id: str) -> ToolResult[Booking]:
        booking = Booking(
            pnr=f"PNR-{flight_id[-3:]}{passenger_id[-3:]}",
            flight_id=flight_id,
            passenger_id=passenger_id,
            confirmed_at=dt.datetime.now(dt.timezone.utc),
        )
        self._record("book_flight", {"flight_id": flight_id, "passenger_id": passenger_id},
                     ok=True, summary=f"PNR {booking.pnr}")
        return ToolResult[Booking](ok=True, data=booking)

    # --- issue_voucher -------------------------------------------------------
    def issue_voucher(self, passenger_id: str, type: str, amount: float) -> ToolResult[Voucher]:
        v = Voucher(code=f"V-{type[:3].upper()}-{passenger_id[-3:]}", type=type, amount=amount)
        self._record("issue_voucher", {"passenger_id": passenger_id, "type": type, "amount": amount},
                     ok=True, summary=f"{v.code} ${amount:.0f}")
        return ToolResult[Voucher](ok=True, data=v)
