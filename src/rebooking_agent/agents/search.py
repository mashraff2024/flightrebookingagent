"""Travel Search Agent -- the legwork specialist. Returns options, never decisions."""

from __future__ import annotations

from enum import Enum

from ..models import RankedShortlist, State
from .base import Agent


class SearchOutcome(str, Enum):
    OK = "ok"
    TOOL_FAILURE = "tool_failure"      # upstream API failed (timeout etc.)
    NO_AVAILABILITY = "no_availability"  # API ok, zero results


class SearchAgent(Agent):
    name = "search"
    system_prompt = (
        "You are the Travel Search agent. You are given a disruption assessment and the "
        "results of flight/hotel tool calls. Your only job is to rank the candidate flights "
        "best-first (legal cabin, then cost, then arrival time) and trim to a shortlist. "
        "You never book anything and you never invent a flight that is not in the results."
    )

    def run(self, state: State) -> SearchOutcome:
        a = state.assessment
        leg0 = state.disruption.original_itinerary.legs[0]
        origin, dest = leg0.origin, state.disruption.original_itinerary.legs[-1].destination

        flights_res = self.toolbox.search_flights(
            origin=origin,
            destination=dest,
            depart_after=leg0.depart,
            cabin_max=a.entitlement.cabin_ceiling,
            pax_count=state.passenger_profile.pax_count,
        )
        if not flights_res.ok:
            state.log_decision(self.name, "flight search failed", why=flights_res.error or "")
            return SearchOutcome.TOOL_FAILURE
        if not flights_res.data:
            state.log_decision(self.name, "no flights available", why="empty result set")
            return SearchOutcome.NO_AVAILABILITY

        candidates = flights_res.data

        # Soft LLM ranking -- guardrails still enforce the hard rules later.
        ranked: RankedShortlist = self._reason(
            state,
            task="rank_flights",
            user=f"Rank these {len(candidates)} flights for the passenger and trim to 5.",
            context={
                "candidates": candidates,
                "cabin_ceiling": a.entitlement.cabin_ceiling,
                "cabin_floor": a.entitlement.cabin_floor,
            },
            schema=RankedShortlist,
        )
        order = {fid: i for i, fid in enumerate(ranked.order)}
        candidates.sort(key=lambda f: order.get(f.flight_id, 999))
        state.flight_candidates = candidates[:5]

        # Hotels only if an overnight is in scope.
        if a.needs_overnight and a.entitlement.hotel_eligible:
            hotels_res = self.toolbox.search_hotels(
                location=origin,
                checkin=state.disruption.date,
                checkout=state.disruption.date,  # one night
                max_nightly_rate=a.entitlement.max_nightly_rate,
                loyalty_tier=state.passenger_profile.loyalty_tier,
            )
            state.hotel_candidates = hotels_res.data or []

        state.log_decision(
            self.name,
            f"{len(state.flight_candidates)} flight candidate(s), "
            f"{len(state.hotel_candidates)} hotel(s)",
            why=ranked.rationale,
        )
        return SearchOutcome.OK
