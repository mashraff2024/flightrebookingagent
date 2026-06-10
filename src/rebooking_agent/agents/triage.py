"""Triage Agent -- the intake specialist. Frames the problem; does no searching."""

from __future__ import annotations

from ..mock_data import EVENING
from ..models import Assessment, State
from .base import Agent


class TriageAgent(Agent):
    name = "triage"
    system_prompt = (
        "You are the Triage agent for airline irregular-operations. Given a passenger "
        "profile, a disruption event, and the computed entitlement, produce a structured "
        "assessment: severity, what the passenger needs (rebooking / overnight / meal "
        "voucher), and carry the hard constraints forward. You do not search for flights."
    )

    def run(self, state: State) -> None:
        # Deterministic policy math comes from a tool, not the model.
        ent_res = self.toolbox.get_passenger_entitlement()
        entitlement = ent_res.data

        leg = state.disruption.original_itinerary.legs[0]
        stranded_evening = state.disruption.original_itinerary.legs[0].depart.hour >= 18
        context = {
            "entitlement": entitlement,
            "cancelled": state.disruption.cancelled,
            "stranded_evening": stranded_evening,
            "must_arrive_by": state.disruption.original_itinerary.must_arrive_by,
        }
        user = (
            f"Disruption: {state.disruption.flight_number} {leg.origin}->{leg.destination} "
            f"cancelled={state.disruption.cancelled} reason={state.disruption.reason.value}. "
            f"Entitlement: cap=${entitlement.budget_cap:.0f}, "
            f"cabin_ceiling={entitlement.cabin_ceiling.label}, "
            f"hotel_eligible={entitlement.hotel_eligible}. "
            f"Produce the assessment."
        )
        assessment: Assessment = self._reason(
            state, task="triage", user=user, context=context, schema=Assessment)

        state.assessment = assessment
        state.log_decision(
            self.name,
            f"severity={assessment.severity.value}, overnight={assessment.needs_overnight}",
            why=entitlement.rationale,
        )
