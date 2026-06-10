"""Communication / Resolution Agent -- the closer. Writes prose; makes no policy calls."""

from __future__ import annotations

from ..models import DraftMessage, State
from .base import Agent


class CommunicationAgent(Agent):
    name = "comms"
    system_prompt = (
        "You are the Communication agent for an airline. Given a single, already-approved "
        "resolution (a flight, optionally a hotel and meal voucher), write a clear, warm, "
        "on-brand message to the traveler explaining what has been arranged and what, if "
        "anything, they need to do. Do not make or second-guess policy decisions."
    )

    def run(self, state: State) -> None:
        opt = state.approved_option
        f = opt.flight
        context = {
            "name": state.passenger_profile.name,
            "flight": {
                "carrier": f.carrier,
                "cabin_label": f.cabin.label,
                "depart": str(f.depart),
            },
            "hotel": {"name": opt.hotel.name} if opt.hotel else None,
            "meal_voucher_amount": opt.meal_voucher_amount,
        }
        draft: DraftMessage = self._reason(
            state,
            task="draft_message",
            user="Write the traveler message for the approved resolution.",
            context=context,
            schema=DraftMessage,
        )
        state.final_message = draft.message
        state.log_decision(self.name, "drafted traveler message", why="resolution communicated")
