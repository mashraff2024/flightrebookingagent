"""
Policy & Compliance Agent -- the gate.

This agent makes NO LLM calls. Every approve/reject verdict is computed by the
deterministic guardrails over real numbers. The model is not trusted with money,
legality, or cabin eligibility. The agent selects the best *approved* option and
flags whether it crosses the human-in-the-loop threshold.
"""

from __future__ import annotations

from .. import guardrails as gr
from ..models import ApprovedOption, PolicyVerdict, State, Verdict
from .base import Agent


class PolicyAgent(Agent):
    name = "policy"
    system_prompt = "(deterministic gate; no model reasoning)"

    def run(self, state: State, hitl_threshold: float) -> None:
        a = state.assessment
        ent = a.entitlement
        profile = state.passenger_profile

        # Pick the cheapest eligible hotel if an overnight is owed.
        hotel = None
        if a.needs_overnight and ent.hotel_eligible and state.hotel_candidates:
            hotel = state.hotel_candidates[0]
        hotel_cost = hotel.nightly_rate if hotel else 0.0

        # Care gap: overnight owed but nothing bookable -> cannot self-resolve.
        if a.needs_overnight and ent.hotel_eligible and not state.hotel_candidates:
            state.escalation_reason = "Overnight care owed but no hotel available within policy."
            state.log_decision(self.name, "cannot fulfill hotel care duty",
                               why=state.escalation_reason)

        from ..observability import TRACER
        verdicts: list[PolicyVerdict] = []
        for f in state.flight_candidates:
            total = round(f.price + hotel_cost, 2)
            named_checks = [
                ("cabin", gr.cabin_ok(f, ent)),
                ("budget", gr.budget_ok(total, ent)),
                ("visa", gr.visa_ok(f, profile)),
                ("mct", gr.mct_ok(f)),
            ]
            for cname, (ok, reason) in named_checks:
                TRACER.check(f.flight_id, cname, ok, reason)
            checks = [c for _, c in named_checks]
            passed = all(ok for ok, _ in checks)
            reasons = [reason for ok, reason in checks if (not ok) or passed]
            verdicts.append(PolicyVerdict(
                flight_id=f.flight_id,
                verdict=Verdict.APPROVED if passed else Verdict.REJECTED,
                reasons=reasons,
                total_cost=total,
            ))
        state.policy_verdicts = verdicts

        # Select the first approved option in (already-ranked) order.
        approved_ids = state.approved_flight_ids()
        chosen = next((f for f in state.flight_candidates if f.flight_id in approved_ids), None)

        if chosen is None or (a.needs_overnight and ent.hotel_eligible and hotel is None):
            state.log_decision(self.name, "no compliant option approved",
                               why="all candidates rejected or care gap")
            return

        total = round(chosen.price + hotel_cost, 2)
        needs_human = gr.requires_human_approval(total, hitl_threshold)
        state.approved_option = ApprovedOption(
            flight=chosen,
            hotel=hotel,
            total_cost=total,
            requires_human_approval=needs_human,
            meal_voucher_amount=ent.meal_voucher_amount if a.needs_meal_voucher else 0.0,
        )
        verdict_summary = next(v for v in verdicts if v.flight_id == chosen.flight_id)
        state.log_decision(
            self.name,
            f"APPROVED {chosen.flight_id} @ ${total:.0f} (hitl={needs_human})",
            why="; ".join(verdict_summary.reasons),
        )
