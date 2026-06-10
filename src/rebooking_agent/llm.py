"""
Pluggable LLM client.

The whole system talks to the model through one method:

    llm.structured(task=..., system=..., user=..., context=..., schema=Model) -> Model

Two backends implement it identically:

  * AnthropicLLM -- real Claude. Forces schema-valid output by exposing the target
    Pydantic schema as a single tool and pinning tool_choice to it. This is the
    robust way to get structured JSON out of a tool-calling model.
  * StubLLM -- deterministic stand-in. Produces the *same* validated schema using
    simple heuristics over `context`. This is what lets the pipeline run with no
    API key and lets tests be reproducible without a live model (per the brief).

Either way, hard decisions (money/legality/cabin) are NOT made here -- they live
in guardrails.py. The model only proposes, ranks, and writes prose.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from .models import (
    Assessment,
    DraftMessage,
    Entitlement,
    RankedShortlist,
    Severity,
)


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def structured(self, *, task: str, system: str, user: str,
                   context: dict, schema: type[BaseModel]) -> BaseModel: ...


# --------------------------------------------------------------------------- #
# Deterministic stub
# --------------------------------------------------------------------------- #
class StubLLM:
    """Deterministic reasoning stand-in. Dispatches on `task`."""

    name = "stub"

    def structured(self, *, task: str, system: str, user: str,
                   context: dict, schema: type[BaseModel]) -> BaseModel:
        handler = getattr(self, f"_task_{task}", None)
        if handler is None:
            raise ValueError(f"StubLLM has no handler for task '{task}'")
        return handler(context, schema)

    # --- triage ------------------------------------------------------------- #
    def _task_triage(self, ctx: dict, schema) -> Assessment:
        ent: Entitlement = ctx["entitlement"]
        cancelled: bool = ctx["cancelled"]
        evening: bool = ctx["stranded_evening"]
        needs_overnight = bool(cancelled and evening)
        severity = Severity.SEVERE if needs_overnight else (
            Severity.MODERATE if cancelled else Severity.MINOR)
        return Assessment(
            severity=severity,
            needs_rebooking=bool(cancelled),
            needs_overnight=needs_overnight,
            needs_meal_voucher=ent.meal_voucher_eligible,
            entitlement=ent,
            must_arrive_by=ctx.get("must_arrive_by"),
            notes=f"Auto-triaged: {severity.value}; overnight={needs_overnight}.",
        )

    # --- rank_flights ------------------------------------------------------- #
    def _task_rank_flights(self, ctx: dict, schema) -> RankedShortlist:
        cands = ctx["candidates"]  # list[FlightOption]
        ceiling = ctx["cabin_ceiling"]
        floor = ctx.get("cabin_floor", ceiling)
        # Soft preference only — the Policy gate is authoritative. We rank
        # cabin-appropriate options first: anything above the ceiling is an
        # over-correction and anything below the floor is an unowed downgrade,
        # so both sink to the bottom. Within the owed band we prefer the
        # higher cabin (what the traveler is actually entitled to), then the
        # cheaper fare, then the earlier arrival.
        def key(f):
            out_of_band = (f.cabin > ceiling) or (f.cabin < floor)
            return (out_of_band, -int(f.cabin), f.price, f.arrive)

        ordered = sorted(cands, key=key)
        return RankedShortlist(
            order=[f.flight_id for f in ordered],
            rationale=(
                "Ranked by cabin appropriateness (within the owed band first), "
                "then by entitled cabin, price, and arrival time."
            ),
        )

    # --- draft_message ------------------------------------------------------ #
    def _task_draft_message(self, ctx: dict, schema) -> DraftMessage:
        name = ctx["name"]
        f = ctx["flight"]
        hotel = ctx.get("hotel")
        voucher = ctx.get("meal_voucher_amount", 0.0)
        dep = f["depart"] if isinstance(f, dict) else f.depart
        carrier = f["carrier"] if isinstance(f, dict) else f.carrier
        cabin = f["cabin_label"] if isinstance(f, dict) else f.cabin.label
        lines = [
            f"Dear {name},",
            "",
            "We're sorry your flight was disrupted. Here's how we've rebooked you:",
            f"  - Flight: {carrier} in {cabin}, departing {dep}.",
        ]
        if hotel:
            hname = hotel["name"] if isinstance(hotel, dict) else hotel.name
            lines.append(f"  - Hotel: {hname} for tonight, arranged at no cost to you.")
        if voucher:
            lines.append(f"  - Meal voucher: ${voucher:.0f} for use at the airport.")
        lines += ["", "No action is needed on your part. Safe travels.",
                  "— Customer Care"]
        return DraftMessage(message="\n".join(lines))


# --------------------------------------------------------------------------- #
# Real Anthropic client (optional import)
# --------------------------------------------------------------------------- #
class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str | None = None):
        import anthropic  # imported lazily so the stub path needs no dependency
        self._client = anthropic.Anthropic()
        self._model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    def structured(self, *, task: str, system: str, user: str,
                   context: dict, schema: type[BaseModel]) -> BaseModel:
        from .observability import TRACER
        tool = {
            "name": "emit",
            "description": f"Emit the result for task '{task}' as structured data.",
            "input_schema": schema.model_json_schema(),
        }
        kwargs = dict(
            model=self._model,
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit"},
        )

        if TRACER.wants_tokens():
            # Live feed: with forced tool-use the model streams the tool input
            # JSON (its structured answer) token by token. Surface those deltas.
            with self._client.messages.stream(**kwargs) as stream:
                for event in stream:
                    if getattr(event, "type", None) == "content_block_delta":
                        delta = event.delta
                        piece = getattr(delta, "partial_json", None) or getattr(delta, "text", None)
                        if piece:
                            TRACER.token(piece)
                msg = stream.get_final_message()
            TRACER.token("\n")
        else:
            msg = self._client.messages.create(**kwargs)

        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                return schema.model_validate(block.input)
        raise RuntimeError("model did not return structured tool_use output")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_llm(force_stub: bool = False) -> LLMClient:
    """Real Claude if a key + SDK are present, else the deterministic stub."""
    if force_stub or not os.environ.get("ANTHROPIC_API_KEY"):
        return StubLLM()
    try:
        return AnthropicLLM()
    except Exception:
        return StubLLM()
