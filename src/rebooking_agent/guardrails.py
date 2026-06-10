"""
Deterministic guardrails -- the authoritative layer.

Nothing here asks the model anything. These functions compute, in plain Python
over real data, whether an option is legal, affordable, and within entitlement.
The Policy Agent *explains* a rejection in prose; the rejection itself is decided
here. This is the "hard guardrail" in the defense-in-depth stack:

    prompt (soft, helpful)  ->  these checks (hard, authoritative)  ->  HITL (backstop)
"""

from __future__ import annotations

import re

from .models import (
    CabinClass,
    DisruptionEvent,
    DisruptionReason,
    Entitlement,
    FlightOption,
    LoyaltyTier,
    PassengerProfile,
)

# Causes for which the carrier is liable (drives US care duties; EU261 care
# applies regardless of cause, only cash compensation is waived for the rest).
CARRIER_LIABLE = {DisruptionReason.TECHNICAL, DisruptionReason.CREW, DisruptionReason.IT_OUTAGE}

_BASE_CAP_BY_CABIN = {
    CabinClass.BASIC_ECONOMY: 600.0,
    CabinClass.ECONOMY: 1000.0,
    CabinClass.PREMIUM_ECONOMY: 1800.0,
    CabinClass.BUSINESS: 3500.0,
    CabinClass.FIRST: 6000.0,
}
_LOYALTY_MULT = {
    LoyaltyTier.NONE: 1.0,
    LoyaltyTier.SILVER: 1.1,
    LoyaltyTier.GOLD: 1.2,
    LoyaltyTier.PLATINUM: 1.3,
}
_MEAL_VOUCHER = 25.0
_INTL_MCT_MINUTES = 60  # required minimum connection time for international connections

DEFAULT_HITL_THRESHOLD = 3000.0


# --------------------------------------------------------------------------- #
# Entitlement: what the airline OWES (a ceiling, not what's available)
# --------------------------------------------------------------------------- #
def compute_entitlement(profile: PassengerProfile, disruption: DisruptionEvent) -> Entitlement:
    liable = disruption.reason in CARRIER_LIABLE

    # Cabin band: floor = the cabin the passenger actually paid for (never
    # downgrade below it; Basic-Economy floors up to standard Economy as a
    # reroute product). Ceiling = max allowed; Platinum gets a one-class
    # courtesy bump (capped at Business) when the carrier is at fault.
    floor = max(profile.fare_class, CabinClass.ECONOMY)
    ceiling = floor
    if liable and profile.loyalty_tier >= LoyaltyTier.PLATINUM:
        ceiling = CabinClass(min(int(CabinClass.BUSINESS), int(floor) + 1))

    cap = _BASE_CAP_BY_CABIN[ceiling] * _LOYALTY_MULT[profile.loyalty_tier]

    # Care duties (hotel/meal): EU261 -> always on overnight; US/DOT -> only if liable.
    eu = disruption.jurisdiction.upper() == "EU"
    care = eu or liable
    max_nightly = 250.0 if profile.loyalty_tier >= LoyaltyTier.GOLD else 200.0

    return Entitlement(
        budget_cap=round(cap, 2),
        cabin_ceiling=ceiling,
        cabin_floor=floor,
        hotel_eligible=care,
        max_nightly_rate=max_nightly,
        meal_voucher_eligible=care,
        meal_voucher_amount=_MEAL_VOUCHER if care else 0.0,
        rationale=(
            f"jurisdiction={disruption.jurisdiction}, reason={disruption.reason.value}, "
            f"carrier_liable={liable}, tier={profile.loyalty_tier.label}"
        ),
    )


# --------------------------------------------------------------------------- #
# Per-option hard checks. Each returns (ok, reason).
# --------------------------------------------------------------------------- #
def cabin_ok(option: FlightOption, ent: Entitlement) -> tuple[bool, str]:
    if option.cabin > ent.cabin_ceiling:
        return False, f"cabin {option.cabin.label} exceeds ceiling {ent.cabin_ceiling.label}"
    if option.cabin < ent.cabin_floor:
        return False, f"cabin {option.cabin.label} is a downgrade below {ent.cabin_floor.label}"
    return True, f"cabin {option.cabin.label} within band [{ent.cabin_floor.label}..{ent.cabin_ceiling.label}]"


def budget_ok(total_cost: float, ent: Entitlement) -> tuple[bool, str]:
    if total_cost <= ent.budget_cap + 1e-6:
        return True, f"total ${total_cost:.0f} within cap ${ent.budget_cap:.0f}"
    return False, f"total ${total_cost:.0f} exceeds cap ${ent.budget_cap:.0f}"


def visa_ok(option: FlightOption, profile: PassengerProfile) -> tuple[bool, str]:
    permitted = {c.upper() for c in profile.permitted_countries}
    for country in option.transit_countries:
        if country not in permitted:
            return False, f"transits {country}; passenger not permitted to enter/transit"
    return True, "all transit countries permitted"


def mct_ok(option: FlightOption) -> tuple[bool, str]:
    for i, mins in enumerate(option.connection_minutes):
        if mins < _INTL_MCT_MINUTES:
            airport = option.legs[i].destination
            return False, f"connection at {airport} is {mins}m < required {_INTL_MCT_MINUTES}m MCT"
    return True, "all connections meet MCT"


def requires_human_approval(total_cost: float, threshold: float) -> bool:
    return total_cost >= threshold


def relax_might_help(verdict_reasons: str) -> bool:
    """Whether widening the search could change the outcome. Relaxing helps for
    availability/MCT issues, but NOT for hard legality (visa) or budget/cabin
    caps -- those won't change by searching again. Shared by both the plain and
    the LangGraph orchestrators so the retry policy can't drift between them."""
    if ("not permitted" in verdict_reasons
            or "exceeds cap" in verdict_reasons
            or "exceeds ceiling" in verdict_reasons):
        return False
    return True


# --------------------------------------------------------------------------- #
# Input validation + PII / injection hygiene
# --------------------------------------------------------------------------- #
class InputValidationError(ValueError):
    pass


def validate_case(profile: PassengerProfile, disruption: DisruptionEvent) -> None:
    """Reject malformed cases before any agent sees them."""
    if profile.pax_count < 1:
        raise InputValidationError("pax_count must be >= 1")
    leg = disruption.original_itinerary.legs[0]
    if leg.origin == leg.destination:
        raise InputValidationError("origin equals destination")
    if not re.fullmatch(r"[A-Z]{3}", leg.origin) or not re.fullmatch(r"[A-Z]{3}", leg.destination):
        raise InputValidationError("airport codes must be 3 uppercase letters")


_INJECTION_PAT = re.compile(
    r"(ignore (all|previous)|disregard.*instructions|system prompt|you are now)", re.I)


def sanitize_free_text(text: str) -> str:
    """Neutralize obvious prompt-injection in attacker-controllable free text.
    Free-text passenger messages are data, never instructions."""
    return _INJECTION_PAT.sub("[redacted]", text)
