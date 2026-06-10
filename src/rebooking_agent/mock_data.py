"""
Mock world: synthetic-but-deliberate data.

The inventory is hand-authored (not random) so every graded edge case is
reproducible: a sold-out hotel, an MCT-violating connection, a route with zero
availability, an in-budget-but-cabin-capped option, an available-but-over-budget
option, and a reroute that can only be flown through a visa-restricted country.

`MockWorld` also supports *fault injection* (timeout then empty) so the
"API fails -> degrade gracefully, don't hallucinate" scenario is testable.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .models import (
    CabinClass,
    DisruptionEvent,
    DisruptionReason,
    FlightLeg,
    FlightOption,
    HotelOption,
    Itinerary,
    LoyaltyTier,
    PassengerProfile,
)

# Fixed reference clock so runs are deterministic.
DISRUPTION_DATE = dt.date(2025, 11, 15)
EVENING = dt.datetime(2025, 11, 15, 21, 30)  # stranded at ~9:30pm


def _dt(day_offset: int, hour: int, minute: int = 0) -> dt.datetime:
    return dt.datetime.combine(DISRUPTION_DATE, dt.time(hour, minute)) + dt.timedelta(days=day_offset)


# --------------------------------------------------------------------------- #
# Passengers
# --------------------------------------------------------------------------- #
PASSENGERS: dict[str, PassengerProfile] = {
    "PAX-PLAT": PassengerProfile(
        passenger_id="PAX-PLAT",
        name="Dana Okafor",
        loyalty_tier=LoyaltyTier.PLATINUM,
        fare_class=CabinClass.BUSINESS,
        home_airport="JFK",
        passport_country="US",
        passport_number="X1234567",
        permitted_countries=["US", "GB", "IE", "FR", "DE", "NL"],
        pax_count=1,
    ),
    "PAX-BASIC": PassengerProfile(
        passenger_id="PAX-BASIC",
        name="Sam Reyes",
        loyalty_tier=LoyaltyTier.NONE,
        fare_class=CabinClass.BASIC_ECONOMY,
        home_airport="BOS",
        passport_country="US",
        passport_number="X7654321",
        permitted_countries=["US", "GB"],
        pax_count=1,
    ),
    "PAX-VISA": PassengerProfile(
        passenger_id="PAX-VISA",
        name="Lena Vasquez",
        loyalty_tier=LoyaltyTier.GOLD,
        fare_class=CabinClass.ECONOMY,
        home_airport="LAX",
        passport_country="US",
        passport_number="X5559999",
        # Notably does NOT include CN -> a reroute transiting China is illegal.
        permitted_countries=["US", "JP", "KR"],
        pax_count=1,
    ),
}


# --------------------------------------------------------------------------- #
# Flight inventory  (hand-crafted edge cases flagged in comments)
# --------------------------------------------------------------------------- #
def _flight_inventory() -> list[FlightOption]:
    return [
        # --- JFK -> LHR : the happy-path route (Platinum, overnight next morning) ---
        FlightOption(  # in-budget Business, next morning -> triggers overnight
            flight_id="VA-201",
            carrier="Virgin Atlantic",
            cabin=CabinClass.BUSINESS,
            price=2800.0,
            legs=[FlightLeg(origin="JFK", destination="LHR",
                            depart=_dt(1, 8, 30), arrive=_dt(1, 20, 30),
                            transit_country="GB")],
            stops=0,
        ),
        FlightOption(  # cheaper economy alt on same route (cabin below ceiling -> fine)
            flight_id="BA-178",
            carrier="British Airways",
            cabin=CabinClass.ECONOMY,
            price=900.0,
            legs=[FlightLeg(origin="JFK", destination="LHR",
                            depart=_dt(1, 9, 45), arrive=_dt(1, 21, 40),
                            transit_country="GB")],
            stops=0,
        ),
        FlightOption(  # EDGE: First class, way over even a Platinum cap -> budget reject
            flight_id="BA-001",
            carrier="British Airways",
            cabin=CabinClass.FIRST,
            price=9100.0,
            legs=[FlightLeg(origin="JFK", destination="LHR",
                            depart=_dt(1, 10, 0), arrive=_dt(1, 22, 5),
                            transit_country="GB")],
            stops=0,
        ),

        # --- BOS -> LHR : Basic-economy passenger, only option is over the cap ---
        FlightOption(  # EDGE: available but price > basic-economy budget cap -> escalate
            flight_id="AA-302",
            carrier="American",
            cabin=CabinClass.ECONOMY,
            price=1450.0,
            legs=[FlightLeg(origin="BOS", destination="LHR",
                            depart=_dt(1, 7, 0), arrive=_dt(1, 18, 50),
                            transit_country="GB")],
            stops=0,
        ),

        # --- LAX -> NRT : visa passenger; the only reroute transits CN (restricted) ---
        FlightOption(  # EDGE: connection via PEK (China) -> visa/compliance reject
            flight_id="CA-988",
            carrier="Air China",
            cabin=CabinClass.ECONOMY,
            price=1100.0,
            legs=[
                FlightLeg(origin="LAX", destination="PEK",
                          depart=_dt(1, 11, 0), arrive=_dt(1, 16, 0), transit_country="CN"),
                FlightLeg(origin="PEK", destination="NRT",
                          depart=_dt(1, 19, 0), arrive=_dt(1, 22, 30), transit_country="JP"),
            ],
            stops=1,
            connection_minutes=[180],  # MCT fine; it's the visa that kills it
        ),
        FlightOption(  # EDGE: legal routing (via ICN, Korea) BUT violates MCT
            flight_id="OZ-450",
            carrier="Asiana",
            cabin=CabinClass.ECONOMY,
            price=1150.0,
            legs=[
                FlightLeg(origin="LAX", destination="ICN",
                          depart=_dt(1, 12, 0), arrive=_dt(1, 17, 0), transit_country="KR"),
                FlightLeg(origin="ICN", destination="NRT",
                          depart=_dt(1, 17, 25), arrive=_dt(1, 19, 30), transit_country="JP"),
            ],
            stops=1,
            connection_minutes=[25],  # EDGE: 25 min < required MCT -> reject
        ),

        # NB: route SFO -> SIN intentionally has NO inventory -> "no availability".
    ]


# --------------------------------------------------------------------------- #
# Hotel inventory
# --------------------------------------------------------------------------- #
def _hotel_inventory() -> dict[str, list[HotelOption]]:
    return {
        "JFK": [
            HotelOption(hotel_id="HX-JFK-1", name="TWA Hotel", nightly_rate=180.0,
                        distance_km=0.4, available=True, loyalty_partner=True),
            HotelOption(hotel_id="HX-JFK-2", name="Airport Inn", nightly_rate=320.0,
                        distance_km=3.1, available=True, loyalty_partner=False),  # over typical cap
            HotelOption(hotel_id="HX-JFK-3", name="Skyline Suites", nightly_rate=150.0,
                        distance_km=2.0, available=False, loyalty_partner=True),  # EDGE: sold out
        ],
        "BOS": [
            HotelOption(hotel_id="HX-BOS-1", name="Harbor Lodge", nightly_rate=210.0,
                        distance_km=1.2, available=True, loyalty_partner=False),
        ],
        "LAX": [
            HotelOption(hotel_id="HX-LAX-1", name="Pacific Rest", nightly_rate=190.0,
                        distance_km=1.0, available=True, loyalty_partner=True),
        ],
    }


# --------------------------------------------------------------------------- #
# MockWorld with fault injection
# --------------------------------------------------------------------------- #
@dataclass
class FlightFault:
    """Injected fault for a route. `timeouts_then_empty` simulates a flaky API:
    the first N calls raise TimeoutError, subsequent calls return []."""

    timeouts_remaining: int = 0
    then_empty: bool = False


@dataclass
class MockWorld:
    flights: list[FlightOption] = field(default_factory=_flight_inventory)
    hotels: dict[str, list[HotelOption]] = field(default_factory=_hotel_inventory)
    faults: dict[tuple[str, str], FlightFault] = field(default_factory=dict)

    def inject_flight_fault(self, origin: str, dest: str, fault: FlightFault) -> None:
        self.faults[(origin.upper(), dest.upper())] = fault


# --------------------------------------------------------------------------- #
# Disruption builders for the scenarios
# --------------------------------------------------------------------------- #
def cancelled_event(flight_number: str, origin: str, dest: str,
                    reason: DisruptionReason, jurisdiction: str,
                    transit: str | None) -> DisruptionEvent:
    return DisruptionEvent(
        flight_number=flight_number,
        date=DISRUPTION_DATE,
        reason=reason,
        cancelled=True,
        original_itinerary=Itinerary(
            legs=[FlightLeg(origin=origin, destination=dest,
                            depart=EVENING, arrive=_dt(0, 23, 59),
                            transit_country=transit)],
        ),
        jurisdiction=jurisdiction,
    )
