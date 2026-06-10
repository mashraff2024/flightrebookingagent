"""Optional FastAPI service that exposes the mock flight/hotel inventory as a
real HTTP boundary, per the brief's Phase-1 suggestion ("better, a small
FastAPI service so the boundary feels real").

The agent system does NOT require this to run -- the Toolbox talks to MockWorld
directly in-process, which keeps the demo and the tests dependency-free. This
module is here to show the tool boundary can be a network call without changing
any agent code: a thin HTTP-backed Toolbox would call these endpoints instead.

Run:
    pip install fastapi uvicorn
    PYTHONPATH=src uvicorn rebooking_agent.service:app --reload
"""
from __future__ import annotations

import datetime as dt

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "FastAPI is not installed. Run: pip install fastapi uvicorn"
    ) from exc

from .mock_data import MockWorld
from .models import CabinClass, LoyaltyTier
from .tools import (
    FlightStatus,
    SearchFlightsInput,
    SearchHotelsInput,
)

app = FastAPI(title="IRROPS Mock Travel APIs", version="1.0.0")

# A single process-wide world. In a real deployment each of these would be a
# separate upstream provider; here one MockWorld stands in for all of them.
WORLD = MockWorld()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search_flights")
def search_flights(req: SearchFlightsInput) -> list[dict]:
    """Mirror Toolbox.search_flights, minus fault injection (that lives in the
    in-process world used by the demo/tests)."""
    results = [
        f for f in WORLD.flights
        if f.origin == req.origin.upper()
        and f.destination == req.destination.upper()
        and f.depart >= req.depart_after
        and f.cabin <= req.cabin_max
    ]
    return [f.model_dump(mode="json") for f in results]


@app.post("/search_hotels")
def search_hotels(req: SearchHotelsInput) -> list[dict]:
    props = WORLD.hotels.get(req.location.upper(), [])
    results = [h for h in props if h.available and h.nightly_rate <= req.max_nightly_rate]
    results.sort(key=lambda h: (not h.loyalty_partner, h.nightly_rate))
    return [h.model_dump(mode="json") for h in results]


@app.get("/flight_status/{flight_number}")
def flight_status(flight_number: str, date: dt.date) -> dict:
    match = next((f for f in WORLD.flights if f.flight_id == flight_number), None)
    if match is None:
        # Unknown flight number -> report as scheduled rather than erroring.
        return FlightStatus(flight_number=flight_number, cancelled=False,
                            delay_minutes=0).model_dump(mode="json")
    return FlightStatus(flight_number=flight_number, cancelled=False,
                        delay_minutes=0).model_dump(mode="json")
