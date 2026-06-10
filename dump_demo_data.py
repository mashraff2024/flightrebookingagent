"""Capture real agent outputs for all scenarios + engines into demo_data.json,
to be embedded in the standalone HTML demo (so it shows genuine results)."""
import json
import sys, os
sys.path.insert(0, "src")

import app_helpers as h
from rebooking_agent.models import Status


def snapshot(state):
    out = h.outcome_summary(state)
    return {
        "stages": h.derive_stages(state),
        "rows": [
            {
                "flight": r["flight"], "cabin": r["cabin"],
                "price": r["price"], "total": r["total"], "verdict": r["verdict"],
                "checks": {k: {"ok": v[0], "reason": v[1]} for k, v in r["checks"].items()},
            } for r in h.build_check_rows(state)
        ],
        "outcome": out,
    }


def disruption_of(state):
    d = state.disruption
    return {"reason": d.reason.value, "jurisdiction": d.jurisdiction,
            "flight_number": d.flight_number,
            "route": f"{d.original_itinerary.legs[0].origin}\u2192{d.original_itinerary.legs[-1].destination}"}


DEMO = {}
for key in ["happy", "overbudget", "visa", "apifail", "hitl"]:
    title, blurb = h.SCENARIO_INFO[key]
    DEMO[key] = {"title": title, "blurb": blurb, "engines": {}}
    for eng in ["python", "langgraph"]:
        orch, state, trace = h.start_case(key, eng)
        snap = snapshot(state)
        snap["trace"] = trace
        snap["passenger"] = {"name": state.passenger_profile.name,
                             "tier": state.passenger_profile.loyalty_tier.label}
        snap["disruption"] = disruption_of(state)
        # HITL: capture both branches so the page's Approve/Reject buttons are real.
        if state.status == Status.AWAITING_HUMAN_APPROVAL:
            pax = snap["passenger"]; dis = snap["disruption"]
            orch_a, state_a, _ta = h.start_case(key, eng)
            sa2, tr_a2 = h.resume_case(orch_a, True)
            snap["approved"] = snapshot(sa2)
            snap["approved"].update({"passenger": pax, "disruption": dis,
                                     "trace": trace + "\n\n--- after human APPROVED ---\n" + tr_a2})
            orch_r, state_r, _tr = h.start_case(key, eng)
            sr2, tr_r2 = h.resume_case(orch_r, False)
            snap["rejected"] = snapshot(sr2)
            snap["rejected"].update({"passenger": pax, "disruption": dis,
                                     "trace": trace + "\n\n--- after human REJECTED ---\n" + tr_r2})
        DEMO[key]["engines"][eng] = snap

json.dump(DEMO, open("demo_data.json", "w"), indent=1, default=str)
print("wrote demo_data.json:", os.path.getsize("demo_data.json"), "bytes")
print("scenarios:", list(DEMO))
