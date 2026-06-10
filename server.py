"""
Live web server for the flight rebooking agent.

Serves the interactive UI (web/index.html) and runs the REAL agents behind two
endpoints:

    POST /api/run     {scenario, engine}      -> runs the orchestrator
    POST /api/resume  {session, approved}     -> resumes a paused (HITL) case

Everything runs on the deterministic stub backend (no API key, no external
calls). The orchestrator instance for a paused case is held in memory keyed by
a session id, so the Approve/Reject buttons resume the actual run.

Local:
    pip install -r requirements.txt
    PYTHONPATH=src python -m uvicorn server:app --reload --port 8000
    # then open http://localhost:8000

Deploy (Railway): the included Procfile runs
    uvicorn server:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import os
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import app_helpers as h

app = FastAPI(title="Flight Rebooking Agent")

_HERE = os.path.dirname(__file__)
_INDEX = os.path.join(_HERE, "web", "index.html")

# In-memory session store for paused (awaiting-approval) cases.
# {session_id: (orchestrator, created_ts)}. Single-process; fine for a demo.
SESSIONS: dict[str, tuple] = {}
_SESSION_TTL = 3600


def _gc():
    now = time.time()
    for sid in [s for s, (_, ts) in SESSIONS.items() if now - ts > _SESSION_TTL]:
        SESSIONS.pop(sid, None)


@app.get("/", response_class=HTMLResponse)
def index():
    with open(_INDEX, encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health():
    return {"status": "ok", "scenarios": h.SCENARIO_ORDER}


class RunReq(BaseModel):
    scenario: str
    engine: str = "python"


@app.post("/api/run")
def run(req: RunReq):
    if req.scenario not in h.SCENARIO_INFO:
        raise HTTPException(400, "unknown scenario")
    if req.engine not in ("python", "langgraph"):
        raise HTTPException(400, "unknown engine")
    _gc()
    orch, state, trace = h.start_case(req.scenario, req.engine)
    payload = h.snapshot_dict(state, trace)
    if payload["awaiting"]:
        sid = uuid.uuid4().hex
        SESSIONS[sid] = (orch, time.time())
        payload["session"] = sid
    return payload


class ResumeReq(BaseModel):
    session: str
    approved: bool


@app.post("/api/resume")
def resume(req: ResumeReq):
    entry = SESSIONS.get(req.session)
    if not entry:
        raise HTTPException(404, "session expired or not found")
    orch, _ts = entry
    state, trace = h.resume_case(orch, req.approved)
    SESSIONS.pop(req.session, None)
    return h.snapshot_dict(state, trace)
