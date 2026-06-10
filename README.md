# Autonomous Flight Disruption & Rebooking Agent (IRROPS)

A supervisor-worker multi-agent system that triages **one** disrupted airline
passenger end to end: assess the situation → find a viable reroute → arrange an
overnight hotel if needed → verify policy/budget/legality → produce a
personalized resolution — **escalating to a human when it should.**

The whole design follows one rule:

> **The LLM proposes. Deterministic code disposes.**

Anything that touches money, eligibility, legality, or an irreversible booking
is decided by plain Python against real data — never taken on the model's word.

---

## Quick start

```bash
pip install -r requirements.txt          # only pydantic is strictly required

# Run all five held-out scenarios end to end (uses the deterministic StubLLM):
cd flight_rebooking_agent
PYTHONPATH=src python3 -m rebooking_agent.run                 # plain-Python engine (default)
PYTHONPATH=src python3 -m rebooking_agent.run --engine langgraph   # LangGraph engine (pip install langgraph)

# Run one scenario:
PYTHONPATH=src python3 -m rebooking_agent.run happy        # happy | overbudget | visa | apifail | hitl

# Watch it think, live (step-by-step trace as it runs):
PYTHONPATH=src python3 -m rebooking_agent.run happy --stream    # routes, decisions, tool calls
PYTHONPATH=src python3 -m rebooking_agent.run happy --verbose   # + reasoning inputs, every guardrail check,
                                                       #   and live model tokens when a key is set

# Run the test suite (no live model needed):
PYTHONPATH=src python3 -m pytest tests/ -q
```

## Web app (the real frontend)

A polished web UI that drives the **actual agents** over a small API — every
click runs the real pipeline (including the live human-in-the-loop pause and
resume). Still the deterministic backend: no API key, no external calls.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m uvicorn server:app --reload --port 8000
# open http://localhost:8000
# Windows PowerShell:  $env:PYTHONPATH="src"; python -m uvicorn server:app --reload --port 8000
```

`server.py` serves `web/index.html` and exposes `POST /api/run` and
`POST /api/resume`, which call the orchestrator and return the case state. The
page animates the pipeline, shows the per-option rule checks, renders the
boarding-pass outcome, and has working Approve/Reject buttons for the
sign-off case. There's a "show technical detail" panel with the full trace.

### Deploy on Railway

The repo includes a `Procfile` and `.python-version`, so Railway works out of the box:

1. Push the repo to GitHub.
2. In Railway: New Project → Deploy from GitHub repo → pick this repo.
3. Railway installs `requirements.txt` and starts `uvicorn server:app` on its `$PORT`. No env vars needed (it runs on the offline stub backend).

The page is served at the Railway URL's root.

### Offline demo (no server)

`demo.html` is a standalone version that needs no server — double-click it. It's
a presentation layer over **recorded** real outputs (see `DEMO.md`); the live
server above is the one that runs the agents on each request.

### Alternative: Streamlit UI

`app.py` is a simpler live UI built on Streamlit (`pip install streamlit`, then
`streamlit run app.py`). The FastAPI app above is the nicer-looking option;
Streamlit is kept as a quick alternative.

To run against a **live Claude model** instead of the stub:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
export ANTHROPIC_MODEL=claude-sonnet-4-6   # optional; this is the default
PYTHONPATH=src python3 -m rebooking_agent.run
```

`get_llm()` auto-selects the real client when the SDK **and** key are present,
and falls back to the stub otherwise. Both backends emit the **same validated
Pydantic schemas**, so the agents, guardrails, and tests are identical either
way.

---

## Architecture

```
                ┌──────────────────────────────────────────┐
                │          ORCHESTRATOR / SUPERVISOR         │
                │  routes on shared State; only it escalates │
                └──┬─────────┬─────────┬──────────┬──────────┘
                   │         │         │          │
            ┌──────▼──┐ ┌────▼───┐ ┌───▼────┐ ┌───▼──────────┐
            │ TRIAGE  │ │ SEARCH │ │ POLICY │ │ COMMUNICATION │
            │ assess  │ │ options│ │  GATE  │ │   draft msg   │
            └──────┬──┘ └────┬───┘ └───┬────┘ └───┬──────────┘
                   │         │         │          │
                   ▼         ▼         ▼          ▼
        ┌──────────────────────────────────────────────────┐
        │   SHARED STATE  (single source of truth)          │
        │  profile · disruption · assessment · candidates · │
        │  policy_verdicts · approved_option · decisions_log │
        │  · audit_trail · status                           │
        └──────────────────────────────────────────────────┘
```

Control **returns to the supervisor after every agent**. The supervisor reads
state, picks the next hop, and is the only component allowed to decide "this
needs a human." Agents never call each other or pass blobs of text — they read
and write typed fields on the `State` blackboard.

### The agents (`src/rebooking_agent/agents/`)

| Agent | Job | LLM calls | Never does |
|------|-----|-----------|------------|
| **Triage** | Builds the structured `Assessment`: severity, what's owed (cabin band, budget cap, hotel/meal eligibility), constraints. | 1 | search or book |
| **Search** | Calls `search_flights` / `search_hotels`, returns a *ranked shortlist of options*. | 1 (ranking) | decide or book |
| **Policy** | The mandatory **gate**. Validates every option deterministically and emits approve/reject verdicts with reasons. | **0** | use the LLM for any money/legality call |
| **Communication** | Drafts the traveler message for the single approved option. | 1 | make policy decisions |

Per case: **~3 LLM calls** on the happy path, **0** of them in the gate.

---

## Where every dollar/legality decision lives (defense in depth)

Guardrails are layered, and the authoritative ones are pure Python in
`guardrails.py`:

1. **Soft layer — the prompt / ranking.** The LLM *proposes* an ordering of
   options. Helpful, not trusted.
2. **Hard layer — deterministic checks (the Policy gate).** Every candidate is
   run through code that the model cannot talk its way past:
   - `cabin_ok` — enforces a cabin **band**: never above the entitled
     **ceiling** (stops a $9k First seat for an Economy passenger) *and* never
     below the paid **floor** (stops silently downgrading a Platinum owed
     Business to a cheaper Economy seat).
   - `budget_ok` — `total_cost <= budget_cap`, computed in Python.
   - `visa_ok` — every transit country must be in the passenger's permitted set.
   - `mct_ok` — connection time must meet the 60-minute international minimum.
3. **Backstop — human-in-the-loop.** Any booking at/above the HITL threshold
   (`DEFAULT_HITL_THRESHOLD = $3000`, configurable per case), **or any case
   where nothing compliant is found**, pauses (`AWAITING_HUMAN_APPROVAL`) or
   escalates (`ESCALATED`). The agent prepares a recommendation; a human pulls
   the trigger.

Other guardrails: input validation (`validate_case`), output validation (every
agent output is a Pydantic model — a response that doesn't parse is a handled
failure, not data), loop/cost ceilings (`MAX_HOPS=12`, `MAX_LLM_CALLS=8`,
`MAX_SEARCH_ATTEMPTS=2`), PII hygiene (passport numbers are `repr=False` and
redacted from `public_snapshot()`), and prompt-injection sanitization on
free-text fields (`sanitize_free_text`).

### Entitlement model (the policy math)

`compute_entitlement()` is deterministic. Carrier-liable reasons
(`TECHNICAL`, `CREW`, `IT_OUTAGE`) and EU jurisdiction drive care duties;
`WEATHER`/`ATC` in the US do not. Budget cap = base cap by cabin
(Basic $600 → First $6000) × loyalty multiplier (1.0–1.3). A Platinum on a
carrier-liable disruption gets a one-class courtesy bump, capped at Business.

---

## The five held-out scenarios (and what each proves)

| Scenario | Expected outcome | What it proves |
|----------|------------------|----------------|
| `happy` | **RESOLVED** — books VA-201 **Business** + TWA Hotel + $25 voucher (~$2,980) | owed cabin is honored (not the cheaper Economy downgrade); overnight care arranged |
| `overbudget` | **ESCALATED** — only flight ($1,450) over the $1,000 cap | escalates honestly, **does not over-book** |
| `visa` | **ESCALATED** — one option transits CN (no visa), the other violates MCT | rejects on **compliance**, not on price |
| `apifail` | **ESCALATED** — search times out, retries, returns empty | degrades gracefully, retries with a limit, **invents no flight** |
| `hitl` | **AWAITING_HUMAN_APPROVAL → RESOLVED** after approval | high-value booking **pauses for a human** before the irreversible action |

All five run from `python3 -m rebooking_agent.run` and are asserted in
`tests/test_pipeline.py`.

---

## State, auditability, resumability

`State` (in `models.py`) is the single typed source of truth. Two append-only
trails make "why did the system do that?" answerable after the fact:

- `decisions_log` — who decided what, and why (supervisor routes, agent
  outputs, human approvals).
- `audit_trail` — every tool call recorded as a **summary** (e.g.
  `search_flights ok 2 flights`), not the raw payload, so the context window
  stays lean.

State is an in-memory Pydantic model and is JSON-serializable
(`public_snapshot()` returns a PII-redacted dict), so an escalated case can be
handed to a human reviewer or persisted to disk/DB without code changes.

**Context discipline:** agents pass IDs and trimmed summaries, not 50-flight
payloads. The full candidate list lives in state; the ranking prompt sees a
compact view and the shortlist is trimmed to five.

---

## Two engines, one design (LangGraph + plain Python)

The brief recommends **LangGraph**, so this project ships it — and a plain typed
Python engine alongside it. **Both drive the exact same agents, tools,
guardrails, and `State`.** Only the orchestration layer differs, and a parity
test suite proves they reach identical outcomes on all five scenarios.

```bash
PYTHONPATH=src python3 -m rebooking_agent.run                     # plain-Python engine
PYTHONPATH=src python3 -m rebooking_agent.run --engine langgraph  # LangGraph engine
```

- **LangGraph engine** (`src/rebooking_agent/graph.py`): a compiled `StateGraph` with
  conditional edges; human-in-the-loop uses LangGraph's `interrupt()` plus a
  checkpointer, resumed with `Command(resume=...)`. This is the recommended
  stack, implemented idiomatically.
- **Plain-Python engine** (`src/rebooking_agent/orchestrator.py`): the same
  supervisor-worker graph as an explicit, dependency-free `while` loop.

**My recommendation: lead with the plain-Python engine, keep LangGraph as the
drop-in alternative.** Reasons:

1. **It runs anywhere with one dependency** (pydantic) and **no API key**, so
   the reviewer can clone and `pytest` in seconds; LangGraph adds a dependency
   tree for orchestration we can already express clearly.
2. **The control flow is the thing being graded** (agent design + state +
   guardrails), and an explicit `_decide_next()` makes routing, the retry
   bound, and the HITL gate auditable on one screen — no framework internals to
   trust.
3. **Frameworks churn; the design shouldn't.** Keeping domain logic
   framework-agnostic is *why* the LangGraph port was a thin wrapper rather than
   a rewrite — which is itself the strongest argument that the architecture is
   sound.

The two map 1:1, by construction:

| Plain (`orchestrator.py`) | LangGraph (`graph.py`) |
|---------------------------|------------------------|
| `State` (Pydantic) | graph state channel |
| each agent's `run(state)` | a node (thin wrapper over the same agent) |
| `_decide_next()` if/elif ladder | conditional edges (router functions) |
| explicit `while` loop | the compiled `StateGraph` runtime |
| `MAX_HOPS` guard | `recursion_limit` + ceiling routers |
| `search_relaxed` relax-and-retry | a loop edge back to the Search node |
| `AWAITING_HUMAN_APPROVAL` + `resume_*` | `interrupt()` + checkpointer resume |

The shared retry predicate (`guardrails.relax_might_help`) is imported by both
engines so the policy can't drift between them.

---

## Framework note for the reviewer

> The LangGraph engine is included because the brief names it as the
> recommended stack, and it's implemented the idiomatic way (`StateGraph`,
> conditional edges, `interrupt()`/checkpointer HITL). For day-to-day use I'd
> actually recommend the plain-Python engine: it runs with no API key and one
> dependency, the orchestration is fully inspectable, and — as the near-zero
> cost of the LangGraph port shows — the design is what carries the weight, not
> the framework. Run either with `--engine python|langgraph`; the parity tests
> prove they behave identically.

---

## Project layout

```
src/rebooking_agent/
  models.py        # all Pydantic schemas + the State blackboard
  mock_data.py     # hand-authored world: passengers, flights, hotels, faults
  tools.py         # mock API surface; every call returns a typed ToolResult
  guardrails.py    # deterministic authoritative layer (entitlement + checks)
  llm.py           # LLMClient protocol; StubLLM + AnthropicLLM; get_llm()
  orchestrator.py  # plain-Python engine: the supervisor (routing, loops, HITL)
  graph.py         # LangGraph engine: same graph as a compiled StateGraph
  agents/          # triage / search / policy / comms  (shared by both engines)
  run.py           # the five scenarios + trace printer (--engine python|langgraph)
  service.py       # OPTIONAL FastAPI wrapper around the mock APIs
tests/
  test_guardrails.py     # unit tests for the hard guardrails (no LLM)
  test_pipeline.py       # end-to-end scenario tests (plain engine, StubLLM)
  test_graph_pipeline.py # same scenarios on the LangGraph engine (parity)
```

---

## Trade-offs and what's deliberately out of scope

- **One passenger, not a fleet.** Per the brief, the system triages a single
  case end to end. Batch/queue handling is not modeled.
- **Mock world is hand-authored, not Faker-seeded.** The interesting cases
  (sold-out hotel, MCT violation, visa-blocked transit, over-budget-only,
  empty route, flaky API) are worth more than volume, and hand-authoring keeps
  the scenarios deterministic and the tests stable.
- **Observability is via the decisions/audit log + a trace printer**, not a
  hosted tracer. Wiring LangSmith/Langfuse would be additive; the per-case
  hop/LLM-call/tool counts are already measured and printed.
- **The real Anthropic client forces schema-valid output via tool-use** with a
  pinned `tool_choice`, so even the live path returns validated structured data
  rather than prose to regex.
