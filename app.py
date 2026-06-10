"""
Friendly web UI for the flight rebooking agent.

Run it with:
    pip install streamlit
    $env:PYTHONPATH="src"          # PowerShell (Windows)
    streamlit run app.py
or on macOS/Linux:
    PYTHONPATH=src streamlit run app.py

This is purely a front end. It runs on the deterministic stub backend
(no API key, no network, no real Claude calls) and reuses the exact same
agents, guardrails, and state as the command line. Nothing underneath changes.
"""
import os
import sys
import time

# Make the package under src/ importable without needing PYTHONPATH set.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

import app_helpers as h

st.set_page_config(page_title="Flight Rebooking Agent", page_icon="✈️", layout="wide")

# ----- colours -----
C_DONE = "#2e7d32"; C_ESC = "#c62828"; C_AWAIT = "#ef6c00"
C_PEND = "#9aa0a6"; C_OK = "#e6f4ea"; C_BAD = "#fde8e8"
C_OKT = "#1e7e34"; C_BADT = "#c62828"


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def stepper_html(stages, upto=None):
    if upto is None:
        upto = len(stages)
    colour = {"done": C_DONE, "escalate": C_ESC, "await": C_AWAIT}
    parts = []
    for i, (label, status) in enumerate(stages):
        if i < upto:
            bg = colour.get(status, C_DONE); fg = "#fff"; border = bg
        else:
            bg = "#fff"; fg = C_PEND; border = C_PEND
        parts.append(
            f'<span style="display:inline-block;padding:8px 16px;border-radius:18px;'
            f'background:{bg};color:{fg};border:1.5px solid {border};font-weight:600;'
            f'font-size:14px;">{label}</span>'
        )
        if i < len(stages) - 1:
            arrow_colour = C_DONE if i + 1 < upto else C_PEND
            parts.append(f'<span style="color:{arrow_colour};font-size:18px;'
                         f'margin:0 6px;">&rarr;</span>')
    return f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px;">{"".join(parts)}</div>'


def decision_table_html(rows):
    head = (
        '<tr style="background:#f1f3f4;">'
        '<th style="text-align:left;padding:8px 10px;">Flight</th>'
        '<th style="text-align:left;padding:8px 10px;">Cabin</th>'
        '<th style="text-align:right;padding:8px 10px;">Total</th>'
        '<th style="padding:8px 10px;">Cabin</th>'
        '<th style="padding:8px 10px;">Budget</th>'
        '<th style="padding:8px 10px;">Visa</th>'
        '<th style="padding:8px 10px;">Connection</th>'
        '<th style="padding:8px 10px;">Verdict</th></tr>'
    )
    body = []
    for r in rows:
        cells = [
            f'<td style="padding:8px 10px;font-weight:600;">{r["flight"]}</td>',
            f'<td style="padding:8px 10px;">{r["cabin"]}</td>',
            f'<td style="padding:8px 10px;text-align:right;">${r["total"]:.0f}</td>',
        ]
        for name in ("Cabin", "Budget", "Visa", "Connection"):
            ok, reason = r["checks"][name]
            bg = C_OK if ok else C_BAD; fg = C_OKT if ok else C_BADT
            mark = "PASS" if ok else "FAIL"
            safe = reason.replace('"', "'")
            cells.append(
                f'<td title="{safe}" style="padding:8px 10px;text-align:center;'
                f'background:{bg};color:{fg};font-weight:600;">{mark}</td>'
            )
        approved = r["verdict"] == "approved"
        vbg = C_OK if approved else C_BAD; vfg = C_OKT if approved else C_BADT
        cells.append(
            f'<td style="padding:8px 10px;text-align:center;background:{vbg};'
            f'color:{vfg};font-weight:700;">{r["verdict"].upper()}</td>'
        )
        body.append(f'<tr style="border-bottom:1px solid #eee;">{"".join(cells)}</tr>')
    return (f'<table style="border-collapse:collapse;width:100%;font-size:14px;">'
            f'{head}{"".join(body)}</table>')


def reasons_list(rows):
    for r in rows:
        for name, (ok, reason) in r["checks"].items():
            if not ok:
                st.markdown(f'- **{r["flight"]} · {name}** — {reason}')


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.title("✈️ Rebooking Agent")
st.sidebar.caption("A friendly view of the disruption-rebooking workflow. "
                   "Runs offline on the deterministic backend — no API key, no real calls.")

engine = st.sidebar.radio("Engine", ["python", "langgraph"],
                          format_func=lambda x: "Plain Python" if x == "python" else "LangGraph",
                          help="Both run the same agents and produce the same result.")

scenario = st.sidebar.selectbox(
    "Scenario", list(h.SCENARIO_INFO.keys()),
    format_func=lambda k: h.SCENARIO_INFO[k][0])

animate = st.sidebar.checkbox("Animate steps", value=True)

run_clicked = st.sidebar.button("Run case", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("Tip: try **Needs sign-off** to see the human-approval pause, "
                   "and switch the engine to confirm the result is identical.")


# --------------------------------------------------------------------------- #
# State handling across reruns
# --------------------------------------------------------------------------- #
def fresh_run():
    orch, state, trace = h.start_case(scenario, engine)
    st.session_state.orch = orch
    st.session_state.state = state
    st.session_state.trace = trace
    st.session_state.scenario = scenario
    st.session_state.engine = engine
    st.session_state.need_animation = animate


if run_clicked:
    fresh_run()

state = st.session_state.get("state")


# --------------------------------------------------------------------------- #
# Main panel
# --------------------------------------------------------------------------- #
title, blurb = h.SCENARIO_INFO[st.session_state.get("scenario", scenario)]
st.title("Flight Disruption & Rebooking")

if state is None:
    st.info("Pick a scenario on the left and press **Run case**.")
    st.stop()

st.subheader(title)
st.write(blurb)
eng_label = "LangGraph" if st.session_state.get("engine") == "langgraph" else "Plain Python"
st.caption(f"Engine: {eng_label} · backend: deterministic stub (offline)")

# --- pipeline stepper (animated on a fresh run) ---
stages = h.derive_stages(state)
step_box = st.empty()
if st.session_state.get("need_animation"):
    for k in range(1, len(stages) + 1):
        step_box.markdown(stepper_html(stages, upto=k), unsafe_allow_html=True)
        time.sleep(0.5)
    st.session_state.need_animation = False
else:
    step_box.markdown(stepper_html(stages), unsafe_allow_html=True)

st.write("")

# --- the decision table (the heart of it) ---
rows = h.build_check_rows(state)
if rows:
    st.markdown("#### What the rules checked")
    st.markdown(decision_table_html(rows), unsafe_allow_html=True)
    st.caption("Hover a PASS/FAIL cell to see the reason. These are the deterministic "
               "checks the Policy agent runs — the AI never decides these.")
    fails = any(not ok for r in rows for ok, _ in r["checks"].values())
    if fails:
        with st.expander("Why options were rejected"):
            reasons_list(rows)
else:
    st.markdown("#### Search returned no options")
    st.caption("The search failed or came back empty, so there was nothing to check.")

st.write("")

# --- outcome ---
out = h.outcome_summary(state)
if out["status"] == "RESOLVED":
    st.success("✅ Resolved")
    c1, c2, c3 = st.columns(3)
    c1.metric("Flight booked", f'{out["flight"]} · {out["cabin"]}')
    c2.metric("Total", f'${out["total"]:.0f}')
    c3.metric("Confirmation", out["booking_pnr"])
    extras = []
    if out["hotel"]:
        extras.append(f'Hotel: {out["hotel"]}')
    if out["voucher"]:
        extras.append(f'Meal voucher: ${out["voucher"]:.0f}')
    if extras:
        st.caption(" · ".join(extras))
    if out["message"]:
        with st.expander("Message sent to the traveller", expanded=True):
            st.text(out["message"])

elif out["status"] == "ESCALATED":
    st.error("⛔ Escalated to a human")
    st.write(f'**Reason:** {out["escalation_reason"]}')
    st.caption("No option cleared the rules, so the agent handed the case to a person "
               "instead of guessing or over-booking.")

elif out["status"] == "AWAITING_HUMAN_APPROVAL":
    st.warning("⏸️ Waiting for your approval")
    st.write(f'This booking is **{out["flight"]} · {out["cabin"]}** for '
             f'**${out["total"]:.0f}**, which is above the sign-off threshold. '
             f'Nothing has been booked yet.')
    a, b, _ = st.columns([1, 1, 3])
    if a.button("✅ Approve", type="primary", use_container_width=True):
        new_state, trace2 = h.resume_case(st.session_state.orch, True)
        st.session_state.state = new_state
        st.session_state.trace += "\n\n--- after human APPROVED ---\n" + trace2
        st.session_state.need_animation = animate
        st.rerun()
    if b.button("⛔ Reject", use_container_width=True):
        new_state, trace2 = h.resume_case(st.session_state.orch, False)
        st.session_state.state = new_state
        st.session_state.trace += "\n\n--- after human REJECTED ---\n" + trace2
        st.session_state.need_animation = False
        st.rerun()

# --- stats + technical trace ---
st.write("")
s1, s2, s3 = st.columns(3)
s1.metric("Pipeline steps", out["hops"])
s2.metric("AI calls", out["llm_calls"])
s3.metric("Search attempts", out["search_attempts"])

with st.expander("Show technical detail (full trace)"):
    st.caption("The same step-by-step trace technical reviewers see from the "
               "command line with --verbose.")
    st.code(st.session_state.get("trace", ""), language="text")
