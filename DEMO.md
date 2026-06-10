# Demo page (`demo.html`)

A standalone, good-looking walkthrough of the five scenarios — open it by
double-clicking `demo.html` in any browser. No install, no server, no API key.

It is a **presentation layer over real, recorded agent outputs**: the data baked
into the page was produced by running the actual agents (see `dump_demo_data.py`
-> `demo_data.json` -> `build_demo_html.py`). It shows the animated pipeline, the
per-option rule checks, the boarding-pass outcome, and a working
Approve/Reject moment for the human-in-the-loop case, plus the full technical
trace under "show technical detail."

For a version that runs the agents **live** (real pause/resume against the
Python backend), use the Streamlit app instead: `streamlit run app.py`.

To regenerate the demo data after changing the agents:

    PYTHONPATH=src python3 dump_demo_data.py
    python3 build_demo_html.py
