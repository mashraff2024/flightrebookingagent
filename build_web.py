"""Generate web/index.html (the LIVE, API-driven UI) from demo.html.

demo.html is the offline version with baked-in data. This script reuses its
exact styling and render code, and swaps the data source: instead of reading an
embedded blob, the page calls the FastAPI endpoints (/api/run, /api/resume) so
every click runs the real agents. Keeping one visual source avoids drift.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
import json
from app_helpers import SCENARIO_INFO

INFO = {k: {"title": t, "blurb": b} for k, (t, b) in SCENARIO_INFO.items()}
INFO_JSON = json.dumps(INFO, separators=(",", ":"))

src = open(os.path.join(HERE, "demo.html"), encoding="utf-8").read()

# 1) drop the baked data blob (single physical line ending in ";\n")
src = re.sub(r"const DEMO = .*?;\n", "", src, count=1, flags=re.S)

# 2) state vars: no phase; add live state + static label map for chips
src = src.replace(
    'const ORDER = ["happy","overbudget","visa","apifail","hitl"];\nlet cur = "happy", eng = "python", phase = "base";',
    'const ORDER = ["happy","overbudget","visa","apifail","hitl"];\n'
    'const INFO = __INFO_JSON__;\n'
    'let cur = "happy", eng = "python";\nlet CURRENT=null, SESSION=null;'
)

# 2b) chips read their labels from INFO, not the (now absent) DEMO blob
src = src.replace('ORDER.forEach(k=>{const c=DEMO[k];', 'ORDER.forEach(k=>{const c=INFO[k];')

# 3) remove snap() (server provides snapshots now)
src = src.replace(
    'function snap(){const base=DEMO[cur].engines[eng];\n'
    '  if(phase==="approved"&&base.approved) return base.approved;\n'
    '  if(phase==="rejected"&&base.rejected) return base.rejected;\n'
    '  return base;}\n', ''
)

# 4) chips onclick -> fetch a fresh run
src = src.replace(
    'el.onclick=()=>{cur=k; phase="base"; markChips(); render(true);}; chipsEl.appendChild(el);',
    'el.onclick=()=>{cur=k; markChips(); runCase();}; chipsEl.appendChild(el);'
)

# 5) engine toggle -> re-run current case (unique trailing token)
src = src.replace('render(true);});', 'runCase();});')

# 6) approve/reject -> resume the live session
src = src.replace(
    'window.decide=function(p){phase=p; render(true);};',
    "window.decide=function(p){resumeCase(p==='approved');};"
)

# 7) master render + live controller
old_master = (
    'let gen=0;\n'
    'async function render(animate){\n'
    '  const tok=++gen;\n'
    '  const s=snap();\n'
    '  renderBoard(s);\n'
    '  document.getElementById("trace").textContent=s.trace||"";\n'
    '  await renderPipe(s,animate,tok); if(tok!==gen) return;\n'
    '  await renderTable(s,animate,tok); if(tok!==gen) return;\n'
    '  renderOutcome(s);\n'
    '  renderStats(s);\n'
    '}\n'
    'render(true);'
)
new_master = (
    'let gen=0;\n'
    'async function render(s,animate){\n'
    '  const tok=++gen;\n'
    '  renderBoard(s);\n'
    '  document.getElementById("trace").textContent=s.trace||"";\n'
    '  await renderPipe(s,animate,tok); if(tok!==gen) return;\n'
    '  await renderTable(s,animate,tok); if(tok!==gen) return;\n'
    '  renderOutcome(s);\n'
    '  renderStats(s);\n'
    '}\n'
    'function showRunning(){document.getElementById("outcome").innerHTML='
    '\'<div class="card" style="margin:0"><p class="sub" style="margin:0">Running the agent\\u2026</p></div>\';}\n'
    'async function api(path,body){const r=await fetch(path,{method:"POST",'
    'headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});'
    'if(!r.ok){throw new Error("HTTP "+r.status);}return await r.json();}\n'
    'async function runCase(){showRunning();\n'
    '  try{CURRENT=await api("/api/run",{scenario:cur,engine:eng});SESSION=CURRENT.session||null;}\n'
    '  catch(e){document.getElementById("outcome").innerHTML='
    '\'<div class="alert"><div class="h">Server not reachable</div>'
    '<div class="r">Start the server and reload. (\'+e.message+\')</div></div>\';return;}\n'
    '  render(CURRENT,true);}\n'
    'async function resumeCase(approved){if(!SESSION)return;showRunning();\n'
    '  try{CURRENT=await api("/api/resume",{session:SESSION,approved});SESSION=CURRENT.session||null;}\n'
    '  catch(e){document.getElementById("outcome").innerHTML='
    '\'<div class="alert"><div class="h">Could not resume</div><div class="r">\'+e.message+\'</div></div>\';return;}\n'
    '  render(CURRENT,true);}\n'
    'runCase();'
)
assert old_master in src, "master render block not found — demo.html changed?"
src = src.replace(old_master, new_master)

# tweak the title/footers slightly so it's clearly the live one
src = src.replace("<title>Runway Ops — Autonomous Rebooking</title>",
                  "<title>Runway Ops — Live Rebooking Agent</title>")
src = src.replace("Real agent outputs · deterministic offline backend",
                  "Live agent · deterministic backend")

os.makedirs(os.path.join(HERE, "web"), exist_ok=True)
out = os.path.join(HERE, "web", "index.html")
src = src.replace("__INFO_JSON__", INFO_JSON)
open(out, "w", encoding="utf-8").write(src)
print("wrote", out, len(src), "bytes")
