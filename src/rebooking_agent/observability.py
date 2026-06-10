"""
Live observability / tracing.

By default the system is silent while running and the demo prints a full trace
at the end. Turning the tracer up makes each step appear *as it happens*:

  * STREAM  -> supervisor routes, agent decisions, and tool calls, live.
  * VERBOSE -> the above, plus dumps of the raw reasoning inputs (the context
               handed to each LLM step), every individual guardrail check, and
               -- when a real Claude key is set -- the model's output tokens
               streaming in as they generate.

Nothing here makes decisions; it only observes. The hooks are wired through the
existing State.log_decision / log_tool helpers and the agents' reasoning funnel,
so enabling it changes output, never behavior. Default level is QUIET, which is
why the test suite sees no extra output and behaves identically.
"""
from __future__ import annotations

import sys
from enum import IntEnum
from typing import Any


class Level(IntEnum):
    QUIET = 0    # no live output (default; used by tests)
    STREAM = 1   # live routes, decisions, tool calls
    VERBOSE = 2  # + reasoning-input dumps, per-check results, model tokens


# ANSI colours (auto-disabled when not a TTY so piped/captured output stays clean)
_USE_COLOR = sys.stdout.isatty()
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

_ACTOR_COLOR = {
    "supervisor": "36", "triage": "32", "search": "34",
    "policy": "33", "comms": "35", "human": "1;31", "claude": "90",
}


class Tracer:
    def __init__(self, level: Level = Level.QUIET):
        self.level = level
        self._mid_token_line = False
        self.capture = False        # when True, lines go to self.sink instead of stdout
        self.sink: list[str] = []

    # ----- configuration ----------------------------------------------------
    def enabled(self, lvl: Level) -> bool:
        return self.level >= lvl

    def _emit(self, lvl: Level, plain: str, colored: str) -> None:
        if not self.enabled(lvl):
            return
        if self.capture:
            self.sink.append(plain)
        else:
            self._break_token_line()
            print(colored, flush=True)

    # ----- live events (STREAM) ---------------------------------------------
    def decision(self, actor: str, decision: str, why: str = "") -> None:
        plain = f"  [{actor:<10}] {decision}" + (f"  ({why})" if why else "")
        tag = _c(_ACTOR_COLOR.get(actor, "37"), f"[{actor:<10}]")
        tail = _c("90", f"  ({why})") if why else ""
        self._emit(Level.STREAM, plain, f"  {tag} {decision}{tail}")

    def tool(self, tool: str, ok: bool, summary: str = "", error: str = "") -> None:
        plain = f"    \u2192 tool {tool:<24} {'ok' if ok else 'ERR:'+error:<14} {summary}"
        flag = _c("32", "ok") if ok else _c("31", f"ERR:{error}")
        arrow = _c("90", "    \u2192 tool")
        self._emit(Level.STREAM, plain, f"  {arrow} {tool:<24} {flag:<14} {summary}")

    # ----- verbose dumps (VERBOSE) ------------------------------------------
    def dump(self, label: str, data: Any) -> None:
        if not self.enabled(Level.VERBOSE):
            return
        lines = [f"      \u00B7 {label}:"] + [f"          {ln}" for ln in _format(data)]
        if self.capture:
            self.sink.extend(lines)
        else:
            self._break_token_line()
            for ln in lines:
                print(_c("90", ln), flush=True)

    def check(self, flight_id: str, name: str, ok: bool, reason: str) -> None:
        plain = f"      \u00B7 check {flight_id} {name:<7} {'PASS' if ok else 'FAIL'}  {reason}"
        mark = _c("32", "PASS") if ok else _c("31", "FAIL")
        colored = _c("90", f"      \u00B7 check {flight_id} {name:<7} ") + f"{mark}  " + _c("90", reason)
        self._emit(Level.VERBOSE, plain, colored)

    # ----- model token stream (VERBOSE, real Claude only) -------------------
    def wants_tokens(self) -> bool:
        return self.enabled(Level.VERBOSE) and not self.capture

    def token(self, text: str) -> None:
        if not self.enabled(Level.VERBOSE) or not text:
            return
        if self.capture:
            self.sink.append(text)
            return
        if not self._mid_token_line:
            sys.stdout.write(_c("90", "      \u00B7 claude \u00BB "))
            self._mid_token_line = True
        sys.stdout.write(_c("90", text))
        sys.stdout.flush()

    def _break_token_line(self) -> None:
        if self._mid_token_line:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._mid_token_line = False


def _format(data: Any) -> list[str]:
    """Compact, human-readable rendering of common reasoning inputs."""
    out: list[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                out.append(f"{k}: [{len(v)} item(s)]")
                for item in v[:8]:
                    out.append(f"    - {_one(item)}")
                if len(v) > 8:
                    out.append(f"    \u2026 (+{len(v) - 8} more)")
            else:
                out.append(f"{k}: {_one(v)}")
    elif isinstance(data, list):
        for item in data[:8]:
            out.append(f"- {_one(item)}")
        if len(data) > 8:
            out.append(f"\u2026 (+{len(data) - 8} more)")
    else:
        out.append(_one(data))
    return out


def _one(item: Any) -> str:
    """One-line summary of a value, with friendly handling of flights/hotels."""
    # FlightOption-like
    if hasattr(item, "flight_id") and hasattr(item, "cabin") and hasattr(item, "price"):
        cabin = getattr(item.cabin, "label", item.cabin)
        try:
            route = f"{item.origin}->{item.destination}"
        except Exception:
            route = ""
        return f"{item.flight_id} {cabin} ${item.price:.0f} {route}".strip()
    # HotelOption-like
    if hasattr(item, "hotel_id") and hasattr(item, "nightly_rate"):
        return f"{item.name} ${item.nightly_rate:.0f}/night" + (" [partner]" if getattr(item, "loyalty_partner", False) else "")
    # Pydantic model
    if hasattr(item, "model_dump"):
        d = item.model_dump()
        keys = list(d)[:5]
        return ", ".join(f"{k}={d[k]}" for k in keys)
    s = repr(item)
    return s if len(s) <= 120 else s[:117] + "\u2026"


# --------------------------------------------------------------------------- #
# Global tracer
# --------------------------------------------------------------------------- #
TRACER = Tracer(Level.QUIET)


def configure_tracer(level: Level) -> None:
    TRACER.level = level


def get_tracer() -> Tracer:
    return TRACER
