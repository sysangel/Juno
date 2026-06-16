#!/usr/bin/env python
"""design_pass.py - ONE fusion deliberation on the three hard Juno forks.

Concentrates expensive multi-model thinking into a single call, so the cheap driver can
just execute settled decisions during the implementation run. Writes the decisions to
runs/fusion_design_decisions.md for baking into the implementation prompt.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import httpx

from fusion_probe import BASE, HEADERS  # reuse base url + auth headers

PROMPT = """You are advising on three hard implementation decisions for a Python terminal app
("Juno"): a prompt_toolkit 3 based REPL running on Windows (Python 3.12). For each, give a
CONCRETE, implementable decision as "DECISION: <what to do>" followed by 2-4 sentences of
rationale plus the key code-level specifics. Be opinionated and specific - this is handed to a
coding agent as settled guidance, not a survey of options.

1. WAITING INDICATOR. After the user submits, show a calm animated "nebula signal . . ." while a
   BLOCKING provider HTTP call runs, then print the assistant reply. Constraints:
   - Must render correctly on BOTH legacy Windows conhost AND Windows Terminal.
   - Must NOT cause the bottom-line blackout / one-line upward shift that naive prints cause.
   - In non-TTY / piped / plain mode it must be deterministic: a single static line, NO ANSI, no thread.
   - Provide a JUNO_WAIT_ANIMATION=0 kill switch forcing the static line.
   Decide between: a background thread with carriage-return repaint, vs prompt_toolkit
   run_in_terminal / patch_stdout, vs a prompt_toolkit float/progress. Specify exactly how to repaint
   without flicker/artifacts on conhost and how to clear the line cleanly before printing the reply.

2. THEME CENTRALIZATION. Colors/labels/glyphs are today scattered module-level constants (JUNO_*,
   raw ANSI) and existing tests import some of those names. We want a frozen @dataclass JunoTheme +
   DEFAULT_THEME and helpers that take theme=DEFAULT_THEME. Decide the LOWEST-RISK migration: keep old
   constant names as aliases bound to DEFAULT_THEME fields? thread theme through formatter signatures
   with a default arg? How to avoid breaking tests that reference the old names? Give the concrete pattern.

3. POPUPS. /model, /skills, /menu. Start with prompt_toolkit radiolist_dialog over static rows, or
   invest now in a custom prompt_toolkit Application overlay? Decide the pragmatic path and the explicit
   threshold at which migrating to a custom Application becomes worth it.

Keep the total answer under ~900 words. No preamble."""


def main() -> None:
    if "--go" not in sys.argv:
        print("add --go to spend (~one fusion call, panel+judge)")
        return
    print("consulting fusion (panel + judge; ~30-60s)...", flush=True)
    r = httpx.post(f"{BASE}/chat/completions", headers=HEADERS, json={
        "model": "openrouter/fusion",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 1600,
        "temperature": 0.3,
    }, timeout=600)
    if not r.is_success:
        print(f"ERROR {r.status_code}: {r.text[:400]}")
        sys.exit(1)
    data = r.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or "(no content)"
    usage = data.get("usage") or {}
    out = Path(__file__).with_name("runs") / "fusion_design_decisions.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print("\n" + content)
    print(f"\n[usage {json.dumps(usage)}]  written to {out}")


if __name__ == "__main__":
    main()
