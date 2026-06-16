#!/usr/bin/env python
"""review.py - fusion review gate.

Diffs the current working tree against a pre-run backup (runs/prerun_backup_*), so it sees
EXACTLY the implementation agent's edits (not your earlier uncommitted work), then asks
OpenRouter Fusion (panel + judge) to grade the diff against the done-criteria.

Usage:
  python review.py                       # diff vs the most recent prerun_backup_*
  python review.py runs/prerun_backup_X  # diff vs a specific backup
"""
from __future__ import annotations
import difflib
import sys
from pathlib import Path

import httpx

from fusion_probe import BASE, HEADERS

ROOT = Path(__file__).parent

DOD = """Done-criteria to grade against:
- `pytest test_harness_ui.py` is green.
- Plain-mode waiting text contains 'nebula signal' and NEVER 'claude-code', 'opus', or ' with '; no ANSI in plain mode.
- WaitingDots: background thread + carriage-return repaint; branch on TTY / JUNO_WAIT_ANIMATION BEFORE spawning the thread; never emits '\\n' during animation; pads to columns-1; clears with space-pad (NOT \\x1b[2K); deterministic single static line in non-TTY/plain.
- --tui maps to the prompt UI, --cli maps to plain, error if both passed; documented model-resolution order honored.
- /skills and /menu open without crashing (including when zero skills are installed); /menu can dispatch /status.
- JunoTheme is a frozen dataclass + DEFAULT_THEME singleton; legacy JUNO_* names are aliases that are the SAME object; theme threaded through formatters as a keyword-only default arg; a guard test asserts alias identity.
- harness.py sends NO per-request `provider` block on its OpenRouter call(s); any test asserting the old block is updated/removed.
- HARNESS.md documents the new flags/commands.
- No unrelated edits or regressions."""


def latest_backup() -> Path | None:
    cands = sorted(ROOT.glob("runs/prerun_backup_*"))
    return cands[-1] if cands else None


def build_diff(backup: Path) -> str:
    chunks: list[str] = []
    for bf in sorted(backup.glob("*")):
        if not bf.is_file():
            continue
        cur = ROOT / bf.name
        old = bf.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        new = (cur.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
               if cur.exists() else [])
        d = list(difflib.unified_diff(old, new, fromfile=f"a/{bf.name}", tofile=f"b/{bf.name}"))
        if d:
            chunks.append("".join(d))
    return "\n".join(chunks)


def main() -> None:
    arg = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
    backup = Path(arg) if arg else latest_backup()
    if not backup or not backup.exists():
        print("No backup dir found. Pass one, e.g. python review.py runs/prerun_backup_YYYYMMDD-HHMMSS")
        sys.exit(1)
    print(f"diffing working tree vs {backup.name} ...")
    diff = build_diff(backup)
    if not diff.strip():
        print("No changes vs backup - nothing to review.")
        return
    if len(diff) > 60000:
        diff = diff[:60000] + "\n... [diff truncated for review]"

    prompt = (
        "Output ONLY the graded review and begin IMMEDIATELY with the first criterion verdict. Do NOT "
        "write any preamble such as 'I'll grade this' or 'let me get a panel analysis first'.\n\n"
        "You are a strict, skeptical code reviewer. Grade the DIFF below against the done-criteria. "
        "For EACH criterion output `PASS` / `FAIL` / `UNCLEAR` with a one-line reason. Then list any real "
        "bugs or regressions visible in the diff (quote the line). End with `VERDICT: SHIP` or "
        "`VERDICT: FIX` followed by the top 3 fixes if FIX.\n\n"
        + DOD + "\n\nDIFF:\n" + diff
    )

    def call(model: str) -> str:
        r = httpx.post(f"{BASE}/chat/completions", headers=HEADERS, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2400,
            "temperature": 0.1,
        }, timeout=600)
        if not r.is_success:
            return f"(ERROR {r.status_code}: {r.text[:300]})"
        return (r.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""

    print("fusion reviewing (panel + judge; ~30-60s)...", flush=True)
    content = call("openrouter/fusion")
    if len(content.strip()) < 200:  # fusion's agentic wrapper stalled on the big diff
        print("  fusion returned no usable verdict; falling back to a direct strong reviewer (claude-opus-4.8)...")
        content = call("anthropic/claude-opus-4.8")
    if not content.strip():
        content = "(no content from either reviewer)"
    out = ROOT / "runs" / "fusion_review.md"
    out.write_text(content, encoding="utf-8")
    print("\n" + content)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):  # never crash a print on Windows cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    main()
