#!/usr/bin/env python
"""fusion_probe3.py — bare confirmation (privacy now enforced ACCOUNT-SIDE on OpenRouter).

No `provider` field anywhere. Confirms the two things docs leave unconfirmed:
  B1 bare fusion consult     : model=openrouter/fusion, no tools  -> 200 + content?
  B2 server-tool coexistence : deepseek + fusion tool + read_file tool -> does the
                               driver still round-trip a read_file tool_call alongside fusion?
"""
from __future__ import annotations
import json
import sys

from fusion_probe import READ_TOOL, post_chat
from fusion_probe2 import FUSION_TOOL, RISK_Q


def show(label: str, v: dict) -> None:
    print(f"\n[{label}]")
    print(json.dumps(v, indent=2, default=str))


def run() -> None:
    if "--go" not in sys.argv:
        print("Bare plan (add --go): B1 fusion consult; B2 fusion+read_file coexistence. No provider gating.")
        return

    b1 = post_chat({"model": "openrouter/fusion",
                    "messages": [{"role": "user", "content": RISK_Q}],
                    "max_tokens": 120, "temperature": 0}, timeout=300)
    show("B1 bare fusion consult", b1)

    b2 = post_chat({
        "model": "deepseek/deepseek-v4-pro",
        "messages": [{"role": "user", "content": "Read ./notes.txt with the read_file tool, then deliberate and answer: " + RISK_Q}],
        "tools": [FUSION_TOOL, READ_TOOL],
        "tool_choice": "auto",
        "max_tokens": 250, "temperature": 0,
    }, timeout=300)
    show("B2 fusion + read_file coexistence", b2)

    print("\n" + "=" * 72)
    print("BARE VERDICT")
    print("=" * 72)
    print(f"  B1 fusion works    : ok={b1.get('ok')} status={b1.get('http_status')} served={b1.get('served_provider')}")
    print(f"  B2 coexistence     : ok={b2.get('ok')} status={b2.get('http_status')} tool_calls={b2.get('has_tool_calls')} names={b2.get('tool_names')} finish={b2.get('finish_reason')}")


if __name__ == "__main__":
    run()
