#!/usr/bin/env python
"""fusion_probe2.py — find the fusion invocation that WORKS and stays no-train / CN-free.

Round 1 finding: model="openrouter/fusion" + a strict `only` allowlist -> 404
("no endpoints"). The meta-model isn't served by a single allowlisted provider.
This round isolates the cause and finds the production-safe screen.

  R1 bare fusion             : no `provider` field at all.
  R2 no-train only           : provider={data_collection:"deny"}.
  R3 no-train + CN denylist  : provider={data_collection:"deny", ignore:CN}  <- candidate prod screen.
  R4 server-tool form        : outer deepseek (carries UNION screen) + fusion tool + read_file tool.
                               tests (a) fusion via the form that lets the DRIVER hold the screen,
                               and (b) custom-tool coexistence with fusion in one request.

Only spends with --go.
"""
from __future__ import annotations
import json
import sys

from fusion_probe import CN_PROVIDERS, READ_TOOL, UNION_ONLY, post_chat, prefs

FUSION_TOOL = {
    "type": "openrouter:fusion",
    "parameters": {
        "analysis_models": ["deepseek/deepseek-v4-pro"],
        "model": "deepseek/deepseek-v4-pro",
    },
}

RISK_Q = "In one sentence: what is the single biggest risk when running an LLM agent in an unattended file-editing loop?"


def show(label: str, verdict: dict) -> None:
    print(f"\n[{label}]")
    print(json.dumps(verdict, indent=2, default=str))


def run() -> None:
    if "--go" not in sys.argv:
        print("Round 2 plan (add --go to spend):")
        print("  R1 openrouter/fusion, no provider field")
        print("  R2 openrouter/fusion, provider={data_collection:deny}")
        print("  R3 openrouter/fusion, provider={data_collection:deny, ignore:CN}")
        print("  R4 server-tool form: deepseek + fusion tool + read_file tool, UNION screen")
        return

    base_msgs = [{"role": "user", "content": RISK_Q}]

    r1 = post_chat({"model": "openrouter/fusion", "messages": base_msgs,
                    "max_tokens": 120, "temperature": 0}, timeout=300)
    show("R1 bare fusion", r1)

    r2 = post_chat({"model": "openrouter/fusion", "messages": base_msgs,
                    "max_tokens": 120, "temperature": 0,
                    "provider": {"data_collection": "deny"}}, timeout=300)
    show("R2 no-train only", r2)

    r3 = post_chat({"model": "openrouter/fusion", "messages": base_msgs,
                    "max_tokens": 120, "temperature": 0,
                    "provider": {"data_collection": "deny", "ignore": CN_PROVIDERS}}, timeout=300)
    show("R3 no-train + CN denylist (candidate prod screen)", r3)

    r4 = post_chat({
        "model": "deepseek/deepseek-v4-pro",
        "messages": [{"role": "user", "content": "Read ./notes.txt with the read_file tool, then deliberate and answer: " + RISK_Q}],
        "tools": [FUSION_TOOL, READ_TOOL],
        "tool_choice": "auto",
        "max_tokens": 250,
        "temperature": 0,
        "provider": prefs(UNION_ONLY),
    }, timeout=300)
    show("R4 server-tool form (driver holds screen; fusion+read_file coexist)", r4)

    print("\n" + "=" * 72)
    print("ROUND 2 VERDICT")
    print("=" * 72)
    print(f"  R1 bare fusion           : ok={r1.get('ok')} status={r1.get('http_status')} served={r1.get('served_provider')}")
    print(f"  R2 data_collection:deny  : ok={r2.get('ok')} status={r2.get('http_status')} served={r2.get('served_provider')}")
    print(f"  R3 deny + CN denylist    : ok={r3.get('ok')} status={r3.get('http_status')} served={r3.get('served_provider')}  <- prod screen")
    print(f"  R4 server-tool form      : ok={r4.get('ok')} status={r4.get('http_status')} tool_calls={r4.get('has_tool_calls')} names={r4.get('tool_names')}")


if __name__ == "__main__":
    run()
