"""Feasibility test: does a Western-only allowlist serve both models?

A denylist can't be proven complete (it let StreamLake/Kuaishou through). An
allowlist (provider.only) is the only way to GUARANTEE non-CN routing - but it
errors if no allowed provider serves a model. This checks that risk directly.
"""
from __future__ import annotations

import json
import os
import sys

import httpx

MODELS = ["deepseek/deepseek-v4-pro", "moonshotai/kimi-k2.7-code"]
WESTERN = ["DeepInfra", "Together", "Fireworks", "GMICloud", "Baseten",
           "Lambda", "Hyperbolic", "Nebius", "Parasail"]
SCREEN = {"data_collection": "deny", "only": WESTERN, "sort": "price"}


def main() -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("NO_KEY_IN_ENV")
        return 2
    print("allowlist screen =", json.dumps(SCREEN))
    headers = {"Authorization": f"Bearer {key}",
               "HTTP-Referer": "https://github.com/angelsystems/agent-loop",
               "X-Title": "agent-loop-screen-test"}
    all_ok = True
    for model in MODELS:
        body = {"model": model,
                "messages": [{"role": "user", "content": "Reply with: ok"}],
                "max_tokens": 3, "temperature": 0, "provider": SCREEN}
        r = httpx.post("https://openrouter.ai/api/v1/chat/completions",
                       headers=headers, json=body, timeout=90)
        d = r.json()
        if r.status_code == 200:
            print(f"{model}: HTTP 200 served_provider={d.get('provider')!r} "
                  f"served_model={d.get('model')!r}")
        else:
            all_ok = False
            print(f"{model}: HTTP {r.status_code} ERROR={json.dumps(d)[:300]}")
    print("ALLOWLIST_RESULT:", "BOTH_SERVED" if all_ok else "SOME_UNSERVED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
