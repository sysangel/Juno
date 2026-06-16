"""Prove the data-secure provider screen reaches OpenRouter and routes off CN.

For each model it:
  1. Asserts get_llm() injects the loop's actual OPENROUTER_PROVIDER_PREFS as
     extra_body={"provider": ...} onto the ChatOpenAI the loop uses.
  2. Makes a raw OpenRouter call with that screen and reports the serving
     provider + model, flagging it if the provider is on the CN denylist.
  3. Invokes the same get_llm() client end-to-end and reports token usage.

Reads OPENROUTER_API_KEY from the environment; never prints it.
"""
from __future__ import annotations

import json
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_loop  # noqa: E402

MODELS = ["deepseek/deepseek-v4-pro"]
SCREEN = agent_loop.OPENROUTER_PROVIDER_PREFS
CN = {p.lower() for p in agent_loop.OPENROUTER_CN_PROVIDERS}


def _check_model(key: str, model: str) -> bool:
    ok = True
    print(f"\n########## {model} ##########")

    print("=== Check 1: get_llm() injects the screen ===")
    llm = agent_loop.get_llm(provider="openrouter", model=model)
    eb = getattr(llm, "extra_body", None)
    print("get_llm.extra_body =", json.dumps(eb))
    assert eb == {"provider": SCREEN}, f"screen missing/wrong: {eb}"
    print("ASSERT_OK: screen present.")

    print("=== Check 2: raw OpenRouter call under the screen ===")
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/angelsystems/agent-loop",
        "X-Title": "agent-loop-verify",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single token: ok"}],
        "max_tokens": 3,
        "temperature": 0,
        "provider": SCREEN,
    }
    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers, json=body, timeout=90,
    )
    print("HTTP", r.status_code)
    d = r.json()
    if r.status_code != 200:
        print("ERROR body:", json.dumps(d)[:800])
        return False
    served = d.get("provider")
    print("served_provider =", served)
    print("served_model    =", d.get("model"))
    print("usage           =", json.dumps(d.get("usage")))
    if served and str(served).lower() in CN:
        print(f"FAIL: served by CN provider {served!r} despite denylist.")
        ok = False
    elif served and "deepseek" in str(served).lower():
        print(f"FAIL: served by a DeepSeek-named provider {served!r}.")
        ok = False
    else:
        print(f"OK: served by non-CN provider {served!r}.")

    print("=== Check 3: end-to-end via the loop's own client ===")
    resp = llm.invoke("Reply with the single token: ok")
    md = getattr(resp, "response_metadata", None) or {}
    print("langchain.model_name   =", md.get("model_name"))
    print("langchain.model_provider =", md.get("model_provider"))
    print("usage_metadata         =", json.dumps(getattr(resp, "usage_metadata", None)))
    return ok


def main() -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("NO_KEY_IN_ENV")
        return 2
    print("screen =", json.dumps(SCREEN))
    all_ok = True
    for m in MODELS:
        all_ok = _check_model(key, m) and all_ok
    print("\n==============================")
    print("VERIFY_RESULT:", "ALL_PASS" if all_ok else "SOME_FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
