#!/usr/bin/env python
"""fusion_probe.py — verify OpenRouter Fusion behavior BEFORE building the agentic harness.

Free phase (always runs):
  L1. GET /providers  -> confirm exact provider slugs for the union allowlist.
  L2. GET /models     -> auto-pick a cheap OpenAI + Anthropic slug for the closed-provider reach test.

Paid phase (only with --go; prints the plan first):
  P1. driver tool-calling  : deepseek-v4-pro + custom read_file tool -> do tool_calls round-trip? (well-doc'd sanity)
  P2. fusion + custom tool  : model=openrouter/fusion + read_file tool -> does fusion return tool_calls? (the /brain mode)
  P3. pure fusion consult   : model=openrouter/fusion, no tools, tiny reasoning prompt -> content + served provider.
  P4. closed-provider reach : cheap OpenAI slug through the UNION allowlist -> 200 + served by a Western closed provider?

Privacy: union allowlist (Western closed + open), data_collection:deny, CN denied, allow_fallbacks:false.
All paid calls cap max_tokens hard to keep cost to cents.
"""
from __future__ import annotations
import json
import os
import sys

import httpx

BASE = "https://openrouter.ai/api/v1"
KEY = os.getenv("OPENROUTER_API_KEY")

# --- privacy screen (union: Western closed + open) ---------------------------
CN_PROVIDERS = [
    "Baidu", "DeepSeek", "Moonshot AI", "Moonshot", "Alibaba", "Alibaba Cloud",
    "Qwen", "Zhipu", "Zhipu AI", "Z.AI", "ByteDance", "Volcengine", "Tencent",
    "Hunyuan", "MiniMax", "StepFun", "01.AI", "SiliconFlow", "iFlytek",
    "StreamLake", "Kuaishou", "SenseTime", "Baichuan", "InternLM",
]
WESTERN_OPEN = [
    "DeepInfra", "Together", "Fireworks", "GMICloud", "Baseten",
    "Lambda", "Hyperbolic", "Nebius", "Parasail",
]
# Western first-party / closed-model providers. Slugs confirmed against /providers (L1).
WESTERN_CLOSED = [
    "OpenAI", "Anthropic", "Google", "Google AI Studio", "Amazon Bedrock", "Azure",
]
UNION_ONLY = WESTERN_OPEN + WESTERN_CLOSED


def prefs(only: list[str]) -> dict:
    return {
        "data_collection": "deny",
        "only": only,
        "ignore": CN_PROVIDERS,
        "allow_fallbacks": False,
        "require_parameters": True,
    }


HEADERS = {
    "Authorization": f"Bearer {KEY}",
    "HTTP-Referer": "https://github.com/angelsystems/agent-loop",
    "X-Title": "fusion-probe",
    "Content-Type": "application/json",
}

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file from disk and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to read"}},
            "required": ["path"],
        },
    },
}


def _die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def post_chat(body: dict, timeout: float = 120.0) -> dict:
    """Return a compact verdict dict for a /chat/completions call."""
    try:
        r = httpx.post(f"{BASE}/chat/completions", headers=HEADERS, json=body, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "transport_error": repr(e)}
    out: dict = {"http_status": r.status_code, "ok": r.is_success}
    try:
        data = r.json()
    except Exception:  # noqa: BLE001
        out["raw"] = r.text[:400]
        return out
    if not r.is_success:
        out["error"] = data.get("error", data)
        return out
    out["served_model"] = data.get("model")
    out["served_provider"] = data.get("provider")
    out["gen_id"] = data.get("id")
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message", {}) or {}
        tcs = msg.get("tool_calls") or []
        out["has_tool_calls"] = bool(tcs)
        out["tool_names"] = [tc.get("function", {}).get("name") for tc in tcs]
        content = msg.get("content")
        out["content_snippet"] = (content or "")[:160] if isinstance(content, str) else content
        out["finish_reason"] = choices[0].get("finish_reason")
    usage = data.get("usage") or {}
    out["usage"] = {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in usage}
    return out


# --- free lookups ------------------------------------------------------------
def lookup_providers() -> list[str]:
    r = httpx.get(f"{BASE}/providers", headers=HEADERS, timeout=60)
    names = []
    if r.is_success:
        for p in r.json().get("data", []):
            n = p.get("name") or p.get("slug")
            if n:
                names.append(n)
    return names


def lookup_cheap_closed(models_json: dict) -> dict:
    """Pick the cheapest openai/* and anthropic/* slugs for a reachability test."""
    picks: dict = {}
    for prefix in ("openai/", "anthropic/"):
        best = None
        best_price = None
        for m in models_json.get("data", []):
            mid = m.get("id", "")
            if not mid.startswith(prefix):
                continue
            price = m.get("pricing", {}) or {}
            try:
                p = float(price.get("prompt", "0") or 0)
            except (TypeError, ValueError):
                continue
            if p <= 0:
                continue
            if best_price is None or p < best_price:
                best_price, best = p, mid
        if best:
            picks[prefix.rstrip("/")] = best
    return picks


def main() -> None:
    if not KEY:
        _die("OPENROUTER_API_KEY is not set. Inject it from the User env var before running.")
    go = "--go" in sys.argv

    print("=" * 72)
    print("FREE LOOKUPS")
    print("=" * 72)

    provider_names = lookup_providers()
    print(f"\n[L1] /providers returned {len(provider_names)} providers.")
    missing_closed = [p for p in WESTERN_CLOSED if p not in provider_names]
    present_closed = [p for p in WESTERN_CLOSED if p in provider_names]
    missing_open = [p for p in WESTERN_OPEN if p not in provider_names]
    print(f"     Western-closed present in catalog : {present_closed}")
    print(f"     Western-closed NOT matched (slug?) : {missing_closed}")
    print(f"     Western-open NOT matched (slug?)   : {missing_open}")

    models_r = httpx.get(f"{BASE}/models", headers=HEADERS, timeout=60)
    cheap_closed = lookup_cheap_closed(models_r.json()) if models_r.is_success else {}
    print(f"\n[L2] cheapest closed slugs for reach test: {cheap_closed}")

    if not go:
        print("\n" + "=" * 72)
        print("PAID PLAN (re-run with --go to execute). Each call caps max_tokens.")
        print("=" * 72)
        print("  P1 deepseek/deepseek-v4-pro + read_file tool        (~1 completion)")
        print("  P2 openrouter/fusion + read_file tool               (~panel+judge)")
        print("  P3 openrouter/fusion pure consult, max_tokens=120   (~panel+judge)")
        print(f"  P4 {cheap_closed.get('openai','openai/<cheap>')} via UNION allowlist  (~1 completion)")
        print("\nNo tokens spent. Add --go to run the paid checks.")
        return

    print("\n" + "=" * 72)
    print("PAID CHECKS (--go)")
    print("=" * 72)

    # P1 — driver tool-calling sanity (open-host screen; deepseek is open-weight)
    p1 = post_chat({
        "model": "deepseek/deepseek-v4-pro",
        "messages": [{"role": "user", "content": "Read the file ./notes.txt for me. Call the read_file tool."}],
        "tools": [READ_TOOL],
        "tool_choice": "auto",
        "max_tokens": 200,
        "temperature": 0,
        "provider": prefs(WESTERN_OPEN),
    })
    print("\n[P1] driver tool-calling (deepseek-v4-pro):")
    print(json.dumps(p1, indent=2, default=str))

    # P2 — fusion + custom tool (the /brain promote mode): do tool_calls round-trip?
    p2 = post_chat({
        "model": "openrouter/fusion",
        "messages": [{"role": "user", "content": "Read the file ./notes.txt for me. Call the read_file tool."}],
        "tools": [READ_TOOL],
        "tool_choice": "auto",
        "max_tokens": 200,
        "temperature": 0,
        "provider": prefs(UNION_ONLY),
    }, timeout=300)
    print("\n[P2] fusion + custom tool (the /brain mode):")
    print(json.dumps(p2, indent=2, default=str))

    # P3 — pure fusion consult: deliberation returns content + which provider served it
    p3 = post_chat({
        "model": "openrouter/fusion",
        "messages": [{"role": "user", "content": "In one sentence: what is the single biggest risk when running an LLM agent in an unattended file-editing loop?"}],
        "max_tokens": 120,
        "temperature": 0,
        "provider": prefs(UNION_ONLY),
    }, timeout=300)
    print("\n[P3] pure fusion consult:")
    print(json.dumps(p3, indent=2, default=str))

    # P4 — closed-provider reachability through the UNION allowlist.
    # Prefer a genuinely-closed model (Claude is only served first-party by Anthropic);
    # an open-weight slug like gpt-oss-* would route to an open host and not test the closed provider.
    closed_slug = cheap_closed.get("anthropic") or cheap_closed.get("openai") or "anthropic/claude-3-haiku"
    p4 = post_chat({
        "model": closed_slug,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 8,
        "temperature": 0,
        "provider": prefs(UNION_ONLY),
    })
    print(f"\n[P4] closed-provider reach ({closed_slug}):")
    print(json.dumps(p4, indent=2, default=str))

    print("\n" + "=" * 72)
    print("VERDICT SUMMARY")
    print("=" * 72)
    print(f"  driver tools work        : {p1.get('has_tool_calls')}  (served {p1.get('served_provider')})")
    print(f"  fusion+tools round-trip  : {p2.get('has_tool_calls')}  (status {p2.get('http_status')})  <- gates /brain mode")
    print(f"  fusion consult works     : {p3.get('ok')}  (served {p3.get('served_provider')})")
    print(f"  closed Western reachable : {p4.get('ok')}  (served {p4.get('served_provider')})  <- gates union allowlist")


if __name__ == "__main__":
    main()
