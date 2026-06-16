"""Root-cause sweep: WHY is Kimi K2.7 Code bad at write_tests?

Isolates model-vs-provider-vs-prompt by firing a small matrix of OpenRouter
calls concurrently and tabulating: elapsed, content length, test-def count,
output tokens, reasoning tokens, finish_reason, served provider, error.

Reads OPENROUTER_API_KEY from env; never prints it. Writes probe_matrix.json.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_loop  # noqa: E402
import headtohead  # noqa: E402

URL = "https://openrouter.ai/api/v1/chat/completions"
KIMI = "moonshotai/kimi-k2.7-code"
DEEPSEEK = "deepseek/deepseek-v4-pro"
MAXTOK = 12000

_TASK = headtohead.DEFAULT_TASK
TESTS_PROMPT = (
    f"Write a pytest test file named `{agent_loop.TEST_FILENAME}` for this task:\n\n"
    f"TASK:\n{_TASK}\n\n"
    "Requirements:\n- Import from `solution`.\n- Concrete deterministic assertions.\n"
    "- Cover normal + edge cases.\n- Aim for ~15-30 focused tests; be decisive.\n"
    "Return the complete file contents only."
)
CODE_PROMPT = (
    "Write a complete, self-contained Python module implementing a Thompson-NFA "
    "regex engine with fullmatch(pattern: str, text: str) -> bool supporting "
    "literals, '.', '*', '+', '?', '|', '(' ')', and a RegexError(ValueError). "
    "Return the full file contents only."
)

# (label, model, provider_only, reasoning_override, prompt_kind)
CONFIGS = [
    ("kimi_tests_parasail",  KIMI, "Parasail",  None,               "tests"),
    ("kimi_tests_fireworks", KIMI, "Fireworks", None,               "tests"),
    ("kimi_tests_deepinfra", KIMI, "DeepInfra", None,               "tests"),
    ("kimi_tests_together",  KIMI, "Together",  None,               "tests"),
    ("kimi_tests_baseten",   KIMI, "Baseten",   None,               "tests"),
    ("kimi_tests_noReason_parasail",  KIMI, "Parasail",  {"enabled": False}, "tests"),
    ("kimi_tests_noReason_fireworks", KIMI, "Fireworks", {"enabled": False}, "tests"),
    ("kimi_code_cheapest",   KIMI, None,        None,               "code"),
    ("deepseek_tests_ctrl",  DEEPSEEK, None,     None,              "tests"),
]


def run_one(key: str, cfg) -> dict:
    label, model, only, reasoning, kind = cfg
    system = agent_loop._TESTS_SYSTEM if kind == "tests" else agent_loop._CODE_SYSTEM
    user = TESTS_PROMPT if kind == "tests" else CODE_PROMPT
    provider = {"data_collection": "deny"}
    if only:
        provider["only"] = [only]
    else:
        provider["only"] = agent_loop.OPENROUTER_WESTERN_PROVIDERS
        provider["sort"] = "price"
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": MAXTOK,
        "temperature": 0.1,
        "provider": provider,
    }
    if reasoning is not None:
        body["reasoning"] = reasoning
    headers = {"Authorization": f"Bearer {key}",
               "HTTP-Referer": "https://github.com/angelsystems/agent-loop",
               "X-Title": "agent-loop-matrix"}
    t = time.time()
    try:
        r = httpx.post(URL, headers=headers, json=body, timeout=300)
        dt = time.time() - t
        d = r.json()
        if r.status_code != 200:
            return {"label": label, "status": r.status_code, "elapsed": round(dt, 1),
                    "error": (d.get("error") or {}).get("message", str(d))[:160]}
        choice = (d.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        usage = d.get("usage") or {}
        ctd = usage.get("completion_tokens_details") or {}
        return {
            "label": label, "status": 200, "elapsed": round(dt, 1),
            "content_len": len(content), "n_test_defs": content.count("def test_"),
            "out_tok": usage.get("completion_tokens"),
            "reasoning_tok": ctd.get("reasoning_tokens"),
            "finish": choice.get("finish_reason"),
            "served": d.get("provider"),
            "verdict": "WORKS" if content.strip() else "EMPTY",
        }
    except Exception as exc:  # noqa: BLE001
        return {"label": label, "status": "EXC", "elapsed": round(time.time() - t, 1),
                "error": f"{type(exc).__name__}: {exc}"[:160]}


def main() -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("NO_KEY_IN_ENV")
        return 2
    print(f"firing {len(CONFIGS)} configs concurrently (max_tokens={MAXTOK}) ...", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(CONFIGS)) as ex:
        futs = {ex.submit(run_one, key, c): c[0] for c in CONFIGS}
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            results.append(res)
            print("DONE", json.dumps(res), flush=True)
    order = {c[0]: i for i, c in enumerate(CONFIGS)}
    results.sort(key=lambda r: order.get(r["label"], 99))
    print("\n=== SUMMARY ===")
    for r in results:
        if r.get("status") == 200:
            print(f"{r['label']:32s} {r['verdict']:5s} {r['elapsed']:6.1f}s "
                  f"content={r['content_len']:5d} tests={r['n_test_defs']:2d} "
                  f"out={r['out_tok']} reason={r['reasoning_tok']} "
                  f"finish={r['finish']} served={r['served']}")
        else:
            print(f"{r['label']:32s} ERR status={r['status']} {r.get('error','')}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "probe_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("PROBE_MATRIX_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
