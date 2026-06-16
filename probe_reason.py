"""Check whether the bounded test prompt lets each model return real content.

Tests both default models through the loop's actual get_llm() config (16K cap +
screen + reasoning param) using the now-softened _TESTS_SYSTEM. Reads key from env.
"""
from __future__ import annotations

import os
import sys
import time

from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_loop  # noqa: E402
import headtohead  # noqa: E402

MODELS = ["moonshotai/kimi-k2.7-code", "deepseek/deepseek-v4-pro"]


def build_prompt() -> str:
    task = headtohead.DEFAULT_TASK
    return (
        f"Write a pytest test file named `{agent_loop.TEST_FILENAME}` for this task:\n\n"
        f"TASK:\n{task}\n\n"
        "Requirements:\n"
        f"- Import the implementation from `solution` (e.g. `from solution import ...`).\n"
        "- Encode the task's requirements as concrete, deterministic assertions.\n"
        "- Cover normal cases AND edge cases (empty input, boundaries, errors).\n"
        "- Aim for ~15-30 focused tests; be decisive.\n"
        "Return the complete file contents only."
    )


def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("NO_KEY_IN_ENV")
        return 2
    prompt = build_prompt()
    for model in MODELS:
        llm = agent_loop.get_llm(provider="openrouter", model=model)
        print(f"\n### {model} (max_tokens={agent_loop.OPENROUTER_MAX_OUTPUT_TOKENS}) ...",
              flush=True)
        t = time.time()
        try:
            resp = llm.invoke([
                SystemMessage(content=agent_loop._TESTS_SYSTEM),
                HumanMessage(content=prompt),
            ])
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED after {time.time()-t:.1f}s: {type(exc).__name__}: {exc}")
            continue
        dt = time.time() - t
        text = agent_loop._content_to_text(resp.content)
        stripped = agent_loop.strip_code_fences(text)
        md = getattr(resp, "response_metadata", None) or {}
        um = getattr(resp, "usage_metadata", None) or {}
        rtok = (um.get("output_token_details") or {}).get("reasoning")
        ndefs = stripped.count("def test_")
        print(f"  elapsed={dt:.1f}s content_len={len(text)} stripped_len={len(stripped)} "
              f"n_test_defs={ndefs} out_tok={um.get('output_tokens')} reasoning={rtok} "
              f"finish={md.get('finish_reason')}")
        verdict = "WORKS" if (stripped.strip() and ndefs >= 1) else "EMPTY/BAD"
        print(f"  -> {verdict}")
    print("\nPROBE_REASON_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
