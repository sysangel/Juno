"""Time a single realistic Kimi code-generation call through the loop's client.

Confirms large-output OpenRouter calls complete (or fail fast at the new
timeout) before committing to a long head-to-head run. Reads the key from env.
"""
from __future__ import annotations

import os
import sys
import time

from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_loop  # noqa: E402

MODEL = "moonshotai/kimi-k2.7-code"
PROMPT = (
    "Write a complete, self-contained Python module implementing a Thompson-NFA "
    "regular-expression engine with a function fullmatch(pattern: str, text: str) "
    "-> bool supporting literals, '.', '*', '+', '?', '|', and '(' ')'. Include a "
    "RegexError(ValueError). Return the full file contents only."
)


def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("NO_KEY_IN_ENV")
        return 2
    llm = agent_loop.get_llm(provider="openrouter", model=MODEL)
    print(f"calling {MODEL} (timeout={agent_loop.OPENROUTER_REQUEST_TIMEOUT}s, "
          f"max_retries={agent_loop.OPENROUTER_MAX_RETRIES}) ...", flush=True)
    t = time.time()
    try:
        resp = llm.invoke([
            SystemMessage(content=agent_loop._CODE_SYSTEM),
            HumanMessage(content=PROMPT),
        ])
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED after {time.time() - t:.1f}s: {type(exc).__name__}: {exc}")
        return 1
    dt = time.time() - t
    text = agent_loop._content_to_text(resp.content)
    md = getattr(resp, "response_metadata", None) or {}
    um = getattr(resp, "usage_metadata", None) or {}
    print(f"OK in {dt:.1f}s | chars={len(text)} | provider_model={md.get('model_name')} "
          f"| in={um.get('input_tokens')} out={um.get('output_tokens')}")
    print("PROBE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
