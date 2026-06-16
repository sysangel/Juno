"""Diagnose why write_tests returned an empty/degenerate spec.

Reproduces the exact write_tests call (system prompt + prompt) for the regex
task and dumps the raw response so we can see whether content is empty, a
content-block list, fenced, or truncated by finish_reason. Reads key from env.
"""
from __future__ import annotations

import os
import sys
import time

from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_loop  # noqa: E402
import headtohead  # noqa: E402

MODEL = "moonshotai/kimi-k2.7-code"


def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("NO_KEY_IN_ENV")
        return 2
    task = headtohead.DEFAULT_TASK
    prompt = (
        f"Write a pytest test file named `{agent_loop.TEST_FILENAME}` for this task:\n\n"
        f"TASK:\n{task}\n\n"
        "Requirements:\n"
        f"- Import the implementation from `solution` (e.g. `from solution import ...`).\n"
        "- Encode the task's requirements as concrete, deterministic assertions.\n"
        "- Cover normal cases AND edge cases (empty input, boundaries, errors).\n"
        "- Use plain `def test_*()` functions and bare `assert`. No network, "
        "no filesystem, no randomness without a fixed seed.\n"
        "- The tests are the FIXED contract; do not leave them trivially passing.\n"
        "Return the complete file contents only."
    )
    llm = agent_loop.get_llm(provider="openrouter", model=MODEL)
    print(f"calling {MODEL} write_tests ...", flush=True)
    t = time.time()
    resp = llm.invoke([
        SystemMessage(content=agent_loop._TESTS_SYSTEM),
        HumanMessage(content=prompt),
    ])
    dt = time.time() - t
    c = resp.content
    md = getattr(resp, "response_metadata", None) or {}
    um = getattr(resp, "usage_metadata", None) or {}
    print(f"elapsed={dt:.1f}s")
    print("content_type:", type(c).__name__)
    if isinstance(c, str):
        print("content_len:", len(c))
        print("content_repr_head:", repr(c)[:400])
    else:
        print("content_list:", repr(c)[:600])
    print("finish_reason:", md.get("finish_reason"))
    print("response_metadata_keys:", sorted(md.keys()))
    print("usage_metadata:", um)
    coerced = agent_loop._content_to_text(c)
    stripped = agent_loop.strip_code_fences(coerced)
    print("coerced_len:", len(coerced))
    print("stripped_len:", len(stripped))
    print("stripped_head:", repr(stripped)[:400])
    print("PROBE_TESTS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
