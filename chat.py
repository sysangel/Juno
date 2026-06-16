"""chat.py - direct, privacy-screened chat with any OpenRouter model slug.

Talk to a model (default: Kimi) through the SAME data-secure provider screen the
agent loop uses (no-train + Western allowlist), to get a feel for it. Multi-turn,
history kept in-process only. OPENROUTER_API_KEY is read from env, never printed.
Run with the venv python; this is INTERACTIVE, so run it in your own terminal.

Usage:
  python chat.py [model-slug] [--system "..."] [--provider-only Together]
    default slug: moonshotai/kimi-k2.7-code
    --provider-only pins one OpenRouter provider (use Together for Kimi to avoid
      the Parasail uncappable-reasoning spiral if it gets rambly/truncates).

In-chat commands: /exit  /reset  /system <text>  /cost
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import agent_loop  # noqa: E402


def _turn_cost(slug: str, in_tok: int, out_tok: int) -> float:
    price = getattr(agent_loop, "_MODEL_PRICES", {}).get(slug)
    if not price:
        return 0.0
    return in_tok / 1e6 * price.get("in", 0) + out_tok / 1e6 * price.get("out", 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Privacy-screened chat with an OpenRouter model.")
    ap.add_argument("model", nargs="?", default="moonshotai/kimi-k2.7-code")
    ap.add_argument("--system", default=None, help="optional system prompt")
    ap.add_argument("--provider-only", default=None,
                    help="pin to one OpenRouter provider (e.g. Together)")
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("NO_KEY_IN_ENV - inject OPENROUTER_API_KEY first (see loopy/swarm skill).")
        return 2

    llm = agent_loop.get_llm(provider="openrouter", model=args.model)
    if args.provider_only:
        eb = dict(getattr(llm, "extra_body", {}) or {})
        prov = dict(eb.get("provider", {}))
        prov["only"] = [args.provider_only]
        eb["provider"] = prov
        llm.extra_body = eb

    history: list[tuple[str, str]] = []
    if args.system:
        history.append(("system", args.system))

    tag = args.model.split("/")[-1]
    via = f" via {args.provider_only}" if args.provider_only else ""
    tot_in = tot_out = 0
    tot_cost = 0.0
    print(f"chat with {args.model}{via}  (no-train + Western screen)")
    print("commands: /exit  /reset  /system <text>  /cost")

    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            history = [h for h in history if h[0] == "system"]
            print("(conversation cleared; system prompt kept)")
            continue
        if user == "/cost":
            print(f"(in={tot_in} out={tot_out} tok  est ${tot_cost:.4f})")
            continue
        if user.startswith("/system "):
            history.append(("system", user[len("/system "):]))
            print("(system message added)")
            continue

        history.append(("human", user))
        try:
            resp = llm.invoke(history)
        except Exception as e:
            history.pop()
            print(f"[error] {e}")
            continue
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        history.append(("ai", text))
        print(f"\n{tag}> {text}")

        um = getattr(resp, "usage_metadata", None) or {}
        i, o = int(um.get("input_tokens", 0)), int(um.get("output_tokens", 0))
        tot_in += i
        tot_out += o
        tot_cost += _turn_cost(args.model, i, o)
        served = (getattr(resp, "response_metadata", None) or {}).get("model_provider")
        print(f"   (turn {i}/{o} tok | running ${tot_cost:.4f}"
              + (f" | served by {served}" if served else "") + ")")

    print(f"\nsession: in={tot_in} out={tot_out} tok  est ${tot_cost:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
