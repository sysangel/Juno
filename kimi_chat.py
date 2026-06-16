"""kimi_chat.py - direct chat with Kimi via the Moonshot AI first-party API.

OpenAI-compatible. Reads MOONSHOT_API_KEY from env (never printed). Multi-turn,
history in-process only. Run in your OWN terminal (it's interactive).

PRIVACY: Moonshot is a first-party Chinese API and is OUTSIDE the OpenRouter
no-train / Western-allowlist screen the rest of agent-loop uses. Do NOT send
internal / sensitive data through it. For screened routing use chat.py instead.

Usage:
  python kimi_chat.py --list                  # show available model ids, then pick
  python kimi_chat.py [model-id] [--system "..."] [--base-url URL] [--temperature T]
    default model-id: kimi-k2.7-code   (confirm the exact id with --list first)
Commands in chat: /exit  /reset  /system <text>  /usage
"""
from __future__ import annotations

import argparse
import os
import sys

from openai import OpenAI

DEFAULT_BASE = "https://api.moonshot.ai/v1"  # international endpoint (.ai, not .cn)
DEFAULT_MODEL = "kimi-k2.7-code"


def _text_of(msg) -> str:
    # k2.7 is a thinking model; the final answer is in .content, but guard for
    # responses that only carry reasoning_content.
    if getattr(msg, "content", None):
        return msg.content
    extra = getattr(msg, "model_extra", None) or {}
    return extra.get("reasoning_content", "") or ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Direct Kimi chat via the Moonshot API.")
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("--list", action="store_true", help="list available model ids and exit")
    ap.add_argument("--system", default=None, help="optional system prompt")
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    key = os.environ.get("MOONSHOT_API_KEY")
    if not key:
        print("NO_KEY: set MOONSHOT_API_KEY first (get one at https://platform.moonshot.ai).")
        return 2
    client = OpenAI(api_key=key, base_url=args.base_url)

    if args.list:
        try:
            for m in client.models.list().data:
                print(m.id)
        except Exception as e:
            print(f"[error listing models] {e}")
            return 1
        return 0

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    tot_in = tot_out = 0
    print(f"chat with {args.model} @ {args.base_url}  (Moonshot first-party - UNSCREENED)")
    print("commands: /exit  /reset  /system <text>  /usage")

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
            messages = [m for m in messages if m["role"] == "system"]
            print("(conversation cleared; system prompt kept)")
            continue
        if user == "/usage":
            print(f"(in={tot_in} out={tot_out} tok)")
            continue
        if user.startswith("/system "):
            messages.append({"role": "system", "content": user[len("/system "):]})
            print("(system message added)")
            continue

        messages.append({"role": "user", "content": user})
        try:
            resp = client.chat.completions.create(
                model=args.model, messages=messages, temperature=args.temperature,
            )
        except Exception as e:
            messages.pop()
            print(f"[error] {e}")
            continue
        msg = resp.choices[0].message
        text = _text_of(msg)
        messages.append({"role": "assistant", "content": text})
        print(f"\n{args.model}> {text}")
        u = getattr(resp, "usage", None)
        if u:
            tot_in += u.prompt_tokens
            tot_out += u.completion_tokens
            print(f"   (turn {u.prompt_tokens}/{u.completion_tokens} tok | total {tot_in}/{tot_out})")

    print(f"\nsession: in={tot_in} out={tot_out} tok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
