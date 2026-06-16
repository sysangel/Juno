from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .models import *  # re-export compatibility surface
from .providers import *
from .runtime import *
from .stores import *
from .theme import *
from .tool_registry import *
from .ui import PlainReplUI, PromptToolkitUI, _prompt_toolkit_available, print_banner, run_repl


def run_once(args: argparse.Namespace, client: ProviderClient, memory: MemoryStore | None, sessions: SessionStore) -> int:
    if getattr(args, "mode", "chat") == "agent":
        runtime = HarnessRuntime.create(args, memory, sessions)
        try:
            result = runtime.submit_user_message(args.query)
        finally:
            runtime.close()
        session_id = runtime.session_id
    else:
        session_id = sessions.start_session(args.provider, args.model, title=args.query[:80])
        messages = build_initial_messages(args.system, memory, mode="chat", provider=args.provider)
        messages.append({"role": "user", "content": args.query})
        sessions.append(session_id, "user", args.query)
        result = client.complete(messages)
        sessions.append(session_id, "assistant", result.text)
        sessions.end_session(session_id)
    color = sys.stdout.isatty()

    print(format_assistant_response(result.text, provider=args.provider, model=args.model, color=color))
    print()
    print(format_usage_line(result.input_tokens, result.output_tokens, session_id=session_id, color=color))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Custom CLI agent harness with OpenRouter/OpenAI/Anthropic/Claude-Code providers.")
    parser.add_argument("-q", "--query", help="single-turn query; omit for interactive REPL")
    parser.add_argument("--provider", choices=["openrouter", "openai", "anthropic", "claude-code", "echo"], default=os.getenv("HARNESS_PROVIDER", "openrouter"))
    parser.add_argument("--model", default=None, help="model id/slug; defaults depend on provider")
    parser.add_argument("--default-model", default=None, metavar="KEY|PROVIDER/MODEL", help="persist a default model in config.json, then run")
    parser.add_argument("--base-url", default=None, help="override provider endpoint/base URL")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=os.getenv("JUNO_CLAUDE_EFFORT"),
        help="Claude Code reasoning effort for claude-code provider (low, medium, high, xhigh, max)",
    )
    parser.add_argument("--system", default="You are a careful senior engineer working from a CLI harness.")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME, help="state directory (default: ~/.agent-harness or AGENT_HARNESS_HOME)")
    parser.add_argument("--mode", choices=["chat", "agent"], default=os.getenv("JUNO_MODE", "agent"), help="agent enables tools by default; chat disables tools for testing")
    parser.add_argument("--approval", choices=["never", "on-request", "always", "dangerous"], default=os.getenv("JUNO_APPROVAL", "on-request"), help="tool approval policy for --mode agent")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="workspace root for agent-mode tools")
    parser.add_argument("--max-tool-turns", type=int, default=int(os.getenv("JUNO_MAX_TOOL_TURNS", "8")), help="maximum provider/tool iterations per user turn")
    parser.add_argument("--no-memory", action="store_true", help="do not load/inject memory")
    parser.add_argument(
        "--ui",
        choices=["auto", "plain", "prompt"],
        default=os.getenv("HARNESS_UI", "auto"),
        help="interactive UI mode: auto (Juno prompt_toolkit on a TTY, else plain), "
             "plain (basic REPL), or prompt (force Juno; errors if prompt_toolkit missing)",
    )
    ui_group = parser.add_mutually_exclusive_group()
    ui_group.add_argument(
        "--tui",
        action="store_true",
        dest="_tui",
        help="alias for --ui prompt (Hermes-compatible flag)",
    )
    ui_group.add_argument(
        "--cli",
        action="store_true",
        dest="_cli",
        help="alias for --ui plain (Hermes-compatible flag)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    # Reconfigure stdout/stderr to UTF-8 so Unicode glyphs (✦, box-drawing)
    # do not crash on Windows codepage cp1252.  errors="replace" guarantees
    # we never raise UnicodeEncodeError, even on truly broken terminals.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    provider_explicit = any(a == "--provider" or a.startswith("--provider=") for a in raw_argv)
    # --tui / --cli aliases override --ui
    if getattr(args, "_tui", False):
        args.ui = "prompt"
    elif getattr(args, "_cli", False):
        args.ui = "plain"

    config = HarnessConfig(args.home)
    if getattr(args, "default_model", None):
        prov, model = resolve_model_ref(args.default_model)
        config.set_default_model(prov, model)
        print(f"\u2726 default model set \u2192 {prov}/{model}")
    # Model resolution order:
    #   1. CLI --model (already in args.model if given)
    #   2. Juno config default (unless --provider was explicit)
    #   3. default_model(provider) fallback
    if args.model is None:
        cfg_default = config.default_model()
        if cfg_default is not None and not provider_explicit:
            args.provider, args.model = cfg_default
    args.model = args.model or default_model(args.provider)

    print(f"RUN provider={args.provider} model={args.model}", file=sys.stderr)
    memory: MemoryStore | None = None
    if not args.no_memory:
        memory = MemoryStore(args.home)
        memory.load()
    sessions = SessionStore(args.home / "sessions.db")
    client = ProviderClient(
        args.provider,
        args.model,
        args.temperature,
        args.base_url,
        args.effort,
        mode=args.mode,
        approval=args.approval,
        workspace=args.workspace,
        max_tool_turns=args.max_tool_turns,
    )
    if args.query:
        return run_once(args, client, memory, sessions)
    return run_repl(args, client, memory, sessions, config=config)


if __name__ == "__main__":
    raise SystemExit(main())
