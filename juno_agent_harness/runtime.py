from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent_loop import JUNO_AGENT_CAPABILITY_NOTICE, default_tool_registry, run_agent_turn
from .models import (
    DEFAULT_HOME, Provider, PROVIDERS, SkillSummary,
    default_model, discover_local_skills, format_model_catalog,
    model_aliases, model_catalog, model_catalog_by_key, normalize_provider,
    resolve_model_ref, resolve_model_selection,
)
from .providers import ChatResult, ProviderClient
from .stores import MemoryStore, SessionStore
from .theme import RainbowSweep, rainbow, use_rainbow

JUNO_CHAT_ONLY_CAPABILITY_NOTICE = """
Juno capability contract:
- This interface is a chat harness, not a live tool-running agent.
- You cannot inspect files, run shell commands, browse the web, access hidden memory indexes, or call tools from inside this conversation unless the user has explicitly pasted the relevant information into the chat.
- Do not say "I'll check", "let me inspect", "I'll run", or similar action promises unless you can complete the answer from the conversation text alone.
- If the user asks for an action that requires tools, clearly say that this Juno session is chat-only and tell them what command/context to provide or suggest using a tool-enabled agent.
- Think privately if needed, then answer directly and concisely; do not expose hidden chain-of-thought or claim to be using a separate thinking mode.
""".strip()


TOOL_LOOP_PROVIDERS = {"echo", "openai", "openrouter"}
NATIVE_AGENT_PROVIDERS = {"claude-code"}


def provider_supports_juno_tools(provider: str) -> bool:
    """Whether this provider path can return normalized ToolCall objects to Juno.

    These providers return normalized ToolCall objects to Juno. Claude Code is
    handled separately because its native CLI executes tools internally.
    """
    return provider in TOOL_LOOP_PROVIDERS


def provider_supports_agent_mode(provider: str) -> bool:
    return provider_supports_juno_tools(provider) or provider in NATIVE_AGENT_PROVIDERS


def build_initial_messages(
    system_prompt: str,
    memory: MemoryStore | None,
    mode: str = "chat",
    provider: str = "echo",
) -> list[dict[str, str]]:
    base = system_prompt.strip() or "You are a careful, useful CLI agent."
    if mode == "agent" and provider_supports_juno_tools(provider):
        base += "\n\n" + JUNO_AGENT_CAPABILITY_NOTICE
    elif mode == "agent" and provider == "claude-code":
        base += (
            "\n\nClaude Code native tool contract:\n"
            "- This provider path may use Claude Code native tools that Juno enables.\n"
            "- Actually call the native tools; do not print XML, pseudo function calls, or claims that a tool was requested.\n"
            "- In the default approval policy, only read/search/list tools are enabled.\n"
            "- Summarize real tool results concisely."
        )
    else:
        base += "\n\n" + JUNO_CHAT_ONLY_CAPABILITY_NOTICE
        if mode == "agent":
            base += (
                "\n\nProvider note: this provider path is not wired to Juno's "
                "tool loop yet, so behave as chat-only until the harness reports "
                "actual tool results."
            )
    if memory:
        block = memory.system_prompt_block()
        if block:
            base += "\n\nPersistent memory snapshot follows. Treat it as background context, not as a new user request.\n" + block
    return [{"role": "system", "content": base}]


# ---------------------------------------------------------------------------
# UI-agnostic command vocabulary + rendering helpers
# ---------------------------------------------------------------------------

# Slash commands offered to the completer and shown in /help.
SLASH_COMMANDS = [
    "/help",
    "/status",
    "/model",
    "/memory",
    "/search",
    "/reset",
    "/clear",
    "/skills",
    "/menu",
    "/exit",
]
# Provider/model tokens accepted after `/model` (canonical names, aliases, catalog keys).
PROVIDER_WORDS = [
    "codex",
    "gpt5",
    "gpt-5.1",
    "claude-code",
    "claude",
    "sonnet",
    "opus",
    "anthropic",
    "openai",
    "chatgpt",
    "openrouter",
    "or",
    "dp",
    "deepseek",
    "fm3v4",
    "fbudget",
    "fhigh",
    "echo",
]
MEMORY_SUBCOMMANDS = ["list", "add", "replace", "remove", "reload"]
MEMORY_TARGETS = ["memory", "user"]


def format_help() -> str:
    return (
        "commands\n"
        "  /help                         show this command list\n"
        "  /status                       provider/model/session/memory/token totals\n"
        "  /model                        open segmented Juno picker; plain mode prints catalog\n"
        "  /model sonnet                 switch to Claude Code Sonnet\n"
        "  /model codex                  switch to OpenAI/Codex default\n"
        "  /model dp                     switch to OpenRouter DeepSeek shortcut\n"
        "  /model openai gpt-5.1         switch to raw provider/model pair\n"
        "  /skills                       list local skills (SKILL.md under <home>/skills)\n"
        "  /menu                         command palette of common actions\n"
        "  /memory list                  list durable memory\n"
        "  /memory add memory <entry>    add a memory entry\n"
        "  /memory add user <entry>      add a user-profile entry\n"
        "  /memory replace <t> <old> => <new>   replace an entry\n"
        "  /memory remove <t> <old>      remove an entry\n"
        "  /memory reload                reload durable memory into active prompt\n"
        "  /search <query>               FTS search across sessions.db\n"
        "  /reset                        clear conversation, keep memory snapshot\n"
        "  /clear                        alias for /reset\n"
        "  /exit                         exit (Ctrl-D also works)\n"
        "aliases: claude->claude-code, chatgpt/gpt->openai, or->openrouter"
    )


MENU_ACTIONS = [
    ("model", "Model selector", "/model"),
    ("skills", "Skills hub", "/skills"),
    ("status", "Status", "/status"),
    ("memory", "Memory list", "/memory list"),
    ("search", "Search sessions", "/search"),
    ("reset", "Reset conversation", "/reset"),
    ("clear", "Clear screen", "/clear"),
    ("exit", "Exit", "/exit"),
]


def format_menu_palette() -> str:
    """Render the command palette (Task 9). Plain text so it renders in every UI."""
    lines = ["menu - common actions (type the command to run it)"]
    for _key, label, command in MENU_ACTIONS:
        lines.append(f"  {label:<22} {command}")
    return "\n".join(lines)


def format_memory_help() -> str:
    return (
        "usage:\n"
        "  /memory list\n"
        "  /memory add <memory|user> <entry>\n"
        "  /memory replace <memory|user> <old substring> => <new entry>\n"
        "  /memory remove <memory|user> <old substring>\n"
        "  /memory reload"
    )


def format_status(runtime: "HarnessRuntime") -> str:
    args = runtime.args
    lines = [
        f"provider : {args.provider}",
        f"model    : {args.model}",
        f"mode     : {getattr(args, 'mode', 'chat')}",
        f"approval : {getattr(args, 'approval', 'on-request')}",
        f"workspace: {getattr(args, 'workspace', Path.cwd())}",
        f"home     : {args.home}",
        f"session  : {runtime.session_id}",
    ]
    if runtime.memory is not None:
        mem_count = len(runtime.memory.memory_entries)
        user_count = len(runtime.memory.user_entries)
        lines.append(f"memory   : on ({mem_count} memory / {user_count} user entries)")
    else:
        lines.append("memory   : off")
    turns = sum(1 for m in runtime.messages if m["role"] != "system")
    lines.append(f"tokens   : in/out = {runtime.total_in}/{runtime.total_out}")
    lines.append(f"transcript turns : {turns}")
    if args.provider == "claude-code":
        lines.append("note     : claude-code uses local `claude` OAuth bridge (no ANTHROPIC_API_KEY)")
        if getattr(args, "mode", "chat") == "agent":
            lines.append("tools    : claude-code agent mode may use approved local Claude Code tools")
        else:
            lines.append("tools    : chat-only; Juno disables Claude Code tools and runs one response per turn")
        if getattr(args, "effort", None):
            lines.append(f"effort   : {args.effort}")
    return "\n".join(lines)


@dataclass
class CommandResult:
    """Outcome of a slash command: whether to keep looping + text to render."""

    should_continue: bool
    output: str = ""


@dataclass
class HarnessRuntime:
    """Owns mutable session state and command dispatch, UI-agnostic.

    UIs call methods on this object instead of mutating locals; both PlainReplUI
    and PromptToolkitUI (Juno) drive the same runtime so behavior stays identical.
    """

    args: argparse.Namespace
    config: "HarnessConfig | None"
    memory: MemoryStore | None
    sessions: SessionStore
    session_id: str
    messages: list[dict[str, str]]
    client: ProviderClient
    total_in: int = 0
    total_out: int = 0
    tool_registry: Any | None = None

    @classmethod
    def create(
        cls,
        args: argparse.Namespace,
        memory: MemoryStore | None,
        sessions: SessionStore,
        *,
        config: "HarnessConfig | None" = None,
    ) -> "HarnessRuntime":
        session_id = sessions.start_session(args.provider, args.model)
        messages = build_initial_messages(args.system, memory, mode=getattr(args, "mode", "chat"), provider=args.provider)
        client = ProviderClient(
            args.provider,
            args.model,
            args.temperature,
            args.base_url,
            getattr(args, "effort", None),
            mode=getattr(args, "mode", "chat"),
            approval=getattr(args, "approval", "on-request"),
            workspace=getattr(args, "workspace", None),
            max_tool_turns=getattr(args, "max_tool_turns", 8),
        )
        return cls(
            args=args,
            config=config,
            memory=memory,
            sessions=sessions,
            session_id=session_id,
            messages=messages,
            client=client,
            tool_registry=default_tool_registry() if (
                getattr(args, "mode", "chat") == "agent" and provider_supports_juno_tools(args.provider)
            ) else None,
        )

    # -- presentation -------------------------------------------------------

    def status_line(self) -> str:
        """Compact one-line status for the bottom toolbar."""
        mem = "on" if self.memory is not None else "off"
        return (
            f"provider={self.args.provider} | model={self.args.model} | "
            f"memory={mem} | session={self.session_id} | "
            f"tokens in/out={self.total_in}/{self.total_out} | /help"
        )

    def prompt_label(self) -> str:
        return f"{self.args.provider}/{self.args.model} \u203a "

    # -- state transitions --------------------------------------------------

    def reset_conversation(self) -> str:
        self.messages = build_initial_messages(self.args.system, self.memory, mode=getattr(self.args, "mode", "chat"), provider=self.args.provider)
        return "conversation reset; current memory snapshot kept"

    def reload_memory(self) -> str:
        if self.memory is None:
            return "memory disabled"
        self.memory.load()
        conversation = [m for m in self.messages if m["role"] != "system"]
        self.messages = build_initial_messages(self.args.system, self.memory, mode=getattr(self.args, "mode", "chat"), provider=self.args.provider) + conversation
        return "memory reloaded and re-injected into the active system prompt"

    def switch_provider(self, provider_text: str, model_text: str | None = None) -> str:
        provider, model, note = resolve_model_selection(provider_text, model_text)
        self.args.provider = provider
        self.args.model = model
        self.client = ProviderClient(
            provider,
            model,
            self.args.temperature,
            self.args.base_url,
            getattr(self.args, "effort", None),
            mode=getattr(self.args, "mode", "chat"),
            approval=getattr(self.args, "approval", "on-request"),
            workspace=getattr(self.args, "workspace", None),
            max_tool_turns=getattr(self.args, "max_tool_turns", 8),
        )
        self.tool_registry = default_tool_registry() if (
            getattr(self.args, "mode", "chat") == "agent" and provider_supports_juno_tools(provider)
        ) else None
        conversation = [m for m in self.messages if m["role"] != "system"]
        self.messages = build_initial_messages(
            self.args.system,
            self.memory,
            mode=getattr(self.args, "mode", "chat"),
            provider=provider,
        ) + conversation
        self.sessions.update_session_model(self.session_id, provider, model)
        prefix = f"{note}\n" if note else ""
        return f"{prefix}switched to provider={provider} model={model}"

    # -- command dispatch ---------------------------------------------------

    def handle_command(self, line: str) -> CommandResult:
        """Dispatch a slash command. Never raises; errors become output text."""
        line = line.strip()
        try:
            if line in {"/exit", "/quit"}:
                return CommandResult(False, "")
            if line == "/help":
                return CommandResult(True, format_help())
            if line == "/status":
                return CommandResult(True, format_status(self))
            if line in {"/reset", "/clear"}:
                return CommandResult(True, self.reset_conversation())
            if line.startswith("/model") or line.startswith("/provider"):
                parts = line.split(maxsplit=2)
                if len(parts) == 1:
                    body = (
                        f"current provider={self.args.provider} model={self.args.model}\n\n"
                        + format_model_catalog(self.args.provider, self.args.model)
                    )
                    return CommandResult(True, body)
                if parts[1].lower() == "default":
                    return CommandResult(True, self._handle_model_default(parts[2] if len(parts) > 2 else None))
                msg = self.switch_provider(parts[1], parts[2] if len(parts) > 2 else None)
                return CommandResult(True, msg)
            if line.startswith("/memory"):
                return CommandResult(True, self._handle_memory(line))
            if line.startswith("/search "):
                return CommandResult(True, self._handle_search(line[len("/search "):].strip()))
            if line == "/skills":
                return CommandResult(True, self._list_local_skills())
            if line == "/menu":
                return CommandResult(True, format_menu_palette())
            return CommandResult(True, "unknown command; try /help")
        except Exception as exc:  # noqa: BLE001 - command errors must not crash the UI
            return CommandResult(True, f"[command error] {exc}")

    def _handle_model_default(self, ref: str | None) -> str:
        """Get or set the persistent default model.

        `/model default` reports the current default; `/model default <ref>`
        resolves either a catalog KEY or PROVIDER/MODEL and persists it.
        """
        if not ref:
            current = getattr(self, "_config_default_model", None)
            if current is None and self.config is not None:
                current = self.config.default_model()
            if current is None:
                return "no default model set; use /model default <KEY|PROVIDER/MODEL>"
            prov, model = current
            return f"current default model → {prov}/{model}"
        prov, model = resolve_model_ref(ref)
        if self.config is not None:
            self.config.set_default_model(prov, model)
        self._config_default_model = (prov, model)
        return f"\u2726 default model set \u2192 {prov}/{model}"

    def _handle_memory(self, line: str) -> str:
        if self.memory is None:
            return "memory disabled"
        if line == "/memory" or line == "/memory help":
            return format_memory_help()
        if line == "/memory list":
            return self.memory.list_text()
        if line == "/memory reload":
            return self.reload_memory()
        if line.startswith("/memory add "):
            _, _, rest = line.partition("/memory add ")
            target, _, content = rest.partition(" ")
            return self.memory.add(target, content)
        if line.startswith("/memory replace "):
            _, _, rest = line.partition("/memory replace ")
            target, _, rest = rest.partition(" ")
            old, sep, new = rest.partition("=>")
            if not sep:
                raise ValueError("usage: /memory replace <memory|user> <old substring> => <new entry>")
            return self.memory.replace(target, old.strip(), new.strip())
        if line.startswith("/memory remove "):
            _, _, rest = line.partition("/memory remove ")
            target, _, old = rest.partition(" ")
            return self.memory.remove(target, old.strip())
        return format_memory_help()

    def _handle_search(self, query: str) -> str:
        if not query:
            return "usage: /search <fts-query>"
        rows = self.sessions.search(query)
        if not rows:
            return "(no matches)"
        out = []
        for row in rows:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["started_at"]))
            out.append(
                f"{row['session_id']} {when} {row['role']} "
                f"{row['provider']}/{row['model']}: {row['snippet']}"
            )
        return "\n".join(out)

    def _list_local_skills(self) -> str:
        """Discover local skills and return a text listing."""
        roots_env = os.getenv("JUNO_SKILLS_PATH", "")
        extra_roots: list[Path] = []
        if roots_env:
            extra_roots = [Path(p.strip()) for p in roots_env.split(os.pathsep) if p.strip()]
        skills = discover_local_skills(self.args.home, extra_roots)
        if not skills:
            return (
                "No local skills found.\n"
                "Place SKILL.md files under <home>/skills/<category>/SKILL.md\n"
                "or set JUNO_SKILLS_PATH for additional scan roots."
            )
        out: list[str] = []
        # Rainbow "boost" header (req 6): animate once on first open, static
        # thereafter. Gated behind use_rainbow() so plain/piped mode stays
        # SGR-free (the subprocess plain tests assert "\x1b[" absent).
        if use_rainbow():
            opened = getattr(self, "_skills_opened", False)
            header_text = "✦ skills ✧"
            if not opened:
                RainbowSweep(header_text, cycles=1, fps=20).stop()
                self._skills_opened = True
            else:
                out.append(rainbow(header_text, color=True))
        last_category = ""
        for s in skills:
            if s.category != last_category:
                out.append(f"\n  {s.category}")
                last_category = s.category
            desc = f" — {s.description}" if s.description else ""
            out.append(f"    {s.name}{desc}")
        return "\n".join(out)

    # -- provider turn ------------------------------------------------------

    def submit_user_message(self, line: str) -> ChatResult:
        if getattr(self.args, "mode", "chat") == "agent":
            if provider_supports_juno_tools(self.args.provider):
                return run_agent_turn(self, line)
            if self.args.provider not in NATIVE_AGENT_PROVIDERS:
                text = (
                    f"Juno is running in --mode agent, but provider={self.args.provider} is not wired "
                    "to Juno's tool loop yet. I will not pretend to read files or call tools through "
                    "this provider. Use `--provider echo` for the verified offline tool slice, or "
                    "use OpenAI/OpenRouter once credentials are configured."
                )
                self.sessions.append(self.session_id, "user", line)
                self.sessions.append(self.session_id, "assistant", text)
                return ChatResult(text=text)
        self.messages.append({"role": "user", "content": line})
        try:
            result = self.client.complete(self.messages)
        except Exception:
            self.messages.pop()  # roll back the user turn so input is recoverable
            raise
        self.sessions.append(self.session_id, "user", line)
        self.messages.append({"role": "assistant", "content": result.text})
        self.sessions.append(self.session_id, "assistant", result.text)
        self.total_in += result.input_tokens
        self.total_out += result.output_tokens
        return result

    def close(self) -> None:
        self.sessions.end_session(self.session_id)

