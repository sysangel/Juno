#!/usr/bin/env python3
"""Small CLI agent harness with Hermes-inspired memory.

This is intentionally separate from agent_loop.py: it is a general-purpose chat
harness that can talk to OpenRouter, OpenAI/ChatGPT, or Anthropic first-party
APIs from the CLI while reusing the design ideas that make Hermes pleasant:

- frozen, bounded, curated memory injected at session start
- durable session transcript storage in SQLite with FTS search
- provider-neutral message history
- explicit provider/model line before paid API calls

It is not a full Hermes clone yet. The next layers should add tool calling,
sandboxed shell/file tools, and agent-loop/orchestrator integration.
"""
from __future__ import annotations

import argparse
import random
import shutil
import textwrap
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from dotenv import load_dotenv
from openai import OpenAI

# Reuse the project's OpenRouter privacy screen constants instead of duplicating
# policy-critical allow/deny lists.
import agent_loop

ENTRY_DELIMITER = "\n§\n"
DEFAULT_HOME = Path(os.getenv("AGENT_HARNESS_HOME", Path.home() / ".agent-harness"))
DEFAULT_OPENROUTER_MODEL = agent_loop.DEFAULT_MODEL_OPENROUTER
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
DEFAULT_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
DEFAULT_CLAUDE_CODE_MODEL = os.getenv("CLAUDE_CODE_MODEL", "sonnet")
# Hermes-style/Fusion shortcuts are env-overridable because account-local
# OpenRouter aliases or panel slugs can differ between machines.
DEFAULT_OPENROUTER_DP_MODEL = os.getenv("HARNESS_DP_MODEL", "deepseek/deepseek-v4-pro")
DEFAULT_OPENROUTER_FM3V4_MODEL = os.getenv("HARNESS_FM3V4_MODEL", "deepseek/deepseek-v4-pro")
DEFAULT_OPENROUTER_FBUDGET_MODEL = os.getenv("HARNESS_FBUDGET_MODEL", "openrouter/auto")
DEFAULT_OPENROUTER_FHIGH_MODEL = os.getenv("HARNESS_FHIGH_MODEL", "openrouter/auto")

Provider = Literal["openrouter", "openai", "anthropic", "claude-code", "echo"]
Role = Literal["system", "user", "assistant"]
PROVIDERS = {"openrouter", "openai", "anthropic", "claude-code", "echo"}


@dataclass(frozen=True)
class ModelOption:
    key: str
    section: str
    label: str
    provider: Provider
    model: str
    description: str = ""


@dataclass
class ChatResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict[str, Any] | None = None


class MemoryStore:
    """Bounded file-backed memory modeled after Hermes MEMORY.md / USER.md."""

    def __init__(self, home: Path, memory_limit: int = 2200, user_limit: int = 1375) -> None:
        self.home = home
        self.memory_dir = home / "memories"
        self.memory_limit = memory_limit
        self.user_limit = user_limit
        self.memory_entries: list[str] = []
        self.user_entries: list[str] = []
        self._snapshot: dict[str, str] = {"memory": "", "user": ""}

    def load(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_entries = self._read_entries("memory")
        self.user_entries = self._read_entries("user")
        self._snapshot = {
            "memory": self._render_block("memory", self.memory_entries),
            "user": self._render_block("user", self.user_entries),
        }

    def system_prompt_block(self) -> str:
        parts = [p for p in [self._snapshot["user"], self._snapshot["memory"]] if p]
        return "\n\n".join(parts)

    def add(self, target: str, content: str) -> str:
        entries = self._entries(target)
        content = content.strip()
        if not content:
            raise ValueError("memory content cannot be empty")
        if content in entries:
            return "entry already exists"
        limit = self._limit(target)
        new_entries = entries + [content]
        new_total = len(ENTRY_DELIMITER.join(new_entries))
        if new_total > limit:
            raise ValueError(f"{target} memory would exceed {limit} chars ({new_total}/{limit}); shorten or remove entries")
        entries.append(content)
        self._write_entries(target, entries)
        return "entry added"

    def replace(self, target: str, old_text: str, content: str) -> str:
        entries = self._entries(target)
        matches = [i for i, entry in enumerate(entries) if old_text in entry]
        if not matches:
            raise ValueError(f"no {target} memory entry matched {old_text!r}")
        if len(matches) > 1:
            raise ValueError(f"multiple {target} memory entries matched {old_text!r}; be more specific")
        candidate = entries.copy()
        candidate[matches[0]] = content.strip()
        limit = self._limit(target)
        new_total = len(ENTRY_DELIMITER.join(candidate))
        if new_total > limit:
            raise ValueError(f"replacement would exceed {limit} chars ({new_total}/{limit})")
        entries[matches[0]] = content.strip()
        self._write_entries(target, entries)
        return "entry replaced"

    def remove(self, target: str, old_text: str) -> str:
        entries = self._entries(target)
        matches = [i for i, entry in enumerate(entries) if old_text in entry]
        if not matches:
            raise ValueError(f"no {target} memory entry matched {old_text!r}")
        if len(matches) > 1:
            raise ValueError(f"multiple {target} memory entries matched {old_text!r}; be more specific")
        entries.pop(matches[0])
        self._write_entries(target, entries)
        return "entry removed"

    def list_text(self, target: str | None = None) -> str:
        targets = [target] if target else ["user", "memory"]
        blocks: list[str] = []
        for t in targets:
            entries = self._entries(t)
            used = len(ENTRY_DELIMITER.join(entries)) if entries else 0
            blocks.append(f"{t.upper()} ({used}/{self._limit(t)} chars)")
            blocks.extend(f"  {i}. {entry}" for i, entry in enumerate(entries, 1))
            if not entries:
                blocks.append("  (empty)")
        return "\n".join(blocks)

    def _entries(self, target: str) -> list[str]:
        if target == "user":
            return self.user_entries
        if target == "memory":
            return self.memory_entries
        raise ValueError("target must be 'memory' or 'user'")

    def _limit(self, target: str) -> int:
        return self.user_limit if target == "user" else self.memory_limit

    def _path(self, target: str) -> Path:
        return self.memory_dir / ("USER.md" if target == "user" else "MEMORY.md")

    def _read_entries(self, target: str) -> list[str]:
        path = self._path(target)
        if not path.exists():
            return []
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return []
        return list(dict.fromkeys(entry.strip() for entry in raw.split(ENTRY_DELIMITER) if entry.strip()))

    def _write_entries(self, target: str, entries: list[str]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(target)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")
        os.replace(tmp, path)

    def _render_block(self, target: str, entries: list[str]) -> str:
        if not entries:
            return ""
        content = ENTRY_DELIMITER.join(entries)
        limit = self._limit(target)
        pct = int(min(100, len(content) / limit * 100)) if limit else 0
        label = "USER PROFILE (who the user is)" if target == "user" else "MEMORY (your personal notes)"
        sep = "═" * 46
        return f"{sep}\n{label} [{pct}% — {len(content):,}/{limit:,} chars]\n{sep}\n{content}"


class SessionStore:
    """SQLite transcript store with basic FTS5 search."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY,
              title TEXT,
              provider TEXT NOT NULL,
              model TEXT NOT NULL,
              started_at REAL NOT NULL,
              ended_at REAL
            );
            CREATE TABLE IF NOT EXISTS messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
              content, session_id UNINDEXED, role UNINDEXED, content='messages', content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
              INSERT INTO messages_fts(rowid, content, session_id, role) VALUES (new.id, new.content, new.session_id, new.role);
            END;
            CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
              INSERT INTO messages_fts(messages_fts, rowid, content, session_id, role) VALUES('delete', old.id, old.content, old.session_id, old.role);
            END;
            CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
              INSERT INTO messages_fts(messages_fts, rowid, content, session_id, role) VALUES('delete', old.id, old.content, old.session_id, old.role);
              INSERT INTO messages_fts(rowid, content, session_id, role) VALUES (new.id, new.content, new.session_id, new.role);
            END;
            """
        )
        self.conn.commit()

    def start_session(self, provider: str, model: str, title: str = "") -> str:
        sid = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        self.conn.execute(
            "INSERT INTO sessions(id, title, provider, model, started_at) VALUES (?, ?, ?, ?, ?)",
            (sid, title, provider, model, time.time()),
        )
        self.conn.commit()
        return sid

    def append(self, session_id: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )
        self.conn.commit()

    def end_session(self, session_id: str) -> None:
        self.conn.execute("UPDATE sessions SET ended_at=? WHERE id=?", (time.time(), session_id))
        self.conn.commit()

    def update_session_model(self, session_id: str, provider: str, model: str) -> None:
        self.conn.execute("UPDATE sessions SET provider=?, model=? WHERE id=?", (provider, model, session_id))
        self.conn.commit()

    def search(self, query: str, limit: int = 5) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT m.id, m.session_id, m.role, snippet(messages_fts, 0, '[', ']', ' … ', 12) AS snippet,
                       s.started_at, s.provider, s.model
                FROM messages_fts
                JOIN messages m ON m.id = messages_fts.rowid
                JOIN sessions s ON s.id = m.session_id
                WHERE messages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            )
        )


class ProviderClient:
    def __init__(self, provider: Provider, model: str, temperature: float, base_url: str | None = None) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.base_url = base_url

    def complete(self, messages: list[dict[str, str]]) -> ChatResult:
        if self.provider == "echo":
            last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return ChatResult(text=f"echo: {last}", input_tokens=len(json.dumps(messages)), output_tokens=len(last))
        if self.provider == "openrouter":
            return self._openai_compatible(
                messages,
                api_key_env="OPENROUTER_API_KEY",
                base_url=self.base_url or agent_loop.OPENROUTER_BASE_URL,
                default_headers=agent_loop.OPENROUTER_HEADERS,
                extra_body={
                    "provider": agent_loop.OPENROUTER_PROVIDER_PREFS,
                    "reasoning": {"max_tokens": agent_loop.OPENROUTER_REASONING_MAX_TOKENS},
                },
            )
        if self.provider == "openai":
            return self._openai_compatible(messages, api_key_env="OPENAI_API_KEY", base_url=self.base_url)
        if self.provider == "anthropic":
            return self._anthropic(messages)
        if self.provider == "claude-code":
            return self._claude_code(messages)
        raise AssertionError(f"unknown provider {self.provider}")

    def _openai_compatible(
        self,
        messages: list[dict[str, str]],
        *,
        api_key_env: str,
        base_url: str | None,
        default_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set")
        client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
        resp = client.chat.completions.create(
            model=self.model,
            messages=list(self._strip_empty_system(messages)),
            temperature=self.temperature,
            max_tokens=8192,
            extra_body=extra_body,
        )
        msg = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        return ChatResult(
            text=msg.content or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            raw={"id": getattr(resp, "id", None), "model": getattr(resp, "model", None)},
        )

    def _anthropic(self, messages: list[dict[str, str]]) -> ChatResult:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        system_parts = [m["content"] for m in messages if m["role"] == "system" and m["content"].strip()]
        anthropic_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in {"user", "assistant"}
        ]
        body = {
            "model": self.model,
            "max_tokens": 8192,
            "temperature": self.temperature,
            "messages": anthropic_messages,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        req = urllib.request.Request(
            self.base_url or "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"Anthropic HTTP {exc.code}: {detail}") from exc
        text_parts = [part.get("text", "") for part in data.get("content", []) if part.get("type") == "text"]
        usage = data.get("usage", {}) or {}
        return ChatResult(
            text="".join(text_parts),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            raw={"id": data.get("id"), "model": data.get("model")},
        )

    def _claude_code(self, messages: list[dict[str, str]]) -> ChatResult:
        """Call Claude Code through its existing OAuth login, no API key needed."""
        system_parts = [m["content"] for m in messages if m["role"] == "system" and m["content"].strip()]
        prompt_lines: list[str] = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            label = "User" if msg["role"] == "user" else "Assistant"
            prompt_lines.append(f"{label}: {msg['content']}")
        prompt_lines.append("Assistant:")
        prompt = "\n\n".join(prompt_lines)

        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--max-turns",
            "1",
            "--tools",
            "",
        ]
        if self.model:
            cmd.extend(["--model", self.model])
        if system_parts:
            cmd.extend(["--append-system-prompt", "\n\n".join(system_parts)])

        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=600)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()
            raise RuntimeError(f"Claude Code exited {proc.returncode}: {detail}")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return ChatResult(text=proc.stdout.strip(), raw={"stderr": proc.stderr})

        usage = data.get("usage", {}) or {}
        return ChatResult(
            text=data.get("result", "") or "",
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            raw=data,
        )

    @staticmethod
    def _strip_empty_system(messages: Iterable[dict[str, str]]) -> Iterable[dict[str, str]]:
        for m in messages:
            if m["role"] == "system" and not m["content"].strip():
                continue
            yield m


def default_model(provider: Provider) -> str:
    if provider == "openrouter":
        return DEFAULT_OPENROUTER_MODEL
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    if provider == "anthropic":
        return DEFAULT_ANTHROPIC_MODEL
    if provider == "claude-code":
        return DEFAULT_CLAUDE_CODE_MODEL
    return "echo"


def normalize_provider(value: str) -> Provider:
    aliases = {
        "claude": "claude-code",
        "cc": "claude-code",
        "chatgpt": "openai",
        "gpt": "openai",
        "codex": "openai",
        "or": "openrouter",
    }
    provider = aliases.get(value.strip().lower(), value.strip().lower())
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {value!r}; choose one of: {', '.join(sorted(PROVIDERS))}")
    return provider  # type: ignore[return-value]


def model_catalog() -> list[ModelOption]:
    """Return grouped model choices for `/model` and the Juno picker.

    Keep this list small and high-signal. Users can still type any raw model slug
    with `/model <provider> <slug>`.
    """
    return [
        ModelOption(
            key="codex",
            section="Codex / OpenAI",
            label=f"Codex / OpenAI default ({DEFAULT_OPENAI_MODEL})",
            provider="openai",
            model=DEFAULT_OPENAI_MODEL,
            description="OpenAI-compatible API path using OPENAI_API_KEY.",
        ),
        ModelOption(
            key="gpt5",
            section="Codex / OpenAI",
            label="GPT-5.1",
            provider="openai",
            model="gpt-5.1",
            description="General OpenAI model; useful fallback if OPENAI_MODEL is unset.",
        ),
        ModelOption(
            key="claude",
            section="Anthropic / Claude",
            label=f"Claude Code default ({DEFAULT_CLAUDE_CODE_MODEL})",
            provider="claude-code",
            model=DEFAULT_CLAUDE_CODE_MODEL,
            description="Local Claude Code OAuth bridge; no ANTHROPIC_API_KEY required.",
        ),
        ModelOption(
            key="sonnet",
            section="Anthropic / Claude",
            label="Claude Code Sonnet",
            provider="claude-code",
            model="sonnet",
            description="Fast Claude Code OAuth model alias.",
        ),
        ModelOption(
            key="opus",
            section="Anthropic / Claude",
            label="Claude Code Opus",
            provider="claude-code",
            model="opus",
            description="Heavier Claude Code OAuth model alias if your account allows it.",
        ),
        ModelOption(
            key="anthropic",
            section="Anthropic / Claude",
            label=f"Anthropic API default ({DEFAULT_ANTHROPIC_MODEL})",
            provider="anthropic",
            model=DEFAULT_ANTHROPIC_MODEL,
            description="Direct Anthropic Messages API; needs ANTHROPIC_API_KEY.",
        ),
        ModelOption(
            key="or",
            section="OpenRouter",
            label=f"OpenRouter default ({DEFAULT_OPENROUTER_MODEL})",
            provider="openrouter",
            model=DEFAULT_OPENROUTER_MODEL,
            description="Uses the repo privacy-screened OpenRouter provider preferences.",
        ),
        ModelOption(
            key="dp",
            section="OpenRouter",
            label=f"DeepSeek Pro ({DEFAULT_OPENROUTER_DP_MODEL})",
            provider="openrouter",
            model=DEFAULT_OPENROUTER_DP_MODEL,
            description="Hermes-style dp shortcut.",
        ),
        ModelOption(
            key="fm3v4",
            section="OpenRouter",
            label=f"Fusion M3/V4 ({DEFAULT_OPENROUTER_FM3V4_MODEL})",
            provider="openrouter",
            model=DEFAULT_OPENROUTER_FM3V4_MODEL,
            description="Env override: HARNESS_FM3V4_MODEL.",
        ),
        ModelOption(
            key="fbudget",
            section="OpenRouter",
            label=f"Fusion budget ({DEFAULT_OPENROUTER_FBUDGET_MODEL})",
            provider="openrouter",
            model=DEFAULT_OPENROUTER_FBUDGET_MODEL,
            description="Env override: HARNESS_FBUDGET_MODEL.",
        ),
        ModelOption(
            key="fhigh",
            section="OpenRouter",
            label=f"Fusion high ({DEFAULT_OPENROUTER_FHIGH_MODEL})",
            provider="openrouter",
            model=DEFAULT_OPENROUTER_FHIGH_MODEL,
            description="Env override: HARNESS_FHIGH_MODEL.",
        ),
        ModelOption(
            key="echo",
            section="Offline",
            label="Echo smoke-test provider",
            provider="echo",
            model="echo",
            description="No API key; returns `echo: <input>`.",
        ),
    ]


def model_aliases() -> dict[str, ModelOption]:
    aliases: dict[str, ModelOption] = {}
    for option in model_catalog():
        aliases[option.key.lower()] = option
    # Friendly synonyms.
    aliases.update({
        "openai": aliases["codex"],
        "chatgpt": aliases["codex"],
        "gpt": aliases["codex"],
        "gpt-5.1": aliases["gpt5"],
        "claude-code": aliases["claude"],
        "cc": aliases["claude"],
        "openrouter": aliases["or"],
        "deepseek": aliases["dp"],
    })
    return aliases


def resolve_model_selection(first: str, second: str | None = None) -> tuple[Provider, str, str]:
    """Resolve `/model ...` input.

    Returns (provider, model, note). Supports:
    - `/model <catalog-key>` e.g. `/model sonnet`, `/model dp`, `/model fhigh`
    - `/model <provider> [model]` e.g. `/model openrouter deepseek/...`
    - provider aliases (`claude`, `codex`, `or`) and raw provider/model pairs.
    """
    token = first.strip().lower()
    if second is None:
        option = model_aliases().get(token)
        if option is not None:
            return option.provider, option.model, f"selected {option.section}: {option.label}"
    provider = normalize_provider(first)
    model = second or default_model(provider)
    return provider, model, ""


def format_model_catalog(current_provider: str | None = None, current_model: str | None = None) -> str:
    lines = [
        "model catalog",
        "  Select in Juno with /model, or type one of:",
        "  /model <key>                  e.g. /model sonnet, /model dp, /model fhigh",
        "  /model <provider> [model]     e.g. /model openrouter deepseek/deepseek-v4-pro",
    ]
    current = (current_provider, current_model)
    last_section = None
    for option in model_catalog():
        if option.section != last_section:
            lines.append(f"\n{option.section}")
            last_section = option.section
        marker = "*" if (option.provider, option.model) == current else " "
        desc = f" — {option.description}" if option.description else ""
        lines.append(f" {marker} {option.key:<10} {option.provider:<12} {option.model:<34} {option.label}{desc}")
    lines.append("\n* = current model")
    return "\n".join(lines)


def build_initial_messages(system_prompt: str, memory: MemoryStore | None) -> list[dict[str, str]]:
    base = system_prompt.strip() or "You are a careful, useful CLI agent."
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
    memory: MemoryStore | None
    sessions: SessionStore
    session_id: str
    messages: list[dict[str, str]]
    client: ProviderClient
    total_in: int = 0
    total_out: int = 0

    @classmethod
    def create(
        cls,
        args: argparse.Namespace,
        memory: MemoryStore | None,
        sessions: SessionStore,
    ) -> "HarnessRuntime":
        session_id = sessions.start_session(args.provider, args.model)
        messages = build_initial_messages(args.system, memory)
        client = ProviderClient(args.provider, args.model, args.temperature, args.base_url)
        return cls(
            args=args,
            memory=memory,
            sessions=sessions,
            session_id=session_id,
            messages=messages,
            client=client,
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
        self.messages = build_initial_messages(self.args.system, self.memory)
        return "conversation reset; current memory snapshot kept"

    def reload_memory(self) -> str:
        if self.memory is None:
            return "memory disabled"
        self.memory.load()
        conversation = [m for m in self.messages if m["role"] != "system"]
        self.messages = build_initial_messages(self.args.system, self.memory) + conversation
        return "memory reloaded and re-injected into the active system prompt"

    def switch_provider(self, provider_text: str, model_text: str | None = None) -> str:
        provider, model, note = resolve_model_selection(provider_text, model_text)
        self.args.provider = provider
        self.args.model = model
        self.client = ProviderClient(provider, model, self.args.temperature, self.args.base_url)
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
                msg = self.switch_provider(parts[1], parts[2] if len(parts) > 2 else None)
                return CommandResult(True, msg)
            if line.startswith("/memory"):
                return CommandResult(True, self._handle_memory(line))
            if line.startswith("/search "):
                return CommandResult(True, self._handle_search(line[len("/search "):].strip()))
            return CommandResult(True, "unknown command; try /help")
        except Exception as exc:  # noqa: BLE001 - command errors must not crash the UI
            return CommandResult(True, f"[command error] {exc}")

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

    # -- provider turn ------------------------------------------------------

    def submit_user_message(self, line: str) -> ChatResult:
        self.messages.append({"role": "user", "content": line})
        self.sessions.append(self.session_id, "user", line)
        try:
            result = self.client.complete(self.messages)
        except Exception:
            self.messages.pop()  # roll back the user turn so input is recoverable
            raise
        self.messages.append({"role": "assistant", "content": result.text})
        self.sessions.append(self.session_id, "assistant", result.text)
        self.total_in += result.input_tokens
        self.total_out += result.output_tokens
        return result

    def close(self) -> None:
        self.sessions.end_session(self.session_id)


def run_once(args: argparse.Namespace, client: ProviderClient, memory: MemoryStore | None, sessions: SessionStore) -> int:
    session_id = sessions.start_session(args.provider, args.model, title=args.query[:80])
    messages = build_initial_messages(args.system, memory)
    messages.append({"role": "user", "content": args.query})
    sessions.append(session_id, "user", args.query)
    result = client.complete(messages)
    sessions.append(session_id, "assistant", result.text)
    sessions.end_session(session_id)
    color = sys.stdout.isatty()
    print(format_assistant_response(result.text, provider=args.provider, model=args.model, color=color))
    print(format_usage_line(result.input_tokens, result.output_tokens, session_id=session_id, color=color))
    return 0


# ---------------------------------------------------------------------------
# UI layer
# ---------------------------------------------------------------------------

# "Juno" brand palette: deep blue -> indigo -> violet.
JUNO_NAME = "Juno"
JUNO_BANNER = "Juno \u00b7 agent-harness"
RESET = "\033[0m"
PURPLE = "\033[38;5;99m"
PURPLE_DIM = "\033[38;5;98m"
PURPLE_BOLD = "\033[1;38;5;99m"
PURPLE_ITALIC = "\033[3;38;5;99m"
LAVENDER = "\033[38;5;183m"
RED_BOLD = "\033[31;1m"

THINKING_MESSAGES = [
    "braiding moonlit context",
    "asking the purple oracle",
    "tuning the tiny inference harp",
    "polishing the response crystal",
    "consulting the indigo star chart",
    "brewing a little nebula of tokens",
    "listening for the useful answer",
]


def terminal_width(default: int = 88) -> int:
    return max(60, min(shutil.get_terminal_size((default, 24)).columns, 120))


def colorize(text: str, ansi: str, *, color: bool) -> str:
    return f"{ansi}{text}{RESET}" if color else text


def wrap_text_block(text: str, width: int) -> list[str]:
    """Wrap prose while preserving blank lines, code fences, and structured lines."""
    lines: list[str] = []
    in_code = False
    raw_lines = text.splitlines() or [""]
    for raw in raw_lines:
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            lines.append(raw[:width] if raw else "")
            continue
        if not stripped:
            lines.append("")
            continue
        structured = in_code or stripped.startswith(("- ", "* ", "+ ", "> ", "|")) or raw.startswith(("    ", "\t"))
        if structured:
            if len(raw) <= width:
                lines.append(raw)
            else:
                lines.extend(textwrap.wrap(raw, width=width, break_long_words=False, break_on_hyphens=False) or [""])
            continue
        lines.extend(textwrap.wrap(
            raw,
            width=width,
            replace_whitespace=False,
            drop_whitespace=True,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""])
    return lines


def format_box(text: str, *, title: str, subtitle: str = "", color: bool = True, width: int | None = None) -> str:
    cols = width or terminal_width()
    inner = max(44, cols - 4)
    border = "─" * inner
    header = f" {title}"
    if subtitle:
        header += f" · {subtitle}"
    header = header[:inner]
    body = wrap_text_block(text.rstrip(), inner)
    rendered = "\n".join([
        f"╭{border}╮",
        f"│{header.ljust(inner)}│",
        f"├{border}┤",
        *(f"│{line[:inner].ljust(inner)}│" for line in body),
        f"╰{border}╯",
    ])
    return colorize(rendered, PURPLE, color=color)


def format_assistant_response(text: str, *, provider: str, model: str, color: bool = True, width: int | None = None) -> str:
    return format_box(text, title="assistant", subtitle=f"{provider}/{model}", color=color, width=width)


def format_command_output(text: str, *, color: bool = True) -> str:
    if "\n" in text:
        return format_box(text, title="juno", subtitle="command output", color=color)
    return colorize(f"juno · {text}", PURPLE_DIM, color=color)


def format_thinking(provider: str, model: str, *, color: bool = True) -> str:
    message = random.choice(THINKING_MESSAGES)
    return colorize(f"✦ {message} with {provider}/{model}…", PURPLE_ITALIC, color=color)


def format_usage_line(input_tokens: int, output_tokens: int, total_in: int | None = None, total_out: int | None = None, *, session_id: str | None = None, color: bool = True) -> str:
    parts = [f"turn {input_tokens}/{output_tokens} tok"]
    if total_in is not None and total_out is not None:
        parts.append(f"total {total_in}/{total_out}")
    if session_id:
        parts.append(f"session {session_id}")
    return colorize("   " + " · ".join(parts), PURPLE_DIM, color=color)


def format_error(text: str, *, color: bool = True) -> str:
    return colorize(text, RED_BOLD, color=color)


def print_banner(runtime: HarnessRuntime, *, color: bool) -> None:
    print(colorize(JUNO_BANNER, PURPLE_BOLD, color=color))
    print(f"{'home':<8} {runtime.args.home}")
    print(f"{'session':<8} {runtime.session_id}")
    print(
        f"{'model':<8} {runtime.args.provider}/{runtime.args.model} "
        f"· memory={'on' if runtime.memory is not None else 'off'}"
    )
    print("Type /help for commands. Ctrl-D exits. Esc+Enter inserts a newline.")
    print()


class PlainReplUI:
    """Fallback REPL using plain input()/print(). Used for pipes, CI, and when
    prompt_toolkit is unavailable. Drives the same HarnessRuntime as Juno."""

    def __init__(self, runtime: HarnessRuntime) -> None:
        self.runtime = runtime

    def run(self) -> int:
        rt = self.runtime
        print_banner(rt, color=False)
        try:
            while True:
                try:
                    line = input(f"\n{rt.prompt_label()}").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not line:
                    continue
                if line.startswith("/"):
                    res = rt.handle_command(line)
                    if res.output:
                        print()
                        print(format_command_output(res.output, color=False))
                        print()
                    if not res.should_continue:
                        break
                    continue
                print()
                print(format_thinking(rt.args.provider, rt.args.model, color=False))
                try:
                    result = rt.submit_user_message(line)
                except Exception as exc:  # noqa: BLE001 - keep CLI recoverable
                    print(format_error(f"[provider error] {exc}", color=False))
                    continue
                print()
                print(format_assistant_response(
                    result.text,
                    provider=rt.args.provider,
                    model=rt.args.model,
                    color=False,
                ))
                print(format_usage_line(
                    result.input_tokens,
                    result.output_tokens,
                    rt.total_in,
                    rt.total_out,
                    color=False,
                ))
                print()
        finally:
            rt.close()
        return 0


class PromptToolkitUI:
    """Juno: lightweight prompt_toolkit REPL with deep blue/purple styling,
    persistent history, slash/provider/memory completion, and a live status
    toolbar. Imports prompt_toolkit lazily so the module loads without it."""

    def __init__(self, runtime: HarnessRuntime) -> None:
        # Import lazily so the module loads without prompt_toolkit. Construction
        # is cheap and import-only here; the actual PromptSession (which probes
        # the console and can fail under non-Windows-console terminals) is built
        # in run() so run_repl can fall back gracefully.
        import prompt_toolkit  # noqa: F401  (ensures ImportError surfaces early)

        self.runtime = runtime
        self._pt_html = self._import_html()
        self.session = None

    def _build_session(self):
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style

        history_path = Path(self.runtime.args.home)
        history_path.mkdir(parents=True, exist_ok=True)

        # Deep blue -> purple Juno theme; the toolbar reverses light lavender
        # onto a deep indigo background.
        style = Style.from_dict({
            "prompt": "#b38cff bold",
            "bottom-toolbar": "#e7dcff bg:#24133f",
            "bottom-toolbar.key": "#d6b4ff bold",
            "banner": "#b38cff bold",
            "banner.sub": "#8a7cff",
            "meta": "#a78bfa",
            "thinking": "italic #c084fc",
            "error": "ansired bold",
            "assistant": "#e9d5ff",
            "assistant.border": "#a855f7",
        })

        return PromptSession(
            history=FileHistory(str(history_path / "history.txt")),
            completer=self._build_completer(),
            complete_while_typing=True,
            bottom_toolbar=self._bottom_toolbar,
            style=style,
            multiline=True,
            prompt_continuation=lambda width, line_number, wrap_count: "." * (width - 1) + " ",
            key_bindings=self._build_keybindings(),
        )

    @staticmethod
    def _import_html():
        from prompt_toolkit.formatted_text import HTML
        return HTML

    def _build_completer(self):
        from prompt_toolkit.completion import NestedCompleter

        # NestedCompleter maps a token tree; None = stop completing (free text).
        return NestedCompleter.from_nested_dict({
            "/help": None,
            "/status": None,
            "/model": {p: None for p in PROVIDER_WORDS},
            "/provider": {p: None for p in PROVIDER_WORDS},
            "/memory": {
                "list": None,
                "reload": None,
                "add": {t: None for t in MEMORY_TARGETS},
                "replace": {t: None for t in MEMORY_TARGETS},
                "remove": {t: None for t in MEMORY_TARGETS},
            },
            "/search": None,
            "/reset": None,
            "/clear": None,
            "/exit": None,
        })

    def _build_keybindings(self):
        # Enter submits; Esc-Enter (Meta-Enter / Alt-Enter) inserts a newline.
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            event.current_buffer.validate_and_handle()

        @kb.add("escape", "enter")
        def _(event):
            event.current_buffer.insert_text("\n")

        return kb

    def _bottom_toolbar(self):
        return self._pt_html(
            f"<b>{self.runtime.args.provider}/{self.runtime.args.model}</b>  "
            f"memory={'on' if self.runtime.memory is not None else 'off'}  "
            f"session={self.runtime.session_id}  "
            f"tokens in/out={self.runtime.total_in}/{self.runtime.total_out}  "
            f"<b>/help</b>"
        )

    def _prompt_fragments(self):
        # Styled prompt label using the 'prompt' style class.
        return [("class:prompt", self.runtime.prompt_label())]

    def _print_banner(self) -> None:
        print_banner(self.runtime, color=True)

    def _run_model_picker(self) -> str:
        from prompt_toolkit.shortcuts import radiolist_dialog

        current = (self.runtime.args.provider, self.runtime.args.model)
        values = []
        for option in model_catalog():
            marker = "*" if (option.provider, option.model) == current else " "
            values.append((
                option.key,
                f"{marker} [{option.section}] {option.key}: {option.provider}/{option.model} — {option.label}",
            ))
        chosen = radiolist_dialog(
            title="Juno model selection",
            text="Choose a model route. Raw slugs still work with /model <provider> <model>.",
            values=values,
        ).run()
        if not chosen:
            return "model selection cancelled"
        option = model_aliases()[chosen]
        return self.runtime.switch_provider(option.provider, option.model)

    def run(self) -> int:
        rt = self.runtime
        if self.session is None:
            self.session = self._build_session()
        self._print_banner()
        try:
            while True:
                try:
                    line = self.session.prompt(self._prompt_fragments())
                except KeyboardInterrupt:
                    continue  # Ctrl-C clears the current line, keeps the session
                except EOFError:
                    print()
                    break
                line = line.strip()
                if not line:
                    continue
                if line.startswith("/"):
                    if line in {"/model", "/provider"}:
                        try:
                            print()
                            print(format_command_output(self._run_model_picker(), color=True))
                            print()
                        except Exception as exc:  # noqa: BLE001 - keep UI recoverable
                            print(format_error(f"[model picker error] {exc}", color=True))
                            print(format_command_output(rt.handle_command(line).output, color=True))
                        continue
                    res = rt.handle_command(line)
                    if res.output:
                        print()
                        print(format_command_output(res.output, color=True))
                        print()
                    if not res.should_continue:
                        break
                    continue
                print()
                print(format_thinking(rt.args.provider, rt.args.model, color=True))
                try:
                    result = rt.submit_user_message(line)
                except Exception as exc:  # noqa: BLE001 - keep UI recoverable
                    print(format_error(f"[provider error] {exc}", color=True))
                    continue
                print()
                print(format_assistant_response(
                    result.text,
                    provider=rt.args.provider,
                    model=rt.args.model,
                    color=True,
                ))
                print(format_usage_line(
                    result.input_tokens,
                    result.output_tokens,
                    rt.total_in,
                    rt.total_out,
                    color=True,
                ))
                print()
        finally:
            rt.close()
        return 0


def _prompt_toolkit_available() -> tuple[bool, str]:
    """Return (ok, reason). Checks both import and that a real console output
    can be created — prompt_toolkit raises NoConsoleScreenBufferError under
    MSYS/Git-Bash xterm shells on Windows, which is not an ImportError."""
    try:
        import prompt_toolkit  # noqa: F401
    except ImportError as exc:
        return False, f"prompt_toolkit not installed ({exc})"
    try:
        from prompt_toolkit.output.defaults import create_output

        create_output()
    except Exception as exc:  # noqa: BLE001 - any console-probe failure means no Juno
        return False, f"no usable console for prompt_toolkit ({exc.__class__.__name__})"
    return True, ""


def run_repl(args: argparse.Namespace, client: ProviderClient, memory: MemoryStore | None, sessions: SessionStore) -> int:
    runtime = HarnessRuntime.create(args, memory, sessions)
    # Plain for explicit --ui plain, or whenever stdin is not a TTY (pipes, CI).
    if args.ui == "plain" or not sys.stdin.isatty():
        return PlainReplUI(runtime).run()
    ok, reason = _prompt_toolkit_available()
    if not ok:
        if args.ui == "prompt":
            raise RuntimeError(
                f"--ui prompt requested but Juno cannot start: {reason}. "
                "Install with `.venv/Scripts/python.exe -m pip install \"prompt_toolkit>=3,<4\"`, "
                "run in a real Windows console (cmd.exe/Windows Terminal/PowerShell), "
                "or use --ui plain."
            )
        print(f"[note] {reason}; falling back to plain REPL", file=sys.stderr)
        return PlainReplUI(runtime).run()
    return PromptToolkitUI(runtime).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Custom CLI agent harness with OpenRouter/OpenAI/Anthropic/Claude-Code providers.")
    parser.add_argument("-q", "--query", help="single-turn query; omit for interactive REPL")
    parser.add_argument("--provider", choices=["openrouter", "openai", "anthropic", "claude-code", "echo"], default=os.getenv("HARNESS_PROVIDER", "openrouter"))
    parser.add_argument("--model", default=None, help="model id/slug; defaults depend on provider")
    parser.add_argument("--base-url", default=None, help="override provider endpoint/base URL")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--system", default="You are a careful senior engineer working from a CLI harness.")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME, help="state directory (default: ~/.agent-harness or AGENT_HARNESS_HOME)")
    parser.add_argument("--no-memory", action="store_true", help="do not load/inject memory")
    parser.add_argument(
        "--ui",
        choices=["auto", "plain", "prompt"],
        default=os.getenv("HARNESS_UI", "auto"),
        help="interactive UI mode: auto (Juno prompt_toolkit on a TTY, else plain), "
             "plain (basic REPL), or prompt (force Juno; errors if prompt_toolkit missing)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    args.model = args.model or default_model(args.provider)
    print(f"RUN provider={args.provider} model={args.model}", file=sys.stderr)
    memory: MemoryStore | None = None
    if not args.no_memory:
        memory = MemoryStore(args.home)
        memory.load()
    sessions = SessionStore(args.home / "sessions.db")
    client = ProviderClient(args.provider, args.model, args.temperature, args.base_url)
    if args.query:
        return run_once(args, client, memory, sessions)
    return run_repl(args, client, memory, sessions)


if __name__ == "__main__":
    raise SystemExit(main())
