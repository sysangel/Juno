from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

from . import provider_policy as policy
from .models import Provider
from .tool_registry import ToolCall, parse_tool_arguments

@dataclass
class ChatResult:
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict[str, Any] | None = None
    tool_calls: list[ToolCall] | None = None


class ProviderClient:
    def __init__(
        self,
        provider: Provider,
        model: str,
        temperature: float,
        base_url: str | None = None,
        effort: str | None = None,
        mode: str = "chat",
        approval: str = "on-request",
        workspace: str | Path | None = None,
        max_tool_turns: int = 8,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.effort = effort
        self.mode = mode
        self.approval = approval
        self.workspace = Path(workspace) if workspace is not None else None
        self.max_tool_turns = max(1, int(max_tool_turns or 1))

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ChatResult:
        if self.provider == "echo":
            return self._echo(messages, tools)
        if self.provider == "openrouter":
            return self._openai_compatible(
                messages,
                api_key_env="OPENROUTER_API_KEY",
                base_url=self.base_url or policy.OPENROUTER_BASE_URL,
                default_headers=policy.OPENROUTER_HEADERS,
                extra_body={
                    "provider": policy.OPENROUTER_PROVIDER_PREFS,
                    "reasoning": {"max_tokens": policy.OPENROUTER_REASONING_MAX_TOKENS},
                },
                tools=tools,
            )
        if self.provider == "openai":
            return self._openai_compatible(messages, api_key_env="OPENAI_API_KEY", base_url=self.base_url, tools=tools)
        if self.provider == "anthropic":
            return self._anthropic(messages)
        if self.provider == "claude-code":
            return self._claude_code(messages)
        raise AssertionError(f"unknown provider {self.provider}")

    def _echo(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ChatResult:
        last_tool = next((m for m in reversed(messages) if m.get("role") == "tool"), None)
        if last_tool is not None:
            content = str(last_tool.get("content", ""))
            return ChatResult(text=f"echo tool result: {content}", input_tokens=len(json.dumps(messages, default=str)), output_tokens=len(content))
        last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if tools:
            import re
            match = re.search(r"([\w./\\-]+\.\w+)", str(last))
            if match and "read_file" in json.dumps(tools):
                path = match.group(1).replace("\\", "/")
                return ChatResult(
                    text="",
                    input_tokens=len(json.dumps(messages, default=str)),
                    tool_calls=[ToolCall(id="echo-call-1", name="read_file", arguments={"path": path})],
                    raw={"provider": "echo", "tool_call": "read_file"},
                )
        return ChatResult(text=f"echo: {last}", input_tokens=len(json.dumps(messages, default=str)), output_tokens=len(str(last)))

    def _openai_compatible(
        self,
        messages: list[dict[str, Any]],
        *,
        api_key_env: str,
        base_url: str | None,
        default_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set")
        client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(self._strip_empty_system(messages)),
            "temperature": self.temperature,
            "max_tokens": 8192,
            "extra_body": extra_body,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            tool_calls.append(ToolCall(
                id=str(getattr(tc, "id", "")),
                name=str(getattr(fn, "name", "")),
                arguments=parse_tool_arguments(getattr(fn, "arguments", {}) or {}),
            ))
        usage = getattr(resp, "usage", None)
        return ChatResult(
            text=msg.content or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            raw={"id": getattr(resp, "id", None), "model": getattr(resp, "model", None)},
            tool_calls=tool_calls or None,
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
            str(self.max_tool_turns if self.mode == "agent" else 1),
        ]
        cwd = str(self.workspace.resolve()) if self.workspace is not None else None
        if self.mode == "agent":
            workspace = self.workspace.resolve() if self.workspace is not None else Path.cwd().resolve()
            workspace.mkdir(parents=True, exist_ok=True)
            cwd = str(workspace)
            cmd.extend(["--add-dir", str(workspace)])
            if self.approval == "dangerous":
                cmd.extend(["--tools", "default", "--permission-mode", "bypassPermissions"])
                cmd.append("--dangerously-skip-permissions")
            else:
                # First wired Claude Code slice is read-only. Read/Grep/Glob/LS are
                # enough for repository inspection while preventing edits/shell.
                cmd.extend(["--tools", "Read,Grep,Glob,LS"])
                if self.approval == "never":
                    cmd.extend(["--permission-mode", "dontAsk"])
        else:
            cmd.extend(["--tools", ""])
        if self.model:
            cmd.extend(["--model", self.model])
        if self.effort:
            cmd.extend(["--effort", self.effort])
        if system_parts:
            prompt_flag = "--append-system-prompt" if self.mode == "agent" else "--system-prompt"
            cmd.extend([prompt_flag, "\n\n".join(system_parts)])

        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=600, encoding="utf-8", errors="replace", cwd=cwd)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Claude Code exited {proc.returncode}: {detail}")
        if proc.stdout is None or not proc.stdout.strip():
            raise RuntimeError(
                f"Claude Code produced no readable stdout (stderr: {(proc.stderr or '').strip()})"
            )
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

