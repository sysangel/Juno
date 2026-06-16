from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class ToolRisk(str, Enum):
    SAFE_READ = "safe_read"
    LOCAL_WRITE = "local_write"
    PROCESS = "process"
    NETWORK = "network"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True)
class ToolContext:
    workspace: Path
    home: Path
    session_id: str
    approval: str = "never"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    risk: ToolRisk
    handler: Callable[[dict[str, Any], ToolContext], "ToolResult"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    ok: bool
    content: str
    metadata: dict[str, Any] | None = None

    def to_message_content(self) -> str:
        prefix = "ok" if self.ok else "error"
        return f"{prefix}: {self.content}"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._tools)

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema,
                },
            }
            for spec in self._tools.values()
        ]

    def anthropic_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
            }
            for spec in self._tools.values()
        ]

    def provider_schemas(self, provider: str) -> list[dict[str, Any]]:
        if provider == "anthropic":
            return self.anthropic_schemas()
        return self.openai_schemas()


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, context: ToolContext) -> None:
        self.registry = registry
        self.context = context

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            spec = self.registry.get(call.name)
            denied = self._denied(spec)
            if denied:
                return ToolResult(call.id, call.name, False, denied)
            return spec.handler(call.arguments, self.context)
        except Exception as exc:  # noqa: BLE001 - tool errors must become model-visible results
            return ToolResult(call.id, call.name, False, f"{type(exc).__name__}: {exc}")

    def _denied(self, spec: ToolSpec) -> str | None:
        approval = self.context.approval
        if approval == "dangerous":
            return None
        if spec.risk == ToolRisk.SAFE_READ and approval in {"never", "on-request"}:
            return None
        if approval == "never":
            return f"tool {spec.name!r} requires approval ({spec.risk.value})"
        # Non-interactive first slice: represent approvals as denied unless a future UI callback grants them.
        if approval in {"on-request", "always"} and spec.risk != ToolRisk.SAFE_READ:
            return f"tool {spec.name!r} requires interactive approval ({spec.risk.value})"
        return None


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw in {None, ""}:
        return {}
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must decode to an object")
        return parsed
    raise TypeError(f"unsupported tool arguments type: {type(raw).__name__}")
