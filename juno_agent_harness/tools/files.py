from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..tool_registry import ToolContext, ToolRegistry, ToolResult, ToolRisk, ToolSpec


def resolve_workspace_path(workspace: Path, user_path: str = ".") -> Path:
    root = workspace.resolve()
    target = (root / (user_path or ".")).resolve()
    if target != root and root not in target.parents:
        raise PermissionError(f"path escapes workspace: {user_path}")
    return target


def _line_window(lines: list[str], offset: int, limit: int) -> str:
    offset = max(1, int(offset or 1))
    limit = max(1, min(int(limit or 200), 2000))
    start = offset - 1
    chunk = lines[start:start + limit]
    return "\n".join(f"{start + i + 1}|{line}" for i, line in enumerate(chunk))


def list_files(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    base = resolve_workspace_path(ctx.workspace, str(args.get("path") or "."))
    glob = str(args.get("glob") or "*")
    limit = max(1, min(int(args.get("limit") or 100), 1000))
    if not base.exists():
        raise FileNotFoundError(base)
    paths = sorted(base.glob(glob), key=lambda p: (not p.is_dir(), str(p).lower()))[:limit]
    root = ctx.workspace.resolve()
    content = "\n".join(("/" if p.is_dir() else "") + str(p.resolve().relative_to(root)).replace("\\", "/") for p in paths)
    return ToolResult(str(args.get("_call_id", "")), "list_files", True, content or "(no matches)")


def read_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = resolve_workspace_path(ctx.workspace, str(args["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    content = _line_window(text.splitlines(), int(args.get("offset") or 1), int(args.get("limit") or 200))
    return ToolResult(str(args.get("_call_id", "")), "read_file", True, content)


def search_files(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    root = resolve_workspace_path(ctx.workspace, str(args.get("path") or "."))
    pattern = re.compile(str(args["pattern"]))
    glob = str(args.get("glob") or "*.py")
    limit = max(1, min(int(args.get("limit") or 50), 500))
    workspace = ctx.workspace.resolve()
    matches: list[str] = []
    for path in sorted(root.rglob(glob) if root.is_dir() else [root]):
        if len(matches) >= limit:
            break
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, 1):
            if pattern.search(line):
                rel = str(path.resolve().relative_to(workspace)).replace("\\", "/")
                matches.append(f"{rel}:{line_no}: {line}")
                if len(matches) >= limit:
                    break
    return ToolResult(str(args.get("_call_id", "")), "search_files", True, "\n".join(matches) or "(no matches)")


def write_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = resolve_workspace_path(ctx.workspace, str(args["path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    content = str(args.get("content") or "")
    path.write_text(content, encoding="utf-8")
    return ToolResult(str(args.get("_call_id", "")), "write_file", True, f"wrote {len(content)} chars to {path.name}")


def patch_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = resolve_workspace_path(ctx.workspace, str(args["path"]))
    text = path.read_text(encoding="utf-8")
    old = str(args["old"])
    new = str(args.get("new") or "")
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one match, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return ToolResult(str(args.get("_call_id", "")), "patch_file", True, f"patched {path.name}")


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolSpec(
        "list_files", "List files under the workspace.",
        _schema({"path": {"type": "string"}, "glob": {"type": "string"}, "limit": {"type": "integer"}}),
        ToolRisk.SAFE_READ, list_files,
    ))
    registry.register(ToolSpec(
        "read_file", "Read a UTF-8 text file from the workspace with line numbers.",
        _schema({"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["path"]),
        ToolRisk.SAFE_READ, read_file,
    ))
    registry.register(ToolSpec(
        "search_files", "Regex search text files under the workspace.",
        _schema({"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}, "limit": {"type": "integer"}}, ["pattern"]),
        ToolRisk.SAFE_READ, search_files,
    ))
    registry.register(ToolSpec(
        "write_file", "Write a text file inside the workspace.",
        _schema({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        ToolRisk.LOCAL_WRITE, write_file,
    ))
    registry.register(ToolSpec(
        "patch_file", "Replace one exact string in a workspace file.",
        _schema({"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, ["path", "old", "new"]),
        ToolRisk.LOCAL_WRITE, patch_file,
    ))
    return registry
