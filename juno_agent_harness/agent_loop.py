from __future__ import annotations

import json
from pathlib import Path

from .providers import ChatResult
from .tool_registry import ToolCall, ToolContext, ToolExecutor, ToolRegistry
from .tools import create_default_registry


JUNO_AGENT_CAPABILITY_NOTICE = """
Juno tool-use contract:
- You may request tools from the provided tool schema.
- Do not claim a tool result until it is returned by the harness.
- Prefer read/search tools before writes.
- Explain risky actions before requesting them.
- Stop calling tools when the task is complete and provide a concise final answer.
""".strip()


def default_tool_registry() -> ToolRegistry:
    return create_default_registry()


def run_agent_turn(runtime: "HarnessRuntime", line: str) -> ChatResult:  # noqa: F821
    """Run one user turn with a bounded provider/tool loop."""
    runtime.messages.append({"role": "user", "content": line})
    runtime.sessions.append(runtime.session_id, "user", line)

    registry = getattr(runtime, "tool_registry", None) or default_tool_registry()
    runtime.tool_registry = registry
    context = ToolContext(
        workspace=Path(getattr(runtime.args, "workspace", Path.cwd())),
        home=Path(runtime.args.home),
        session_id=runtime.session_id,
        approval=getattr(runtime.args, "approval", "never"),
    )
    executor = ToolExecutor(registry, context)
    max_turns = max(1, int(getattr(runtime.args, "max_tool_turns", 8)))

    last_result: ChatResult | None = None
    for _step in range(max_turns):
        result = runtime.client.complete(
            runtime.messages,
            tools=registry.provider_schemas(runtime.args.provider),
        )
        runtime.total_in += result.input_tokens
        runtime.total_out += result.output_tokens
        last_result = result
        if not result.tool_calls:
            runtime.messages.append({"role": "assistant", "content": result.text})
            runtime.sessions.append(runtime.session_id, "assistant", result.text)
            return result

        runtime.messages.append({
            "role": "assistant",
            "content": result.text or "",
            "tool_calls": [call.__dict__ for call in result.tool_calls],
        })
        runtime.sessions.append(
            runtime.session_id,
            "assistant_tool_call",
            json.dumps([call.__dict__ for call in result.tool_calls], ensure_ascii=False),
        )
        for call in result.tool_calls:
            tool_result = executor.execute(call)
            message_content = tool_result.to_message_content()
            runtime.messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": message_content,
            })
            runtime.sessions.append(runtime.session_id, "tool", message_content)

    raise RuntimeError(f"max tool turns exceeded ({max_turns}); last result={last_result}")
