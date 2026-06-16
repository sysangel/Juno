"""Smoke + unit tests for the harness UI layer (Juno + plain fallback).

Run from the agent-loop directory:

    .venv/Scripts/python.exe -m pytest test_harness_ui.py -v

These tests avoid any paid API call: they use the `echo` provider and the plain
UI, and they construct HarnessRuntime / PromptToolkitUI objects directly. The
prompt_toolkit PromptSession itself is NOT instantiated under pytest because it
requires a real console; we test the completer/toolbar/keybindings instead.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Make `import harness` (and its `import agent_loop`) resolve when pytest is run
# from anywhere.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import harness  # noqa: E402

PYEXE = sys.executable


def _make_runtime(tmp_path: Path, provider: str = "echo") -> harness.HarnessRuntime:
    args = argparse.Namespace(
        provider=provider,
        model=harness.default_model(provider),
        temperature=0.2,
        base_url=None,
        system="You are a test harness.",
        home=tmp_path,
        ui="plain",
    )
    memory = harness.MemoryStore(args.home)
    memory.load()
    sessions = harness.SessionStore(args.home / "sessions.db")
    return harness.HarnessRuntime.create(args, memory, sessions)


# --- normalize_provider aliases -------------------------------------------

@pytest.mark.parametrize("alias,expected", [
    ("claude", "claude-code"),
    ("cc", "claude-code"),
    ("chatgpt", "openai"),
    ("gpt", "openai"),
    ("codex", "openai"),
    ("or", "openrouter"),
    ("openrouter", "openrouter"),
    ("ECHO", "echo"),
])
def test_normalize_provider_aliases(alias, expected):
    assert harness.normalize_provider(alias) == expected


def test_normalize_provider_rejects_unknown():
    with pytest.raises(ValueError):
        harness.normalize_provider("not-a-provider")


# --- model catalog and aliases ---------------------------------------------

@pytest.mark.parametrize("token,provider,model", [
    ("sonnet", "claude-code", "sonnet"),
    ("opus", "claude-code", "opus"),
    ("codex", "openai", harness.DEFAULT_OPENAI_MODEL),
    ("dp", "openrouter", harness.DEFAULT_OPENROUTER_DP_MODEL),
    ("fhigh", "openrouter", harness.DEFAULT_OPENROUTER_FHIGH_MODEL),
    ("echo", "echo", "echo"),
])
def test_resolve_model_selection_catalog_aliases(token, provider, model):
    resolved_provider, resolved_model, note = harness.resolve_model_selection(token)
    assert resolved_provider == provider
    assert resolved_model == model
    assert note.startswith("selected")


def test_resolve_model_selection_raw_provider_pair():
    provider, model, note = harness.resolve_model_selection("openrouter", "some/raw-model")
    assert provider == "openrouter"
    assert model == "some/raw-model"
    assert note == ""


def test_format_model_catalog_is_segmented():
    text = harness.format_model_catalog("echo", "echo")
    assert "Codex / OpenAI" in text
    assert "Anthropic / Claude" in text
    assert "OpenRouter" in text
    assert "* echo" in text


# --- HarnessRuntime state transitions -------------------------------------

def test_switch_provider_preserves_conversation(tmp_path):
    rt = _make_runtime(tmp_path)
    rt.submit_user_message("hello there")
    before = [m for m in rt.messages if m["role"] != "system"]
    assert len(before) == 2  # user + assistant
    msg = rt.switch_provider("claude", "sonnet")
    assert "claude-code" in msg and "sonnet" in msg
    assert rt.args.provider == "claude-code"
    assert rt.args.model == "sonnet"
    after = [m for m in rt.messages if m["role"] != "system"]
    assert after == before  # conversation preserved across switch
    rt.close()


def test_reload_memory_preserves_conversation_and_injects(tmp_path):
    rt = _make_runtime(tmp_path)
    rt.submit_user_message("remember this turn")
    rt.memory.add("memory", "durable fact about the project")
    out = rt.reload_memory()
    assert "reloaded" in out
    # system prompt now contains the memory entry
    system = next(m for m in rt.messages if m["role"] == "system")
    assert "durable fact about the project" in system["content"]
    # non-system conversation preserved
    convo = [m for m in rt.messages if m["role"] != "system"]
    assert len(convo) == 2
    rt.close()


def test_reset_conversation_keeps_memory(tmp_path):
    rt = _make_runtime(tmp_path)
    rt.memory.add("memory", "keep me")
    rt.reload_memory()
    rt.submit_user_message("throwaway")
    rt.reset_conversation()
    convo = [m for m in rt.messages if m["role"] != "system"]
    assert convo == []  # conversation cleared
    system = next(m for m in rt.messages if m["role"] == "system")
    assert "keep me" in system["content"]  # memory snapshot retained
    rt.close()


def test_handle_command_unknown_does_not_raise(tmp_path):
    rt = _make_runtime(tmp_path)
    res = rt.handle_command("/bogus")
    assert res.should_continue is True
    assert "unknown command" in res.output
    rt.close()


def test_handle_command_exit(tmp_path):
    rt = _make_runtime(tmp_path)
    assert rt.handle_command("/exit").should_continue is False
    assert rt.handle_command("/quit").should_continue is False
    rt.close()


def test_memory_add_and_list_roundtrip(tmp_path):
    rt = _make_runtime(tmp_path)
    rt.handle_command("/memory add memory smoke fact")
    listing = rt.handle_command("/memory list").output
    assert "smoke fact" in listing
    rt.close()


# --- UI formatting helpers -------------------------------------------------

def test_format_assistant_response_wraps_in_box():
    text = " ".join(["word"] * 40)
    rendered = harness.format_assistant_response(
        text,
        provider="echo",
        model="echo",
        color=False,
        width=60,
    )
    lines = rendered.splitlines()
    assert lines[0].startswith("╭")
    assert "assistant · echo/echo" in rendered
    assert lines[-1].startswith("╰")
    assert all(len(line) <= 60 for line in lines)


def test_format_assistant_response_preserves_code_fences():
    text = "before\n\n```python\nprint('hello')\n```\n\nafter"
    rendered = harness.format_assistant_response(text, provider="echo", model="echo", color=False, width=70)
    assert "```python" in rendered
    assert "print('hello')" in rendered
    assert "after" in rendered


def test_format_thinking_has_juno_vibe_and_provider():
    rendered = harness.format_thinking("echo", "echo", color=False)
    assert rendered.startswith("✦ ")
    assert rendered.endswith(" with echo/echo…")
    assert any(msg in rendered for msg in harness.THINKING_MESSAGES)


# --- PromptToolkitUI construction (no console required) --------------------

def test_juno_completer_covers_vocabulary(tmp_path):
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    rt = _make_runtime(tmp_path)
    ui = harness.PromptToolkitUI(rt)
    comp = ui._build_completer()

    def comps(text):
        doc = Document(text, len(text))
        return {c.text for c in comp.get_completions(doc, CompleteEvent())}

    slash = comps("/")
    for cmd in ["/help", "/status", "/model", "/memory", "/search", "/reset", "/clear", "/exit"]:
        assert cmd in slash
    assert {"claude-code", "sonnet", "codex", "dp", "fhigh", "openai", "openrouter", "echo"} <= comps("/model ")
    assert {"list", "add", "reload"} <= comps("/memory ")
    assert {"memory", "user"} <= comps("/memory add ")
    rt.close()


def test_juno_toolbar_and_prompt_label(tmp_path):
    rt = _make_runtime(tmp_path)
    ui = harness.PromptToolkitUI(rt)
    toolbar = str(ui._bottom_toolbar().value)
    assert "echo/echo" in toolbar and "memory=on" in toolbar
    assert rt.prompt_label().startswith("echo/echo")
    rt.close()


# --- subprocess smoke: plain + auto fallback ------------------------------

def _run_harness(tmp_path: Path, ui: str, script: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(HERE))
    return subprocess.run(
        [PYEXE, str(HERE / "harness.py"), "--provider", "echo",
         "--ui", ui, "--home", str(tmp_path / f"runs_{ui}")],
        input=script, text=True, capture_output=True, timeout=120, env=env,
    )


def test_plain_pipe_smoke(tmp_path):
    proc = _run_harness(tmp_path, "plain", "/model echo\nhello\n/exit\n")
    assert proc.returncode == 0
    assert "echo: hello" in proc.stdout


def test_plain_turn_uses_thinking_and_assistant_box(tmp_path):
    proc = _run_harness(tmp_path, "plain", "hello\n/exit\n")
    assert proc.returncode == 0
    out = proc.stdout
    assert "✦ " in out and " with echo/echo…" in out
    assert "╭" in out and "assistant · echo/echo" in out and "╰" in out
    assert "echo: hello" in out
    assert "turn " in out and "tok · total" in out
    assert "\x1b[" not in out


def test_plain_command_output_has_no_thinking_or_assistant_box(tmp_path):
    proc = _run_harness(tmp_path, "plain", "/status\n/exit\n")
    assert proc.returncode == 0
    out = proc.stdout
    assert "provider : echo" in out
    assert "✦ " not in out
    assert "assistant · echo/echo" not in out
    assert "tok · total" not in out


def test_auto_falls_back_to_plain_on_pipe(tmp_path):
    # stdin is a pipe (not a TTY), so auto must use the plain UI and exit 0.
    proc = _run_harness(tmp_path, "auto", "/status\nhello\n/exit\n")
    assert proc.returncode == 0
    assert "echo: hello" in proc.stdout


def test_one_shot_query_still_works(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(HERE))
    proc = subprocess.run(
        [PYEXE, str(HERE / "harness.py"), "--provider", "echo", "-q", "hello",
         "--home", str(tmp_path / "runs_q")],
        text=True, capture_output=True, timeout=120, env=env,
    )
    assert proc.returncode == 0
    assert "echo: hello" in proc.stdout
