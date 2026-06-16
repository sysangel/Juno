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
import asyncio
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
    assert "juno · echo/echo" in rendered
    assert lines[-1].startswith("╰")
    assert all(len(line) <= 60 for line in lines)


def test_format_assistant_response_preserves_code_fences():
    text = "before\n\n```python\nprint('hello')\n```\n\nafter"
    rendered = harness.format_assistant_response(text, provider="echo", model="echo", color=False, width=70)
    assert "```python" in rendered
    assert "print('hello')" in rendered
    assert "after" in rendered


def test_format_thinking_has_juno_vibe_and_no_provider():
    rendered = harness.format_thinking(color=False)
    assert rendered.startswith("✦ ")
    assert "nebula signal" in rendered
    assert " with " not in rendered  # no provider/model injected


# --- Task 1: requested wording tests (should FAIL before implementation) ---

def test_format_thinking_contains_nebula_signal():
    rendered = harness.format_thinking("claude-code", "opus", color=False)
    assert "nebula signal" in rendered


def test_format_thinking_no_provider_model_injection():
    rendered = harness.format_thinking("claude-code", "opus", color=False)
    assert "claude-code" not in rendered
    assert "opus" not in rendered
    assert " with " not in rendered


def test_format_waiting_static_if_exists():
    # If the static helper already exists, test it; if not, skip gracefully.
    fn = getattr(harness, "format_waiting_static", None)
    if fn is None:
        pytest.skip("format_waiting_static not yet implemented")
    result = fn(color=False)
    assert "nebula signal" in result
    assert "claude-code" not in result
    assert "opus" not in result
    assert " with " not in result


# --- Task 5: --tui / --cli alias tests -----------------------------------

def test_parser_tui_alias_maps_to_prompt():
    parser = harness.build_parser()
    ns = parser.parse_args(["--tui"])
    assert getattr(ns, "_tui") is True
    # After main() post-processing, args.ui would become "prompt"


def test_parser_cli_alias_maps_to_plain():
    parser = harness.build_parser()
    ns = parser.parse_args(["--cli"])
    assert getattr(ns, "_cli") is True
    # After main() post-processing, args.ui would become "plain"


def test_parser_tui_and_cli_conflict():
    parser = harness.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--tui", "--cli"])


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


def test_waitingdots_symbol_is_gone():
    # WaitingDots was deleted; run() now animates via the bottom toolbar. A
    # lingering reference would NameError on submit in the live TUI.
    assert not hasattr(harness, "WaitingDots")


def test_awaiting_toolbar_renders_nebula_signal(tmp_path):
    # The same toolbar callback prompt_toolkit polls renders the waiting state
    # when _awaiting is set, so the bar stays pinned (no screen flip). Assert the
    # provider/model + "nebula signal" structure rather than exact dots.
    rt = _make_runtime(tmp_path)
    ui = harness.PromptToolkitUI(rt)
    ui._awaiting = True
    ui._tick = 1
    toolbar = str(ui._bottom_toolbar().value)
    assert "echo/echo" in toolbar
    assert "nebula signal" in toolbar
    rt.close()


def test_run_async_provider_error_clears_awaiting_no_thread_leak(tmp_path, monkeypatch, capsys):
    # Core invariant of the B-persistent restructure: when the off-thread
    # provider call raises inside _handle_turn, the try/except prints the error
    # and the finally clears `_awaiting` so the persistent prompt stays usable —
    # without leaking the waiting state. There is NO ticker thread anymore (the
    # app's refresh_interval replaced it), so the active-thread count must not
    # grow. We drive the REAL _handle_turn (not a hand-replica): JUNO_WAIT_ANIMATION=0
    # forces the static-line path so no Application is started.
    import threading

    # JUNO_WAIT_ANIMATION=0 makes `animate` False regardless of isatty (it's an
    # `and`), so the static-line path is taken and no Application is started.
    monkeypatch.setenv("JUNO_WAIT_ANIMATION", "0")
    rt = _make_runtime(tmp_path)
    ui = harness.PromptToolkitUI(rt)

    def _boom(line):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(rt, "submit_user_message", _boom)

    before = threading.active_count()
    asyncio.run(ui._handle_turn("hi"))

    assert ui._awaiting is False
    # _handle_turn swallows the provider error (background tasks must not raise)
    # and prints it instead of propagating.
    assert "provider error" in capsys.readouterr().out
    # No ticker thread is spun up; nothing leaked.
    assert threading.active_count() <= before
    rt.close()


# --- subprocess smoke: plain + auto fallback ------------------------------

def _run_harness(tmp_path: Path, ui: str, script: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(HERE), PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [PYEXE, str(HERE / "harness.py"), "--provider", "echo",
         "--ui", ui, "--home", str(tmp_path / f"runs_{ui}")],
        input=script, text=True, capture_output=True, timeout=120, env=env,
        encoding="utf-8", errors="replace",
    )


def test_plain_pipe_smoke(tmp_path):
    proc = _run_harness(tmp_path, "plain", "/model echo\nhello\n/exit\n")
    assert proc.returncode == 0
    assert "echo: hello" in proc.stdout


def test_plain_turn_uses_thinking_and_assistant_box(tmp_path):
    proc = _run_harness(tmp_path, "plain", "hello\n/exit\n")
    assert proc.returncode == 0
    out = proc.stdout
    assert "✦ " in out and "nebula signal" in out
    assert " with echo/echo…" not in out
    assert "╭" in out and "juno · echo/echo" in out and "╰" in out
    assert "echo: hello" in out
    assert "turn " in out and "tok · total" in out
    assert "\x1b[" not in out


def test_plain_command_output_has_no_thinking_or_assistant_box(tmp_path):
    proc = _run_harness(tmp_path, "plain", "/status\n/exit\n")
    assert proc.returncode == 0
    out = proc.stdout
    assert "provider : echo" in out
    assert " with echo/echo…" not in out
    assert "juno · echo/echo" not in out
    assert "tok · total" not in out


def test_auto_falls_back_to_plain_on_pipe(tmp_path):
    # stdin is a pipe (not a TTY), so auto must use the plain UI and exit 0.
    proc = _run_harness(tmp_path, "auto", "/status\nhello\n/exit\n")
    assert proc.returncode == 0
    assert "echo: hello" in proc.stdout


def test_one_shot_query_still_works(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(HERE), PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [PYEXE, str(HERE / "harness.py"), "--provider", "echo", "-q", "hello",
         "--home", str(tmp_path / "runs_q")],
        text=True, capture_output=True, timeout=120, env=env,
        encoding="utf-8", errors="replace",
    )
    assert proc.returncode == 0
    assert "echo: hello" in proc.stdout


# --- Task 9: /menu palette must render (regression guard for the undefined
# format_menu_palette NameError that the rest of the suite did not exercise) ---

def test_menu_palette_renders_without_crashing():
    out = harness.format_menu_palette()
    assert "/status" in out
    assert "/skills" in out
    assert "Model selector" in out


def test_help_lists_skills_and_menu():
    h = harness.format_help()
    assert "/skills" in h
    assert "/menu" in h


# --- Task 1: banner trim ---

def test_banner_trimmed_keeps_sigil_drops_subtitles(tmp_path, capsys):
    rt = _make_runtime(tmp_path)
    harness.print_banner(rt, color=False)
    out = capsys.readouterr().out
    assert "J U N O" in out
    assert "sleek galaxy terminal" not in out
    assert "galaxy console" not in out


# --- Task 2: default-model setter ---

import json  # noqa: E402


@pytest.mark.parametrize("ref,expected", [
    ("gpt5", ("openai", "gpt-5.1")),
    ("sonnet", ("claude-code", "sonnet")),
    ("openai/gpt-5.1", ("openai", "gpt-5.1")),
    ("claude-code/opus", ("claude-code", "opus")),
])
def test_resolve_model_ref_both_forms(ref, expected):
    assert harness.resolve_model_ref(ref) == expected


def test_resolve_model_ref_unknown_key_raises():
    with pytest.raises(KeyError):
        harness.resolve_model_ref("nope-not-a-key")


def test_model_default_persists(tmp_path):
    cfg = harness.HarnessConfig(home=tmp_path)
    prov, model = harness.resolve_model_ref("gpt5")
    cfg.set_default_model(prov, model)
    on_disk = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert on_disk["model"] == {"provider": prov, "default": model}
    # Reload a fresh config object from the same path and confirm round-trip.
    assert harness.HarnessConfig(home=tmp_path).default_model() == (prov, model)


def test_model_default_command_sets_and_reports(tmp_path):
    cfg = harness.HarnessConfig(home=tmp_path)
    args = argparse.Namespace(
        provider="echo", model=harness.default_model("echo"), temperature=0.2,
        base_url=None, system="t", home=tmp_path, ui="plain",
    )
    memory = harness.MemoryStore(args.home); memory.load()
    sessions = harness.SessionStore(args.home / "sessions.db")
    rt = harness.HarnessRuntime.create(args, memory, sessions, config=cfg)
    res = rt.handle_command("/model default gpt5")
    assert res.should_continue
    assert "openai/gpt-5.1" in res.output
    assert cfg.default_model() == ("openai", "gpt-5.1")
    # Bare reports current default.
    res2 = rt.handle_command("/model default")
    assert "openai/gpt-5.1" in res2.output


# --- Task 3: model strength + icons + ultracode ---

def test_every_openai_anthropic_option_has_tier_and_icon():
    for opt in harness.model_catalog():
        if opt.provider in ("openai", "claude-code", "anthropic"):
            tier = harness.tier_for(opt)
            assert tier in harness.TIERS
            assert harness.strength_icon(opt, color=True) != ""
            assert harness.strength_icon(opt, color=False) != ""


def test_strength_bars_are_constant_width():
    glyph_widths = {len(g) for g, _a, _c in harness.TIERS.values()}
    ascii_widths = {len(a) for _g, a, _c in harness.TIERS.values()}
    assert glyph_widths == {5}
    assert ascii_widths == {len("[#####]")}


def test_ultracode_routes_exist():
    tiers = {(o.provider, o.key): harness.tier_for(o) for o in harness.model_catalog()}
    assert tiers[("openai", "codex")] == "ultracode"
    # At least one Anthropic/claude-code ultracode route.
    assert "ultracode" in {harness.tier_for(o) for o in harness.model_catalog()
                            if o.provider == "claude-code"}


def test_format_model_row_does_not_change_key():
    for opt in harness.model_catalog():
        row = harness.format_model_row(opt, color=True)
        # The catalog key must never be embedded as the row identity.
        assert isinstance(row, str)
    # Key resolution is unaffected by decoration.
    assert harness.model_aliases()["codex"].key == "codex"


def test_ascii_fallback_has_bar_and_ultra_badge():
    codex = next(o for o in harness.model_catalog() if o.key == "codex")
    row = harness.format_model_row(codex, color=False)
    assert "[#####]" in row
    assert "*ULTRA*" in row
    # No ANSI escapes in plain mode.
    assert "\033" not in row


def test_format_model_row_truecolor_has_sgr():
    codex = next(o for o in harness.model_catalog() if o.key == "codex")
    row = harness.format_model_row(codex, color=True)
    assert "\033[" in row
    assert "ULTRACODE" in row


# --- Task 4: calm the picker (style= kwarg) ---

def test_model_picker_passes_style_kwarg(tmp_path, monkeypatch):
    # B-persistent: _run_model_picker is async and runs `await dialog.run_async()`
    # inside `async with in_terminal()`. The fake dialog exposes an async
    # run_async() and a settable full_screen attribute; we patch in_terminal to a
    # no-op async context manager so only the dialog-kwarg contract is exercised
    # (no live console). The style= kwarg assertion is unchanged.
    pytest.importorskip("prompt_toolkit")
    import contextlib
    import prompt_toolkit.shortcuts as ptshort

    captured = {}

    class _FakeDialog:
        def __init__(self):
            self.full_screen = True  # settable; picker must flip this to False

        async def run_async(self):
            return None  # simulate cancel

    made = {}

    def _fake_radiolist_dialog(*args, **kwargs):
        captured.update(kwargs)
        d = _FakeDialog()
        made["dialog"] = d
        return d

    @contextlib.asynccontextmanager
    async def _noop_in_terminal(*args, **kwargs):
        yield

    monkeypatch.setattr(ptshort, "radiolist_dialog", _fake_radiolist_dialog)
    # _run_model_picker does `from prompt_toolkit.application.run_in_terminal
    # import in_terminal` at call time. NOTE: `import ...run_in_terminal as x`
    # binds the *function* (re-exported on the package), not the submodule, so
    # resolve the submodule explicitly to patch its `in_terminal` symbol.
    import importlib
    ptrit = importlib.import_module("prompt_toolkit.application.run_in_terminal")
    monkeypatch.setattr(ptrit, "in_terminal", _noop_in_terminal)

    rt = _make_runtime(tmp_path)
    ui = harness.PromptToolkitUI(rt)
    result = asyncio.run(ui._run_model_picker())
    assert "style" in captured, "radiolist_dialog must receive a style= kwarg"
    assert captured["style"] is not None
    assert result == "model selection cancelled"
    # B-persistent: the picker must force inline (no alt-screen flip) and resolve
    # via the async run_async path.
    assert made["dialog"].full_screen is False


def test_enter_binding_ignores_submit_while_awaiting_and_schedules_otherwise(tmp_path):
    # The non-accepting Enter handler is the core of B-persistent: while a turn is
    # in flight (_awaiting) it must do nothing; otherwise it appends to history,
    # clears the buffer, and schedules _handle_turn as a background task. We drive
    # the real handler with a fake event/app/buffer (no live Application).
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.keys import Keys

    rt = _make_runtime(tmp_path)
    ui = harness.PromptToolkitUI(rt)
    kb = ui._build_keybindings()

    def _handler_for(*keys):
        want = tuple(keys)
        for b in kb.bindings:
            if tuple(b.keys) == want:
                return b.handler
        raise AssertionError(f"no binding for {want}")

    enter = _handler_for(Keys.ControlM)

    class _FakeHistory:
        def __init__(self):
            self.appended = []

        def append_string(self, s):
            self.appended.append(s)

    class _FakeBuffer:
        def __init__(self, text):
            self.text = text
            self.history = _FakeHistory()
            self.reset_called = False

        def reset(self):
            self.reset_called = True

    class _FakeApp:
        def __init__(self):
            self.scheduled = []

        def create_background_task(self, coro):
            self.scheduled.append(coro)
            coro.close()  # we are not running it; avoid "never awaited" warning

    class _FakeEvent:
        def __init__(self, buf, app):
            self.current_buffer = buf
            self.app = app

    # 1) While awaiting: ignore the submit entirely.
    ui._awaiting = True
    buf = _FakeBuffer("hello")
    app = _FakeApp()
    enter(_FakeEvent(buf, app))
    assert app.scheduled == []
    assert buf.reset_called is False
    assert buf.history.appended == []

    # 2) Not awaiting, non-empty: append history, reset, schedule a turn.
    ui._awaiting = False
    buf = _FakeBuffer("do a thing")
    app = _FakeApp()
    enter(_FakeEvent(buf, app))
    assert buf.history.appended == ["do a thing"]
    assert buf.reset_called is True
    assert len(app.scheduled) == 1

    # 3) Not awaiting, blank: do nothing.
    buf = _FakeBuffer("   ")
    app = _FakeApp()
    enter(_FakeEvent(buf, app))
    assert app.scheduled == []
    assert buf.reset_called is False
    rt.close()


def test_ctrl_d_exits_only_on_empty_buffer_ctrl_c_clears(tmp_path):
    # Ctrl-D exits only when the buffer is empty; Ctrl-C always clears, never exits.
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.keys import Keys

    rt = _make_runtime(tmp_path)
    ui = harness.PromptToolkitUI(rt)
    kb = ui._build_keybindings()

    def _handler_for(*keys):
        want = tuple(keys)
        for b in kb.bindings:
            if tuple(b.keys) == want:
                return b.handler
        raise AssertionError(f"no binding for {want}")

    class _FakeBuffer:
        def __init__(self, text):
            self.text = text
            self.reset_called = False

        def reset(self):
            self.reset_called = True

    class _FakeApp:
        def __init__(self):
            self.exited = False

        def exit(self, result=None):
            self.exited = True

    class _FakeEvent:
        def __init__(self, buf, app):
            self.current_buffer = buf
            self.app = app

    cd = _handler_for(Keys.ControlD)
    cc = _handler_for(Keys.ControlC)

    # Ctrl-D on empty buffer -> exit.
    app = _FakeApp()
    cd(_FakeEvent(_FakeBuffer(""), app))
    assert app.exited is True
    # Ctrl-D on non-empty buffer -> no exit.
    app = _FakeApp()
    cd(_FakeEvent(_FakeBuffer("x"), app))
    assert app.exited is False
    # Ctrl-C clears, never exits.
    app = _FakeApp()
    buf = _FakeBuffer("typed text")
    cc(_FakeEvent(buf, app))
    assert buf.reset_called is True
    assert app.exited is False
    rt.close()


def test_handle_turn_normal_path_prints_assistant_and_usage(tmp_path, monkeypatch, capsys):
    # _handle_turn's normal-turn path: clears _awaiting after the executor call
    # and prints the assistant box + usage line. JUNO_WAIT_ANIMATION=0 keeps it
    # off the live-Application path.
    monkeypatch.setenv("JUNO_WAIT_ANIMATION", "0")
    rt = _make_runtime(tmp_path)
    ui = harness.PromptToolkitUI(rt)
    asyncio.run(ui._handle_turn("hello"))
    out = capsys.readouterr().out
    assert ui._awaiting is False
    assert "echo: hello" in out
    assert "turn " in out and "tok" in out
    rt.close()


# --- Task 6: rainbow "boost" text (req 6) ---

def test_rainbow_color_disabled_is_identity():
    # Color off -> input returned unchanged, with NO SGR escape codes at all.
    assert harness.rainbow("hello world", color=False) == "hello world"
    assert "\033[" not in harness.rainbow("xyz", color=False)


def test_rainbow_truecolor_emits_24bit_and_resets(monkeypatch):
    monkeypatch.setenv("COLORTERM", "truecolor")
    out = harness.rainbow("abc", color=True)
    assert "38;2;" in out          # 24-bit truecolor SGR
    assert out.endswith(harness.RESET)


def test_rainbow_degrades_to_256_without_truecolor(monkeypatch):
    monkeypatch.delenv("COLORTERM", raising=False)
    out = harness.rainbow("abc", color=True)
    assert "38;5;" in out          # 256-color degrade
    assert "38;2;" not in out
    assert out.endswith(harness.RESET)


def test_rainbow_sweep_uses_erase_to_eol_not_space_pad():
    import inspect
    src = inspect.getsource(harness.RainbowSweep)
    # The repaint clears with erase-to-EOL, not by writing a run of spaces.
    assert "\\r\\033[K" in src
    # No space-pad clear: there must be no string literal of >=4 spaces used to
    # blank the line (that pattern caused the white-bar artifact).
    assert '"    ' not in src and "'    " not in src


# --- JunoTheme Pass 1: define + rebind, identity preserved (req 7a) --------

def test_default_theme_identity():
    import dataclasses
    # DEFAULT_THEME is a frozen JunoTheme dataclass instance.
    assert isinstance(harness.DEFAULT_THEME, harness.JunoTheme)
    assert dataclasses.is_dataclass(harness.DEFAULT_THEME)
    assert harness.JunoTheme.__dataclass_params__.frozen is True
    # Frozen: assigning to a field raises FrozenInstanceError.
    with pytest.raises(dataclasses.FrozenInstanceError):
        harness.DEFAULT_THEME.purple = "x"  # type: ignore[misc]
    # Every legacy name IS the corresponding DEFAULT_THEME field (same object,
    # not a copy) — the core Pass 1 guarantee that call sites stay valid.
    assert harness.PURPLE is harness.DEFAULT_THEME.purple
    assert harness.PURPLE_DIM is harness.DEFAULT_THEME.purple_dim
    assert harness.PURPLE_BOLD is harness.DEFAULT_THEME.purple_bold
    assert harness.PURPLE_ITALIC is harness.DEFAULT_THEME.purple_italic
    assert harness.LAVENDER is harness.DEFAULT_THEME.lavender
    assert harness.RED_BOLD is harness.DEFAULT_THEME.red_bold
    assert harness.RESET is harness.DEFAULT_THEME.reset
    assert harness.JUNO_NAME is harness.DEFAULT_THEME.name
    assert harness.JUNO_BANNER is harness.DEFAULT_THEME.banner
    assert harness.JUNO_SIGIL is harness.DEFAULT_THEME.sigil


# --- JunoTheme Pass 2: theme threaded through formatters (req 7a) -----------

def test_formatter_accepts_custom_theme():
    # A custom frozen theme with sentinel SGR values overrides the formatter
    # rendering when passed via the keyword-only `theme=` arg.
    custom = harness.JunoTheme(purple_italic="<<PI>>", purple="<<P>>", red_bold="<<RB>>")
    # Threaded formatter picks up the sentinel from the custom theme...
    thinking = harness.format_thinking(color=True, theme=custom)
    assert "<<PI>>" in thinking
    assert harness.format_box("body", title="t", color=True, theme=custom).find("<<P>>") != -1
    assert "<<RB>>" in harness.format_error("boom", color=True, theme=custom)
    # ...and omitting theme= still renders with the default singleton (unchanged).
    default_thinking = harness.format_thinking(color=True)
    assert "<<PI>>" not in default_thinking
    assert harness.DEFAULT_THEME.purple_italic in default_thinking


# --- Model picker: labels wrapped as ANSI formatted-text (jumbled-text fix) ---

def test_model_picker_wraps_labels_as_ansi(tmp_path, monkeypatch):
    # RadioList renders a plain str label as ONE literal fragment and does NOT
    # parse embedded \033[...m SGR, so the raw ANSI from format_model_row printed
    # verbatim (jumbled text). The fix wraps every label in ANSI(...). Drive the
    # picker with a fake dialog that captures values= (same harness as
    # test_model_picker_passes_style_kwarg) and assert each label is an ANSI
    # instance and no label is a raw str containing an escape.
    pytest.importorskip("prompt_toolkit")
    import contextlib
    import importlib
    import prompt_toolkit.shortcuts as ptshort
    from prompt_toolkit.formatted_text import ANSI

    captured = {}

    class _FakeDialog:
        def __init__(self):
            self.full_screen = True

        async def run_async(self):
            return None  # simulate cancel

    def _fake_radiolist_dialog(*args, **kwargs):
        captured.update(kwargs)
        return _FakeDialog()

    @contextlib.asynccontextmanager
    async def _noop_in_terminal(*args, **kwargs):
        yield

    monkeypatch.setattr(ptshort, "radiolist_dialog", _fake_radiolist_dialog)
    ptrit = importlib.import_module("prompt_toolkit.application.run_in_terminal")
    monkeypatch.setattr(ptrit, "in_terminal", _noop_in_terminal)

    rt = _make_runtime(tmp_path)
    ui = harness.PromptToolkitUI(rt)
    asyncio.run(ui._run_model_picker())

    values = captured.get("values")
    assert values, "radiolist_dialog must receive a non-empty values= list"
    for key, label in values:
        assert isinstance(label, ANSI), f"label for {key!r} must be ANSI, got {type(label)}"
        # No raw str carrying a verbatim SGR escape (the jumbled-text symptom).
        assert not (isinstance(label, str) and "\033[" in label)
    rt.close()


def test_initial_messages_explain_juno_is_chat_only():
    messages = harness.build_initial_messages("base system", None)
    content = messages[0]["content"]
    assert "base system" in content
    assert "chat harness, not a live tool-running agent" in content
    assert "cannot inspect files" in content
    assert 'Do not say "I\'ll check"' in content


def test_claude_code_status_explains_chat_only_mode(tmp_path):
    rt = _make_runtime(tmp_path, provider="claude-code")
    rt.args.model = "opus"
    status = harness.format_status(rt)

    assert "chat-only" in status
    assert "disables Claude Code tools" in status
    rt.close()


# --- claude-code provider: None-stdout guard (UnicodeDecodeError fix) ---


def test_claude_code_effort_flag_is_passed(monkeypatch):
    captured = {}
    chat = harness.ProviderClient("claude-code", "opus", 0.0, effort="high")

    class _FakeProc:
        returncode = 0
        stdout = '{"result": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}}'
        stderr = ""

    def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(harness.subprocess, "run", _fake_run)

    result = chat._claude_code([{"role": "user", "content": "hi"}])
    assert result.text == "ok"
    assert "--effort" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--effort") + 1] == "high"
    assert captured["cmd"][captured["cmd"].index("--max-turns") + 1] == "1"
    assert captured["cmd"][captured["cmd"].index("--tools") + 1] == ""

def test_claude_code_raises_on_none_stdout(monkeypatch):
    # On Windows text=True with no encoding= can decode as cp1252 and die,
    # leaving proc.stdout None; json.loads(None) then raised a cryptic TypeError.
    # The guard turns that into a clear RuntimeError. Construct the provider
    # client directly (its __init__ is trivial) and monkeypatch subprocess.run.
    chat = harness.ProviderClient("claude-code", "some-model", 0.0)

    class _FakeProc:
        returncode = 0
        stdout = None
        stderr = "boom"

    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **k: _FakeProc())

    with pytest.raises(RuntimeError) as exc:
        chat._claude_code([{"role": "user", "content": "hi"}])
    assert "no readable stdout" in str(exc.value)


# --- Milestone 1: tool-enabled agent mode ----------------------------------

def test_parser_agent_mode_flags(tmp_path):
    args = harness.build_parser().parse_args([
        "--mode", "agent",
        "--approval", "never",
        "--workspace", str(tmp_path),
        "--max-tool-turns", "3",
        "--provider", "echo",
        "-q", "hello",
    ])
    assert args.mode == "agent"
    assert args.approval == "never"
    assert args.workspace == tmp_path
    assert args.max_tool_turns == 3


def test_parser_defaults_to_agent_mode(monkeypatch):
    monkeypatch.delenv("JUNO_MODE", raising=False)
    args = harness.build_parser().parse_args([])
    assert args.mode == "agent"


def test_juno_mode_env_can_still_force_chat(monkeypatch):
    monkeypatch.setenv("JUNO_MODE", "chat")
    args = harness.build_parser().parse_args([])
    assert args.mode == "chat"


def test_initial_messages_agent_mode_allows_tools(tmp_path):
    memory = harness.MemoryStore(tmp_path)
    memory.load()
    messages = harness.build_initial_messages("You are Juno.", memory, mode="agent")
    content = messages[0]["content"]
    assert "You may request tools" in content
    assert "cannot inspect files" not in content


def test_tool_registry_exports_openai_and_anthropic_schemas():
    registry = harness.ToolRegistry()
    registry.register(harness.ToolSpec(
        name="demo_tool",
        description="Demo",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        risk=harness.ToolRisk.SAFE_READ,
        handler=lambda args, ctx: harness.ToolResult(call_id="c1", name="demo_tool", ok=True, content="ok"),
    ))
    assert registry.openai_schemas()[0]["function"]["name"] == "demo_tool"
    assert registry.anthropic_schemas()[0]["name"] == "demo_tool"
    with pytest.raises(ValueError):
        registry.register(registry.get("demo_tool"))


def test_agent_mode_fake_provider_executes_read_file_tool(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("hello from tool", encoding="utf-8")
    args = argparse.Namespace(
        provider="echo",
        model="echo",
        temperature=0.2,
        base_url=None,
        effort=None,
        system="You are a test harness.",
        home=tmp_path / "home",
        ui="plain",
        mode="agent",
        approval="never",
        workspace=workspace,
        max_tool_turns=4,
    )
    memory = harness.MemoryStore(args.home)
    memory.load()
    sessions = harness.SessionStore(args.home / "sessions.db")
    rt = harness.HarnessRuntime.create(args, memory, sessions)
    result = rt.submit_user_message("please read note.txt")
    assert "hello from tool" in result.text
    assert any(m.get("role") == "tool" and "hello from tool" in m.get("content", "") for m in rt.messages)
    search = sessions.search("hello")
    assert search
    rt.close()

def test_agent_prompt_is_provider_aware_for_claude_code():
    messages = harness.build_initial_messages("base", None, mode="agent", provider="claude-code")
    content = messages[0]["content"]

    assert "Claude Code native tool contract" in content
    assert "pseudo function calls" in content
    assert "You may request tools" not in content


def test_switch_provider_rebuilds_capability_prompt(tmp_path):
    rt = _make_runtime(tmp_path, provider="echo")
    rt.args.mode = "agent"
    rt.messages = harness.build_initial_messages(rt.args.system, rt.memory, mode="agent", provider="echo")
    assert "You may request tools" in rt.messages[0]["content"]

    msg = rt.switch_provider("claude-code", "opus")

    assert "switched to provider=claude-code model=opus" in msg
    assert "Claude Code native tool contract" in rt.messages[0]["content"]
    assert "You may request tools" not in rt.messages[0]["content"]


def test_prompt_toolkit_prompt_message_is_dynamic_callable(tmp_path, monkeypatch):
    rt = _make_runtime(tmp_path, provider="echo")
    ui = harness.PromptToolkitUI.__new__(harness.PromptToolkitUI)
    ui.runtime = rt
    called = {}

    class FakeSession:
        async def prompt_async(self, message):
            called["callable"] = callable(message)
            called["rendered"] = message() if callable(message) else message

    import contextlib
    import prompt_toolkit.patch_stdout as patch_stdout_mod
    monkeypatch.setattr(patch_stdout_mod, "patch_stdout", lambda raw=True: contextlib.nullcontext())
    ui.session = FakeSession()
    ui._print_banner = lambda: None
    import asyncio
    asyncio.run(ui._run_main())

    assert called["callable"] is True
    assert called["rendered"] == [("class:prompt.star", "✦ "), ("class:prompt", "echo/echo › ")]

def test_agent_mode_unsupported_provider_refuses_tool_claims(tmp_path):
    rt = _make_runtime(tmp_path, provider="anthropic")
    rt.args.mode = "agent"
    rt.messages = harness.build_initial_messages(rt.args.system, rt.memory, mode="agent", provider="anthropic")

    result = rt.submit_user_message("read harness.py")

    assert "not wired to Juno's tool loop" in result.text
    assert "--provider echo" in result.text or "OpenAI/OpenRouter" in result.text

def test_claude_code_agent_mode_uses_native_read_tools(monkeypatch, tmp_path):
    captured = {}
    workspace = tmp_path / "ws"
    workspace.mkdir()
    chat = harness.ProviderClient(
        "claude-code",
        "opus",
        0.0,
        effort="high",
        mode="agent",
        approval="never",
        workspace=workspace,
        max_tool_turns=5,
    )

    class _FakeProc:
        returncode = 0
        stdout = '{"result": "read ok", "usage": {"input_tokens": 3, "output_tokens": 2}}'
        stderr = ""

    def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(harness.subprocess, "run", _fake_run)

    result = chat._claude_code([{"role": "user", "content": "read note.txt"}])

    assert result.text == "read ok"
    assert captured["cmd"][captured["cmd"].index("--max-turns") + 1] == "5"
    assert captured["cmd"][captured["cmd"].index("--tools") + 1] == "Read,Grep,Glob,LS"
    assert "--add-dir" in captured["cmd"]
    assert str(workspace) in captured["cmd"]
    assert captured["kwargs"]["cwd"] == str(workspace)
    assert "--effort" in captured["cmd"]


def test_claude_code_agent_mode_is_not_refused_by_runtime(tmp_path, monkeypatch):
    rt = _make_runtime(tmp_path, provider="claude-code")
    rt.args.mode = "agent"
    rt.args.approval = "never"
    rt.args.workspace = tmp_path
    rt.args.max_tool_turns = 4
    rt.client = harness.ProviderClient("claude-code", "opus", 0.0, mode="agent", approval="never", workspace=tmp_path, max_tool_turns=4)

    def _fake_complete(messages, tools=None):
        assert tools is None
        assert messages[-1]["role"] == "user"
        return harness.ChatResult(text="native claude tools ran", input_tokens=1, output_tokens=2)

    rt.client.complete = _fake_complete

    result = rt.submit_user_message("read harness.py")

    assert result.text == "native claude tools ran"
    assert "not wired" not in result.text
    assert rt.messages[-1]["content"] == "native claude tools ran"
    rt.close()

def test_claude_code_chat_mode_uses_system_prompt_not_append(monkeypatch):
    captured = {}
    chat = harness.ProviderClient("claude-code", "opus", 0.0, mode="chat")

    class _FakeProc:
        returncode = 0
        stdout = '{"result": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}}'
        stderr = ""

    def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(harness.subprocess, "run", _fake_run)
    chat._claude_code([{"role": "system", "content": "chat only"}, {"role": "user", "content": "hi"}])

    assert "--system-prompt" in captured["cmd"]
    assert "--append-system-prompt" not in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--tools") + 1] == ""

def test_banner_has_blank_after_title_not_after_help(tmp_path, capsys):
    rt = _make_runtime(tmp_path)
    harness.print_banner(rt, color=False)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("✦")
    assert "J U N O" in lines[0]
    assert lines[1] == ""
    assert lines[-1].startswith("Type /help")
    rt.close()


def test_prompt_turn_output_uses_in_terminal_and_compact_spacing(tmp_path, monkeypatch):
    """Prompt-mode turn output should be printed above the live prompt and avoid
    the extra blank line after the usage line that visually separates the next box.
    """
    import asyncio
    import contextlib
    import importlib

    rt = _make_runtime(tmp_path)
    ui = harness.PromptToolkitUI(rt)
    monkeypatch.setenv("JUNO_WAIT_ANIMATION", "0")

    events = []

    @contextlib.asynccontextmanager
    async def _fake_in_terminal(*args, **kwargs):
        events.append("enter")
        yield
        events.append("exit")

    ptrit = importlib.import_module("prompt_toolkit.application.run_in_terminal")
    monkeypatch.setattr(ptrit, "in_terminal", _fake_in_terminal)

    asyncio.run(ui._handle_turn("hello"))
    assert events == ["enter", "exit", "enter", "exit"]
    rt.close()

