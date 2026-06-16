from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from .models import model_aliases, model_catalog
from .runtime import HarnessRuntime, MEMORY_TARGETS, PROVIDER_WORDS

from .theme import (
    DEFAULT_THEME, JunoTheme, LAVENDER, PURPLE, PURPLE_BOLD, PURPLE_DIM,
    RESET, colorize, format_assistant_response,
    format_command_output, format_error, format_model_row, format_usage_line,
    format_waiting_static,
)

def print_banner(runtime: HarnessRuntime, *, color: bool, theme: JunoTheme = DEFAULT_THEME) -> None:
    # Give the title a little more visual weight, then separate it from metadata.
    # Do NOT print a trailing blank after the help hint; prompt_toolkit already
    # moves to the prompt line, and the extra newline made startup feel loose.
    print(colorize(theme.sigil, theme.purple_bold, color=color))
    print()
    print(f"{'home':<8} {runtime.args.home}")
    print(f"{'session':<8} {runtime.session_id}")
    print(
        f"{'model':<8} {runtime.args.provider}/{runtime.args.model} "
        f"· memory={'on' if runtime.memory is not None else 'off'}"
    )
    print("Type /help for commands. Ctrl-D exits. Esc+Enter inserts a newline.")


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
                    line = input(f"{rt.prompt_label()}").strip()
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
                    if not res.should_continue:
                        break
                    continue
                try:
                    # Plain / non-TTY path: one deterministic static line, no
                    # animation, no ANSI (pipes / CI stay stable).
                    print()
                    print(format_waiting_static(color=False))
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
                print()
                print(format_usage_line(
                    result.input_tokens,
                    result.output_tokens,
                    rt.total_in,
                    rt.total_out,
                    color=False,
                ))
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
        # Waiting-state animation flags (req 3+4): driven by an invalidate/tick
        # daemon while the provider call runs off the UI thread.
        self._awaiting = False
        self._tick = 0

    def _build_session(self):
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style

        history_path = Path(self.runtime.args.home)
        history_path.mkdir(parents=True, exist_ok=True)

        # Galaxy theme: starlight prompt/toolbar over deep indigo space.
        style = Style.from_dict({
            "prompt": "#f8fbff bold",
            "prompt.star": "#7dd3fc bold",
            "bottom-toolbar": "#f8fbff bg:#111827",
            "bottom-toolbar.key": "#93c5fd bold",
            "bottom-toolbar.star": "#bfdbfe bold",
            "banner": "#e0f2fe bold",
            "banner.sub": "#93c5fd",
            "meta": "#bfdbfe",
            "thinking": "italic #7dd3fc",
            "error": "ansired bold",
            "assistant": "#f8fbff",
            "assistant.border": "#60a5fa",
        })

        return PromptSession(
            history=FileHistory(str(history_path / "history.txt")),
            completer=self._build_completer(),
            complete_while_typing=True,
            bottom_toolbar=self._bottom_toolbar,
            style=style,
            multiline=True,
            # Drives the bottom-toolbar animation on the app's own timer (copied
            # onto the internal app at run). Replaces the old ticker thread.
            refresh_interval=0.25,
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
            "/skills": None,
            "/menu": None,
            "/reset": None,
            "/clear": None,
            "/exit": None,
        })

    def _build_keybindings(self):
        # Design B-persistent: Enter NEVER accepts/exits. A single prompt_async
        # stays alive for the whole session; each Enter spawns a background turn
        # instead of resolving the app future, so the toolbar keeps animating via
        # refresh_interval through the blocking provider call. Our explicit
        # enter/c-d/c-c override the prompt's built-in bindings (user bindings win
        # for the same key), so the default accept/exit path is never reached.
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            buf = event.current_buffer
            if self._awaiting:
                return  # a turn is in flight; ignore submit
            text = buf.text
            if not text.strip():
                return
            # Replicate the accept_handler's append_to_history side effect.
            buf.history.append_string(text)
            buf.reset()  # clear the input line (no accept/exit)
            event.app.create_background_task(self._handle_turn(text))

        @kb.add("escape", "enter")  # Alt/Meta-Enter inserts a newline
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add("c-d")  # Ctrl-D on an EMPTY buffer exits the session
        def _(event):
            if not event.current_buffer.text:
                event.app.exit(result=None)

        @kb.add("c-c")  # Ctrl-C clears the line, never exits
        def _(event):
            event.current_buffer.reset()

        return kb

    def _bottom_toolbar(self):
        provider = self.runtime.args.provider
        model = self.runtime.args.model
        if self._awaiting:
            # Waiting state: same toolbar callback prompt_toolkit already polls,
            # so the bar stays pinned (no full-screen flip). Dots are clock-driven
            # so they advance purely from the app's own refresh_interval repaints
            # (no ticker thread). _tick is retained for tests but unused here.
            dots = ". " * (1 + int(time.monotonic() / 0.2) % 3)
            return self._pt_html(
                f"\u2726 <b>{provider}/{model}</b>  "
                f"\u2727 nebula signal {dots}"
            )
        return self._pt_html(
            f"✦ <b>{provider}/{model}</b>  "
            f"memory={'on' if self.runtime.memory is not None else 'off'}  "
            f"session={self.runtime.session_id}  "
            f"tokens in/out={self.runtime.total_in}/{self.runtime.total_out}  "
            f"✧ <b>/help</b>"
        )

    def _prompt_fragments(self):
        # Styled prompt label using the 'prompt' style class.
        return [("class:prompt.star", "✦ "), ("class:prompt", self.runtime.prompt_label())]

    def _print_banner(self) -> None:
        print_banner(self.runtime, color=True)

    async def _print_above_prompt(self, *lines: str) -> None:
        """Print durable output above the live prompt_toolkit prompt.

        Direct prints while the app is rendering can be swallowed or repainted
        away by prompt_toolkit. `in_terminal()` temporarily yields the terminal to
        ordinary stdout, then restores the prompt so chat turns stay visible.
        """
        from prompt_toolkit.application.run_in_terminal import in_terminal

        async with in_terminal():
            for line in lines:
                print(line)

    async def _run_model_picker(self) -> str:
        from prompt_toolkit.shortcuts import radiolist_dialog
        from prompt_toolkit.styles import Style
        from prompt_toolkit.application.run_in_terminal import in_terminal
        from prompt_toolkit.formatted_text import ANSI

        current = (self.runtime.args.provider, self.runtime.args.model)
        # default_model tracking will be populated by HarnessConfig (Task 6);
        # for now it is always None so only "(current)" appears.
        _default = getattr(self.runtime, "_config_default_model", None) or (None, None)

        # Group by section for Hermes-style segmented display.
        options = sorted(model_catalog(), key=lambda o: (
            # push the current section to the top
            0 if o.section == next(
                (x.section for x in model_catalog()
                 if (x.provider, x.model) == current), ""
            ) else 1,
            o.section,
            o.key,
        ))
        values = []
        last_section = None
        for option in options:
            if option.section != last_section:
                # Insert a non-selectable section header row.
                # radiolist_dialog treats every row as selectable, so we use
                # a disabled-looking prefix to signal a section break.
                values.append((
                    f"__section__{option.section}",
                    ANSI(f"  {option.section}"),
                ))
                last_section = option.section
            tags = []
            if (option.provider, option.model) == current:
                tags.append("current")
            if (option.provider, option.model) == _default:
                tags.append("default")
            badge = ""
            if tags:
                badge = f"  ({', '.join(tags)})"
            row = format_model_row(option, color=True)
            values.append((
                option.key,
                ANSI(f"  {row}  —  {option.label}{badge}"),
            ))
        juno_picker_style = Style.from_dict({
            "dialog":              "bg:#1c1b29",
            "dialog frame.border": "#8a7fb8",
            "dialog frame.label":  "bg:#1c1b29 #c9c2e8",
            "dialog.body":         "bg:#1c1b29 #d6d2e8",
            "radio":               "#d6d2e8",
            "radio-selected":      "#c9c2e8",
            "radio-checked":       "bold #b9a8ff",
            "button":              "#d6d2e8",
            "button.focused":      "bg:#8a7fb8 #1c1b29",
        })
        dialog = radiolist_dialog(
            title="Juno model selection",
            text="Choose a model route. Raw slugs still work with /model <provider> <model>.",
            values=values,
            style=juno_picker_style,
        )
        # radiolist_dialog builds its app full_screen=True (alt-screen flip);
        # force inline so the picker renders in the primary buffer with no flip.
        dialog.full_screen = False
        # Suspend + erase the outer persistent app, detach its input, run the
        # nested dialog app, then restore the outer app on exit (the validated
        # in_terminal recipe — see spec §3).
        async with in_terminal():
            chosen = await dialog.run_async()
        if not chosen or chosen.startswith("__section__"):
            return "model selection cancelled"
        option = model_aliases()[chosen]
        return self.runtime.switch_provider(option.provider, option.model)

    def run(self) -> int:
        """Synchronous entrypoint. The whole session is one long-lived async
        prompt (Design B-persistent): a single ``prompt_async`` stays alive for
        the entire session so the bottom-toolbar animation can repaint via the
        app's own ``refresh_interval`` while a blocking provider call runs in an
        executor. External callers/tests keep this plain ``ui.run()`` signature."""
        return asyncio.run(self._run_main())

    async def _run_main(self) -> int:
        from prompt_toolkit.patch_stdout import patch_stdout

        rt = self.runtime
        if self.session is None:
            self.session = self._build_session()  # refresh_interval=0.25 set here
        self._print_banner()
        try:
            with patch_stdout(raw=True):
                # ONE call for the whole session. Enter never accepts, so this
                # returns only when app.exit() is called (Ctrl-D on an empty
                # buffer, or a /command with should_continue False). Each Enter
                # spawns a background turn; the app keeps running between turns
                # and the toolbar animates on its own refresh_interval timer.
                await self.session.prompt_async(self._prompt_fragments)
        finally:
            rt.close()
        return 0

    async def _handle_turn(self, text: str) -> None:
        """Handle one submission as a background task on the app loop. Must never
        raise (a raising background task hits the loop exception handler), so the
        broad ``except`` keeps the UI recoverable."""
        from prompt_toolkit.application import get_app

        rt = self.runtime
        line = text.strip()
        if not line:
            return
        try:
            if line in {"/model", "/provider"}:
                try:
                    picked = await self._run_model_picker()
                except Exception as exc:  # noqa: BLE001 - fall back to text command
                    await self._print_above_prompt(
                        format_error(f"[model picker error] {exc}", color=True),
                        format_command_output(rt.handle_command(line).output, color=True),
                    )
                    return
                await self._print_above_prompt("", format_command_output(picked, color=True))
                return
            if line.startswith("/"):
                res = rt.handle_command(line)
                if res.output:
                    await self._print_above_prompt("", format_command_output(res.output, color=True))
                if not res.should_continue:
                    get_app().exit(result=None)  # /exit etc.
                return
            # Normal turn — block in an executor so the app loop keeps repainting
            # (and the toolbar animating) via refresh_interval. Gate the static
            # waiting line behind isatty() + JUNO_WAIT_ANIMATION; when animating,
            # the live toolbar carries the waiting state instead.
            animate = (
                sys.stdout.isatty()
                and os.environ.get("JUNO_WAIT_ANIMATION") != "0"
            )
            self._awaiting = True
            if animate:
                # Deterministically repaint the live toolbar while the blocking
                # provider call runs in the executor. refresh_interval alone is
                # flaky, so drive app.invalidate() from an explicit async ticker.
                app = get_app()

                async def _spin():
                    while self._awaiting:
                        app.invalidate()
                        await asyncio.sleep(0.2)

                spin = asyncio.ensure_future(_spin())
            else:
                spin = None
                await self._print_above_prompt(format_waiting_static(color=True))
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, rt.submit_user_message, line)
            except Exception as exc:  # noqa: BLE001 - keep UI recoverable
                await self._print_above_prompt(format_error(f"[provider error] {exc}", color=True))
                return
            finally:
                self._awaiting = False
                if spin is not None:
                    spin.cancel()
                    try:
                        await spin
                    except asyncio.CancelledError:
                        pass
            await self._print_above_prompt(
                "",
                format_assistant_response(
                    result.text,
                    provider=rt.args.provider,
                    model=rt.args.model,
                    color=True,
                ),
                "",
                format_usage_line(
                    result.input_tokens,
                    result.output_tokens,
                    rt.total_in,
                    rt.total_out,
                    color=True,
                ),
            )
        finally:
            self._awaiting = False


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


def run_repl(args: argparse.Namespace, client: ProviderClient, memory: MemoryStore | None, sessions: SessionStore, *, config: HarnessConfig | None = None) -> int:
    runtime = HarnessRuntime.create(args, memory, sessions, config=config)
    if config is not None:
        runtime._config_default_model = config.default_model()
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
