from __future__ import annotations

import colorsys
import os
import random
import shutil
import sys
import textwrap
import threading
import time
from dataclasses import dataclass

# "Juno" brand palette: galaxy black -> deep blue -> starlight white.
#
# Pass 1 of the JunoTheme refactor (req 7a): the palette + sigil/banner strings
# now live as field defaults on a frozen `JunoTheme` dataclass with a single
# `DEFAULT_THEME` instance. The legacy module-level names are rebound to that
# instance's fields so every existing call site keeps working unchanged AND
# identity holds (`PURPLE is DEFAULT_THEME.purple`) — the theme is frozen and
# the default is a singleton, so the rebind names the SAME `str` object.
# Pass 2 (threading `theme=` through the formatters) is a SEPARATE later task;
# nothing here touches a formatter signature or call site.
@dataclass(frozen=True)
class JunoTheme:
    name: str = "Juno"
    banner: str = "Juno · galaxy console"
    sigil: str = "✦    J U N O    ✧    AGENT-HARNESS    ✦"
    reset: str = "\033[0m"
    purple: str = "\033[38;5;111m"
    purple_dim: str = "\033[38;5;103m"
    purple_bold: str = "\033[1;38;5;189m"
    purple_italic: str = "\033[3;38;5;153m"
    lavender: str = "\033[38;5;231m"
    red_bold: str = "\033[31;1m"


DEFAULT_THEME = JunoTheme()

# Rebind the legacy names to the SAME objects on DEFAULT_THEME (identity, not a
# copy). These must precede `TIERS` below, which references PURPLE/PURPLE_BOLD/
# PURPLE_DIM, so the rebind is already in effect when `TIERS` is built.
JUNO_NAME = DEFAULT_THEME.name
JUNO_BANNER = DEFAULT_THEME.banner
JUNO_SIGIL = DEFAULT_THEME.sigil
RESET = DEFAULT_THEME.reset
PURPLE = DEFAULT_THEME.purple
PURPLE_DIM = DEFAULT_THEME.purple_dim
PURPLE_BOLD = DEFAULT_THEME.purple_bold
PURPLE_ITALIC = DEFAULT_THEME.purple_italic
LAVENDER = DEFAULT_THEME.lavender
RED_BOLD = DEFAULT_THEME.red_bold

# --- Rainbow "boost" text (req 6) -----------------------------------------
# Claude-Code-style spectrum sweep. Sits next to the ANSI palette so the SGR
# helpers live together. Gating mirrors the rest of the file: color is keyed
# off ``sys.stdout.isatty()`` (see ``run_once`` at the top of the UI layer and
# the run-loop animate gate). ``truecolor`` decides 24-bit vs 256-color
# degrade; both honor ``NO_COLOR``.

# 256-color degrade cycle when 24-bit truecolor is unavailable. These are the
# raw color numbers from the existing PURPLE / LAVENDER / PURPLE_DIM palette
# so the boost text stays on-brand on legacy terminals.
_RAINBOW_256 = (111, 189, 231, 153, 103)


def _truecolor_supported() -> bool:
    """24-bit color if the terminal advertises it via COLORTERM."""
    return os.environ.get("COLORTERM", "") in ("truecolor", "24bit")


# Module-level truecolor gate (mirrors how `color` is decided elsewhere from
# the environment). Recomputed lazily inside `rainbow` so tests can monkeypatch
# COLORTERM, but exposed here as the spec's named symbol.
truecolor = _truecolor_supported()


def use_rainbow() -> bool:
    """Gate for animated rainbow output.

    Matches the file's color convention: a real TTY, not suppressed by
    NO_COLOR. (PromptToolkitUI has no ``color_enabled`` attribute; the spec's
    ``self.color_enabled`` does not exist, so we gate on ``isatty()`` directly
    like ``run_once`` and the run-loop animate check do.)
    """
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


# Back-compat module symbol named by the spec. Evaluated at import time; call
# ``use_rainbow()`` for a live decision.
USE_RAINBOW = use_rainbow()


def _rgb(h: float) -> str:
    """Map a hue in [0,1) to a ``"r;g;b"`` truecolor triple."""
    r, g, b = colorsys.hls_to_rgb(h % 1.0, 0.6, 1.0)
    return f"{int(r * 255)};{int(g * 255)};{int(b * 255)}"


def rainbow(text: str, *, phase: float = 0.0, spread: float = 0.06, color: bool = True) -> str:
    """Color each character along a sweeping spectrum (Claude-Code boost style).

    - color disabled -> returns ``text`` unchanged, with NO SGR at all.
    - truecolor terminal -> 24-bit ``38;2;r;g;b`` per character.
    - otherwise -> degrades to a 256-color ``38;5;N`` cycle from the palette.
    Always ends with RESET when any color is emitted. Spaces are passed through
    uncolored so word gaps stay clean.
    """
    if not color or os.environ.get("NO_COLOR"):
        return text
    out: list[str] = []
    use_24bit = _truecolor_supported()
    for i, ch in enumerate(text):
        if ch == " ":
            out.append(ch)
            continue
        if use_24bit:
            out.append(f"\033[38;2;{_rgb((phase + i * spread) % 1.0)}m{ch}")
        else:
            n = _RAINBOW_256[i % len(_RAINBOW_256)]
            out.append(f"\033[38;5;{n}m{ch}")
    return "".join(out) + RESET


class RainbowSweep:
    """Animated one-line rainbow sweep for boost activation.

    Repaints a single line in place via carriage-return + erase-to-EOL
    (``\\r\\033[K``) — it NEVER space-pads to clear (space-pad caused the
    white-bar artifact). On ``.stop()`` it prints the final static
    ``rainbow(text)`` plus a newline. Animation only runs on a real TTY; off a
    TTY ``.stop()`` still prints the final static line so output is consistent.
    Usable as a context manager.
    """

    def __init__(self, text: str, *, cycles: int = 2, fps: int = 20, color: bool = True) -> None:
        self.text = text
        self.cycles = cycles
        self.fps = max(1, fps)
        self.color = color
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _animate(self) -> None:
        frames = max(1, int(self.cycles * self.fps))
        delay = 1.0 / self.fps
        for f in range(frames):
            if self._stop.is_set():
                break
            phase = (f / self.fps) % 1.0
            # Erase-to-EOL, never space-pad.
            sys.stdout.write("\r\033[K" + rainbow(self.text, phase=phase, color=self.color))
            sys.stdout.flush()
            self._stop.wait(delay)

    def start(self) -> "RainbowSweep":
        if self.color and sys.stdout.isatty():
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        # Final static frame: erase the line then print the settled gradient.
        if self.color and sys.stdout.isatty():
            sys.stdout.write("\r\033[K")
        sys.stdout.write(rainbow(self.text, color=self.color) + "\n")
        sys.stdout.flush()

    def __enter__(self) -> "RainbowSweep":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


# --- Model strength tiers + icons (req 5) ---------------------------------
# Each tier is a fixed-width 5-cell bar so column widths never shift.
# (glyph_bar, ascii_bar, ansi_color)
TIERS: dict[str, tuple[str, str, str]] = {
    "ultracode": ("\u25c6\u25c6\u25c6\u25c6\u25c6", "[#####]", PURPLE_BOLD),
    "flagship":  ("\u25c6\u25c6\u25c6\u25c6\u25c7", "[####.]", PURPLE),
    "strong":    ("\u25c6\u25c6\u25c6\u25c7\u25c7", "[###..]", PURPLE),
    "standard":  ("\u25c6\u25c6\u25c7\u25c7\u25c7", "[##...]", PURPLE_DIM),
    "light":     ("\u25c6\u25c7\u25c7\u25c7\u25c7", "[#....]", PURPLE_DIM),
}

# Icon per route: (glyph, ascii). Keyed by a derived route token.
STRENGTH_ICONS: dict[str, tuple[str, str]] = {
    "codex":       ("\u232c", "Cx"),  # OpenAI / Codex
    "gpt5":        ("\u2738", "G5"),  # OpenAI gpt5
    "claude-code": ("\u2748", "Cc"),  # Anthropic claude-code
    "sonnet":      ("\u2736", "So"),  # sonnet
    "opus":        ("\u2726", "Op"),  # opus
    "anthropic":   ("\u2749", "An"),  # Anthropic API
    "generic":     ("\u2022", "*"),   # fallback
}


def _route_token(opt: "ModelOption") -> str:
    """Map a ModelOption to an icon key for STRENGTH_ICONS."""
    model = (opt.model or "").lower()
    if opt.provider == "openai":
        return "gpt5" if "gpt-5" in model or opt.key == "gpt5" else "codex"
    if opt.provider == "claude-code":
        if "opus" in model:
            return "opus"
        if "sonnet" in model:
            return "sonnet"
        return "claude-code"
    if opt.provider == "anthropic":
        return "anthropic"
    return "generic"


def tier_for(opt: "ModelOption") -> str:
    """Return the strength tier for a model option.

    Honors an explicit `opt.tier` override; otherwise derives a sensible tier.
    The single strongest coding route per provider gets `ultracode`.
    """
    if opt.tier and opt.tier != "standard":
        return opt.tier
    model = (opt.model or "").lower()
    if opt.provider == "openai":
        return "ultracode" if opt.key == "codex" else "flagship"
    if opt.provider == "claude-code":
        if "opus" in model or opt.key == "opus":
            return "ultracode"
        if "sonnet" in model:
            return "strong"
        return "ultracode" if opt.key == "claude" else "strong"
    if opt.provider == "anthropic":
        return "flagship"
    if opt.provider == "echo":
        return "light"
    return "standard"


def strength_icon(opt: "ModelOption", *, color: bool) -> str:
    """Return the icon glyph (or ASCII fallback) for an option."""
    if opt.icon:
        return opt.icon
    glyph, ascii_ = STRENGTH_ICONS.get(_route_token(opt), STRENGTH_ICONS["generic"])
    return glyph if color else ascii_


def strength_bar(tier: str, *, color: bool) -> str:
    """Return the constant-width strength bar for a tier."""
    glyph, ascii_, _ = TIERS.get(tier, TIERS["standard"])
    return glyph if color else ascii_


def format_model_row(opt: "ModelOption", *, color: bool) -> str:
    """Render a fixed-width, colored picker row for a model option.

    Widths are computed on the PLAIN text (before SGR is applied) so escape
    codes never count toward column width (same discipline as the white-bar
    pad fix). The returned string is display-only; the picker `key` is never
    altered by this decoration.
    """
    tier = tier_for(opt)
    is_ultra = tier == "ultracode"
    bar_color = TIERS.get(tier, TIERS["standard"])[2]
    icon = strength_icon(opt, color=color)
    bar = strength_bar(tier, color=color)
    route = f"{opt.provider}/{opt.model}"

    # Fixed-width plain cells.
    sigil = ("\u26a1" if color else "!") if is_ultra else " "
    icon_cell = f"{icon:<2}"
    route_cell = f"{route:<22}"
    if is_ultra:
        badge = "ULTRACODE" if color else "*ULTRA*"
    else:
        badge = ""

    if not color:
        # Plain text, no SGR.
        return f"{sigil} {icon_cell} {route_cell} {bar} {badge}".rstrip()

    colored_bar = colorize(bar, bar_color, color=color)
    if is_ultra:
        colored_badge = colorize(badge, LAVENDER, color=color)
        colored_sigil = colorize(sigil, PURPLE_BOLD, color=color)
    else:
        colored_badge = badge
        colored_sigil = sigil
    return f"{colored_sigil} {icon_cell} {route_cell} {colored_bar} {colored_badge}".rstrip()

THINKING_MESSAGES = [
    "charting a clean orbit through context",
    "reading the white-star telemetry",
    "tuning the nebula signal",
    "polishing the response crystal",
    "consulting the indigo star chart",
    "brewing a quiet nebula of tokens",
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


def format_box(text: str, *, title: str, subtitle: str = "", color: bool = True, width: int | None = None, theme: JunoTheme = DEFAULT_THEME) -> str:
    cols = width or terminal_width()
    inner = max(44, cols - 4)
    border = "─" * inner
    header = f" ✦ {title}"
    if subtitle:
        header += f" · {subtitle}"
    header += "  ✧"
    header = header[:inner]
    body = wrap_text_block(text.rstrip(), inner)
    rendered = "\n".join([
        f"╭{border[:-2]}✦─╮",
        f"│{header.ljust(inner)}│",
        f"├{border}┤",
        *(f"│{line[:inner].ljust(inner)}│" for line in body),
        f"╰─✧{border[2:]}╯",
    ])
    return colorize(rendered, theme.purple, color=color)


def format_assistant_response(text: str, *, provider: str, model: str, color: bool = True, width: int | None = None) -> str:
    return format_box(text, title="juno", subtitle=f"{provider}/{model}", color=color, width=width)


def format_command_output(text: str, *, color: bool = True, theme: JunoTheme = DEFAULT_THEME) -> str:
    if "\n" in text:
        return format_box(text, title="juno", subtitle="command output", color=color, theme=theme)
    return colorize(f"✧ juno · {text}", theme.purple_dim, color=color)


def format_thinking(provider: str = "", model: str = "", *, color: bool = True, theme: JunoTheme = DEFAULT_THEME) -> str:
    """Model-free waiting label. Provider/model kept in signature for
    backward compatibility but intentionally NOT rendered — the prompt is
    always the same nebula-branded phrase."""
    return colorize("✦ nebula signal…", theme.purple_italic, color=color)


def format_waiting_static(*, color: bool = True, theme: JunoTheme = DEFAULT_THEME) -> str:
    """Deterministic, ANSI-free-friendly waiting line for non-TTY / plain paths."""
    return colorize("✦ nebula signal . . .", theme.purple_italic, color=color)


# ---------------------------------------------------------------------------
# Waiting state (req 3+4)
#
# The old daemon-thread `\r` repaint pad-cleared with spaces (`.ljust(cols-1)`),
# and on conhost `get_terminal_size` is off-by-one vs usable width, so the
# styled trailing cell rendered as a bright white block. The animation now
# lives in the prompt_toolkit bottom toolbar (clock-driven dots, repainted by
# the persistent app's refresh_interval — see `_bottom_toolbar` / `_run_main`),
# which the renderer erases with its own cell-diff (never pads to `cols`) — so
# the white bar cannot recur and the status bar stays pinned (no full-screen flip).
#
# `format_waiting_static` stays for the plain / non-TTY / pipe path.
# ---------------------------------------------------------------------------


def format_usage_line(input_tokens: int, output_tokens: int, total_in: int | None = None, total_out: int | None = None, *, session_id: str | None = None, color: bool = True, theme: JunoTheme = DEFAULT_THEME) -> str:
    parts = [f"turn {input_tokens}/{output_tokens} tok"]
    if total_in is not None and total_out is not None:
        parts.append(f"total {total_in}/{total_out}")
    if session_id:
        parts.append(f"session {session_id}")
    return colorize("   ✧ " + " · ".join(parts), theme.purple_dim, color=color)


def format_error(text: str, *, color: bool = True, theme: JunoTheme = DEFAULT_THEME) -> str:
    return colorize(text, theme.red_bold, color=color)
