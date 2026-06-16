#!/usr/bin/env python
"""forge.py - a Fusion-backed agentic coding harness for the agent-loop repo.

Hybrid design (validated by fusion_probe*.py against the live OpenRouter API):
  - A strong, tool-capable DRIVER model runs the file-edit loop cheaply
    (read_file / write_file / edit_file / list_dir / run_command).
    Confirmed: deepseek/deepseek-v4-pro round-trips tool_calls.
  - Fusion is "on tap":
      * consult_fusion(question)  -> a one-shot `openrouter/fusion` deliberation
        (panel of models + judge) for hard design / correctness calls. Slow + costly.
      * /brain fusion             -> promotes fusion to a server-tool ALONGSIDE the
        file tools, so the driver can deliberate mid-loop. Confirmed: custom tools
        coexist with the `openrouter:fusion` server tool in one request.
  - Privacy is enforced ACCOUNT-SIDE on OpenRouter (data policy + provider allowlist),
    so this harness sends NO per-request provider screen.

Quickstart:
  # interactive REPL
  python forge.py
  # one-shot task, auto-approving edits/commands
  python forge.py --task "implement Task 3 from .hermes/plans/2026-06-15_...md" --yolo
  # stronger driver and/or fusion brain
  python forge.py --model anthropic/claude-4.8-opus-20260528 --brain fusion

REPL slash commands:
  /help                 show commands
  /model <slug>         switch the driver model
  /brain fusion|off     toggle fusion deliberation as a driver tool
  /auto on|off          toggle auto-approval of writes/commands (default off)
  /tokens               show token + rough cost usage this session
  /reset                clear the conversation (keeps the system prompt)
  /exit                 quit
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

BASE = "https://openrouter.ai/api/v1"
DEFAULT_DRIVER = os.getenv("FORGE_MODEL", "deepseek/deepseek-v4-pro")
FUSION_MODEL = "openrouter/fusion"
# Fusion panel for /brain mode (server-tool form). Slugs confirmed present 2026-06-16.
FUSION_PANEL = os.getenv(
    "FORGE_FUSION_PANEL",
    "anthropic/claude-opus-4.8,deepseek/deepseek-v4-pro",
).split(",")
FUSION_JUDGE = os.getenv("FORGE_FUSION_JUDGE", "anthropic/claude-opus-4.8")

MAX_STEPS = int(os.getenv("FORGE_MAX_STEPS", "150"))
READ_CAP = 24000          # max chars returned from read_file
CMD_OUT_CAP = 8000        # max chars returned from run_command
CMD_TIMEOUT = int(os.getenv("FORGE_CMD_TIMEOUT", "300"))

# Rough $/Mtok for the session cost estimate (in/out). Override via env if you care.
PRICE_IN = float(os.getenv("FORGE_PRICE_IN", "0.40"))
PRICE_OUT = float(os.getenv("FORGE_PRICE_OUT", "1.60"))

# ---------------------------------------------------------------------------- colors
_NOCOLOR = bool(os.getenv("NO_COLOR")) or not sys.stdout.isatty()


def c(text: str, code: str) -> str:
    if _NOCOLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def dim(t):     return c(t, "38;5;103")
def cyan(t):    return c(t, "38;5;111")
def green(t):   return c(t, "32;1")
def yellow(t):  return c(t, "33;1")
def red(t):     return c(t, "31;1")
def violet(t):  return c(t, "38;5;141")


# ---------------------------------------------------------------------------- key
def load_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    envp = Path(__file__).with_name(".env")
    if envp.exists():
        for line in envp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    print(red("FATAL: OPENROUTER_API_KEY not set (env or .env)."), file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------- tool schemas
FILE_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file. For large files, pass `offset` (1-based start line) and "
                       "`limit` (line count) to read just a range instead of the whole file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path, relative to the repo root or absolute within it."},
            "offset": {"type": "integer", "description": "1-based start line (optional)."},
            "limit": {"type": "integer", "description": "Number of lines to read (optional)."},
        }, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search_text",
        "description": "Regex-search files under the repo and return `path:line: text` matches. Use to LOCATE "
                       "a symbol/string in a large file, then read_file with offset/limit to read that region.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Python regex."},
            "path": {"type": "string", "description": "File or dir to search (default: whole repo)."},
            "glob": {"type": "string", "description": "Filename glob (default '*.py')."},
            "max_results": {"type": "integer", "description": "Cap (default 100)."},
        }, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List the entries of a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path (default '.')."},
        }, "required": []}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content. Use for new files or full rewrites.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace an exact, unique substring in a file. Prefer this for surgical edits. "
                       "`old` must appear exactly once.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old": {"type": "string", "description": "Exact text to find (must be unique in the file)."},
            "new": {"type": "string", "description": "Replacement text."},
        }, "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a shell command in the repo root and return stdout/stderr (truncated). "
                       "Use for tests, e.g. .venv\\Scripts\\python.exe -m pytest <file> -v",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
        }, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "consult_fusion",
        "description": "Ask OpenRouter Fusion (a panel of frontier models + a judge) a hard design or "
                       "correctness question and get back a deliberated answer. SLOW and EXPENSIVE - "
                       "use only for genuinely hard calls, never for routine edits.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "A focused, self-contained question."},
            "context": {"type": "string", "description": "Optional extra context (code, constraints)."},
        }, "required": ["question"]}}},
]

FUSION_SERVER_TOOL = {
    "type": "openrouter:fusion",
    "parameters": {"analysis_models": FUSION_PANEL, "model": FUSION_JUDGE},
}


def system_prompt(root: Path) -> str:
    return (
        f"You are Forge, an autonomous coding agent operating inside the repository at {root} on Windows.\n"
        "Your job: implement the user's task by reading files, editing them, and running commands/tests "
        "until the task is genuinely done and verified.\n\n"
        "Tooling discipline:\n"
        "- ALWAYS read a file (read_file) before you edit it. Never invent file contents.\n"
        "- For large files, use search_text to locate a symbol/string, then read_file with offset/limit "
        "to read just that region - do not read whole large files into context.\n"
        "- Prefer small, surgical edit_file calls over rewriting whole files with write_file.\n"
        "- Use run_command to run tests and checks. The project venv Python is .venv\\Scripts\\python.exe ; "
        "run tests with `.venv\\Scripts\\python.exe -m pytest <file> -v`.\n"
        "- Use consult_fusion ONLY for hard design/correctness decisions - it is slow and costly. "
        "Do not use it for routine edits.\n\n"
        "Rules:\n"
        "- Work in small steps; after each meaningful change, run the relevant tests.\n"
        "- Privacy/provider screening is handled at the OpenRouter ACCOUNT level - do NOT add any "
        "per-request provider block to code you write.\n"
        "- If a command fails or you are blocked, report the failure with its output instead of guessing.\n"
        "- When the task is complete and tests pass, STOP calling tools and reply with a concise summary "
        "of exactly what you changed and the final test result."
    )


# ---------------------------------------------------------------------------- agent
class ForgeAgent:
    def __init__(self, root: Path, model: str, key: str, *, auto: bool, brain: bool):
        self.root = root.resolve()
        self.model = model
        self.key = key
        self.auto = auto
        self.brain = brain
        self.messages: list[dict] = [{"role": "system", "content": system_prompt(self.root)}]
        self.tok_in = 0
        self.tok_out = 0

    # ----- http
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.key}",
            "HTTP-Referer": "https://github.com/angelsystems/agent-loop",
            "X-Title": "juno-forge",
            "Content-Type": "application/json",
        }

    def _post(self, body: dict, timeout: float = 300.0, *, retries: int = 5) -> dict:
        """POST with retry+backoff on transient network errors and 429/5xx responses.

        A single dropped connection (WinError 10054) or rate-limit must NOT kill a long
        multi-step run, so transient failures are retried with exponential backoff.
        """
        last = ""
        for attempt in range(retries):
            try:
                r = httpx.post(f"{BASE}/chat/completions", headers=self._headers(), json=body, timeout=timeout)
            except httpx.TransportError as e:  # ConnectError/timeouts/protocol errors
                last = f"{type(e).__name__}: {e}"
            else:
                if r.status_code not in (408, 409, 429, 500, 502, 503, 504):
                    if not r.is_success:
                        try:
                            err = r.json().get("error", r.text)
                        except Exception:  # noqa: BLE001
                            err = r.text
                        raise RuntimeError(f"OpenRouter {r.status_code}: {err}")
                    data = r.json()
                    u = data.get("usage") or {}
                    self.tok_in += int(u.get("prompt_tokens") or 0)
                    self.tok_out += int(u.get("completion_tokens") or 0)
                    return data
                last = f"OpenRouter {r.status_code}"
            if attempt < retries - 1:
                wait = min(30, 2 * (2 ** attempt))  # 2, 4, 8, 16, 30
                print(dim(f"  ! {last}; retry {attempt + 1}/{retries - 1} in {wait}s"))
                time.sleep(wait)
        raise RuntimeError(f"request failed after {retries} attempts: {last}")

    def _tools(self) -> list[dict]:
        tools = list(FILE_TOOLS)
        if self.brain:
            tools.append(FUSION_SERVER_TOOL)
        return tools

    # ----- safety
    def _safe_path(self, path: str) -> Path:
        p = Path(path)
        p = (self.root / p) if not p.is_absolute() else p
        p = p.resolve()
        try:
            p.relative_to(self.root)
        except ValueError as e:
            raise PermissionError(f"refusing path outside repo root ({self.root}): {p}") from e
        return p

    def _confirm(self, desc: str, preview: str = "") -> bool:
        if self.auto:
            print(dim(f"  auto-approved: {desc}"))
            return True
        if preview:
            print(preview)
        try:
            ans = input(yellow(f"  ? {desc} [y/N] ")).strip().lower()
        except EOFError:
            return False
        return ans in ("y", "yes")

    # ----- tools
    def _t_read_file(self, args) -> str:
        p = self._safe_path(args["path"])
        if not p.exists():
            return f"ERROR: no such file: {p}"
        rel = p.relative_to(self.root)
        text = p.read_text(encoding="utf-8", errors="replace")
        offset, limit = args.get("offset"), args.get("limit")
        if offset is not None or limit is not None:
            lines = text.splitlines()
            total = len(lines)
            start = max(1, int(offset or 1))
            n = int(limit or 200)
            chunk = "\n".join(lines[start - 1:start - 1 + n])
            end = min(total, start - 1 + n)
            if len(chunk) > READ_CAP:
                chunk = chunk[:READ_CAP] + "\n... [truncated; narrow the range]"
            return f"[{rel} lines {start}-{end} of {total}]\n{chunk}"
        if len(text) > READ_CAP:
            nlines = text.count("\n") + 1
            return (f"[{rel}: {len(text)} chars / {nlines} lines - showing first {READ_CAP}; "
                    f"use offset/limit (or search_text) to target a region]\n" + text[:READ_CAP])
        return text

    def _t_search_text(self, args) -> str:
        import re
        try:
            rx = re.compile(args["pattern"])
        except re.error as e:
            return f"ERROR: bad regex: {e}"
        glob = args.get("glob", "*.py")
        target = args.get("path")
        if target:
            tp = self._safe_path(target)
            files = [tp] if tp.is_file() else sorted(tp.rglob(glob))
        else:
            files = sorted(self.root.rglob(glob))
        cap = int(args.get("max_results", 100))
        out: list[str] = []
        for f in files:
            if not f.is_file():
                continue
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        out.append(f"{f.relative_to(self.root)}:{i}: {line.strip()[:160]}")
                        if len(out) >= cap:
                            return "\n".join(out) + "\n... [hit cap; refine pattern/glob]"
            except Exception:  # noqa: BLE001
                continue
        return "\n".join(out) or "(no matches)"

    def _t_list_dir(self, args) -> str:
        p = self._safe_path(args.get("path", "."))
        if not p.is_dir():
            return f"ERROR: not a directory: {p}"
        rows = []
        for child in sorted(p.iterdir()):
            tag = "d" if child.is_dir() else "f"
            size = child.stat().st_size if child.is_file() else ""
            rows.append(f"{tag} {child.name}{(' ' + str(size) + 'b') if size != '' else ''}")
        return "\n".join(rows) or "(empty)"

    def _t_write_file(self, args) -> str:
        p = self._safe_path(args["path"])
        content = args["content"]
        rel = p.relative_to(self.root)
        action = "overwrite" if p.exists() else "create"
        preview = dim(f"  {action} {rel}  ({len(content)} chars)")
        if not self._confirm(f"{action} {rel}?", preview):
            return "DENIED by user."
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: wrote {rel} ({len(content)} chars)."

    def _t_edit_file(self, args) -> str:
        p = self._safe_path(args["path"])
        if not p.exists():
            return f"ERROR: no such file: {p}"
        text = p.read_text(encoding="utf-8")
        old, new = args["old"], args["new"]
        n = text.count(old)
        if n == 0:
            return "ERROR: `old` not found in file (read it again and match exactly)."
        if n > 1:
            return f"ERROR: `old` appears {n} times; make it unique."
        rel = p.relative_to(self.root)
        preview = (dim(f"  edit {rel}") + "\n" +
                   red("  - " + old[:200].replace("\n", "\n  - ")) + "\n" +
                   green("  + " + new[:200].replace("\n", "\n  + ")))
        if not self._confirm(f"apply edit to {rel}?", preview):
            return "DENIED by user."
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"OK: edited {rel}."

    def _t_run_command(self, args) -> str:
        cmd = args["command"]
        if not self._confirm(f"run: {cmd}", dim(f"  $ {cmd}")):
            return "DENIED by user."
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=self.root, capture_output=True, text=True, timeout=CMD_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {CMD_TIMEOUT}s."
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        if len(out) > CMD_OUT_CAP:
            out = out[:CMD_OUT_CAP] + f"\n... [truncated {len(out) - CMD_OUT_CAP} chars]"
        return f"exit={proc.returncode}\n{out}"

    def _t_consult_fusion(self, args) -> str:
        q = args["question"]
        ctx = args.get("context", "")
        prompt = q if not ctx else f"{q}\n\nContext:\n{ctx}"
        print(violet("  [fusion] consulting (panel + judge; slow + costs more)..."))
        data = self._post({
            "model": FUSION_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1200,
            "temperature": 0.2,
        }, timeout=600)
        choices = data.get("choices") or [{}]
        return (choices[0].get("message", {}) or {}).get("content") or "(fusion returned no content)"

    def _dispatch(self, name: str, args: dict) -> str:
        handler = {
            "read_file": self._t_read_file,
            "search_text": self._t_search_text,
            "list_dir": self._t_list_dir,
            "write_file": self._t_write_file,
            "edit_file": self._t_edit_file,
            "run_command": self._t_run_command,
            "consult_fusion": self._t_consult_fusion,
        }.get(name)
        if not handler:
            return f"ERROR: unknown tool {name}"
        try:
            return handler(args)
        except Exception as e:  # noqa: BLE001
            return f"ERROR in {name}: {e}"

    # ----- main turn loop
    def run_turn(self, task: str) -> None:
        self.messages.append({"role": "user", "content": task})
        for step in range(1, MAX_STEPS + 1):
            try:
                data = self._post({
                    "model": self.model,
                    "messages": self.messages,
                    "tools": self._tools(),
                    "tool_choice": "auto",
                    "temperature": 0.1,
                    "max_tokens": 8000,
                })
            except RuntimeError as e:
                print(red(f"\n{e}"))
                return
            msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
            self.messages.append(msg)
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                content = msg.get("content") or "(no content)"
                print("\n" + cyan("forge") + dim(f"  -  step {step}"))
                print(content)
                return

            for tc in tool_calls:
                fn = tc.get("function", {}) or {}
                name = fn.get("name", "")
                try:
                    cargs = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    cargs = {}
                label = cargs.get("path") or cargs.get("command") or cargs.get("question") or ""
                print(dim(f"  -> {name}({str(label)[:80]})"))
                result = self._dispatch(name, cargs)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "name": name,
                    "content": result,
                })
        print(red(f"\nStopped: hit MAX_STEPS={MAX_STEPS} without finishing."))

    # ----- session bookkeeping
    def cost_line(self) -> str:
        cost = self.tok_in / 1e6 * PRICE_IN + self.tok_out / 1e6 * PRICE_OUT
        return (f"tokens in={self.tok_in:,} out={self.tok_out:,}  "
                f"~ ${cost:.3f} (rough; fusion calls cost more)")


# ---------------------------------------------------------------------------- repl
BANNER = r"""
  +-- forge --------------------------------------------+
  |  fusion-backed agentic coding harness               |
  |  driver runs edits - fusion on tap for hard calls   |
  +-----------------------------------------------------+"""

HELP = """commands:
  /help              this help
  /model <slug>      switch driver model (e.g. anthropic/claude-4.8-opus-20260528)
  /brain fusion|off  toggle fusion deliberation as a driver tool
  /auto on|off       toggle auto-approval of writes/commands
  /tokens            session token + cost usage
  /reset             clear conversation (keeps system prompt)
  /exit              quit
anything else is a task for the agent."""


def repl(agent: ForgeAgent) -> None:
    if not _NOCOLOR:
        print(violet(BANNER))
    print(dim(f"  root   : {agent.root}"))
    print(dim(f"  driver : {agent.model}"))
    print(dim(f"  brain  : {'fusion' if agent.brain else 'off'}   auto: {'on' if agent.auto else 'off'}"))
    print(dim("  /help for commands\n"))
    while True:
        try:
            line = input(cyan("forge > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            rest = rest.strip()
            if cmd in ("exit", "quit"):
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "model":
                if rest:
                    agent.model = rest
                    print(dim(f"  driver -> {rest}"))
                else:
                    print(dim(f"  driver = {agent.model}"))
            elif cmd == "brain":
                agent.brain = (rest == "fusion")
                print(dim(f"  brain -> {'fusion' if agent.brain else 'off'}"))
            elif cmd == "auto":
                agent.auto = (rest == "on")
                print(dim(f"  auto -> {'on' if agent.auto else 'off'}"))
            elif cmd == "tokens":
                print(dim("  " + agent.cost_line()))
            elif cmd == "reset":
                agent.messages = agent.messages[:1]
                print(dim("  conversation cleared."))
            else:
                print(red(f"  unknown command /{cmd} (try /help)"))
            continue
        try:
            agent.run_turn(line)
        except KeyboardInterrupt:
            print(red("\n  interrupted."))
        print(dim("  " + agent.cost_line()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fusion-backed agentic coding harness.")
    ap.add_argument("--model", default=DEFAULT_DRIVER, help=f"driver model slug (default {DEFAULT_DRIVER})")
    ap.add_argument("--root", default=str(Path(__file__).parent), help="repo root the agent may touch")
    ap.add_argument("--task", default=None, help="run a single task non-interactively, then exit")
    ap.add_argument("--brain", choices=["fusion", "off"], default="off", help="fusion deliberation tool on the driver")
    ap.add_argument("--yolo", action="store_true", help="auto-approve all writes/commands (use with care)")
    args = ap.parse_args()

    # Windows consoles default to cp1252; force UTF-8 so model output / glyphs never crash a print.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    agent = ForgeAgent(
        root=Path(args.root),
        model=args.model,
        key=load_key(),
        auto=args.yolo,
        brain=(args.brain == "fusion"),
    )

    if args.task:
        print(dim(f"forge - {agent.model} - brain={'fusion' if agent.brain else 'off'} - auto={'on' if agent.auto else 'off'}"))
        agent.run_turn(args.task)
        print(dim("\n" + agent.cost_line()))
    else:
        repl(agent)


if __name__ == "__main__":
    main()
