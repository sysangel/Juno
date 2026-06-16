# Custom CLI agent harness

`harness.py` is the first working slice of the custom Hermes-like CLI harness.
It is deliberately smaller than Hermes, but it pulls over the pieces that matter
for this project:

- provider-neutral chat history
- OpenRouter (no-train + provider screening now enforced at the OpenRouter
  **account** level; the harness no longer injects a per-request provider block)
- OpenAI/ChatGPT through `OPENAI_API_KEY`
- Anthropic through the first-party Messages API and `ANTHROPIC_API_KEY`
- Claude Code through its existing OAuth login (`claude auth login`) so a
  Claude Pro/Max account can be used without an Anthropic API key
- Hermes-style bounded memory files: `MEMORY.md` and `USER.md`, separated by `§`
- frozen memory snapshot injected at session start, so writes are durable but do
  not mutate the active prompt mid-session
- SQLite transcript storage with FTS search

## Run it

Preferred from any terminal on this machine:

```bash
juno
juno --provider openrouter -q "Say hello in one sentence."
juno --provider openai --model gpt-5.1
juno --provider anthropic --model claude-sonnet-4-20250514
juno --provider claude-code --model sonnet -q "Say hello."
```

The `juno` launchers live in:

- `C:\Users\Core\bin\juno` for Git-Bash/MSYS
- `C:\Users\Core\bin\juno.cmd` for cmd.exe, PowerShell, and Windows Terminal
- `C:\Users\Core\AppData\Local\hermes\bin\juno.cmd` as an additional PowerShell-friendly shim

`C:\Users\Core\bin` has also been added to the persisted User PATH. Existing
PowerShell windows may need to be restarted or have `$env:Path` refreshed before
`juno` resolves.

Direct invocation from `C:\Users\Core\src\agent-loop` still works:

```bash
.venv/Scripts/python.exe harness.py --provider openrouter -q "Say hello in one sentence."
.venv/Scripts/python.exe harness.py --provider claude-code --model sonnet -q "Say hello."
```

Offline smoke test, no API key needed:

```bash
.venv/Scripts/python.exe harness.py --provider echo -q "hello" --home runs/harness_smoke
```

## Juno: the interactive UI

When you launch without `-q`, the harness starts **Juno**, a lightweight
in-terminal UI (deep blue/purple theme) built on `prompt_toolkit`. It gives you:

- a **status toolbar** at the bottom showing `provider/model`, memory on/off,
  session id, and running token totals — it updates live after `/model`
- **command/argument completion**: slash commands, model catalog keys after
  `/model`, and `list/add/replace/remove/reload` + `memory|user` after `/memory`
- **segmented model picker**: in full Juno mode, `/model` opens a selection list
  grouped by `Codex / OpenAI`, `Anthropic / Claude`, and `OpenRouter`. Each row
  shows a fixed-width strength bar and a tier icon; the single strongest coding
  route per provider is tagged `ULTRACODE` (a ⚡ sigil + bar), and selecting it
  fires a one-shot rainbow boost sweep on a color TTY
- **persistent history** across runs at `<harness-home>/history.txt`
  (use Up/Down to recall previous input)
- **multiline input**: `Enter` submits, `Esc` then `Enter` (or `Alt-Enter`)
  inserts a newline
- **Hermes-like response boxes**: Juno replies render in bordered boxes
  so they are visually distinct from prompts, command output, and usage metadata
- **inline waiting state**: while a blocking provider call runs, Juno keeps the
  bottom toolbar in place and animates it there — no screen flip. The animation
  is gated on a real TTY and `JUNO_WAIT_ANIMATION` (set `JUNO_WAIT_ANIMATION=0`
  to force the static waiting line and a synchronous call)
- a token/session usage line after each model response

### UI modes (`--ui`)

```bash
.venv/Scripts/python.exe harness.py --provider claude-code --model sonnet            # auto
.venv/Scripts/python.exe harness.py --provider echo --ui plain                       # basic REPL
.venv/Scripts/python.exe harness.py --provider echo --ui prompt                       # force Juno
```

- `--ui auto` (default): use Juno when stdin is a TTY and a real console is
  available; otherwise fall back to the plain REPL. Honors `HARNESS_UI`.
- `--ui plain`: always use the basic `input()`/`print()` REPL. Best for pipes,
  CI, and scripted smoke tests.
- `--ui prompt`: force Juno; raises an actionable error if `prompt_toolkit` or a
  usable console is missing.
- `-q/--query`: one-shot mode, no UI (unchanged).
- `--tui`: alias for `--ui prompt` (force Juno); `--cli`: alias for `--ui plain`
  (passing both is an error).

A persisted default model in `<harness-home>/config.json` is honored at startup;
resolution order is CLI `--provider`/`--model` > config.json default > built-in
`default_model(provider)` fallback. Use `--default-model KEY|PROVIDER/MODEL` to
persist a new default into `config.json` (it prints the new default, then runs).

> **Windows / Git Bash note:** `prompt_toolkit` needs a real Windows console
> (cmd.exe, Windows Terminal, or PowerShell). Under MSYS/Git-Bash (`xterm`),
> Juno cannot attach a console, so `--ui auto` automatically falls back to the
> plain REPL instead of crashing. Run from a Windows console to get full Juno.

The `HARNESS_UI` env var sets the default mode (`auto`, `plain`, or `prompt`).

### Color and the rainbow boost

Juno's animated rainbow boost (the ULTRACODE sweep) runs only when stdout is a
real TTY and `NO_COLOR` is unset; setting `NO_COLOR` (to any value) disables it
along with other color output. There is no dedicated rainbow on/off env var.
`COLORTERM=truecolor` (or `24bit`) selects the 24-bit gradient; otherwise the
sweep degrades to a 256-color cycle.

## State layout

Default home is `~/.agent-harness`, override with `AGENT_HARNESS_HOME` or
`--home`.

```text
.agent-harness/
  memories/
    MEMORY.md
    USER.md
  sessions.db
  history.txt        # Juno input history (prompt_toolkit)
```

## REPL commands

```text
/help
/status
/model                      full Juno: open model picker; plain: print catalog
/model <key>                e.g. /model sonnet, /model codex, /model dp
/model <provider> [model]   e.g. /model openrouter deepseek/deepseek-v4-pro
/model default              report the current persisted default model
/model default <ref>        persist a default (KEY or PROVIDER/MODEL) to config.json
/skills                     list local skills (SKILL.md under <home>/skills)
/menu                       command palette of common actions
/exit
/reset                      clear conversation, keep memory snapshot
/clear                      alias for /reset
/memory list
/memory add memory <entry>
/memory add user <entry>
/memory replace memory <old substring> => <new entry>
/memory remove memory <old substring>
/memory reload
/search <fts query>
```

### Model catalog and shortcuts

In full Juno mode, `/model` opens a segmented picker:

- **Codex / OpenAI**: `codex`, `gpt5`
- **Anthropic / Claude**: `claude`, `sonnet`, `opus`, `anthropic`
- **OpenRouter**: `or`, `dp`, `fm3v4`, `fbudget`, `fhigh`
- **Offline**: `echo`

Plain mode prints the same catalog. You can switch providers mid-session without
losing the current conversation:

```text
/model sonnet
/model codex
/model dp
/model fhigh
/model openai gpt-5.1
/model openrouter deepseek/deepseek-v4-pro
```

Raw provider/model pairs still work for any model not in the catalog. OpenRouter
Fusion shortcuts are env-overridable:

```text
HARNESS_DP_MODEL=deepseek/deepseek-v4-pro
HARNESS_FM3V4_MODEL=deepseek/deepseek-v4-pro
HARNESS_FBUDGET_MODEL=openrouter/auto
HARNESS_FHIGH_MODEL=openrouter/auto
```

Provider aliases work too: `claude` -> `claude-code`, `codex`/`chatgpt`/`gpt` ->
`openai`, and `or` -> `openrouter`.

## Design notes copied from Hermes research

Hermes' current memory system is simple and strong: two bounded curated files,
`MEMORY.md` for environment/project facts and `USER.md` for user preferences,
injected as a frozen system-prompt snapshot. Tool writes update disk immediately,
but the live prompt does not change until a new session starts. That preserves
prompt-cache stability and prevents mid-turn memory churn.

Hermes augments that with SQLite sessions + FTS5 (`state.db`) for cheap recall of
past conversations. This harness mirrors that distinction: memory is for facts
that should always be in context; session search is for long-tail recall.

Memory retention is durable across harness runs because `MEMORY.md` and
`USER.md` are stored under the harness home. Like Hermes, the memory snapshot is
loaded into the system prompt at session start. If you add/edit memory inside a
live REPL, use `/memory reload` to refresh the active prompt immediately; either
way, the next harness run will load it automatically.

## Anthropic without an API key

The official Anthropic Messages API still needs `ANTHROPIC_API_KEY`. The no-key
path is not the API; it is Claude Code's OAuth login. If `claude auth status`
shows a Claude Pro/Max/Enterprise account, `--provider claude-code` shells out to
`claude -p` in print mode and captures its JSON result. This lets the harness use
Claude while keeping its own memory and transcript store.

That path depends on the installed `claude` CLI and the account's Claude Code
entitlements/limits. It is not the same as direct API access and should be
treated as a local CLI bridge.

## Next layers

1. Add tool calling: shell, file read/write/search, and patch as explicit tools.
2. Add a permission/approval gate for destructive shell commands.
3. Add model routing aliases for the OpenRouter Fusion panels already configured
   in Hermes (`fbudget`, `fhigh`, `fm3v4`, `dp`).
4. Connect the existing `agent_loop.py` and `orchestrator.py` as callable tools or
   subcommands so the harness can launch code→test→fix workers.
5. Add memory write approval/staging if automatic memory extraction is added.
