# Juno

Juno is a lightweight, Hermes-inspired terminal agent harness. It keeps the core loop small while adding the parts that make an agent pleasant to use from a terminal: persistent memory, session search, provider switching, a purple prompt UI, and safe fallback behavior for pipes and CI.

## Features

- **Juno CLI UI**: a deep blue/purple `prompt_toolkit` REPL with history, completions, a status toolbar, model picker, and clearer assistant response boxes.
- **Plain fallback**: automatically falls back to a basic `input()`/`print()` REPL for pipes, CI, and terminals where `prompt_toolkit` cannot attach.
- **Model catalog**: `/model` opens a segmented picker in the prompt UI or prints a grouped catalog in plain mode.
  - Codex / OpenAI: `codex`, `gpt5`
  - Anthropic / Claude: `claude`, `sonnet`, `opus`, `anthropic`
  - OpenRouter: `or`, `dp`, `fm3v4`, `fbudget`, `fhigh`
  - Offline: `echo`
- **Provider routes**:
  - OpenRouter with the repo's no-train / Western-provider privacy screen
  - OpenAI-compatible API via `OPENAI_API_KEY`
  - Anthropic Messages API via `ANTHROPIC_API_KEY`
  - Claude Code OAuth bridge via local `claude -p` (`--provider claude-code`)
  - Offline `echo` provider for smoke tests
- **Persistent memory**: bounded `MEMORY.md` and `USER.md` files under the harness home directory.
- **Session storage**: SQLite transcript store with FTS search via `/search`.
- **Agent loop utilities**: the original LangGraph code→test→fix loop remains available in `agent_loop.py`.

## Security note

Juno can run LLM-generated code through the agent loop utilities, and the harness can send prompts to third-party model providers. Treat this like any other local agent runtime:

- keep API keys in `.env` or environment variables; never commit `.env`
- run generated-code workflows in throwaway directories/VMs
- do not point agent workspaces at sensitive source trees unless you understand the risk
- review provider privacy/data-retention policies before sending sensitive content

The repo's `.gitignore` excludes `.env`, virtualenvs, generated run outputs, local Hermes metadata, and private handoff notes.

## Installation

```bash
git clone https://github.com/sysangel/Juno.git
cd Juno
python -m venv .venv
.venv/Scripts/python.exe -m pip install -U pip
.venv/Scripts/python.exe -m pip install -e .[dev]
```

On macOS/Linux, replace `.venv/Scripts/python.exe` with `.venv/bin/python`.

## Configuration

Copy `.env.example` and add the keys you want to use:

```bash
cp .env.example .env
```

Common settings:

```text
OPENROUTER_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HARNESS_PROVIDER=openrouter
HARNESS_UI=auto
OPENAI_MODEL=gpt-5.1
ANTHROPIC_MODEL=claude-sonnet-4-20250514
CLAUDE_CODE_MODEL=sonnet
```

OpenRouter shortcut overrides used by `/model`:

```text
HARNESS_DP_MODEL=deepseek/deepseek-v4-pro
HARNESS_FM3V4_MODEL=deepseek/deepseek-v4-pro
HARNESS_FBUDGET_MODEL=openrouter/auto
HARNESS_FHIGH_MODEL=openrouter/auto
```

## Usage

After `pip install -e .`, launch the harness with:

```bash
juno
```

Examples:

```bash
juno --provider echo -q "hello"
juno --provider openrouter -q "Say hello in one sentence."
juno --provider claude-code --model sonnet
juno --provider openai --model gpt-5.1
```

Direct script invocation also works:

```bash
python harness.py --provider echo -q "hello"
```

## REPL commands

```text
/help
/status
/model                      prompt UI: open model picker; plain UI: print catalog
/model <key>                e.g. /model sonnet, /model codex, /model dp
/model <provider> [model]   e.g. /model openrouter deepseek/deepseek-v4-pro
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

## UI modes

```bash
juno --ui auto    # default: prompt UI when possible, plain fallback otherwise
juno --ui plain   # basic REPL for pipes/scripts/CI
juno --ui prompt  # force prompt_toolkit UI
```

On Windows Git-Bash/MSYS, `prompt_toolkit` may not be able to attach to the MSYS pseudo-console. `--ui auto` detects that and falls back to the plain UI; use Windows Terminal, PowerShell, or cmd.exe for the full prompt UI.

## Agent loop utility

The original self-iterating LangGraph code/test loop is still available:

```bash
python agent_loop.py "Write a function is_palindrome(s) that ignores case and non-alphanumerics" --max-iters 6 --workdir ./agent_workspace
```

This writes generated `solution.py` and `test_solution.py` into the selected workspace and runs pytest there. Use throwaway workspaces.

## Development

```bash
python -m pytest test_harness_ui.py -q
python -m py_compile harness.py
```

The `echo` provider is sufficient for UI and smoke tests; no API keys are required.

## License

MIT
