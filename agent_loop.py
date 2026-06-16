"""Self-iterating "programming + testing" agent loop built on LangGraph.

A small autonomous loop that, given a natural-language task:

  1. write_tests  - asks the LLM to write a pytest spec (test_solution.py) ONCE,
     pinning the task requirements as a fixed contract.
  2. write_code   - asks the LLM to write/refine solution.py to satisfy the spec.
  3. run_tests    - runs pytest in a subprocess and reads the result.
  4. route        - loops back to write_code on failure until the tests pass
                    or the iteration budget is exhausted.

The graph is a LangGraph StateGraph wired as
    START -> write_tests -> write_code -> run_tests -> (conditional) -> write_code | END

Providers: OpenRouter (default, OpenAI-compatible) or AWS Bedrock.

============================  SECURITY WARNING  ============================
This program EXECUTES LLM-GENERATED PYTHON CODE ON YOUR LOCAL MACHINE.
pytest imports and runs `solution.py` and `test_solution.py`, both authored
by a language model. That code can do ANYTHING your user account can do:
delete files, exfiltrate data, open network connections, etc.

  * Run this ONLY inside a throwaway directory, a disposable VM, or a
    container with no credentials and no access to anything you care about.
  * NEVER point --workdir at a directory containing real source, secrets,
    or anything irreplaceable.
  * Treat the generated code as untrusted input. Review before reuse.
===========================================================================
"""

from __future__ import annotations

import argparse
import json
import operator
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

from dotenv import load_dotenv

# LangChain message types are provider-agnostic.
from langchain_core.messages import HumanMessage, SystemMessage

# LangGraph state-machine primitives.
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Shared OpenRouter policy/model defaults. The allowlist/denylist lives in the
# lightweight package module so both `agent_loop.py` and the Juno CLI harness use
# the same privacy screen without making the harness import LangChain/LangGraph.
from juno_agent_harness.provider_policy import (
    DEFAULT_BEDROCK_MODEL_ID,
    DEFAULT_CODE_MODEL_OPENROUTER,
    DEFAULT_MODEL_OPENROUTER,
    DEFAULT_TEST_MODEL_OPENROUTER,
    OPENROUTER_BASE_URL,
    OPENROUTER_HEADERS,
    OPENROUTER_MAX_OUTPUT_TOKENS,
    OPENROUTER_MAX_RETRIES,
    OPENROUTER_PROVIDER_PREFS,
    OPENROUTER_REASONING_MAX_TOKENS,
    OPENROUTER_REQUEST_TIMEOUT,
)

SOLUTION_FILENAME = "solution.py"
TEST_FILENAME = "test_solution.py"

# Floor for the LangGraph recursion limit (well above its default of 25). The
# effective limit is derived per-run from the iteration budget; see
# _recursion_limit_for(). Each loop turn costs ~2 LangGraph super-steps
# (write_code + run_tests), so a fixed cap would silently break large budgets.
RECURSION_LIMIT = 100

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Shared state threaded through every node.

    Keys without an Annotated reducer are OVERWRITTEN on each node return
    (last-write-wins). `history` accumulates via operator.add so we keep a
    record of every iteration instead of clobbering it.
    """

    task: str
    tests: str
    code: str
    test_output: str
    critique: str
    passed: bool
    iteration: int
    max_iterations: int
    history: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# LLM construction
# ---------------------------------------------------------------------------


def _fail(message: str) -> None:
    """Print an actionable error to stderr and exit non-zero."""
    print(f"\nERROR: {message}", file=sys.stderr)
    sys.exit(1)


def get_llm(model: Optional[str] = None, provider: Optional[str] = None) -> Any:
    """Build a LangChain chat model for the selected provider.

    provider defaults to env PROVIDER, then "openrouter". A missing API key
    for the selected provider is a hard, clearly-explained exit(1).
    """
    provider = (provider or os.getenv("PROVIDER") or "openrouter").strip().lower()

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            _fail(
                "OPENROUTER_API_KEY is not set.\n"
                "  1. Copy .env.example to .env\n"
                "  2. Put your key from https://openrouter.ai/keys in OPENROUTER_API_KEY\n"
                "  3. Re-run. (Or use --compile-only to smoke-test without a key.)"
            )
        from langchain_openai import ChatOpenAI

        # Force the data-secure routing screen + a reasoning cap onto every call.
        extra_body: dict = {"provider": OPENROUTER_PROVIDER_PREFS}
        if OPENROUTER_REASONING_MAX_TOKENS:
            extra_body["reasoning"] = {"max_tokens": OPENROUTER_REASONING_MAX_TOKENS}

        return ChatOpenAI(
            model=model or os.getenv("MODEL") or DEFAULT_MODEL_OPENROUTER,
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            temperature=0.1,
            default_headers=OPENROUTER_HEADERS,
            # Per-attempt timeout + bounded retries so a hung/slow provider
            # connection fails fast instead of stalling the loop forever.
            timeout=OPENROUTER_REQUEST_TIMEOUT,
            max_retries=OPENROUTER_MAX_RETRIES,
            # Bound total output as a backstop to the reasoning cap above.
            max_tokens=OPENROUTER_MAX_OUTPUT_TOKENS,
            extra_body=extra_body,
        )

    if provider == "bedrock":
        # boto3 resolves credentials from its standard chain; we only need to
        # confirm a region is discoverable so construction does not blow up
        # with an opaque error later.
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        if not region:
            _fail(
                "PROVIDER=bedrock but no AWS region is set.\n"
                "  Set AWS_REGION (e.g. us-east-1) in your .env, matching the\n"
                "  geo prefix of BEDROCK_MODEL_ID (a 'us.' model needs a us-* region)."
            )
        if not (
            os.getenv("AWS_ACCESS_KEY_ID")
            or os.getenv("AWS_PROFILE")
            or os.getenv("AWS_SESSION_TOKEN")
        ):
            _fail(
                "PROVIDER=bedrock but no AWS credentials were found.\n"
                "  Set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, or AWS_PROFILE,\n"
                "  or run on an instance/role with Bedrock access.\n"
                "  (Or use --compile-only to smoke-test without credentials.)"
            )
        from langchain_aws import ChatBedrockConverse

        # NOTE: ChatBedrockConverse uses `model=` (ChatBedrock uses `model_id=`).
        return ChatBedrockConverse(
            model=model or os.getenv("BEDROCK_MODEL_ID") or DEFAULT_BEDROCK_MODEL_ID,
            region_name=region,
            temperature=0.1,
            max_tokens=4096,
        )

    _fail(f"Unknown PROVIDER '{provider}'. Use 'openrouter' or 'bedrock'.")
    raise AssertionError("unreachable")  # for type-checkers; _fail exits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)\r?\n?```",
    re.DOTALL,
)


def strip_code_fences(text: str) -> str:
    """Extract pure Python from an LLM reply.

    Handles the common shapes robustly:
      * One or more ```python ... ``` fenced blocks -> concatenate their bodies.
      * A single dangling ``` opener with no closer -> take everything after it.
      * No fences at all -> return the text as-is (trimmed).
    """
    if not text:
        return ""

    blocks = _FENCE_RE.findall(text)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks).strip() + "\n"

    stripped = text.strip()
    if stripped.startswith("```"):
        # Opener but no matching closer: drop the first fence line, keep the rest,
        # and remove a trailing fence if one snuck in.
        body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        body = re.sub(r"\r?\n?```[ \t]*$", "", body)
        return body.strip() + "\n"

    return stripped + "\n"


def _content_to_text(content: Any) -> str:
    """Coerce an AIMessage.content to a string.

    Text models return a str. Some providers (tool/multimodal) can return a
    list of content blocks; concatenate any text we find so we never crash.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


# --- token / cost accounting ------------------------------------------------
# Per-million-token prices (USD) for head-to-head cost estimation. DeepSeek
# V4 Pro is priced at its cheapest data-secure endpoint (DeepInfra $1.3/$2.6);
# sort:price picks the actual serving provider at request time.
_MODEL_PRICES = {
    "moonshotai/kimi-k2.7-code": {"in": 0.75, "out": 3.50},
    "deepseek/deepseek-v4-pro": {"in": 1.30, "out": 2.60},
}

_USAGE: dict = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "by_model": {}}


def reset_usage() -> None:
    """Zero the token/cost accumulator (call before each independent run)."""
    _USAGE["calls"] = 0
    _USAGE["input_tokens"] = 0
    _USAGE["output_tokens"] = 0
    _USAGE["by_model"] = {}


def _record_usage(response: Any) -> None:
    """Fold one AIMessage's token usage into the global accumulator."""
    usage = getattr(response, "usage_metadata", None) or {}
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    meta = getattr(response, "response_metadata", None) or {}
    model = meta.get("model_name") or meta.get("model") or "unknown"
    _USAGE["calls"] += 1
    _USAGE["input_tokens"] += in_tok
    _USAGE["output_tokens"] += out_tok
    slot = _USAGE["by_model"].setdefault(model, {"calls": 0, "in": 0, "out": 0})
    slot["calls"] += 1
    slot["in"] += in_tok
    slot["out"] += out_tok


def estimate_cost(usage: Optional[dict] = None) -> float:
    """Estimate USD cost from accumulated per-model token counts."""
    usage = usage or _USAGE
    total = 0.0
    for model, slot in usage.get("by_model", {}).items():
        # Unknown/aliased model -> priciest known rate so we never under-report.
        price = _MODEL_PRICES.get(model, {"in": 1.30, "out": 3.50})
        total += slot["in"] / 1e6 * price["in"] + slot["out"] / 1e6 * price["out"]
    return total


def usage_snapshot() -> dict:
    """Copy of the usage accumulator plus its cost estimate."""
    snap = {
        "calls": _USAGE["calls"],
        "input_tokens": _USAGE["input_tokens"],
        "output_tokens": _USAGE["output_tokens"],
        "by_model": {m: dict(s) for m, s in _USAGE["by_model"].items()},
    }
    snap["est_cost_usd"] = estimate_cost(snap)
    return snap


def _invoke_llm(llm: Any, system: str, user: str) -> str:
    """Single LLM turn -> plain text. Surfaces API errors as a clean exit."""
    try:
        response = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
    except Exception as exc:  # noqa: BLE001 - present any provider error cleanly
        _fail(
            f"LLM call failed: {exc}\n"
            "  Check your API key, model slug, network access, and credit balance."
        )
        raise AssertionError("unreachable")  # _fail exits
    _record_usage(response)
    text = _content_to_text(response.content)
    if not text.strip():
        fr = (getattr(response, "response_metadata", None) or {}).get("finish_reason")
        _fail(
            f"LLM returned EMPTY content (finish_reason={fr!r}). A reasoning model "
            "likely consumed its whole output budget thinking. Lower "
            "OPENROUTER_REASONING_MAX_TOKENS or raise OPENROUTER_MAX_OUTPUT_TOKENS."
        )
    return text


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

_TESTS_SYSTEM = (
    "You are a senior adversarial test engineer. Write a FOCUSED, deterministic "
    "pytest suite of about 15-30 tests that pins down the spec: the main "
    "behaviors plus the most important edge cases (empty/degenerate input, "
    "boundaries, precedence/associativity, nesting, errors). These tests are the "
    "fixed contract the implementer must satisfy. Be decisive and converge "
    "quickly - do NOT enumerate endlessly or over-deliberate. Every assertion "
    "must be unambiguously CORRECT per the task. Output ONLY a single Python "
    "file's contents - no prose, no markdown outside code."
)

_CODE_SYSTEM = (
    "You are a senior Python engineer. You write correct, self-contained "
    "implementations. Output ONLY a single Python file's contents - no prose, "
    "no markdown commentary outside code."
)

_CRITIQUE_SYSTEM = (
    "You are an adversarial senior reviewer AND the author of the test "
    "contract. A candidate implementation has FAILED your tests. Diagnose WHY "
    "precisely and tell the implementer exactly what to change. You may NOT "
    "relax or rewrite the tests - they are the fixed contract. Be terse and "
    "concrete: name the failing behavior, the root cause, and the fix "
    "direction. Output plain-text guidance, not code."
)


def make_write_tests(llm: Any):
    """Node factory: generate the pytest spec exactly once (iteration 0)."""

    def write_tests(state: AgentState) -> dict:
        print("\n[write_tests] Generating the test spec (runs once)...")
        prompt = (
            f"Write a pytest test file named `{TEST_FILENAME}` for this task:\n\n"
            f"TASK:\n{state['task']}\n\n"
            "Requirements:\n"
            f"- Import the implementation from `solution` (e.g. `from solution import ...`).\n"
            "- Encode the task's requirements as concrete, deterministic assertions.\n"
            "- Cover normal cases AND edge cases (empty input, boundaries, errors).\n"
            "- Use plain `def test_*()` functions and bare `assert`. No network, "
            "no filesystem, no randomness without a fixed seed.\n"
            "- The tests are the FIXED contract; do not leave them trivially passing.\n"
            "Return the complete file contents only."
        )
        tests = strip_code_fences(_invoke_llm(llm, _TESTS_SYSTEM, prompt))
        print(f"[write_tests] Test spec ready ({tests.count(chr(10)) + 1} lines).")
        return {
            "tests": tests,
            "history": [f"iter 0: wrote {TEST_FILENAME}"],
        }

    return write_tests


def make_write_code(llm: Any):
    """Node factory: write or refine solution.py against the spec + failures."""

    def write_code(state: AgentState) -> dict:
        iteration = state["iteration"]
        previous = state.get("code") or ""
        test_output = state.get("test_output") or ""
        critique = state.get("critique") or ""

        if not previous:
            print("\n[write_code] Writing the first implementation...")
            prompt = (
                f"Write `{SOLUTION_FILENAME}` that satisfies the task and passes "
                f"the test suite below.\n\n"
                f"TASK:\n{state['task']}\n\n"
                f"TEST SUITE ({TEST_FILENAME}) - this is the fixed contract:\n"
                f"{state['tests']}\n\n"
                "Write a complete, self-contained implementation. Define every "
                "name the tests import. Return the full file contents only."
            )
        else:
            print(f"\n[write_code] Refining implementation to fix failures "
                  f"(iteration {iteration})...")
            critique_block = (
                "ADVERSARIAL REVIEWER CRITIQUE (from the test author - act on "
                f"this):\n{critique}\n\n"
                if critique else ""
            )
            prompt = (
                "Your previous implementation FAILED the test suite. Fix the "
                "specific failures shown below. Do not change the tests.\n\n"
                f"TASK:\n{state['task']}\n\n"
                f"TEST SUITE ({TEST_FILENAME}):\n{state['tests']}\n\n"
                f"PREVIOUS {SOLUTION_FILENAME}:\n{previous}\n\n"
                f"PYTEST OUTPUT (the failures to fix):\n{test_output}\n\n"
                f"{critique_block}"
                "Analyze each failure, then return a COMPLETE corrected "
                f"`{SOLUTION_FILENAME}` (full file, not a diff). Code only."
            )

        code = strip_code_fences(_invoke_llm(llm, _CODE_SYSTEM, prompt))
        print(f"[write_code] Implementation ready ({code.count(chr(10)) + 1} lines).")
        return {"code": code}

    return write_code


def make_critique(llm: Any):
    """Node factory: the test-author model critiques a failing implementation.

    Runs on the retry path only (two-model mode). It diagnoses WHY the candidate
    violates the fixed test contract and tells write_code what to change - it may
    not touch the tests. The guidance is threaded into the next write_code prompt
    via state['critique'].
    """

    def critique(state: AgentState) -> dict:
        iteration = state["iteration"]
        print(f"\n[critique] Test author diagnosing the failure "
              f"(iteration {iteration})...")
        prompt = (
            "Your test contract was violated by the implementation below.\n\n"
            f"TASK:\n{state['task']}\n\n"
            f"TEST SUITE ({TEST_FILENAME}) - the FIXED contract:\n{state['tests']}\n\n"
            f"FAILING {SOLUTION_FILENAME}:\n{state.get('code') or ''}\n\n"
            f"PYTEST OUTPUT:\n{state.get('test_output') or ''}\n\n"
            "Give a short, sharp diagnosis: which behavior is wrong, the most "
            "likely root cause, and the specific change the implementer should "
            "make. Do NOT rewrite or relax the tests. Plain-text guidance only, "
            "<= 200 words."
        )
        crit = _invoke_llm(llm, _CRITIQUE_SYSTEM, prompt).strip()
        print(f"[critique] Guidance ready ({len(crit)} chars).")
        return {
            "critique": crit,
            "history": [f"iter {iteration}: critique issued"],
        }

    return critique


def make_run_tests(workdir: Path):
    """Node factory: persist files, run pytest, fold the result into state."""

    def run_tests(state: AgentState) -> dict:
        iteration = state["iteration"]
        workdir.mkdir(parents=True, exist_ok=True)

        solution_path = workdir / SOLUTION_FILENAME
        test_path = workdir / TEST_FILENAME
        solution_path.write_text(state["code"], encoding="utf-8")
        test_path.write_text(state["tests"], encoding="utf-8")

        print(f"[run_tests] Running pytest in {workdir} ...")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=120,
                # No shell=True: argv list is cross-platform and injection-safe.
            )
        except FileNotFoundError:
            # The interpreter itself was missing - unrecoverable.
            output = (
                "Could not launch the Python interpreter to run pytest "
                f"({sys.executable})."
            )
            print(f"[run_tests] {output}")
            return {
                "test_output": output,
                "passed": False,
                "iteration": iteration + 1,
                "history": [f"iter {iteration + 1}: interpreter not found"],
            }
        except subprocess.TimeoutExpired:
            output = "pytest timed out after 120s (possible infinite loop in solution)."
            print(f"[run_tests] {output}")
            return {
                "test_output": output,
                "passed": False,
                "iteration": iteration + 1,
                "history": [f"iter {iteration + 1}: pytest TIMEOUT"],
            }

        output = (proc.stdout or "") + (proc.stderr or "")

        # pytest exit code 5 == "no tests collected"; treat anything nonzero as fail.
        if proc.returncode == 5 and "no tests ran" in output.lower():
            output += (
                "\n[run_tests] pytest collected NO tests - the test file may be "
                "malformed or empty."
            )

        # `python -m pytest` with pytest absent exits 1 with a clear message;
        # surface that actionably rather than as a mysterious failure.
        if "No module named pytest" in output:
            output += (
                "\n[run_tests] pytest is not installed. Run: "
                f"{sys.executable} -m pip install pytest"
            )

        passed = proc.returncode == 0
        status = "PASSED" if passed else "FAILED"
        print(f"[run_tests] pytest exit={proc.returncode} -> {status}")

        # Keep the history record short; the full output lives in test_output.
        summary = output.strip().splitlines()[-1] if output.strip() else "(no output)"
        return {
            "test_output": output,
            "passed": passed,
            "iteration": iteration + 1,
            "history": [f"iter {iteration + 1}: {status} | {summary}"],
        }

    return run_tests


def route(state: AgentState) -> str:
    """Conditional edge: stop on success or budget exhaustion, else loop."""
    if state["passed"]:
        print("[route] Tests pass -> done.")
        return "end"
    if state["iteration"] >= state["max_iterations"]:
        print(
            f"[route] Iteration budget reached "
            f"({state['iteration']}/{state['max_iterations']}) -> stopping."
        )
        return "end"
    print(
        f"[route] Tests failing "
        f"({state['iteration']}/{state['max_iterations']}) -> retry write_code."
    )
    return "retry"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(
    code_llm: Optional[Any],
    test_llm: Optional[Any],
    workdir: Path,
    use_critique: bool = False,
):
    """Wire the StateGraph for the two-model loop.

    START -> write_tests -> write_code -> run_tests -> conditional(route).
    With use_critique the retry path is run_tests -> critique -> write_code, so
    the test-author model diagnoses each failure before the implementer model
    refines; otherwise route maps "retry" straight back to write_code.

    `test_llm` backs write_tests + critique; `code_llm` backs write_code. When
    BOTH are None (compile-only smoke test) the LLM-backed nodes are stubs, so
    the graph compiles and is structurally valid without any API access.
    """
    if code_llm is None and test_llm is None:
        def write_tests(state: AgentState) -> dict:  # pragma: no cover - stub
            return {"tests": "", "history": ["compile-only: write_tests stub"]}

        def write_code(state: AgentState) -> dict:  # pragma: no cover - stub
            return {"code": ""}

        def _crit_stub(state: AgentState) -> dict:  # pragma: no cover - stub
            return {"critique": ""}

        critique = _crit_stub if use_critique else None
    else:
        write_tests = make_write_tests(test_llm)
        write_code = make_write_code(code_llm)
        critique = make_critique(test_llm) if use_critique else None

    run_tests = make_run_tests(workdir)

    builder = StateGraph(AgentState)
    builder.add_node("write_tests", write_tests)
    builder.add_node("write_code", write_code)
    builder.add_node("run_tests", run_tests)

    builder.add_edge(START, "write_tests")
    builder.add_edge("write_tests", "write_code")
    builder.add_edge("write_code", "run_tests")

    # path_map: router key -> destination node / END.
    if critique is not None:
        builder.add_node("critique", critique)
        # Retry path runs the adversarial critique before the next write_code.
        builder.add_conditional_edges(
            "run_tests", route, {"retry": "critique", "end": END}
        )
        builder.add_edge("critique", "write_code")
    else:
        builder.add_conditional_edges(
            "run_tests", route, {"retry": "write_code", "end": END}
        )

    return builder.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _recursion_limit_for(max_iters: int) -> int:
    """Derive a recursion limit that always clears the iteration budget.

    One run costs up to `1 (write_tests) + 3 * max_iters (critique +
    write_code + run_tests per loop) + 1 (final route)` LangGraph super-steps.
    We add a margin and floor it at RECURSION_LIMIT so small budgets keep the
    old generous headroom while large budgets get a limit that scales with them.
    Without this, --max-iters >= ~30 would trip GraphRecursionError before
    route() could stop the loop gracefully.
    """
    return max(RECURSION_LIMIT, 3 * max_iters + 10)


def run_agent_loop(
    task: str,
    max_iters: int = 6,
    model: Optional[str] = None,
    code_model: Optional[str] = None,
    test_model: Optional[str] = None,
    workdir: str = "./agent_workspace",
    provider: Optional[str] = None,
    use_critique: Optional[bool] = None,
) -> AgentState:
    """Run the full loop and return the final state.

    Two-model by default: `code_model` drafts solution.py (write_code) and
    `test_model` authors the adversarial test contract and critiques failures
    (write_tests + critique). pytest stays the judge. `--model`/MODEL env pins
    BOTH roles to one slug ("solo" mode) and disables the critique node, which
    reproduces the original single-model loop for head-to-head baselines.

    Importable for use as a library. Builds the LLMs (exiting cleanly if the
    selected provider's key/credentials are missing), compiles the graph, and
    invokes it with a recursion limit derived from the iteration budget so the
    loop's own route() budget check always fires before LangGraph's recursion
    guard (see _recursion_limit_for).
    """
    reset_usage()
    work_path = Path(workdir).expanduser().resolve()
    work_path.mkdir(parents=True, exist_ok=True)

    prov = (provider or os.getenv("PROVIDER") or "openrouter").strip().lower()
    # --model => "solo" mode: one slug for both roles, no critique.
    # --code-model / --test-model explicitly override everything, including MODEL env.
    # MODEL env is only the fallback when neither role is specified.
    if model:
        code_model = test_model = model
        solo = True
    elif code_model or test_model:
        solo = False
        if prov == "openrouter":
            code_model = code_model or DEFAULT_CODE_MODEL_OPENROUTER
            test_model = test_model or DEFAULT_TEST_MODEL_OPENROUTER
        # bedrock: leave None -> get_llm applies the single bedrock default.
    else:
        # No model flags at all: fallback to MODEL env for legacy solo mode,
        # otherwise use the default two-model team.
        env_model = os.getenv("MODEL")
        if env_model:
            code_model = test_model = env_model
            solo = True
        else:
            solo = False
            if prov == "openrouter":
                code_model = DEFAULT_CODE_MODEL_OPENROUTER
                test_model = DEFAULT_TEST_MODEL_OPENROUTER

    if use_critique is None:
        use_critique = not solo

    print("=" * 70)
    print("Self-iterating programming + testing agent loop")
    print(f"  task        : {task}")
    print(f"  max_iters   : {max_iters}")
    print(f"  provider    : {prov}")
    print(f"  code model  : {code_model or '(provider default)'}  [write_code]")
    print(f"  test model  : {test_model or '(provider default)'}  [write_tests/critique]")
    print(f"  critique    : {'on' if use_critique else 'off (solo)'}")
    if prov == "openrouter":
        print(f"  privacy     : provider={OPENROUTER_PROVIDER_PREFS}")
    print(f"  workdir     : {work_path}")
    print("=" * 70)

    code_llm = get_llm(model=code_model, provider=provider)
    test_llm = get_llm(model=test_model, provider=provider)
    graph = build_graph(code_llm, test_llm, work_path, use_critique=use_critique)

    initial: AgentState = {
        "task": task,
        "tests": "",
        "code": "",
        "test_output": "",
        "passed": False,
        "iteration": 0,
        "max_iterations": max_iters,
        "history": [],
    }

    recursion_limit = _recursion_limit_for(max_iters)
    try:
        final: AgentState = graph.invoke(initial, {"recursion_limit": recursion_limit})
    except GraphRecursionError as exc:
        # Defense in depth: the derived limit should always clear the budget,
        # so route() stops us first. If we still trip the guard, surface it
        # through the program's clean _fail() path instead of a raw traceback.
        _fail(
            f"LangGraph recursion limit ({recursion_limit}) hit before the loop "
            f"finished: {exc}\n"
            "  This should not happen with the derived limit; lower --max-iters."
        )
        raise AssertionError("unreachable")  # _fail exits

    _print_final_report(final, work_path)
    return final


def _print_final_report(final: AgentState, work_path: Path) -> None:
    """Emit the end-of-run summary, file locations, and final solution."""
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    status = "SUCCESS - all tests passed" if final.get("passed") else "FAILED - tests not passing"
    print(f"  status      : {status}")
    print(f"  iterations  : {final.get('iteration')}/{final.get('max_iterations')}")
    print(f"  solution    : {work_path / SOLUTION_FILENAME}")
    print(f"  tests       : {work_path / TEST_FILENAME}")

    print("\n  history:")
    for line in final.get("history", []):
        print(f"    - {line}")

    if not final.get("passed") and final.get("test_output"):
        print("\n  last pytest output (tail):")
        tail = "\n".join(final["test_output"].strip().splitlines()[-15:])
        for line in tail.splitlines():
            print(f"    | {line}")

    snap = usage_snapshot()
    by_model = ", ".join(
        f"{m}={s['calls']}c/{s['in']}in/{s['out']}out"
        for m, s in snap["by_model"].items()
    )
    print(f"\n  llm calls   : {snap['calls']}")
    print(f"  tokens      : in={snap['input_tokens']} out={snap['output_tokens']}")
    print(f"  est. cost   : ${snap['est_cost_usd']:.4f}  ({by_model})")
    print(f"USAGE_JSON: {json.dumps(snap)}")

    print("\n  final solution.py:")
    print("-" * 70)
    print(final.get("code") or "(no code produced)")
    print("-" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    """argparse type: accept only integers >= 1 (a 0/negative budget is a no-op)."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}")
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
    return parsed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent_loop.py",
        description="Self-iterating programming + testing agent loop (LangGraph).",
    )
    parser.add_argument("task", nargs="?", help="Natural-language task description.")
    parser.add_argument(
        "--max-iters", type=_positive_int, default=6,
        help="Max write_code/run_tests iterations (>= 1, default: 6).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Solo mode: pin BOTH roles to one slug and disable critique "
             "(reproduces the original single-model loop).",
    )
    parser.add_argument(
        "--code-model", default=None,
        help="OpenRouter slug for write_code (default: DeepSeek V4 Pro).",
    )
    parser.add_argument(
        "--test-model", default=None,
        help="OpenRouter slug for write_tests/critique (default: DeepSeek V4 Pro).",
    )
    parser.add_argument(
        "--workdir", default="./agent_workspace",
        help="Directory for generated files (default: ./agent_workspace).",
    )
    parser.add_argument(
        "--provider", default=None, choices=["openrouter", "bedrock"],
        help="Override PROVIDER from env.",
    )
    parser.add_argument(
        "--no-critique", action="store_true",
        help="Disable the test-author critique node in two-model mode.",
    )
    parser.add_argument(
        "--compile-only", action="store_true",
        help="Build and compile the graph WITHOUT calling the LLM (no API key needed).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()
    args = _build_arg_parser().parse_args(argv)

    if args.compile_only:
        # Smoke test: prove the graph wires up and compiles with no LLM/API key.
        work_path = Path(args.workdir).expanduser().resolve()
        graph = build_graph(None, None, work_path, use_critique=True)
        nodes = sorted(graph.get_graph().nodes.keys())
        print("compile-only: graph compiled successfully.")
        print(f"  nodes: {nodes}")
        print("  edges: START -> write_tests -> write_code -> run_tests "
              "-> route{retry->critique->write_code, end->END}")
        return 0

    if not args.task:
        _build_arg_parser().error("a TASK description is required (or use --compile-only)")

    final = run_agent_loop(
        task=args.task,
        max_iters=args.max_iters,
        model=args.model,
        code_model=args.code_model,
        test_model=args.test_model,
        workdir=args.workdir,
        provider=args.provider,
        use_critique=False if args.no_critique else None,
    )
    return 0 if final.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
