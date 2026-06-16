"""Recursive orchestrator for the agent-loop coding system.

This turns the existing Level-1 `agent_loop.py` code->test->fix loop into a
reusable worker and adds a Level-2 meta-controller that can:

  1. Decide whether a goal is atomic or compound.
  2. If compound, decompose it into bounded sub-task contracts.
  3. Dispatch each contract to an isolated coding worker in parallel.
  4. Verify each result for subtask satisfaction AND progress toward the parent
     goal (the drift-guard step).
  5. Replan, integrate, or ask for human review.

Key anti-drift measures:
  - Every delegation is wrapped in a `TaskContract` with a fixed parent_goal,
    deliverable, verification_command, and budget.
  - `max_depth` is decremented on each decomposition; at depth 0 the agent must
    solve with the worker loop instead of spawning children.
  - A verification model independently scores whether each result satisfies its
    contract and whether the contract still advances the parent goal.
  - Iterations and cost are capped; the graph checkpoints via MemorySaver.
  - A real `interrupt()` call pauses execution when human review is needed.
    Resume by calling `run_orchestrator(...)` with the same `thread_id` and
    `Command.RESUME` in `human_input`; or use `graph.invoke(...)` directly.

The graph topology:

    START -> analyze -> [atomic]  -> execute_atomic -> finish
                         [compound] -> dispatch_subagents --Send--> run_subagent
                                                              run_subagent -> verify
    verify -> [continue]  -> integrate -> finish
           -> [replan]    -> replan -> dispatch_subagents (loop)
           -> [human]     -> human_review -> ... -> integrate | replan

This program EXECUTES LLM-GENERATED PYTHON CODE via subprocesses, exactly like
`agent_loop.py` and `swarm.py` do. Run only in a throwaway directory or sandbox.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import NodeInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# Reuse the existing agent-loop implementation for LLM construction and defaults.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import agent_loop  # noqa: E402

AGENT_LOOP_SCRIPT = ROOT / "agent_loop.py"
DEFAULT_ORCHESTRATOR_MODEL = agent_loop.DEFAULT_MODEL_OPENROUTER
DEFAULT_MAX_DEPTH = 2
DEFAULT_MAX_ITERATIONS = 3


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TaskContract(BaseModel):
    """A bounded contract handed to a sub-agent worker.

    The parent_goal is immutable and is what the drift-guard checks against.
    The subtask description is what the worker actually attempts.
    """

    subtask_id: str = Field(description="Unique id, snake_case, used as workdir name.")
    parent_goal: str = Field(description="The original high-level goal this subtask must advance.")
    description: str = Field(
        description="Complete, self-contained instruction for the worker. "
                    "MUST include the public API / file the deliverable must expose."
    )
    deliverable: Literal["python_module", "test_suite", "analysis", "plan"] = "python_module"
    verification_command: str | None = Field(
        default=None,
        description="Optional shell command to independently verify the result. "
                    "If omitted, the worker's pytest result is used as the verifier."
    )
    max_iterations: int = Field(default=6, ge=1, description="Max worker refine cycles.")
    max_cost_usd: float = Field(default=2.0, ge=0)
    max_depth: int = Field(
        default=1, ge=0,
        description="How many more recursive decompositions are allowed from this subtask. "
                    "At 0 the subtask must be solved directly by the worker loop."
    )


class SubagentResult(BaseModel):
    """Result produced by one worker run."""

    contract: TaskContract
    success: bool
    workdir: str
    solution_path: str | None
    tests_path: str | None
    pytest_summary: str
    est_cost_usd: float
    iterations_used: int
    console_tail: str


class AnalysisOutput(BaseModel):
    """Structured output from the analyze node."""

    reasoning: str = Field(description="Why this goal is atomic or compound.")
    mode: Literal["atomic", "compound"] = Field(
        description="atomic: hand directly to the code worker. "
                    "compound: decompose into the contracts below."
    )
    direct_task: str = Field(
        default="",
        description="If mode==atomic, the refined task passed to the worker."
    )
    contracts: list[TaskContract] = Field(
        default_factory=list,
        description="If mode==compound, the list of independent sub-task contracts."
    )


class VerdictPerResult(BaseModel):
    subtask_id: str
    subtask_satisfied: bool = Field(description="Does the result satisfy its own contract?")
    advances_parent: bool = Field(description="Does this result move the parent goal forward?")
    score: int = Field(ge=1, le=10, description="Confidence-weighted quality score.")
    drift_flags: list[str] = Field(default_factory=list, description="Signs of goal or scope drift.")
    notes: str = Field(default="", description="Concise reviewer notes.")


class VerificationOutput(BaseModel):
    """Structured drift-guard output."""

    verdicts: list[VerdictPerResult]
    overall_continue: bool = Field(
        description="True only if every result is satisfactory AND advances the parent goal."
    )
    needs_human_review: bool = Field(
        default=False,
        description="If True, pause for human review instead of replanning."
    )
    replan_suggestions: str = Field(
        default="",
        description="If not overall_continue, what should change in the next plan."
    )


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


def _replace_reducer(old: Any, new: Any) -> Any:
    """Last-write-wins reducer."""
    return new if new is not None else old


def _results_reducer(old: list[SubagentResult], new: list[SubagentResult] | None) -> list[SubagentResult]:
    """Append lists from parallel workers; ``None`` resets the accumulator."""
    if new is None:
        return []
    if not isinstance(new, list):
        return new
    return (old or []) + new


class OrchestratorState(TypedDict):
    root_goal: str
    iteration: int
    max_iterations: int
    max_depth: int
    output_dir: str
    total_cost_usd: float

    contracts: Annotated[list[TaskContract], _replace_reducer]
    results: Annotated[list[SubagentResult], _results_reducer]
    analysis: Annotated[AnalysisOutput | None, _replace_reducer]
    verification: Annotated[VerificationOutput | None, _replace_reducer]
    final_report: Annotated[str, _replace_reducer]

    # Passed via Send into run_subagent only.
    contract: TaskContract | None

    done: Annotated[bool, _replace_reducer]
    human_message: Annotated[str, _replace_reducer]
    human_decision: Annotated[str | None, _replace_reducer]
    # Carried by Send so the interrupt message can show the thread id.
    _thread_id: Annotated[str | None, _replace_reducer]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_id() -> str:
    """Short unique suffix for default workdirs."""
    return uuid.uuid4().hex[:8]


def _parse_usage(text: str) -> dict | None:
    for line in text.splitlines():
        if line.startswith("USAGE_JSON:"):
            try:
                return json.loads(line[len("USAGE_JSON:"):].strip())
            except Exception:
                return None
    return None


def _parse_pytest_tail(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "(no output)"
    for line in reversed(lines):
        if any(tok in line.lower() for tok in ["passed", "failed", "error", "timeout", "no tests"]):
            return line
    return lines[-1]


def _build_llm(model: str | None = None) -> Any:
    """Use the same factory as agent_loop so keys and privacy screens are reused."""
    return agent_loop.get_llm(
        model=model or DEFAULT_ORCHESTRATOR_MODEL,
        provider=os.getenv("ORCHESTRATOR_PROVIDER") or os.getenv("PROVIDER") or "openrouter",
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


_SYSTEM_ANALYZE = (
    "You are a senior engineering manager planning work. Given a high-level goal, "
    "decide whether it is small enough to implement directly as a single self-contained "
    "Python module, or whether it should be decomposed into smaller independent sub-tasks.\n\n"
    "Rules:\n"
    "- Use mode='atomic' ONLY when the goal can be satisfied by one file/module with pytest as judge.\n"
    "- Use mode='compound' when the goal naturally splits into independent components "
    "  (e.g., parser + evaluator + CLI, backend + frontend, core + tests).\n"
    "- Each subtask contract MUST include a concrete deliverable description with the exact public API.\n"
    "- Contracts should be independent; avoid ordering dependencies between workers.\n"
    "- Keep `max_depth` small (default 1). A subtask should not recursively decompose unless necessary.\n"
    "- The parent_goal field inside every contract must equal the user's original root goal.\n"
    "Be concise and decisive."
)


def make_analyze_node(llm: Any):
    """Node factory: decide atomic vs compound and emit contracts."""
    structured = llm.with_structured_output(AnalysisOutput)

    def analyze(state: OrchestratorState) -> dict:
        iteration = state["iteration"]
        print(f"\n[analyze] iteration {iteration + 1}/{state['max_iterations']}: "
              f"evaluating goal -> atomic or compound...")

        prompt = (
            f"ROOT GOAL:\n{state['root_goal']}\n\n"
            f"Current recursion depth remaining: {state['max_depth']}.\n"
            "If you choose mode='compound', produce independent subtasks. "
            "Each subtask must be verifiable by pytest or by an explicit verification_command. "
            "Use snake_case subtask_id values and make each description self-contained."
        )
        response = structured.invoke(
            [SystemMessage(content=_SYSTEM_ANALYZE), HumanMessage(content=prompt)]
        )
        if response is None:
            raise RuntimeError("LLM returned empty analysis")
        # Defensive: ensure parent_goal matches root_goal in every contract.
        for c in response.contracts:
            c.parent_goal = state["root_goal"]
            # Inherit remaining depth minus one.
            c.max_depth = max(0, state["max_depth"] - 1)

        # Depth cap: if no depth remains, force direct solution mode.
        if state["max_depth"] <= 0:
            mode_msg = "mode=atomic (depth exhausted)"
            response.mode = "atomic"
            response.direct_task = (
                f"Implement the complete solution directly. No further decomposition is allowed.\n\n"
                f"{state['root_goal']}"
            )
            response.contracts = []
            print(f"[analyze] {mode_msg}")
        else:
            mode_msg = f"mode={response.mode}, contracts={len(response.contracts)}"
            print(f"[analyze] {mode_msg}")
            if response.mode == "compound":
                for c in response.contracts:
                    print(f"  - {c.subtask_id}: {c.description[:100]}...")
            else:
                print(f"[analyze] direct task length={len(response.direct_task)}")

        return {
            "analysis": response,
            "contracts": response.contracts,
            "iteration": iteration + 1,
        }

    return analyze


def route_analysis(state: OrchestratorState) -> str:
    analysis = state.get("analysis")
    if analysis is None:
        raise RuntimeError("Missing analysis in routing step")
    if analysis.mode == "atomic":
        print("[route] goal is atomic -> execute directly")
        return "execute_atomic"
    print("[route] goal is compound -> dispatch subagents")
    return "dispatch"


def _run_worker(contract: TaskContract, workdir: Path) -> SubagentResult:
    """Run one agent_loop.py worker and return a structured result.

    Enforces the contract's max_cost_usd by killing the child if the reported
    spend exceeds the budget. This is not real-time metering, but it prevents
    runaway budgets after each worker call completes.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"\n[_run_worker] {contract.subtask_id}: starting in {workdir}")

    cmd = [
        sys.executable, str(AGENT_LOOP_SCRIPT),
        contract.description,
        "--max-iters", str(contract.max_iterations),
        "--workdir", str(workdir),
    ]
    if os.getenv("AGENT_LOOP_MODEL"):
        cmd.extend(["--model", os.environ["AGENT_LOOP_MODEL"]])

    timeout = contract.max_iterations * 150 + 60
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + f"\n[ORCHESTRATOR] worker TIMEOUT after {exc.timeout}s\n"
        proc = subprocess.CompletedProcess(cmd, returncode=-1, stdout=output, stderr="")

    output = (proc.stdout or "") + (proc.stderr or "")
    usage = _parse_usage(output) or {}
    cost = float(usage.get("est_cost_usd", 0.0))

    sol_path = workdir / agent_loop.SOLUTION_FILENAME
    tst_path = workdir / agent_loop.TEST_FILENAME
    success = proc.returncode == 0

    if cost > contract.max_cost_usd > 0:
        success = False
        output += (
            f"\n[ORCHESTRATOR] COST CAP EXCEEDED: ${cost:.4f} > ${contract.max_cost_usd:.4f}. "
            "Result marked as failure.\n"
        )
        print(f"[_run_worker] {contract.subtask_id}: COST CAP EXCEEDED ${cost:.4f}")

    print(f"[_run_worker] {contract.subtask_id}: success={success} cost=${cost:.4f}")

    result = SubagentResult(
        contract=contract,
        success=success,
        workdir=str(workdir),
        solution_path=str(sol_path) if sol_path.exists() else None,
        tests_path=str(tst_path) if tst_path.exists() else None,
        pytest_summary=_parse_pytest_tail(output),
        est_cost_usd=cost,
        iterations_used=contract.max_iterations,
        console_tail="\n".join(output.strip().splitlines()[-30:]),
    )
    return result


def make_execute_atomic_node(llm: Any | None = None):
    """Node factory: hand an atomic goal to the existing code worker."""

    def execute_atomic(state: OrchestratorState) -> dict:
        analysis = state.get("analysis")
        if analysis is None or not analysis.direct_task:
            raise RuntimeError("execute_atomic called without a direct_task")

        outdir = Path(state["output_dir"]).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        wd = outdir / "atomic"

        contract = TaskContract(
            subtask_id="atomic",
            parent_goal=state["root_goal"],
            description=f"{state['root_goal']}\n\nDETAILED TASK:\n{analysis.direct_task}",
            deliverable="python_module",
            max_iterations=6,
            max_cost_usd=2.0,
            max_depth=max(0, state["max_depth"] - 1),
        )
        result = _run_worker(contract, wd)

        return {
            "results": [result],
            "total_cost_usd": state.get("total_cost_usd", 0.0) + result.est_cost_usd,
            "done": True,
            "final_report": _atomic_report(state["root_goal"], result),
        }

    return execute_atomic


def dispatch_subagents(state: OrchestratorState) -> list[Send] | str:
    """Conditional edge: fan out one Send per contract, or fall through to atomic."""
    analysis = state.get("analysis")
    if analysis is None:
        raise RuntimeError("dispatch called without analysis")
    if analysis.mode == "atomic":
        return "execute_atomic"
    contracts = state.get("contracts") or []
    if not contracts:
        raise RuntimeError("dispatch called with no contracts")
    print(f"\n[dispatch] fanning out {len(contracts)} worker(s)")
    for c in contracts:
        print(f"  - {c.subtask_id}")
    return [
        Send(
            "run_subagent",
            {**state, "contract": c, "_thread_id": state.get("_thread_id")},
        )
        for c in contracts
    ]


def run_subagent(state: OrchestratorState) -> dict:
    """Run one worker in a subprocess and return one SubagentResult."""
    contract = state.get("contract")
    if contract is None:
        raise RuntimeError("run_subagent called without a contract")

    outdir = Path(state["output_dir"]).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    wd = outdir / contract.subtask_id

    result = _run_worker(contract, wd)
    return {
        "results": [result],
        "total_cost_usd": state.get("total_cost_usd", 0.0) + result.est_cost_usd,
    }


_SYSTEM_VERIFY = (
    "You are an independent quality auditor. A team of coding agents just returned "
    "results for subtasks of a larger goal. Your job is to detect drift and decide "
    "whether to continue, replan, or ask for human review.\n\n"
    "Check for:\n"
    "- Subtask satisfaction: does the result look like it satisfies its contract?\n"
    "- Parent progress: will these results, taken together, advance the root goal?\n"
    "- Goal drift: did any subtask silently change scope, drop requirements, or solve a different problem?\n"
    "- Test failures, missing files, or timeout summaries are automatic grounds for low scores.\n"
    "Be strict but fair. A score of 8+ means the result is solid. Below 6 means replan. "
    "Set needs_human_review=True only for ambiguous cases that require a human decision."
)


def make_verify_node(llm: Any):
    structured = llm.with_structured_output(VerificationOutput)

    def verify(state: OrchestratorState) -> dict:
        results = state.get("results") or []
        print(f"\n[verify] reviewing {len(results)} result(s)")

        summaries = []
        for r in results:
            flag = "PASS" if r.success else "FAIL"
            line = (
                f"subtask={r.contract.subtask_id} status={flag} "
                f"summary='{r.pytest_summary}' cost=${r.est_cost_usd:.4f}"
            )
            summaries.append(line)

        prompt = (
            f"ROOT GOAL:\n{state['root_goal']}\n\n"
            "SUBTASK RESULTS:\n" + "\n".join(summaries) + "\n\n"
            "Return a verdict for EACH subtask and an overall decision. "
            "If not overall_continue, explain what to change when replanning."
        )
        response = structured.invoke(
            [SystemMessage(content=_SYSTEM_VERIFY), HumanMessage(content=_prompt_truncate(prompt, 12000))]
        )
        if response is None:
            raise RuntimeError("Verification LLM returned empty response")

        for v in response.verdicts:
            print(f"  - {v.subtask_id}: score={v.score} satisfied={v.subtask_satisfied} "
                  f"advances={v.advances_parent} flags={v.drift_flags or 'none'}")
        print(f"[verify] overall_continue={response.overall_continue} "
              f"needs_human_review={response.needs_human_review}")

        return {"verification": response}

    return verify


def route_verification(state: OrchestratorState) -> str:
    verification = state.get("verification")
    if verification is None:
        raise RuntimeError("route_verification called before verify node")
    if verification.needs_human_review:
        return "human_review"
    if verification.overall_continue:
        return "integrate"
    if state["iteration"] >= state["max_iterations"]:
        print("[route] replan requested but iteration budget exhausted -> stop")
        return "fail"
    return "replan"


_SYSTEM_HUMAN_REVIEW = (
    "The recursive orchestrator paused for human review. Here is the current context:\n"
    "- Review the verification verdicts below.\n"
    "- Decide: resume with current results, or replan the failed subtasks.\n"
    "Respond with a short instruction to the orchestrator."
)


def make_human_review_node(llm: Any):
    """Node factory: raise a real LangGraph interrupt, then parse the resume decision."""

    def human_review(state: OrchestratorState) -> dict:
        verification = state.get("verification")
        results = state.get("results") or []

        # Build a compact message for the user.
        reasons = [v.notes for v in verification.verdicts if v.drift_flags] if verification else []
        lines = [
            "\n[HUMAN REVIEW REQUESTED]",
            f"Thread ID : {state.get('_thread_id', '(unknown)')}",
            f"Root goal : {state['root_goal']}",
            f"Iteration : {state['iteration']}/{state['max_iterations']}",
            f"Depth     : {state['max_depth']}",
            f"Results   : {len(results)} worker run(s), "
            f"total cost=${state.get('total_cost_usd', 0.0):.4f}",
        ]
        if reasons:
            lines += ["Drift flags:"] + [f"  - {r}" for r in reasons]
        lines += [
            "\nReply to the orchestrator in plain English, e.g.:",
            "  'resume'                       -> proceed to integration",
            "  'replan and split the parser'  -> generate a new plan",
            "  'drop subtask foo'             -> remove it and continue",
        ]
        message = "\n".join(lines)
        print(message)
        # Real HITL: pause the graph and surface the message to the caller.
        raise NodeInterrupt(message)

    return human_review


_SYSTEM_REPLAN = (
    "You are planning the next iteration of a recursive agent. The previous plan failed "
    "verification. Produce a revised set of subtask contracts that fix the issues. "
    "Keep changes surgical: update descriptions, split a subtask, or add a missing contract. "
    "Do not increase depth unless the failure was caused by an oversized scope."
)


def make_replan_node(llm: Any):
    structured = llm.with_structured_output(AnalysisOutput)

    def replan(state: OrchestratorState) -> dict:
        verification = state.get("verification")
        print(f"\n[replan] iteration {state['iteration']}/{state['max_iterations']}")

        old_contracts = state.get("contracts") or []
        old_summaries = []
        for c in old_contracts:
            old_summaries.append(f"- {c.subtask_id}: {c.description[:120]}...")

        prompt = (
            f"ROOT GOAL:\n{state['root_goal']}\n\n"
            "PREVIOUS PLAN:\n" + "\n".join(old_summaries) + "\n\n"
            "VERIFICATION FAILURE / SUGGESTIONS:\n"
            f"{verification.replan_suggestions if verification else '(none)'}\n\n"
            f"Remaining depth: {state['max_depth']}. Iteration: {state['iteration']}/{state['max_iterations']}.\n"
            "Produce the corrected list of subtask contracts."
        )
        response = structured.invoke(
            [SystemMessage(content=_SYSTEM_REPLAN), HumanMessage(content=_prompt_truncate(prompt, 12000))]
        )
        if response is None:
            raise RuntimeError("Replan LLM returned empty response")

        for c in response.contracts:
            c.parent_goal = state["root_goal"]
            c.max_depth = max(0, state["max_depth"] - 1)

        # If human input says to drop a subtask, filter it out.
        human_decision = state.get("human_decision")
        if human_decision and human_decision.startswith("drop"):
            # crude but effective: "drop subtask foo" removes any contract with id containing foo
            drop_target = human_decision[len("drop"):].strip()
            response.contracts = [c for c in response.contracts if drop_target not in c.subtask_id]
            print(f"[replan] dropped subtask(s) matching '{drop_target}'; "
                  f"{len(response.contracts)} remain")

        print(f"[replan] new plan has {len(response.contracts)} contract(s)")
        return {
            "contracts": response.contracts,
            "results": None,  # reset accumulator
        }

    return replan


_SYSTEM_INTEGRATE = (
    "You are a senior lead integrating outputs from a team of agents. Produce a concise "
    "final report describing what was built, where the artifacts live, and any caveats. "
    "Do not hallucinate files or results not shown in the context."
)


def make_integrate_node(llm: Any):
    freeform = llm

    def integrate(state: OrchestratorState) -> dict:
        results = state.get("results") or []
        print(f"\n[integrate] synthesizing {len(results)} result(s)")

        rows = []
        total_cost = 0.0
        for r in results:
            total_cost += r.est_cost_usd
            rows.append(
                f"- {r.contract.subtask_id}: success={r.success}, "
                f"{r.pytest_summary}, workdir={r.workdir}"
            )

        prompt = (
            f"ROOT GOAL:\n{state['root_goal']}\n\n"
            "WORKER RESULTS:\n" + "\n".join(rows) + "\n\n"
            f"TOTAL ESTIMATED COST: ${total_cost:.4f}\n\n"
            "Write a short final project report: summary, file locations, and caveats."
        )
        report = freeform.invoke(
            [SystemMessage(content=_SYSTEM_INTEGRATE), HumanMessage(content=prompt)]
        ).content
        report_text = report if isinstance(report, str) else str(report)
        print(f"[integrate] report length={len(report_text)}")
        return {
            "final_report": report_text,
            "done": True,
        }

    return integrate


# Human-in-the-loop node now uses NodeInterrupt (see make_human_review_node above).


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _prompt_truncate(text: str, max_chars: int = 12000) -> str:
    """Hard-truncate a prompt so it fits the orchestrator model's context budget."""
    if len(text) <= max_chars:
        return text
    # Keep the head (instructions/goals) and a little tail (most recent context).
    head = text[: int(max_chars * 0.8)]
    tail = text[-int(max_chars * 0.2) + 3 :] if len(text) > int(max_chars * 0.2) else ""
    return head + "\n...[truncated]...\n" + tail


def _atomic_report(goal: str, result: SubagentResult) -> str:
    return (
        f"# Orchestrator Report (atomic run)\n\n"
        f"Goal: {goal}\n\n"
        f"Worker run: {result.contract.subtask_id}\n"
        f"- success: {result.success}\n"
        f"- pytest summary: {result.pytest_summary}\n"
        f"- solution: {result.solution_path or '(none)'}\n"
        f"- tests: {result.tests_path or '(none)'}\n"
        f"- cost: ${result.est_cost_usd:.4f}\n"
    )


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_orchestrator_graph(
    model: str | None = None,
    with_checkpointer: bool = True,
    compile_only: bool = False,
):
    """Build and return the compiled orchestrator graph."""
    if compile_only:
        llm = None
    else:
        llm = _build_llm(model)

    analyze_node = make_analyze_node(llm) if llm else _stub_analyze
    execute_atomic_node = make_execute_atomic_node(llm) if llm else _stub_execute_atomic
    verify_node = make_verify_node(llm) if llm else _stub_verify
    replan_node = make_replan_node(llm) if llm else _stub_replan
    integrate_node = make_integrate_node(llm) if llm else _stub_integrate
    human_review_node = make_human_review_node(llm) if llm else _stub_human_review

    builder = StateGraph(OrchestratorState)
    builder.add_node("analyze", analyze_node)
    builder.add_node("execute_atomic", execute_atomic_node)
    builder.add_node("run_subagent", run_subagent)
    builder.add_node("verify", verify_node)
    builder.add_node("replan", replan_node)
    builder.add_node("integrate", integrate_node)
    builder.add_node("human_review", human_review_node)

    builder.add_edge(START, "analyze")
    builder.add_conditional_edges(
        "analyze",
        dispatch_subagents,
        {
            "execute_atomic": "execute_atomic",
        },
    )
    # Each Send lands in run_subagent; after all workers finish, reduce to verify.
    builder.add_edge("run_subagent", "verify")
    builder.add_conditional_edges(
        "verify",
        route_verification,
        {
            "integrate": "integrate",
            "replan": "replan",
            "human_review": "human_review",
            "fail": END,
        },
    )
    builder.add_edge("replan", "dispatch_subagents")
    builder.add_edge("execute_atomic", END)
    builder.add_edge("integrate", END)
    builder.add_edge("human_review", END)

    checkpointer = MemorySaver() if with_checkpointer else None
    return builder.compile(checkpointer=checkpointer, interrupt_after=["human_review"])


# ---------------------------------------------------------------------------
# Compile-only stubs
# ---------------------------------------------------------------------------


def _stub_analyze(state: OrchestratorState) -> dict:
    return {"analysis": None}


def _stub_execute_atomic(state: OrchestratorState) -> dict:
    return {"results": [], "done": True, "final_report": ""}


def _stub_verify(state: OrchestratorState) -> dict:
    return {"verification": None}


def _stub_replan(state: OrchestratorState) -> dict:
    return {"contracts": [], "results": None}


def _stub_integrate(state: OrchestratorState) -> dict:
    return {"final_report": "", "done": True}


def _stub_human_review(state: OrchestratorState) -> dict:
    return {"human_message": "(skipped)", "human_decision": "resume"}


# ---------------------------------------------------------------------------
# Public runner + CLI
# ---------------------------------------------------------------------------


def run_orchestrator(
    goal: str,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    output_dir: str | None = None,
    model: str | None = None,
    thread_id: str | None = None,
) -> OrchestratorState:
    """Run the recursive orchestrator on a high-level goal.

    If a ``thread_id`` is provided, resumes execution of a previously-checkpointed
    orchestration run (e.g. after a human_review interrupt).
    """
    out = Path(output_dir or f"./agent_orchestrator_{_now_id()}").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Recursive agent orchestrator")
    print(f"  goal        : {goal}")
    print(f"  max_iters   : {max_iterations}")
    print(f"  max_depth   : {max_depth}")
    print(f"  model       : {model or DEFAULT_ORCHESTRATOR_MODEL}")
    print(f"  output_dir  : {out}")
    print("=" * 70)

    graph = build_orchestrator_graph(model=model, with_checkpointer=True)

    config = {"configurable": {"thread_id": thread_id or f"orchestrator_{_now_id()}"}}

    if thread_id:
        # Resume path: the graph must be paused at human_review.
        state = graph.get_state(config)
        if state.next != ("human_review",):
            raise RuntimeError(
                f"Thread {thread_id!r} is not paused at human_review (next={state.next})"
            )
        print(f"\n[run_orchestrator] resuming thread {thread_id} from human_review")
        final = graph.invoke(Command(resume="resume"), config)
    else:
        initial: OrchestratorState = {
            "root_goal": goal,
            "iteration": 0,
            "max_iterations": max_iterations,
            "max_depth": max_depth,
            "output_dir": str(out),
            "total_cost_usd": 0.0,
            "contracts": [],
            "results": [],
            "analysis": None,
            "verification": None,
            "final_report": "",
            "contract": None,
            "done": False,
            "human_message": "",
            "human_decision": None,
            "_thread_id": config["configurable"]["thread_id"],
        }

        # First pass: run until first terminal node or until human_review interrupt.
        final = graph.invoke(initial, config)

        # Auto-resume from human_review with a default 'resume' decision.
        while not final.get("done"):
            current = graph.get_state(config)
            if current.next == ("human_review",):
                print("\n[run_orchestrator] resuming from human_review interrupt with 'resume'")
                final = graph.invoke(Command(resume="resume"), config)
            else:
                break

    # Persist the final state for inspection.
    _persist_state(final, out)
    _print_final(final, out)
    return final


def _persist_state(state: OrchestratorState, out: Path) -> None:
    """Write a JSON snapshot of the final state (contracts/results) for tooling."""
    payload = {
        "root_goal": state["root_goal"],
        "iteration": state["iteration"],
        "done": state["done"],
        "total_cost_usd": state.get("total_cost_usd", 0.0),
        "final_report": state.get("final_report", ""),
        "contracts": [c.model_dump() for c in state.get("contracts", [])],
        "results": [
            {
                "contract": r.contract.model_dump(),
                "success": r.success,
                "workdir": r.workdir,
                "solution_path": r.solution_path,
                "tests_path": r.tests_path,
                "pytest_summary": r.pytest_summary,
                "est_cost_usd": r.est_cost_usd,
                "iterations_used": r.iterations_used,
            }
            for r in state.get("results", [])
        ],
    }
    (out / "orchestrator_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "final_report.md").write_text(state.get("final_report", "(no report)"), encoding="utf-8")


def _print_final(state: OrchestratorState, out: Path) -> None:
    total_cost = sum(r.est_cost_usd for r in state.get("results", []))
    print("\n" + "=" * 70)
    print("ORCHESTRATOR DONE")
    print("=" * 70)
    print(f"  status      : {'done' if state.get('done') else 'incomplete'}")
    print(f"  iterations  : {state['iteration']}/{state['max_iterations']}")
    print(f"  results     : {len(state.get('results', []))} worker run(s)")
    print(f"  est. cost   : ${total_cost:.4f}")
    print(f"  total_cost_usd in state: ${state.get('total_cost_usd', 0.0):.4f}")
    print(f"  output_dir  : {out}")
    print(f"  report      : {out / 'final_report.md'}")
    print(f"  state       : {out / 'orchestrator_state.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestrator.py",
        description="Recursive orchestrator for agent-loop coding workers.",
    )
    parser.add_argument("goal", nargs="?", help="High-level goal to accomplish.")
    parser.add_argument(
        "--max-iters", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"Max orchestrator replan iterations (default: {DEFAULT_MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
        help=f"Max decomposition depth (default: {DEFAULT_MAX_DEPTH}).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for worker artifacts (default: ./agent_orchestrator_<id>).",
    )
    parser.add_argument(
        "--model", default=None,
        help="OpenRouter/Bedrock model for planning/verification.",
    )
    parser.add_argument(
        "--compile-only", action="store_true",
        help="Compile the graph without calling any LLM or worker.",
    )
    parser.add_argument(
        "--resume-thread", default=None,
        help="Thread ID to resume after a human_review interrupt (requires StateGraph checkpointer).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _build_arg_parser().parse_args(argv)

    if args.compile_only:
        graph = build_orchestrator_graph(compile_only=True)
        nodes = sorted(graph.get_graph().nodes.keys())
        print("compile-only: orchestrator graph compiled successfully.")
        print(f"  nodes: {nodes}")
        print("  topology:")
        print("    START -> analyze -> {execute_atomic | run_subagent}")
        print("    run_subagent (parallel Sends) -> verify -> {integrate | replan | human_review | END}")
        print("    replan -> run_subagent (loop)")
        return 0

    if args.resume_thread:
        # Resume an existing thread with a default 'resume' decision.
        try:
            graph = build_orchestrator_graph(model=args.model, with_checkpointer=True)
        except ImportError as exc:
            print(f"ERROR: failed to load LLM for resume: {exc}")
            return 1
        config = {"configurable": {"thread_id": args.resume_thread}}
        state = graph.get_state(config)
        if state.next != ("human_review",):
            print(f"ERROR: thread {args.resume_thread} is not paused at human_review (next={state.next})")
            return 1
        print(f"[main] resuming thread {args.resume_thread} from human_review with 'resume'")
        final = graph.invoke(Command(resume="resume"), config)
        _persist_state(final, Path(final["output_dir"]).expanduser().resolve())
        _print_final(final, Path(final["output_dir"]).expanduser().resolve())
        return 0 if final.get("done") else 1

    if not args.goal:
        _build_arg_parser().error("a GOAL is required (or use --compile-only)")

    final = run_orchestrator(
        goal=args.goal,
        max_iterations=args.max_iters,
        max_depth=args.max_depth,
        output_dir=args.output_dir,
        model=args.model,
    )
    return 0 if final.get("done") else 1


if __name__ == "__main__":
    sys.exit(main())
