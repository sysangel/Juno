"""Head-to-head: solo Kimi vs two-model (Kimi code + DeepSeek tests/critique).

Runs both configurations on a HARDER task (a regex -> NFA engine) that forces
multiple code->test->fix rotations, then CROSS-EVALUATES each solution against
the other config's test suite. DeepSeek's adversarial suite is the real
yardstick: a solo solution that passed its own easy tests may still fail the
adversarial ones.

Verbose loop output for each run is redirected to headtohead_out/<name>.log so
this driver's stdout stays a compact progress + pointer feed. Final artifacts:
  headtohead_out/results.json   (full structured results)
  headtohead_out/results.md     (comparison tables)
"""
from __future__ import annotations

import contextlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_loop  # noqa: E402

MAX_ITERS = 6

DEFAULT_TASK = r"""
Implement a small regular-expression engine in `solution.py`.

Public API - define EXACTLY these names so tests can import them:
  - `fullmatch(pattern: str, text: str) -> bool`
  - `class RegexError(ValueError)`   # raised for malformed patterns

Supported syntax (NO other features):
  - Literal characters: ASCII letters a-z, A-Z and digits 0-9 match themselves.
  - `.`   matches any single character.
  - `*`   postfix: zero or more of the preceding unit.
  - `+`   postfix: one or more of the preceding unit.
  - `?`   postfix: zero or one of the preceding unit.
  - `|`   alternation (LOWEST precedence).
  - `(` `)` grouping.
  - Concatenation by adjacency.

Matching semantics:
  - `fullmatch(pattern, text)` returns True iff the ENTIRE text is matched by the
    pattern (implicitly anchored at both ends), else False.
  - Precedence, lowest to highest: alternation `|`  <  concatenation  < postfix
    quantifiers `* + ?`. A postfix quantifier binds to the SINGLE preceding unit
    (a literal, `.`, or a parenthesized group). So `ab*` is `a` then `(b*)`, and
    `(ab)*` repeats the whole "ab".
  - The empty pattern matches ONLY the empty string.
  - An alternative may be empty: `a|` and `|a` are valid and one branch is the
    empty string. e.g. fullmatch("a|", "") is True and fullmatch("a|", "a") is True.
  - A quantifier may apply to a group, e.g. `(a*)*` is valid and MUST NOT hang on
    inputs like "aaaa" or "" (no catastrophic / exponential backtracking).

Error handling - raise `RegexError` (a subclass of ValueError) for malformed
patterns, specifically:
  - Unbalanced parentheses: "(", ")", "a(b", "a)b".
  - A postfix quantifier with nothing to quantify. RULE: a `*`, `+`, or `?` is
    valid ONLY immediately after a literal, a `.`, or a closing `)`. Anywhere
    else it is malformed - this covers a leading quantifier ("*a", "?x"), one
    right after "(" or "|" ("(*)", "a|*"), and a quantifier right after another
    quantifier ("a**", "a+?", "a*?").

Constraints:
  - Pure Python standard library, deterministic, no network / filesystem / threads
    / randomness.
  - Must handle pathological patterns in well under a second (use a Thompson NFA /
    memoized matcher; do NOT backtrack exponentially).
""".strip()

# Optional CLI override: `headtohead.py "<task text>"` runs a different task;
# with no positional arg it uses the built-in regex->NFA task above.
TASK = (sys.argv[1].strip()
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
        else DEFAULT_TASK)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "headtohead_out"
OUT.mkdir(exist_ok=True)


def _count_tests(tests: str) -> int:
    return len(re.findall(r"^\s*def test_\w+", tests or "", re.M))


def run_config(name: str, **kwargs) -> dict:
    agent_loop.reset_usage()
    wd = OUT / name
    log = OUT / f"{name}.log"
    with open(log, "w", encoding="utf-8", buffering=1) as f, \
            contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        final = agent_loop.run_agent_loop(
            task=TASK, max_iters=MAX_ITERS, workdir=str(wd), **kwargs
        )
    snap = agent_loop.usage_snapshot()
    return {
        "name": name,
        "passed": bool(final.get("passed")),
        "iterations": final.get("iteration"),
        "max_iters": final.get("max_iterations"),
        "n_tests": _count_tests(final.get("tests", "")),
        "calls": snap["calls"],
        "in_tok": snap["input_tokens"],
        "out_tok": snap["output_tokens"],
        "cost": snap["est_cost_usd"],
        "by_model": snap["by_model"],
        "history": final.get("history", []),
        "workdir": str(wd),
        "solution": str(wd / agent_loop.SOLUTION_FILENAME),
        "tests": str(wd / agent_loop.TEST_FILENAME),
    }


def _parse_pytest(out: str) -> tuple[int, int, int]:
    passed = failed = errors = 0
    for line in out.splitlines():
        m = re.search(r"(\d+) passed", line)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+) failed", line)
        if m:
            failed = int(m.group(1))
        m = re.search(r"(\d+) errors?", line)
        if m:
            errors = int(m.group(1))
    return passed, failed, errors


def cross_eval(name: str, solution_src: str, tests_src: str) -> dict:
    wd = OUT / name
    wd.mkdir(parents=True, exist_ok=True)
    src_ok = Path(solution_src).exists() and Path(tests_src).exists()
    if not src_ok:
        return {"name": name, "passed": 0, "failed": 0, "errors": 0,
                "rc": -2, "summary": "(missing solution or tests file)", "dir": str(wd)}
    shutil.copy(solution_src, wd / agent_loop.SOLUTION_FILENAME)
    shutil.copy(tests_src, wd / agent_loop.TEST_FILENAME)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=no"],
            cwd=str(wd), capture_output=True, text=True, timeout=120,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        out, rc = "TIMEOUT (possible catastrophic backtracking)", -1
    p, f, e = _parse_pytest(out)
    tail = out.strip().splitlines()[-1] if out.strip() else "(no output)"
    return {"name": name, "passed": p, "failed": f, "errors": e,
            "rc": rc, "summary": tail, "dir": str(wd)}


def main() -> int:
    A_label = "solo_deepseek"
    B_label = "two_model"

    print(">>> Run A: solo DeepSeek (DeepSeek writes tests AND code, no critique) ...", flush=True)
    A = run_config(A_label, model="deepseek/deepseek-v4-pro")
    print(f"    A done: green={A['passed']} iters={A['iterations']}/{A['max_iters']} "
          f"own_tests={A['n_tests']} calls={A['calls']} cost=${A['cost']:.4f}", flush=True)

    print(">>> Run B: two-model (Kimi code + DeepSeek tests/critique) ...", flush=True)
    # Pin Kimi explicitly so this stays a Kimi-vs comparison even though the loop
    # default flipped to all-DeepSeek. For an all-DeepSeek team run, drop the
    # code_model override (or set it to deepseek/deepseek-v4-pro).
    B = run_config(B_label, code_model="moonshotai/kimi-k2.7-code",
                   test_model="deepseek/deepseek-v4-pro")
    print(f"    B done: green={B['passed']} iters={B['iterations']}/{B['max_iters']} "
          f"own_tests={B['n_tests']} calls={B['calls']} cost=${B['cost']:.4f}", flush=True)

    print(">>> Cross-evaluating each solution against the other suite ...", flush=True)
    xAB = cross_eval("xeval_soloSol_vs_twoTests", A["solution"], B["tests"])
    xBA = cross_eval("xeval_twoSol_vs_soloTests", B["solution"], A["tests"])
    print(f"    solo solution vs two-model tests: {xAB['summary']}", flush=True)
    print(f"    two-model solution vs solo tests: {xBA['summary']}", flush=True)

    results = {A_label: A, B_label: B,
               "xeval_soloSol_vs_twoTests": xAB, "xeval_twoSol_vs_soloTests": xBA}
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    def row(r):
        return (f"| {r['name']} | {'YES' if r['passed'] else 'NO'} | "
                f"{r['iterations']}/{r['max_iters']} | {r['n_tests']} | {r['calls']} | "
                f"{r['in_tok']}/{r['out_tok']} | ${r['cost']:.4f} |")

    md = [
        "# Head-to-head: solo DeepSeek vs two-model (Kimi code + DeepSeek tests/critique)",
        "",
        f"Task: regex->NFA engine (`fullmatch` over `. * + ? | ( )` + `RegexError`). "
        f"max_iters={MAX_ITERS}.",
        "",
        "## Self-contained runs (each config must satisfy its OWN generated tests)",
        "",
        "| config | reached green | iterations | own tests | llm calls | tokens in/out | est cost |",
        "|---|---|---|---|---|---|---|",
        row(A),
        row(B),
        "",
        "## Cross-evaluation (each solution vs the OTHER config's suite)",
        "",
        "DeepSeek's adversarial suite is the hard yardstick.",
        "",
        "| solution | tested against | passed | failed | errors | pytest summary |",
        "|---|---|---|---|---|---|",
        f"| solo DeepSeek | two-model tests | {xAB['passed']} | {xAB['failed']} "
        f"| {xAB['errors']} | {xAB['summary']} |",
        f"| two-model (Kimi) | solo tests | {xBA['passed']} | {xBA['failed']} "
        f"| {xBA['errors']} | {xBA['summary']} |",
        "",
        f"Full per-run logs: `headtohead_out/{A_label}.log`, `headtohead_out/{B_label}.log`.",
        "Generated solutions/tests under `headtohead_out/<config>/`.",
        "",
    ]
    (OUT / "results.md").write_text("\n".join(md), encoding="utf-8")

    print("HEADTOHEAD_DONE")
    print("RESULTS_MD:", OUT / "results.md")
    print("RESULTS_JSON:", OUT / "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
