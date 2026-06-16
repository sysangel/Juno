"""swarm.py - fan a decomposed coding task across parallel loopy teams.

Conductor model: Claude (the orchestrator) splits ONE coding job into independent,
interface-pinned chunks and writes them to a MANIFEST (JSON). swarm.py runs each
chunk as its OWN `agent_loop.py` SUBPROCESS -- process isolation is required
because agent_loop's usage accounting (`_USAGE`) is module-global and NOT safe to
run concurrently in one process. Each chunk is a full DeepSeek code->test->fix
team judged by pytest. swarm caps concurrency, stops launching NEW chunks once a
cost ceiling is reached (in-flight runs finish), RE-RUNS pytest on each finished
workdir for an authoritative green/red, aggregates cost from each run's
`USAGE_JSON:` line, and copies every green solution.py to `output/<module>` for
the conductor to integrate.

Privacy + key handling are inherited from agent_loop.py UNCHANGED: the OpenRouter
data-secure provider screen is baked into get_llm(); OPENROUTER_API_KEY is read
from the environment by the child process and never printed. Defaults are
all-DeepSeek (both roles), so chunks need no model flags. Run this script with the
venv python so the subprocesses use it too.

Usage:
  python swarm.py <manifest.json>            # spends money -- shows plan first
  python swarm.py <manifest.json> --dry-run  # print the exact run plan, spend $0

Manifest schema:
  {
    "project": "myproj",
    "cost_ceiling_usd": 2.0,     # stop launching new chunks once cumulative
                                 #   COMPLETED cost >= this (in-flight runs finish,
                                 #   so real spend can overshoot by < concurrency runs)
    "max_concurrency": 5,        # how many teams run at once
    "max_iters": 6,              # default per-chunk iteration cap
    "workdir_base": null,        # default: <agent-loop>/runs/<project>
    "chunks": [
      {"name": "parser",
       "module": "parser.py",    # green solution.py copied here under output/
       "max_iters": 6,           # optional per-chunk override
       "task": "Implement ... Public API EXACTLY: def parse(...)..."}
    ]
  }
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AGENT_LOOP = ROOT / "agent_loop.py"

sys.path.insert(0, str(ROOT))
import agent_loop  # noqa: E402  (constants only; the loop runs as a subprocess)


def _truncate(s: str, n: int = 70) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def parse_usage(text: str) -> dict | None:
    for line in text.splitlines():
        if line.startswith("USAGE_JSON:"):
            try:
                return json.loads(line[len("USAGE_JSON:"):].strip())
            except Exception:
                return None
    return None


def parse_pytest(out: str) -> tuple[int, int, int]:
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


def parse_iters(out: str) -> str | None:
    m = re.search(r"[Ii]terations?[:\s]+(\d+)\s*/\s*(\d+)", out)
    return f"{m.group(1)}/{m.group(2)}" if m else None


def run_pytest(wd: Path) -> dict:
    """Authoritative re-judge: pytest is the only thing swarm trusts for green."""
    sol = wd / agent_loop.SOLUTION_FILENAME
    tst = wd / agent_loop.TEST_FILENAME
    if not sol.exists() or not tst.exists():
        return {"passed": False, "p": 0, "f": 0, "e": 0, "summary": "(missing solution/tests)"}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=no"],
            cwd=str(wd), capture_output=True, text=True, timeout=120,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        return {"passed": False, "p": 0, "f": 0, "e": 0,
                "summary": "TIMEOUT (possible catastrophic backtracking)"}
    p, f, e = parse_pytest(out)
    tail = out.strip().splitlines()[-1] if out.strip() else "(no output)"
    return {"passed": rc == 0 and f == 0 and e == 0 and p > 0,
            "p": p, "f": f, "e": e, "summary": tail}


def run_chunk(chunk: dict, base: Path, default_iters: int) -> dict:
    name = chunk["name"]
    iters = int(chunk.get("max_iters") or default_iters)
    wd = base / name
    wd.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(AGENT_LOOP), chunk["task"],
           "--max-iters", str(iters), "--workdir", str(wd)]
    log = wd / "run.log"
    timeout = iters * 700 + 180  # agent_loop has its own 600s/call cap + 1 retry
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=os.environ.copy())
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + f"\n[SWARM] TIMEOUT after {timeout}s\n"
        rc = -1
    log.write_text(out, encoding="utf-8")

    usage = parse_usage(out) or {}
    cost = float(usage.get("est_cost_usd") or 0.0)
    judged = run_pytest(wd)

    module_out = None
    sol = wd / agent_loop.SOLUTION_FILENAME
    if judged["passed"] and sol.exists() and chunk.get("module"):
        out_dir = base / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / chunk["module"]
        shutil.copy(sol, dst)
        module_out = str(dst)

    return {
        "name": name,
        "module": chunk.get("module"),
        "green": judged["passed"],
        "tests": f"{judged['p']}p/{judged['f']}f/{judged['e']}e",
        "pytest_tail": judged["summary"],
        "iters": parse_iters(out),
        "max_iters": iters,
        "cost": cost,
        "calls": usage.get("calls"),
        "in_tok": usage.get("input_tokens"),
        "out_tok": usage.get("output_tokens"),
        "rc": rc,
        "workdir": str(wd),
        "log": str(log),
        "module_out": module_out,
    }


def load_manifest(path: Path) -> dict:
    m = json.loads(path.read_text(encoding="utf-8"))
    if not m.get("chunks"):
        raise SystemExit("manifest has no chunks")
    for c in m["chunks"]:
        if not c.get("name") or not c.get("task"):
            raise SystemExit(f"chunk missing name/task: {c!r}")
    return m


def print_plan(m: dict, base: Path) -> None:
    iters = int(m.get("max_iters", 6))
    conc = int(m.get("max_concurrency", 5))
    ceiling = m.get("cost_ceiling_usd")
    print(f"SWARM PLAN  project={m.get('project')}  chunks={len(m['chunks'])}  "
          f"concurrency={conc}  cost_ceiling=${ceiling}")
    print(f"  workdir_base = {base}")
    print(f"  workers = all-DeepSeek (code + tests/critique = "
          f"{agent_loop.DEFAULT_CODE_MODEL_OPENROUTER}); pytest = judge")
    print("  --- chunks ---")
    for c in m["chunks"]:
        ci = int(c.get("max_iters") or iters)
        print(f"  [{c['name']:>16}] -> {c.get('module') or '(no module)':<18} "
              f"max_iters={ci}  task: {_truncate(c['task'])}")
    print("  NOTE: cost ceiling is checked BETWEEN waves; up to `concurrency` runs "
          "already in flight\n        will finish, so real spend can exceed the ceiling "
          "by < one wave.")


def write_results(m: dict, base: Path, results: list[dict], skipped: list[str]) -> None:
    total = sum(r["cost"] for r in results)
    green = [r for r in results if r["green"]]
    payload = {"project": m.get("project"), "total_cost_usd": round(total, 4),
               "green": len(green), "ran": len(results),
               "skipped_ceiling": skipped, "results": results}
    (base / "swarm_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = ["# Swarm results: " + str(m.get("project")), "",
            f"Ran {len(results)} chunk(s), {len(green)} green. "
            f"Total est cost ${total:.4f}. Skipped (ceiling): {skipped or 'none'}.", "",
            "| chunk | module | green | tests (p/f/e) | iters | est cost | log |",
            "|---|---|---|---|---|---|---|"]
    for r in results:
        rows.append(f"| {r['name']} | {r['module'] or '-'} | "
                    f"{'YES' if r['green'] else 'NO'} | {r['tests']} | "
                    f"{r['iters'] or '-'}/{r['max_iters']} | ${r['cost']:.4f} | "
                    f"`{Path(r['log']).name}` |")
    rows += ["", f"Green modules assembled under `{base / 'output'}` for integration.", ""]
    (base / "swarm_results.md").write_text("\n".join(rows), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fan a decomposed coding task across parallel loopy teams.")
    ap.add_argument("manifest", help="path to the chunk manifest JSON")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the run plan and exit without spending")
    args = ap.parse_args()

    m = load_manifest(Path(args.manifest).resolve())
    project = m.get("project") or "swarm"
    base = Path(m.get("workdir_base") or (ROOT / "runs" / project)).resolve()
    base.mkdir(parents=True, exist_ok=True)

    print_plan(m, base)
    if args.dry_run:
        print("\nDRY RUN -- no money spent. Re-run without --dry-run to execute.")
        return 0

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("NO_KEY_IN_ENV -- inject OPENROUTER_API_KEY before running (see loopy/swarm skill).")
        return 2

    iters = int(m.get("max_iters", 6))
    conc = max(1, int(m.get("max_concurrency", 5)))
    ceiling = m.get("cost_ceiling_usd")

    pending = list(m["chunks"])
    running: dict = {}
    results: list[dict] = []
    cum_cost = 0.0
    stopped = False

    print(f"\n>>> launching swarm: {len(pending)} chunks, up to {conc} at a time\n", flush=True)
    with ThreadPoolExecutor(max_workers=conc) as ex:
        while pending and len(running) < conc:
            c = pending.pop(0)
            running[ex.submit(run_chunk, c, base, iters)] = c
        while running:
            done, _ = wait(set(running), return_when=FIRST_COMPLETED)
            for fut in done:
                c = running.pop(fut)
                r = fut.result()
                results.append(r)
                cum_cost += r["cost"]
                print(f"    [{r['name']}] green={r['green']} {r['tests']} "
                      f"iters={r['iters']} cost=${r['cost']:.4f}  "
                      f"(cum ${cum_cost:.4f})", flush=True)
                if ceiling is not None and cum_cost >= float(ceiling):
                    stopped = True
            if not stopped:
                while pending and len(running) < conc:
                    c = pending.pop(0)
                    running[ex.submit(run_chunk, c, base, iters)] = c

    skipped = [c["name"] for c in pending]
    if skipped:
        print(f"\n!!! cost ceiling ${ceiling} reached at ${cum_cost:.4f} -- "
              f"skipped {len(skipped)} chunk(s): {skipped}", flush=True)

    write_results(m, base, results, skipped)
    green = [r for r in results if r["green"]]
    print("\nSWARM_DONE")
    print(f"  green {len(green)}/{len(results)}  total ${cum_cost:.4f}  "
          f"skipped {len(skipped)}")
    print(f"  RESULTS_MD: {base / 'swarm_results.md'}")
    print(f"  OUTPUT_DIR: {base / 'output'}  (green modules for integration)")
    failed = [r["name"] for r in results if not r["green"]]
    if failed:
        print(f"  RED (need attention): {failed}")
    return 0 if green and not failed and not skipped else 1


if __name__ == "__main__":
    sys.exit(main())
