"""
Execute the runbook notebook cell by cell against a MOCK shim.

This is the only way to prove the notebook itself works without a GPU: it runs
the real cell sources in order, with `get_ipython().system()` stubbed to a
subprocess call, so a typo in a cell (or a wrong command) fails here instead of
halfway through a Colab session.

What is covered: C1 bootstrap, C2 env report, C3 shim launch, C4 probe,
C5 baseline (shortened to --limit 8 for speed), C6 gate, C8 table, C9 eligibility.
What is skipped: C7a/b/c (gated/remote models) and the real-GPU path.

Run:  python tests/check_runbook_exec.py
"""
import builtins
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "colab_runbook.ipynb")
SKIP_MARKERS = ("llama3.2-3b", "smollm2-1.7b", "phi3.5-mini")
PORT = 8000


class Shell:
    """Minimal stand-in for IPython's shell, only .system() is used by the cells."""

    def system(self, cmd):
        proc = subprocess.run(cmd, shell=True, cwd=os.getcwd(),
                              text=True, capture_output=True)
        out = (proc.stdout or "") + (proc.stderr or "")
        print(out, end="" if out.endswith("\n") else "\n")
        return out


def _stop_port(port):
    """Kill whatever still listens on `port` (the harness must not leak servers)."""
    if os.name == "posix":
        subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
        return
    out = subprocess.run(f'netstat -ano | findstr ":{port} "', shell=True,
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "LISTENING" in line:
            pid = line.split()[-1]
            subprocess.run(f"taskkill /PID {pid} /F", shell=True,
                           capture_output=True)
            print(f"[harness] stopped shim pid={pid}")


def run_cell(src, index):
    patched = src
    # keep the baseline cell fast: the mock shim has no GPU, 800 rows is pointless
    if "run_baseline.py --model qwen2.5-3b" in patched and "BOOTSTRAP" not in patched:
        patched = patched.replace("--model qwen2.5-3b ", "--model qwen2.5-3b --limit 8 ")
        print("[harness] patched baseline cell to --limit 8")
    builtins.get_ipython = lambda *a, **k: Shell()
    builtins.__dict__.setdefault("display", lambda *a, **k: None)
    ns = {"__name__": "__main__"}
    exec(compile(patched, f"<cell {index}>", "exec"), ns)
    return ns


def main() -> int:
    if not os.path.exists(NB):
        print("missing colab_runbook.ipynb -> run: python src/make_runbook.py")
        return 1

    with open(NB, encoding="utf-8") as f:
        nb = json.load(f)

    work = tempfile.mkdtemp(prefix="runbook_")
    os.makedirs(os.path.join(work, "data", "processed"), exist_ok=True)
    for name in ("sms_smishing.csv", "phishing_email.csv"):
        shutil.copy(os.path.join(ROOT, "data", "processed", name),
                    os.path.join(work, "data", "processed", name))

    os.environ["SLM_ROOT"] = work
    os.environ["SHIM_MODE"] = "mock"      # inherited by the launched shim
    # In Colab `!python` IS the kernel interpreter. Locally it is not (a bare
    # system python without sklearn/pandas), so put this venv first on PATH to
    # reproduce Colab's behaviour instead of testing a different interpreter.
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    os.chdir(work)
    print(f"workdir: {work}")

    failures = []
    ran = 0
    try:
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if any(m in src for m in SKIP_MARKERS):
                print(f"\n=== cell {i}: SKIPPED (remote/gated model) ===")
                continue
            print(f"\n=== cell {i} ===")
            ran += 1
            try:
                run_cell(src, i)
            except SystemExit as exc:
                # the bootstrap cell ends with sys.exit(0) on success; in a
                # notebook that just stops the cell, so keep going.
                if exc.code not in (0, None):
                    failures.append(f"cell {i}: exited with {exc.code}")
                    print(f"FAILED cell {i}: SystemExit {exc.code}")
                    break
                print(f"[harness] cell {i} ended with SystemExit(0) - expected")
            except Exception as exc:
                failures.append(f"cell {i}: {type(exc).__name__}: {exc}")
                print(f"FAILED cell {i}: {type(exc).__name__}: {exc}")
                break

        # artefacts the later cells promise
        expected = ["results/eval_subset.csv", "results/attack_ch1_input.csv",
                    "results/clean_baseline_results.csv", "results/baseline_table.csv",
                    "results/attack_eligible.csv", "src/build_splits.py",
                    "tests/check_bootstrap.py"]
        for rel in expected:
            path = os.path.join(work, rel)
            if not os.path.exists(path):
                failures.append(f"missing artefact after run: {rel}")
            else:
                print(f"artefact ok: {rel} ({os.path.getsize(path)} bytes)")
    finally:
        _stop_port(PORT)
        os.chdir(ROOT)
        shutil.rmtree(work, ignore_errors=True)

    print("\nRESULT:", "OK" if not failures else f"{len(failures)} problem(s)")
    for f in failures:
        print("  FAIL", f)
    print(f"cells executed: {ran}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
