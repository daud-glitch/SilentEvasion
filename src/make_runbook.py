"""
Generate `colab_runbook.ipynb` + `COLAB_STEPS.md`: the exact cell order to run
the study on a fresh Colab T4, with the screenshot markers for the paper.

One file to upload, then Run-all. The bootstrap cell is embedded, so the
notebook is self-contained: it does not depend on any other file being present.

Regenerate after changing any src/ file or the bootstrap:

    python src/make_bootstrap.py     # refresh the cell contents
    python src/make_runbook.py       # rebuild the notebook from it

`tests/check_bootstrap.py` fails when either generated artifact is stale.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BOOTSTRAP = os.path.join(ROOT, "colab_bootstrap.py")
OUT_NB = os.path.join(ROOT, "colab_runbook.ipynb")
OUT_MD = os.path.join(ROOT, "COLAB_STEPS.md")

MODEL = "Qwen/Qwen2.5-3B-Instruct"


def bootstrap_source():
    with open(BOOTSTRAP, "r", encoding="utf-8") as f:
        return f.read()


# Each entry: kind, title, body, and for code cells: keep (required?) + shot.
CELLS = [
    dict(kind="md", title="Setup rules", body="""\
# SLM Evasion Robustness — Colab Runbook

**Runtime → Change runtime type → T4 GPU**, then run the cells top to bottom.

The cells are numbered `C1 … C14`. Anything marked **REQUIRED** must run, in
order, on every fresh runtime. Anything marked **optional** can be deleted — the
deletion rules are in `COLAB_STEPS.md` and repeated in each cell's header.

Golden rules:
* One cell at a time on the first pass — check the output before moving on.
* Never `--limit` a baseline run if you intend to publish the number.
* `n_unparsed` must be 0. Anything else is plumbing, not a result.
"""),

    dict(kind="md", title="C1 · Bootstrap (REQUIRED, first, once per runtime)",
         body="""\
## C1 — Bootstrap

Rebuilds every `src/*.py` / `tests/*.py`, downloads the datasets only if
`data/processed/` is missing, and rebuilds the frozen splits.

**Do not delete this cell, and do not run anything before it.**
"""),
    dict(kind="code", title="C1 · Bootstrap", body=bootstrap_source(),
         keep=True, shot="📸 SCREENSHOT 1 (paper appendix, reproducibility): the whole "
                         "output ending in `BOOTSTRAP OK` — it shows the split fingerprints "
                         "`a0592a3e…` / `48198ef4…` that every result in the paper depends on."),

    dict(kind="md", title="C2 · GPU check (optional)",
         body="""\
## C2 — GPU / library check

Purely diagnostic. Delete it if you like — it changes no data.
Useful for the paper's *Setup* section.
"""),
    dict(kind="code", title="C2 · GPU check", body='''\
import platform, shutil, subprocess, sys
print("python :", sys.version.split()[0])
print("kernel :", platform.platform())
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv"], capture_output=True, text=True).stdout.strip())
else:
    print("no nvidia-smi -> CPU runtime (Runtime > Change runtime type > T4 GPU)")
try:
    import torch
    print("torch  :", torch.__version__, "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device :", torch.cuda.get_device_name(0))
        print("vram   : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
except ModuleNotFoundError:
    print("torch  : NOT INSTALLED -> real mode will fail")
try:
    import transformers, pandas
    print("transf :", transformers.__version__, "| pandas:", pandas.__version__)
except ModuleNotFoundError as exc:
    print("missing lib:", exc)
''', keep=False,
         shot="📸 SCREENSHOT 2 (optional, hardware table): the line with GPU name + VRAM."),

    dict(kind="md", title="C3 · Start shim (REQUIRED before any model call)",
         body="""\
## C3 — Start the inference shim

Starts `src/shim_colab.py` in real mode on port 8000 and waits for `/health`.

* Must be running before C4, C5, C7, C9–C11.
* The model is loaded lazily on the **first chat request** (C4), so this cell
  finishes in seconds — that is normal, not a failure.
* **Do not delete.** If the runtime restarts, run C1 then C3 again.
"""),
    dict(kind="code", title="C3 · Start shim", body='''\
import json, os, subprocess, sys, time, urllib.request

# SLM_PORT, not PORT: some shells export PORT=0, which made the shim bind a
# random port while every request still went to 8000.
PORT = int(os.environ.get("SLM_PORT", "8000"))

# a stale shim from a previous run would answer with the old model: clear it
if os.name == "posix":
    subprocess.run("fuser -k %d/tcp" % PORT, shell=True, capture_output=True)

log = open("shim.log", "wb")
shim_env = dict(os.environ, PORT=str(PORT))
shim_env.setdefault("SHIM_MODE", "real")
shim = subprocess.Popen([sys.executable, "src/shim_colab.py"], env=shim_env,
                        stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
print("shim pid:", shim.pid)

def health():
    try:
        with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=3) as r:
            return json.load(r)
    except Exception:
        return None

deadline = time.time() + 180
while time.time() < deadline:
    h = health()
    if h and h.get("status") == "ok":
        print("shim up:", h)
        break
    if shim.poll() is not None:
        print("shim exited early, code", shim.returncode)
        break
    time.sleep(3)
else:
    print("SHIM DID NOT START in 180 s")

print("--- shim.log (last lines) ---")
print("".join(open("shim.log", errors="replace").readlines()[-8:]))
''', keep=True),

    dict(kind="md", title="C4 · Template probe (REQUIRED first time, ~2 min)",
         body="""\
## C4 — Is the model actually triaging?

Three controlled messages through the **same** prompt builder the study uses.
This cell exists because a constant classifier and a missing chat template look
identical in an accuracy table.

Read the output before anything else:
* benign → `benign`, malicious → `malicious` ⇒ model works, continue to C5.
* all three → `malicious` (or all `benign`) ⇒ **degenerate**. Still continue to
  C5/C6 (that is a real finding), but understand C6 will flag it.
* `parsed=None` on all ⇒ parser or shim problem, fix before spending GPU time.

First call takes 60–120 s (model load into VRAM). **Do not delete on the first
run of a session** — it is your evidence that inference is real.
"""),
    dict(kind="code", title="C4 · Template probe", body='''\
import json, os, sys, time, urllib.request
sys.path.insert(0, os.getcwd())
from src.prompting import build_messages

PORT = int(os.environ.get("SLM_PORT", "8000"))
MODEL = "__MODEL__"

def ask(text, task="sms_smishing", timeout=300):
    msgs = build_messages(task, text)          # identical to the study harness
    payload = json.dumps({"model": MODEL, "messages": msgs,
                          "temperature": 0.0, "max_tokens": 8, "seed": 42}).encode()
    req = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",
                                 data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    shim = out.get("shim", {})
    return out["choices"][0]["message"]["content"], shim.get("parsed_verdict"), time.time() - t0

cases = [("benign   ", "Hey, are we still meeting at 5?"),
         ("malicious", "Your parcel is held at customs. Pay Rs 200 at http://bit.ly/pay2win"),
         ("benign   ", "ok thanks, see you tomorrow")]
labelled = []
for label, text in cases:
    raw, pred, dt = ask(text)
    labelled.append((label, pred))
    print("%s %6.1fs  raw=%-28r parsed=%s" % (label, dt, raw, pred))

preds = [p for _, p in labelled]
if None in preds:
    print("\\n!! some verdicts did not parse - check parse_verdict / shim envelope")
elif len(set(preds)) == 1:
    print("\\n!! CONSTANT PREDICTION across all three - model is not discriminating")
else:
    print("\\nOK - the model separates benign from malicious text")
''', keep=True,
         shot="📸 SCREENSHOT 3 (paper §Setup): this output — it proves real inference "
              "and shows the model is not stuck before any F1 number is claimed."),

    dict(kind="md", title="C5 · Clean baseline, Qwen2.5-3B (REQUIRED)",
         body="""\
## C5 — Clean baseline (the paper's Table 1, first row)

All **800** samples (400 SMS + 400 email). No `--limit`.

* ~4 min at ~3.9 it/s on a T4.
* Writes `results/clean_baseline_results.csv` — the canonical results file.
* `--limit 400` would give you **SMS only** (`eval_subset.csv` is blocked, not
  interleaved), so never use it for a publishable table.
"""),
    dict(kind="code", title="C5 · Baseline qwen2.5-3b", body='''\
get_ipython().system("python src/run_baseline.py --model qwen2.5-3b "
                     "--base-url http://localhost:8000/v1")
''', keep=True,
         shot="📸 SCREENSHOT 4 (Table 1, row 1): the summary block with "
              "`n=400 n_unparsed=0` per task."),

    dict(kind="md", title="C6 · Degeneracy gate (REQUIRED — do not skip)",
         body="""\
## C6 — Gate: is this accuracy real?

Two failure modes hide inside a single F1 number:

| Output | Meaning | What to do |
|---|---|---|
| `ALL ROWS UNPARSED` | plumbing (dead shim, wrong URL) | fix, re-run C5 |
| `CONSTANT PREDICTION` | degenerate classifier | report as a finding, **no ASR possible** |
| `No degenerate / plumbing issues` | usable baseline | continue to C7 |

You cannot measure attack success against a constant classifier — there is
nothing to flip. This gate is the difference between a paper and a table of
artefacts.

**Never delete this cell.** Delete it and the 0.55 you already saw goes into the
paper as if it meant something.
"""),
    dict(kind="code", title="C6 · analyze_baseline", body='''\
get_ipython().system("python src/analyze_baseline.py")
''', keep=True,
         shot="📸 SCREENSHOT 5 (paper §Limitations or Appendix, and your own decision "
              "log): the confusion matrix + the ISSUES/OK verdict line."),

    dict(kind="md", title="C7 · Remaining models (REQUIRED for RQ5, one cell each)",
         body="""\
## C7 — Other three models

One cell per model so a single OOM cannot take the rest down. Each appends to
the same CSV. Delete the cell of any model you are not reporting on.

* `llama3.2-3b` needs a HuggingFace login + licence click (`meta-llama/…` is gated).
  If it 401s, skip that cell and say so in the paper.
* `phi3.5-mini` needs `trust_remote_code` on some transformers versions; if it
  fails, note the version and skip.
* Model changes mid-session are fine: the shim loads one pipeline per model id
  and keeps them in VRAM. If you hit OOM, *Runtime → Restart*, then C1 → C3 again.
"""),
    dict(kind="code", title="C7a · Baseline llama3.2-3b", body='''\
get_ipython().system("python src/run_baseline.py --model llama3.2-3b "
                     "--base-url http://localhost:8000/v1")
''', keep=False, shot="Optional screenshot: the same summary block for this model."),
    dict(kind="code", title="C7b · Baseline smollm2-1.7b", body='''\
get_ipython().system("python src/run_baseline.py --model smollm2-1.7b "
                     "--base-url http://localhost:8000/v1")
''', keep=False),
    dict(kind="code", title="C7c · Baseline phi3.5-mini", body='''\
get_ipython().system("python src/run_baseline.py --model phi3.5-mini "
                     "--base-url http://localhost:8000/v1")
''', keep=False),

    dict(kind="md", title="C8 · All-model table (REQUIRED before writing)",
         body="""\
## C8 — The table the paper needs

Recomputes metrics from the CSV itself, so it cannot drift from the saved data.
"""),
    dict(kind="code", title="C8 · Summary table", body='''\
import os, sys
sys.path.insert(0, os.getcwd())
import pandas as pd
from src.common import binary_metrics

df = pd.read_csv("results/clean_baseline_results.csv")
rows = []
for (m, t), g in df.groupby(["model", "task"]):
    r = binary_metrics(g.ground_truth, g.pred)
    r.update({"model": m, "task": t})
    rows.append(r)
table = pd.DataFrame(rows)[["model", "task", "accuracy", "precision",
                            "recall", "f1", "n", "n_unparsed"]]
print(table.to_string(index=False))
print()
print("rows per model:", dict(df.groupby("model").size()))
print("errors:", int((df.get("error", pd.Series(dtype=str)).fillna("").astype(str).str.len() > 0).sum()))
table.to_csv("results/baseline_table.csv", index=False)
print("saved -> results/baseline_table.csv")
''', keep=True,
         shot="📸 SCREENSHOT 6 (Table 1, final): the full 4-model × 2-task table."),

    dict(kind="md", title="C9 · Attack eligibility (REQUIRED before any ASR)",
         body="""\
## C9 — Which samples may be attacked

ASR is only defined on samples the clean model got **right** (caught as
malicious). Attacking a message the model already called benign measures
nothing, and attacking a random malicious sample inflates/flattens ASR at
random.

This cell writes `results/attack_eligible.csv`: the honest denominator for
Phase 2. It covers every model × task, and joins each caught row to its message
text from `results/eval_subset.csv` (the baseline CSV is deliberately
`text`-free, so an attack row without the join would send an empty prompt).

**Read the catch-rate number.** If the baseline catches 45% of malicious SMS,
you can only attack those 45% — and the paper must say so, because the ASR is
conditional on detection.
"""),
    dict(kind="code", title="C9 · Attack eligibility", body='''\
get_ipython().system("python src/build_attack_set.py --all-models")
''', keep=True,
         shot="📸 SCREENSHOT 7 (paper §Results, denominator): the catch-rate line."),

    dict(kind="md", title="C10 · Attack (REQUIRED for every ASR number)",
         body="""\
## C10 — The attack phase

`src/run_attack.py` (on top of `src/attack_common.py`) is the only attack code
that may produce a reported ASR. What it does, and why each rule exists:

1. attacks **only** `results/attack_eligible.csv` — samples the clean model
   actually caught, so the denominator is real;
2. **re-queries the original** text instead of assuming it was malicious;
3. reuses `build_messages(task, …)` — the baseline prompt, and the sample's own
   task, so an email is not scored with the SMS prompt;
4. counts a success strictly as `orig_pred == 1 → pert_pred == 0`;
5. counts real edits with `difflib` (not `len(text) * 0.15`) and enforces **one**
   15% budget per sample shared by all operations;
6. reports **word-set Jaccard** under that name. There is no BERTScore here.

Two denominators exist on purpose: `asr` (flips / attacked) goes in the results
table, `asr_of_eligible` (flips / all eligible) in the appendix. Quote the
catch rate from C9 next to both.

`src/attack_char.py` / `attack_word.py` are the old scripts. They hard-code
`is_malicious_original = 1`, never query the original, estimate edits from the
length, and label Jaccard as BERTScore — do not use them for anything the paper
will print.
"""),
    dict(kind="code", title="C10a · Character-level attack", body='''\
get_ipython().system("python src/run_attack.py --model qwen2.5-3b --mode char "
                     "--attempts 3 --base-url http://localhost:8000/v1")
''', keep=True,
         shot="📸 SCREENSHOT 8 (paper §Results): the ASR block + per-task breakdown."),
    dict(kind="code", title="C10b · Word-level attack", body='''\
get_ipython().system("python src/run_attack.py --model qwen2.5-3b --mode word "
                     "--attempts 3 --base-url http://localhost:8000/v1")
''', keep=True,
         shot="📸 SCREENSHOT 9 (paper §Results): the second ASR block."),

    dict(kind="md", title="C11 · What to collect for the paper",
         body="""\
## C11 — Screenshot checklist

| # | Shot | Goes into |
|---|---|---|
| 1 | `BOOTSTRAP OK` + split fingerprints | Appendix / reproducibility statement |
| 2 | GPU name + VRAM | §Experimental setup |
| 3 | Template probe (benign vs malicious) | §Setup evidence of real inference |
| 4 | Baseline summary, row 1 | Table 1, row 1 |
| 5 | Confusion matrix + gate verdict | §Limitations / decision log |
| 6 | Full 4-model × 2-task table | Table 1 |
| 7 | Attack eligibility catch-rate | §Results denominator |

Missing so far (they need C10 to be fixed before they exist):
attack ASR table, cross-model transfer table, defense-probe figure.
"""),
]


def nb_source(text):
    """Notebook cells store source as a list of lines with keepends.

    `__MODEL__` is the one placeholder, so the model id lives in a single place
    (MODEL at the top of this generator) instead of being copy-pasted into cells.
    """
    return text.replace("__MODEL__", MODEL).splitlines(keepends=True)


def build_notebook():
    cells = []
    for c in CELLS:
        if c["kind"] == "md":
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": nb_source(c["body"])})
        else:
            cells.append({"cell_type": "code", "metadata": {},
                          "execution_count": None, "outputs": [],
                          "source": nb_source(c["body"])})
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3",
                           "language": "python"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def build_md():
    lines = [
        "# Colab steps — one sorted list",
        "",
        "Upload **`colab_runbook.ipynb`** to Colab (it is self-contained; the",
        "bootstrap cell is embedded). `Runtime → Change runtime type → T4 GPU`,",
        "then run the cells in this order.",
        "",
        "| Cell | What it does | Required? | Screenshot |",
        "|---|---|---|---|",
    ]
    code_ids = [c["title"] for c in CELLS if c["kind"] == "code"]
    for c in CELLS:
        if c["kind"] != "code":
            continue
        req = "**REQUIRED**" if c.get("keep") else "optional (deletable)"
        shot = c.get("shot") or "—"
        shot = shot.replace("|", "/")
        lines.append("| %s | %s | %s | %s |" % (
            c["title"].replace("|", "/"), _short(c["title"]), req, shot))
    lines += [
        "",
        "## Deletion rules",
        "",
        "* **C1 Bootstrap** — never delete, never run anything before it.",
        "* **C3 Shim** — never delete; without it every model call 404s and the",
        "  CSV fills with `n_unparsed == n` (looks like a model failure, isn't).",
        "* **C2 GPU** — deletable, print-only.",
        "* **C4 Probe** — required on the first run of a session (it is the only",
        "  evidence that inference is real and the model is not stuck); deletable",
        "  on later re-runs once you already have that screenshot.",
        "* **C5 / C6** — never delete. C6 is the difference between a measurement",
        "  and an artefact.",
        "* **C7a/b/c** — delete freely, one per model. Deleting a model's cell just",
        "  removes that row from Table 1; the others are unaffected.",
        "* **C8, C9, C10a/b** — keep; they regenerate the paper's tables from the CSV.",
        "* Deleting a cell in the **middle** is safe as long as the cells after it",
        "  do not need its state: C4/C5/C7/C8/C9 all need C3 running, and everything",
        "  needs C1. State lives in the runtime + `results/*.csv`, not in the cell.",
        "* Markdown cells carry no code — deleting them loses the warning and the",
        "  screenshot checklist, not any result.",
        "",
        "## Re-generating",
        "",
        "```powershell",
        "./venv/Scripts/python.exe src/make_bootstrap.py",
        "./venv/Scripts/python.exe src/make_runbook.py",
        "./venv/Scripts/python.exe tests/check_bootstrap.py   # RESULT: OK",
        "```",
        "",
        "Code cells: %d (%d required)." % (
            len(code_ids), sum(1 for c in CELLS if c["kind"] == "code" and c.get("keep"))),
    ]
    return "\n".join(lines) + "\n"


def _short(title):
    """Human 'what it does' for the table, kept short on purpose."""
    table = {
        "C1 · Bootstrap": "restores all code + data, verifies split fingerprints",
        "C2 · GPU check": "prints GPU/VRAM/torch versions",
        "C3 · Start shim": "launches the /v1 inference server on :8000",
        "C4 · Template probe": "3 known messages — real inference vs degenerate",
        "C5 · Baseline qwen2.5-3b": "800-sample clean baseline, Table 1 row 1",
        "C6 · analyze_baseline": "confusion matrix + degenerate/plumbing gate",
        "C7a · Baseline llama3.2-3b": "same baseline for a second model",
        "C7b · Baseline smollm2-1.7b": "same baseline for a third model",
        "C7c · Baseline phi3.5-mini": "same baseline for a fourth model",
        "C8 · Summary table": "rebuilds the full model × task table from the CSV",
        "C9 · Attack eligibility": "which samples may legitimately be attacked",
        "C10a · Character-level attack": "char-level ASR on the eligible set",
        "C10b · Word-level attack": "word-level ASR on the eligible set",
    }
    return table.get(title, title)


def main():
    if not os.path.exists(BOOTSTRAP):
        print("missing colab_bootstrap.py -> run: python src/make_bootstrap.py")
        return 1
    nb = build_notebook()
    with open(OUT_NB, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_md())
    code_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print("wrote %s (%d cells, %d code) and %s"
          % (os.path.basename(OUT_NB), len(nb["cells"]), code_cells,
             os.path.basename(OUT_MD)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
