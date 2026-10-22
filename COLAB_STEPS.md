# Colab steps — one sorted list

Upload **`colab_runbook.ipynb`** to Colab (it is self-contained; the
bootstrap cell is embedded). `Runtime → Change runtime type → T4 GPU`,
then run the cells in this order.

| Cell | What it does | Required? | Screenshot |
|---|---|---|---|
| C1 · Bootstrap | restores all code + data, verifies split fingerprints | **REQUIRED** | 📸 SCREENSHOT 1 (paper appendix, reproducibility): the whole output ending in `BOOTSTRAP OK` — it shows the split fingerprints `a0592a3e…` / `48198ef4…` that every result in the paper depends on. |
| C2 · GPU check | prints GPU/VRAM/torch versions | optional (deletable) | 📸 SCREENSHOT 2 (optional, hardware table): the line with GPU name + VRAM. |
| C3 · Start shim | launches the /v1 inference server on :8000 | **REQUIRED** | — |
| C4 · Template probe | 3 known messages — real inference vs degenerate | **REQUIRED** | 📸 SCREENSHOT 3 (paper §Setup): this output — it proves real inference and shows the model is not stuck before any F1 number is claimed. |
| C5 · Baseline qwen2.5-3b | 800-sample clean baseline, Table 1 row 1 | **REQUIRED** | 📸 SCREENSHOT 4 (Table 1, row 1): the summary block with `n=400 n_unparsed=0` per task. |
| C6 · analyze_baseline | confusion matrix + degenerate/plumbing gate | **REQUIRED** | 📸 SCREENSHOT 5 (paper §Limitations or Appendix, and your own decision log): the confusion matrix + the ISSUES/OK verdict line. |
| C7a · Baseline llama3.2-3b | same baseline for a second model | optional (deletable) | Optional screenshot: the same summary block for this model. |
| C7b · Baseline smollm2-1.7b | same baseline for a third model | optional (deletable) | — |
| C7c · Baseline phi3.5-mini | same baseline for a fourth model | optional (deletable) | — |
| C8 · Summary table | rebuilds the full model × task table from the CSV | **REQUIRED** | 📸 SCREENSHOT 6 (Table 1, final): the full 4-model × 2-task table. |
| C9 · Attack eligibility | which samples may legitimately be attacked | **REQUIRED** | 📸 SCREENSHOT 7 (paper §Results, denominator): the catch-rate line. |
| C10a · Character-level attack | char-level ASR on the eligible set | **REQUIRED** | 📸 SCREENSHOT 8 (paper §Results): the ASR block + per-task breakdown. |
| C10b · Word-level attack | word-level ASR on the eligible set | **REQUIRED** | 📸 SCREENSHOT 9 (paper §Results): the second ASR block. |

## Deletion rules

* **C1 Bootstrap** — never delete, never run anything before it.
* **C3 Shim** — never delete; without it every model call 404s and the
  CSV fills with `n_unparsed == n` (looks like a model failure, isn't).
* **C2 GPU** — deletable, print-only.
* **C4 Probe** — required on the first run of a session (it is the only
  evidence that inference is real and the model is not stuck); deletable
  on later re-runs once you already have that screenshot.
* **C5 / C6** — never delete. C6 is the difference between a measurement
  and an artefact.
* **C7a/b/c** — delete freely, one per model. Deleting a model's cell just
  removes that row from Table 1; the others are unaffected.
* **C8, C9, C10a/b** — keep; they regenerate the paper's tables from the CSV.
* Deleting a cell in the **middle** is safe as long as the cells after it
  do not need its state: C4/C5/C7/C8/C9 all need C3 running, and everything
  needs C1. State lives in the runtime + `results/*.csv`, not in the cell.
* Markdown cells carry no code — deleting them loses the warning and the
  screenshot checklist, not any result.

## Re-generating

```powershell
./venv/Scripts/python.exe src/make_bootstrap.py
./venv/Scripts/python.exe src/make_runbook.py
./venv/Scripts/python.exe tests/check_bootstrap.py   # RESULT: OK
```

Code cells: 13 (9 required).
