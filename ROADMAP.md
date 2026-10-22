# Step-by-Step Research Plan: SLM Evasion Robustness in Cybersecurity Triage

**Working Title:** *Silent Evasion: Measuring the Adversarial Fragility of Small
Language Models Used for Security Triage*
**Target Venue:** ACM AISec (acmart sigconf). Fallback: IEEE TrustCom.
**Hardware:** local Windows for development, free Kaggle T4 for GPU runs.

Status markers: `[x]` done and backed by a file in this repo, `[~]` code written
but the run has not delivered numbers yet, `[ ]` not started.

---

## Phase 1: Environment & Tooling

- [x] Python 3.12 venv at `venv/`; pandas, numpy, scipy, scikit-learn, tqdm,
      fastapi/uvicorn (shim), nltk+wordnet (saliency candidates), matplotlib
      (figures), pymupdf (local PDF inspection).
- [x] `colab_bootstrap.py` — one cell restores every source file, verified by
      SHA-256, plus the datasets.
- [x] `src/make_kaggle.py` — generates a self-contained Kaggle script kernel
      (bootstrap + probe + baseline + attacks + transfer + saliency + defence).

## Phase 2: Datasets

- [x] SMS smishing: `data/processed/sms_smishing.csv`, 5,574 rows.
- [x] Phishing e-mail: `data/processed/phishing_email.csv`, 2,000 rows.
- [x] Frozen splits: `results/eval_subset.csv` (800 rows, 200 malicious per task)
      and `results/attack_ch1_input.csv`, strictly disjoint (asserted in
      `tests/test_splits.py`).

## Phase 3: Clean Baseline

- [x] One zero-shot prompt for every model (`src/prompting.py`), temperature 0,
      8-token budget, verdict parsed to `malicious`/`benign`/unparsed.
- [x] Three models that ran end to end: Qwen2.5-3B, SmolLM2-1.7B,
      Phi-3.5-mini. Llama-3.2-3B is gated and is **not** in the paper.
- [x] `results/baseline_table.csv` + `tables/t1_baseline.tex`.

## Phase 4: Attack Implementation

- [x] Character level (homoglyph / delete / insert / swap) under ONE shared 15%
      budget, exact Levenshtein accounting — `src/attack_common.py`.
- [x] Word level, uniform-random over a small synonym table. **This is not
      PWWS** and is never described as such.
- [x] Saliency-guided word attack (PWWS ordering) — `src/saliency_common.py`,
      needs the shim's `/v1/verdict_probs` endpoint.
- [ ] Sentence-level paraphrase attack. Not started; it needs a paraphrase model,
      which is a new dependency and a new failure mode, so it stays out of scope
      for this paper.

## Phase 5: Adversarial Evaluation & Cross-Transfer

- [x] ASR per model and attack, with Wilson + bootstrap intervals on every row
      (`results/attack_table.csv`).
- [x] Perturbation rate and word-set Jaccard. **No BERTScore**: the repo does not
      ship a semantic-similarity model, so the paper reports Jaccard by name.
- [x] Transfer matrix (`src/run_transfer.py`, `src/summarize_transfer.py`);
      27 of 3,707 off-diagonal caught messages flipped.
- [x] Self-replay faithfulness check, in the same pass (`--include-self`): 2,079
      replayed rows in 12 cells, every one reproducing the recorded flip count.
      Character and word only — the v6 kernel ran transfer before the saliency
      attack, so saliency transfer cells are missing rather than zero; the phase
      order is fixed and `tests/check_paper.py` now fails if a transferred mode
      has no self-replay.
- [x] Saliency attack numbers, re-scored through the generation channel by
      `src/verify_saliency.py`; 46 flips in 1,001 attacks, 23 queries/message.
- [x] Verdict-channel agreement measured (941 of 1,001) instead of assumed.

## Phase 6: Defence & Paper

- [x] Training-free defences (`src/run_defense.py`): input normalisation and a
      5%-false-positive-budget threshold. Normalisation removes every
      character-level flip; the threshold backfires on the least calibrated
      model. Two bugs found and fixed (threshold direction, flip conditioning).
- [x] LaTeX draft `paper/main.tex`: 12 pages, all eleven sections in the PDF,
      tables and figures generated from CSVs.
- [x] Section order fixed: saliency (6), transfer (7) and defences (8) are
      results and now come before the discussion, ethics and conclusion. The
      drafts of those three used to sit after the conclusion.
- [x] The three results sections are wired through `\IfFileExists`, so they
      appear in the PDF the moment their table exists — and never before.
- [x] `tests/check_paper.py` fails on a stale table, a prose percentage the CSVs
      cannot produce, a section whose CSV was deleted, a citation with no
      bibliography entry, a `\ref` with no label, or a float no sentence points
      at.
- [x] Copy-edit pass: every `\cite` resolves and every bibliography entry has a
      `BIB_VERIFICATION.md` row. WordNet was the one entry without a row; it is
      now verified from its Crossref record because the ACM page returns 403.
- [x] Word build `paper/main.docx` (`src/make_docx.py`): real Word tables and
      embedded figures generated from the same `.tex` and CSVs, 22 pages when
      opened in Word. Audited by `tests/check_docx.py`; `src/render_docx_check.py`
      opens it in Word and exports a PDF, so the container is tested, not assumed.

---

## What remains, in order

1. **Saliency transfer cells** — the one mode missing from the matrix, and the
   cheapest gap to close: the columns exist and the replay now runs after the
   attack, so one kernel run fills them. Until then §7 names the gap instead of
   quoting a rate.
2. **Matched-vocabulary ablation** for the saliency attack: same candidates,
   uniform-random order. This is the experiment that separates ordering from
   candidate set, and §6 currently states the confound instead of removing it.
3. **Sentence-level paraphrase attack**, deliberately out of scope: it needs a
   paraphrase model and a new failure mode.
4. **Defence against the probability channel**, not just the text: the measured
   disagreement (941/1,001) says the readout itself is a target.
5. **Corresponding author's e-mail** — the only open item that cannot be
   produced from the repository: `main.tex` carries a
   `TODO(author): add the corresponding e-mail` comment because an address
   guessed by the tooling would be wrong on the paper's face page.
