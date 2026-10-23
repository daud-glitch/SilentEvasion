# Paper state — what is written, what is measured, what is missing

**Title:** Silent Evasion: Measuring the Adversarial Fragility of Small Language
Models Used for Security Triage
**Target:** ACM AISec (acmart, sigconf). Fallback: IEEE TrustCom.
**Build (typeset):** `.tools/tectonic.exe main.tex` from `paper/` — currently
**16 pages**, no undefined references, no LaTeX errors.
**Build (Word):** `python src/make_docx.py` → `paper/main.docx`, 28 pages when
opened in Word, built from the same sections and tables as the PDF.
**Data:** Kaggle kernel `sadiasadozai/slm-robustness-ablation-transfer-v8`, COMPLETE
(401.4 min), merged into `results/`; older runs are in `archive/` and in the
`kaggle_output*` directories.

The rule for this document: a section is listed as **written** only when every
number in it comes from a CSV that exists. `tests/check_paper.py` enforces it —
it recomputes each quoted count from the CSVs, fails if the number is absent
from the prose, and fails on any percentage the CSVs cannot produce.

## Section order (the argument, not the file names)

| # | section | evidence |
|---|---|---|
| — | Abstract | `attack_table.csv`, `baseline_table.csv`, `saliency_verification.csv`, `transfer_matrix.csv` |
| 1 | Introduction | the four contributions are the four experiments that ran |
| 2 | Related work | `BIB_VERIFICATION.md` (every entry opened on a primary source) |
| 3 | Threat model & protocol | the four rules are encoded in `src/attack_common.py`; the 52 exclusions are in the per-sample records |
| 4 | Experimental setup | splits, model keys, shim config, run timing |
| 5 | Results (random attacks) | `t1_baseline.tex`, `t2_char.tex`, `t3_word.tex`, `t4_budget.tex`, `fig_budget`, `fig_asr` |
| 5.7 | Paired modes | `paired_tests.csv` via `analyze_pairs.py`, `t8_pairs.tex` |
| 6 | Saliency-guided word attack, and the matched ablation | `attack_table.csv` rows with `mode=saliency`, generation-verified, `t7_saliency.tex`; `ablation_contrasts.csv`, `t9_ablation.tex` |
| 7 | Transfer | `transfer_matrix.csv`, `t5_transfer.tex` (applied subset + self-replay) |
| 8 | Defences | `defense_summary.csv`, `defense_thresholds.csv`, `t6_defense.tex` |
| 9 | Discussion, limitations and future work | derived from the tables; channel and defence sections quote measured disagreements |
| 10 | Ethics & availability | the seven defects found in our own pipeline |
| 11 | Conclusion | no new claims |

Sections 6–8 are included by `main.tex` through `\IfFileExists{tables/tX_...}`,
so deleting a CSV removes the section rather than leaving a table behind. They
come *before* the discussion because they are results; the earlier draft put
them after the conclusion, which made the conclusion cite sections the reader
had not reached.

## The three headline numbers (all recomputed from CSVs)

- Random character vs random word, paired: **26 vs 10** flip-only messages,
  exact McNemar **p = 0.011**, 1 of 6 cells significant alone.
- Saliency vs random word, paired: **33 vs 7**, exact McNemar
  **p = 4.2 × 10⁻⁵**, 2 of 6 cells significant alone.
- Transfer, applied subset off-diagonal: **51 of 5,514** caught messages (0.9%),
  and the matched ablation's ordering contrast: **17 vs 0** message-specific
  flips, exact McNemar **p < 0.0001** over 1,001 paired messages.

## Measurement facts a reader must not be spared

- The decision margin is measured, not assumed: `src/analyze_margin.py` writes
  `results/margin_table.csv` (mean $p_{\text{orig}} - 0.5$ over caught rows:
  0.449 Qwen2.5-3B, 0.482 SmolLM2-1.7B, 0.494 Phi-3.5-mini). Phi's two-token
  route saturates — 800 of its 845 stored originals score at least 0.9999 — and
  the margin does **not** order the ASR, so the earlier boundary-proximity claim
  is withdrawn in §5, §1, the abstract, §9 and the conclusion.

- The saliency attack optimises P(malicious) over two answer tokens; the rest of
  the paper reads the generated verdict. They agree on **941 of 1,001** rows:
  perfect for Phi-3.5-mini, 45 probability-only flips on Qwen2.5-3B, 15
  generation-only flips on SmolLM2-1.7B.
- **Messages and queries are reported separately and never conflated.** The two
  random attacks send **2,079** caught messages in **8,047** model queries
  (every original is re-queried and every sample gets up to three attempts);
  the saliency attack sends 1,001 messages in 22,964 queries. Both counts come
  from the per-sample records, and `make_paper_tex.py` prints them so no one
  types them by hand.
- **The 52 exclusions are 43 + 9.** Forty-three are Qwen2.5-3B saliency rows
  whose probability route read the original as benign; nine are word-attack
  rows whose perturbation came out identical to the original. No other row was
  excluded in any run.
- **Saliency edits more, not less:** 9.3% of words against 7.6% for the random
  word attack (Table 4). Normalised per unit edited the advantage is ~1.4x, not
  the 1.7x the raw ASR ratio suggests.
- The defence table uses the probability channel throughout (a threshold needs a
  score), so its baseline column is **not** comparable row-for-row with the
  attack tables. It is read column to column.
- The self-transfer faithfulness check **ran and passed** (kernel v8): 3,080
  replayed rows in 18 model–mode–task cells, all reproducing the recorded flip
  count. `summarize_transfer.py` prints the per-cell table; a mismatch exits 1.
- The transfer matrix now covers **all three attacks**, including the
  saliency-guided one: 51 of 5,514 caught messages (0.9\%) flip off the diagonal,
  9 under the character attack, 18 under the word attack and 24 under the
  saliency attack. The classic flipped-source matrix holds 5 nonzero cells of 31,
  27 of them with fewer than ten caught originals.
- One model–task cell cannot meet the 5% false-positive budget at all
  (Phi-3.5-mini / sms); it is reported as budget-not-met, not as a defence.

## Claims that must never appear

- "state of the art", or any comparison to numbers reported by other papers.
- "Llama-3.2-3B results" — that model is gated and has not been run.
- "robust" without an operating point, or any ASR without its denominator and
  interval.
- "the saliency attack is PWWS" — it implements PWWS ordering on a hand-built
  candidate set; it is not the published attack.
- "the difference is caused by saliency ordering" — the candidate vocabulary
  differs too (`§6` says so explicitly). The ablation now measures both, and the
  candidate set is worth about as much as the ordering: 32 vs 1 discordant
  flips against 17 vs 0, so ordering has to be quoted with the vocabulary it
  was measured on.
- "the random attack is one draw per sample" — it re-queries the original and
  runs up to three attempts (3.9 queries per message).
- "perturbations do not transfer" — 51 off-diagonal flips happened; the honest
  phrasing is *low observed transfer*.
- "TextFooler is weaker than PWWS" — the leave-one-out arm spends 23.0 of its 24
  calls per message probing and has almost no budget left to edit with, so its
  contrast is about a query-starved ranking, not about the published attack.
- "the defence leaves word attacks untouched" — it moves them 7.5% to 6.9%.
- "52 saliency exclusions" — 43 are saliency, 9 are word-attack no-op edits.
- "the saliency attack is a transferable attack" — it moved 24 of 1,807 caught
  messages off the diagonal, which is low observed transfer, not a portable one.

## Checks that guard this file

```
python tests/check_paper.py        # tables, prose numbers, gating, claims, cites/refs
python tests/check_docx.py         # the Word build matches the same sources
python tests/check_attack.py       # ASR accounting and interval arithmetic
python tests/check_transfer.py     # target-owned denominator, self-replay
python tests/check_verify.py       # verdict-channel accounting
python tests/check_defense.py      # threshold rule and flip conditioning
python tests/check_saliency.py     # the alpha-normalisation defect
python tests/check_shim_probs.py   # answer-token lookup, left padding
python tests/check_bootstrap.py    # artefact + kernel embed current sources
python tests/check_runbook_exec.py # the runbook notebook still executes
```

`src/render_docx_check.py` is not a check of the content but of the container:
it opens `main.docx` in the installed Word and exports a PDF, which is the only
way to know that a file written by python-docx is a file Word will open.
