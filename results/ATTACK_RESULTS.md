# Results — what is measured, what it means, what failed first

Run: Kaggle `sadiasadozai/slm-robustness-repair-v5` (2×T4), status **COMPLETE**,
126.8 min, outputs in `kaggle_output8/`, merged into `results/`. Full run report:
[KAGGLE_RUN_REPORT_v5.md](KAGGLE_RUN_REPORT_v5.md).

Every number below is recomputed from the per-sample CSVs by
`src/summarize_attack.py`, `src/analyze_pairs.py`, `src/summarize_transfer.py`
and `src/make_paper_tex.py`. Nothing is typed by hand.

## Protocol (as implemented)

- **Denominator.** Only samples the clean model classified malicious on the
  frozen 800-row split; the original is re-queried inside the attack run, and a
  sample that is no longer read as malicious is excluded with a reason (52 rows).
- **Success.** The generated verdict flips `malicious → benign`, with the same
  prompt as the baseline, temperature 0.
- **Budget.** One shared 15% edit budget per sample, measured by exact
  Levenshtein distance.

## Attack success (generation channel)

| mode | attacked | flips | pooled ASR |
|---|---|---|---|
| char | 1,044 | 12 | 1.1% |
| word | 1,035 | 28 | 2.7% |
| saliency | 1,001 | 46 | 4.6% |

Paired on shared messages: word-vs-char 26 vs 10 (exact McNemar p = 0.011, 1 of
6 cells significant alone); saliency-vs-word 33 vs 7 (p = 4.2 × 10⁻⁵, 2 of 6).
`results/paired_tests.csv`.

## Verdict channels disagree — measured, not assumed

The saliency attack optimises P(malicious) over two answer tokens; everything
else reads the generated verdict. Re-scoring every stored saliency row through
the generation route gives `results/saliency_verification.csv`:

| model | agree | probability-only flips | generation-only flips |
|---|---|---|---|
| phi3.5-mini | 283/283 | 0 | 0 |
| qwen2.5-3b | 291/336 | 45 | 0 |
| smollm2-1.7b | 367/382 | 0 | 15 |

Total 941/1,001. The paper reports the generation channel everywhere and treats
the gap as a result about probability probing.

## Transfer

Applied subset, off the diagonal: 27 of 3,707 caught messages flipped (0.7%);
13 of 24 cells nonzero. The classic flipped-source matrix holds 1–6 candidates
per cell (19 of 19 below ten caught originals), so it is kept in the artefact and
not tabulated.

**The self-replay faithfulness check ran and passed.** Each model's own
perturbations were re-sent through the same path (`--include-self`, kernel v6)
and compared against the flip count its attack recorded: 2,079 rows in 12
model–mode–task cells, 2,079/2,079 rows and 12/12 cells agree exactly. That is
what makes the off-diagonal cells a measurement rather than a description.

Saliency perturbations are **not** in the matrix: the v6 kernel ran the transfer
phase before the saliency attack, so those texts had not been generated yet.
Their transfer cells are missing data, not zero transfer. The phase order is
fixed in `src/make_kaggle.py` (transfer now runs after saliency), so the next
kernel run fills them and `tests/check_paper.py` requires the self-replay to
cover them.

## Defences

Pooled over models and tasks, probability channel throughout:

- **Normalisation** (NFKC + homoglyph folding + invisible-character removal)
  takes char-level ASR from 1.0% to 0.0% in every cell, and barely touches the
  word (2.5% → 2.0%) and saliency (7.5% → 6.9%) attacks. The fix matches the
  mechanism, and the null result on the other families is the evidence that the
  families fail differently.
- **Threshold at a 5% clean false-positive budget** is not free: SmolLM2-1.7B
  loses clean recall 100% → 34.5% / 23.0%, and on its cells the threshold
  *creates* evasions (0.0% → 29.0% and 0.0% → 13.3%), because a cutoff near the
  top of the score distribution sits right where perturbation noise lives.
  Phi-3.5-mini / SMS cannot meet the budget at all (`tau` would have to reject
  everything) and is reported as budget-not-met.

## Bugs found in our own pipeline (each inflated the result)

1. Per-operation edit budgets reaching 22% while the paper claimed 15%.
2. Estimated rather than measured edit counts.
3. A similarity metric labelled BERTScore that was word Jaccard.
4. An attack set drawn from messages the model never caught.
5. Saliency score constant at α = 0.5 without normalisation → greedy became
   left-to-right.
6. **v4:** the answer-token lookup compared the same whitespace token for both
   answers on Phi-3.5's tokenizer, so P(malicious) was exactly 0.5 for all 845
   defence rows. Fixed by skipping the shared prefix; the endpoint now raises
   instead of returning a constant.
7. **v4:** the defence threshold was chosen by *maximising* the cutoff (always
   "reject everything") and flips were counted without requiring the original to
   be malicious under the same rule, producing a 100% ASR for a defence.

Each of 5–7 was caught by a unit test that checks the mechanism, not the
headline number. All seven are now covered by tests.
