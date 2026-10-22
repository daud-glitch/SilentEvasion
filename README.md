# Silent Evasion

Measuring the adversarial fragility of small language models used for security
triage — the artefact behind the paper *"Silent Evasion: Measuring the
Adversarial Fragility of Small Language Models Used for Security Triage"*.

**Repository:** <https://github.com/daud-glitch/SilentEvasion>

Everything the paper reports is recomputed from the CSV files in `results/`.
No number in the text is typed by hand: `src/make_paper_tex.py` builds every
table from the records, and `tests/check_paper.py` fails if a number in the
LaTeX stops matching them.

## What is measured

Three open small models (Qwen2.5-3B-Instruct, Phi-3.5-mini-instruct,
SmolLM2-1.7B-Instruct), zero-shot, one fixed prompt, two triage tasks
(phishing e-mail and SMS smishing). Four adversaries under one shared 15 % edit
budget:

| attack | oracle | search | stored as |
|---|---|---|---|
| character | verdict | uniform random draw | `results/attack_results_char_<model>.csv` |
| word | verdict | uniform random draw over a 14-entry table | `results/attack_results_word_<model>.csv` |
| saliency word | `P(malicious)` | PWWS-style ordering over a lexicon + WordNet | `results/attack_results_saliency_<model>.csv` |
| ablation (5 arms) | see below | see below | `results/ablation_<arm>_<model>.csv` |

## The measurement rules the code enforces

These are not documentation claims; each one has a test that fails if the code
stops implementing it.

1. **Conditional denominator.** Only messages the clean model classified
   malicious are attacked, and the original is re-queried inside the attack run
   (`src/build_attack_set.py`, `src/attack_common.py`).
2. **Same prompt on both sides.** Baseline and attack call the identical
   template, including the sample's own task (`src/prompting.py`).
3. **One shared budget, exact distance.** A single 15 % allowance per sample,
   measured as exact Levenshtein distance (`levenshtein` in
   `src/attack_common.py`), not estimated from a `difflib` opcode tally.
4. **Name the similarity you computed.** Word-set Jaccard is called Jaccard.
   No BERTScore is claimed.
5. **Every rate carries its interval.** Wilson (boundary-safe) plus a percentile
   bootstrap over messages.
6. **Messages and queries are different numbers.** 2,079 attacked messages cost
   8,047 model queries for the random attacks; 1,001 saliency messages cost
   22,964. Both are stored per row, never conflated.

## The ablation: ordering, candidate set, oracle

The paper's headline contrast (random word 2.7 % vs saliency 4.6 %) changes
three things at once. `src/run_ablation.py` holds two of them fixed per
comparison:

| arm | order | oracle | vocabulary |
|---|---|---|---|
| `random-prob` | uniform | `P(malicious)` | lexicon + WordNet |
| `saliency-prob` | saliency | `P(malicious)` | lexicon + WordNet |
| `random-verdict` | uniform | verdict | lexicon + WordNet |
| `group-verdict` | binary search over subsets | verdict | lexicon + WordNet |
| `textfooler-prob` | leave-one-out importance | `P(malicious)` | lexicon + WordNet |

Every arm pays the same probe phase — one query for the original plus one per
single-substitution candidate — under the same `--query-cap`, on the same
messages, so the budget is matched by construction rather than by assumption.

Contrasts: `saliency-prob` vs `random-prob` isolates **ordering**;
`random-prob` vs `random-verdict` isolates the **oracle**; `group-verdict` vs
`random-verdict` isolates how much a one-bit oracle recovers by searching;
`random-verdict` over the small table vs the lexicon isolates the **candidate
set**.

```bash
python src/run_ablation.py --model qwen2.5-3b --arm saliency-prob
python src/summarize_ablation.py
```

## Layout

```
src/            attacks, measurements, table and figure generation
tests/          the checks that guard each measurement rule
paper/          main.tex, sections/, generated tables, refs.bib
results/        per-sample records, aggregate tables, split fingerprints
data/processed/ the two public corpora after src/prepare_data.py
kaggle/         the GPU kernel that re-runs the study end to end
colab_bootstrap.py  self-contained cell that unpacks src/ + tests/ and verifies them
```

## Reproducing

**Data.** Both corpora are public; `python src/prepare_data.py` fetches and
normalises them into `data/processed/`.

**Splits.** Three strictly disjoint slices of the same shuffled pool —
evaluation, attack, and calibration — are built deterministically:

```bash
python src/build_splits.py     # writes results/*.csv + SPLIT_FINGERPRINTS.json
```

Eval and attack fingerprints are `a0592a3e…` and `48198ef4…`; a machine that
builds different data prints `DRIFT` and refuses to pretend otherwise. The
calibration slice is taken from the tail of the pool, so the published numbers
were computed on exactly the files this command rebuilds.

**GPU run.** `kaggle/kernel.py` runs baseline → random attacks → saliency →
ablation → paired tests → transfer → defences in one session and writes
`results/REPORT.md`. It needs a GPU: no result in this repository was produced
on a CPU substitute, and none is simulated.

**Local checks (no GPU).**

```bash
python tests/check_paper.py        # tables + prose numbers vs the CSVs
python tests/check_ablation.py     # the ablation engine's invariants
python tests/check_saliency.py     # the alpha-normalisation defect
python tests/check_attack.py       # ASR accounting and interval arithmetic
python tests/check_transfer.py     # target-owned denominator, self-replay
python tests/check_bootstrap.py    # the embedded cell matches src/ and tests/
```

## Status, stated plainly

- The saliency **transfer** cells and the **ablation** cells are implemented and
  wired into the kernel; their numbers are not in this version of the paper,
  which says so where it matters instead of quoting cells that were never run.
- The seed sweep (`src/run_seed_sweep.py`) exists so seed-to-seed spread can be
  reported next to the sampling interval. It has not been run.
- Llama-3.2-3B-Instruct is gated and has not been run; no result here refers to
  it.

## Citation

```bibtex
@inproceedings{khan2026silentevasion,
  title  = {Silent Evasion: Measuring the Adversarial Fragility of Small
            Language Models Used for Security Triage},
  author = {Khan, Muhammad Daud},
  year   = {2026}
}
```
