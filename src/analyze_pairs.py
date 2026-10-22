"""
Paired comparison of attack modes on the SAME messages.

    python src/analyze_pairs.py

The paper's central claim is that a word-level edit buys more flips than a
character-level one at about half the unit cost. Until now that was a comparison
of two independent fractions per (model, task) cell, which is the wrong test:
both attacks ran on the same eligible samples, so the flips are paired and the
unpaired comparison throws away the pairing (and understates the precision).

This script pairs the per-sample results on (id, task) and reports:

  * the discordant counts b (char fails, word succeeds) and c (char succeeds,
    word fails). The claim needs b > c, and only b and c carry information about
    the difference;
  * an exact McNemar test (two-sided binomial on b/(b+c), the exact version
    rather than the chi-square approximation, because b+c is small here);
  * the paired risk difference (word ASR minus char ASR) with a percentile
    bootstrap that resamples MESSAGES, keeping the pair together. Resampling the
    two modes independently would break the very correlation the pairing exists
    to capture;
  * the same numbers for the saliency attack once it exists.

Writes results/paired_tests.csv and prints the table the paper quotes.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.attack_common import read_attack_csv  # noqa: E402
from src.common import RESULTS_DIR, SEED  # noqa: E402

OUT = os.path.join(RESULTS_DIR, "paired_tests.csv")
COMPARISONS = ((("word", "char")), (("saliency", "word")))


def load_mode(mode):
    """{(model, task, id): success} for every counted row of one mode."""
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR,
                                              "attack_results_%s_*.csv" % mode))):
        df = read_attack_csv(path)
        if "counted" not in df.columns:
            continue
        counted = df[df["counted"] == True]                  # noqa: E712
        # The saliency rows carry success_gen once src/verify_saliency.py has
        # re-scored them through the generated-verdict route; that is the
        # channel the other modes use, so the pairing uses it too.
        vcol = "success_gen" if "success_gen" in counted.columns else "success"
        for r in counted.itertuples(index=False):
            key = (str(r.model), str(r.task), r.id)
            flag = getattr(r, vcol)
            if flag is None or (isinstance(flag, float) and flag != flag):
                continue
            # a sample attacked twice (attempts>1) counts as a success if any
            # attempt flipped it - the same rule the summaries use
            out[key] = bool(out.get(key, False)) or bool(flag)
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(b, n, 0.5, alternative="two-sided").pvalue)
    except Exception:                       # no scipy: exact by hand
        from math import comb
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
        return min(1.0, 2 * tail)


def paired_ci(diffs, n_boot=10000, seed=SEED, alpha=0.05):
    """Percentile bootstrap over MESSAGES of the paired risk difference."""
    arr = np.asarray(diffs, dtype=float)
    if not arr.size:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def compare(a_mode, b_mode):
    """a_mode vs b_mode: positive risk difference means a_mode flips more."""
    A, B = load_mode(a_mode), load_mode(b_mode)
    common = sorted(set(A) & set(B))
    rows = []
    for model in sorted({k[0] for k in common}):
        for task in sorted({k[1] for k in common if k[0] == model}):
            keys = [k for k in common if k[0] == model and k[1] == task]
            if not keys:
                continue
            a_flips = [A[k] for k in keys]
            b_flips = [B[k] for k in keys]
            b_count = sum(1 for x, y in zip(a_flips, b_flips) if (not x) and y)
            c_count = sum(1 for x, y in zip(a_flips, b_flips) if x and (not y))
            diffs = [float(x) - float(y) for x, y in zip(a_flips, b_flips)]
            lo, hi = paired_ci(diffs)
            rows.append({
                "a_mode": a_mode, "b_mode": b_mode, "model": model, "task": task,
                "n_paired": len(keys),
                "a_flips": int(sum(a_flips)), "b_flips": int(sum(b_flips)),
                "a_asr": round(float(np.mean(a_flips)), 4),
                "b_asr": round(float(np.mean(b_flips)), 4),
                "risk_diff": round(float(np.mean(diffs)), 4),
                "diff_ci_low": round(lo, 4), "diff_ci_high": round(hi, 4),
                "only_a_mode": c_count, "only_b_mode": b_count,
                "mcnemar_p": round(mcnemar_exact(b_count, c_count), 6),
            })
    return rows


def main():
    all_rows = []
    for a_mode, b_mode in COMPARISONS:
        rows = compare(a_mode, b_mode)
        if rows:
            all_rows += rows
        else:
            print("no paired rows for %s vs %s (run both attacks first)"
                  % (a_mode, b_mode))
    if not all_rows:
        return 1
    out = pd.DataFrame(all_rows).sort_values(["a_mode", "b_mode", "task", "model"])
    out.to_csv(OUT, index=False)

    print("\n| comparison | task | model | paired | %s flips | %s flips | "
          "risk diff | 95%% CI | A-only / B-only | McNemar p |" % ("A", "B"))
    print("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in out.iterrows():
        print("| %s vs %s | %s | %s | %d | %d | %d | %+.1f%% | %+.1f to %+.1f | "
              "%d/%d | %.3f |"
              % (r["a_mode"], r["b_mode"], r["task"], r["model"],
                 r["n_paired"], r["a_flips"], r["b_flips"],
                 100 * r["risk_diff"], 100 * r["diff_ci_low"],
                 100 * r["diff_ci_high"], r["only_a_mode"],
                 r["only_b_mode"], r["mcnemar_p"]))

    sig = out[out["mcnemar_p"] < 0.05]
    print("\n%d of %d cells show a significant paired difference at p<0.05"
          % (len(sig), len(out)))

    # Pooled test. The caveat is real and belongs next to the number: the six
    # cells are NOT independent, because the same messages are reused across
    # models, so this p-value is a summary of the direction and not a
    # confirmatory test. It is reported because a reader will otherwise ask
    # "so is there a difference or not?" and the per-cell answers are all
    # underpowered.
    for (a_mode, b_mode), g in out.groupby(["a_mode", "b_mode"]):
        only_a = int(g["only_a_mode"].sum())
        only_b = int(g["only_b_mode"].sum())
        print("\npooled %s vs %s: %s-only flips %d, %s-only flips %d, exact "
              "McNemar p=%.4f (cells share messages across models, so this is "
              "a descriptive summary, not a confirmatory test)"
              % (a_mode, b_mode, a_mode, only_a, b_mode, only_b,
                 mcnemar_exact(only_b, only_a)))
    if not len(sig):
        print("NOTE: with this many cells, an unpaired comparison would have to "
              "be reported as a trend, not as a result.")
    print("saved -> %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
