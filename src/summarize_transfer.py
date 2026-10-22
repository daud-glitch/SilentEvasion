"""
Build the cross-model transfer matrix from results/transfer_rows_*.csv.

    python src/summarize_transfer.py

Writes results/transfer_matrix.csv and prints both subsets:
  * applied           - every perturbation the source applied, so the
                        denominator is how often the target caught the original
  * flipped_source    - only perturbations that fooled the source (the classic
                        matrix); small n, so the interval is the story

Also asserts the diagonal: source == target must reproduce the single-model ASR
exactly, because it is the same texts and the same prompt. A mismatch means the
replay is not faithful and the off-diagonal cells cannot be trusted.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.common import RESULTS_DIR  # noqa: E402
from src.transfer_common import matrix_rows, self_transfer  # noqa: E402

OUT = os.path.join(RESULTS_DIR, "transfer_matrix.csv")


def main():
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "transfer_rows_*.csv")))
    if not paths:
        print("no transfer rows in %s - run src/run_transfer.py per target" % RESULTS_DIR)
        return 1
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    rows = matrix_rows(df)
    if not rows:
        print("transfer rows exist but none had a caught original")
        return 1
    table = pd.DataFrame(rows).sort_values(["subset", "mode", "task", "source", "target"])
    table.to_csv(OUT, index=False)

    diagonal = df[df["source"] == df["target"]]
    bad = self_transfer(df)
    if not len(diagonal):
        print("diagonal check: NOT RUN (no source==target rows were replayed; "
              "re-run a target with --include-self to check faithfulness)")
    else:
        cells = diagonal[diagonal["counted"] == True].groupby(   # noqa: E712
            ["source", "mode", "task"])
        print("diagonal check (source == target reproduces the single-model ASR): %s"
              % ("OK" if not bad else "MISMATCH %s" % bad))
        print("| model | mode | task | rows | stored flips | replayed flips | agree |")
        print("|---|---|---|---|---|---|---|")
        n_rows = n_agree = 0
        for (source, mode, task), g in cells:
            stored = int(g["flipped_source"].sum())
            replayed = int(g["success"].sum())
            n_rows += len(g)
            n_agree += int((g["flipped_source"] == g["success"]).sum())
            print("| %s | %s | %s | %d | %d | %d | %s |"
                  % (source, mode, task, len(g), stored, replayed,
                     "yes" if stored == replayed else "NO"))
        print("\nself-replay agreement: %d/%d rows, %d/%d cells"
              % (n_agree, n_rows, len(cells) - len(bad), len(cells)))
    print("targets measured: %s" % ", ".join(sorted(df["target"].unique())))

    for subset in ("flipped_source", "applied"):
        sub = table[table["subset"] == subset]
        if not len(sub):
            continue
        print("\n=== %s ===" % subset)
        print("| source | target | mode | task | candidates | caught | flips | ASR | 95% CI |")
        print("|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            print("| %s | %s | %s | %s | %d | %d | %d | %.1f%% | %.1f-%.1f%% |"
                  % (r["source"], r["target"], r["mode"], r["task"],
                     r["n_candidates"], r["n_target_caught"], r["n_flips"],
                     100 * r["transfer_asr"], 100 * r["ci95_low"], 100 * r["ci95_high"]))

    thin = table[(table["subset"] == "flipped_source") & (table["n_target_caught"] < 10)]
    if len(thin):
        print("\n!! %d classic-matrix cell(s) have fewer than 10 caught originals; "
              "quote the interval, not the point estimate" % len(thin))
    print("\nsaved -> %s" % OUT)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
