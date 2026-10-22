"""
Diagnose a baseline results CSV.

A single accuracy number hides the two failure modes that matter for this
paper:
  * all-unparsed  -> plumbing (wrong URL, dead shim), NOT a model result
  * all-positive  -> degenerate classifier; high recall + poor precision
                     looks like detection but is constant prediction

Usage:
    python src/analyze_baseline.py                       # default results CSV
    python src/analyze_baseline.py path/to/results.csv
"""
import os
import sys

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")


def confusion(g):
    tp = int(((g.ground_truth == 1) & (g.pred == 1)).sum())
    fp = int(((g.ground_truth == 0) & (g.pred == 1)).sum())
    tn = int(((g.ground_truth == 0) & (g.pred == 0)).sum())
    fn = int(((g.ground_truth == 1) & (g.pred == 0)).sum())
    return tp, fp, tn, fn


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        RESULTS_DIR, "clean_baseline_results.csv")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}\n")

    if "pred" not in df.columns:
        print("ERROR: CSV has no 'pred' column; is this a results file?")
        return 1

    problems = []

    for (model, task), g in df.groupby(["model", "task"]):
        n = len(g)
        n_unparsed = int(g["pred"].isna().sum())
        n_pred_pos = int((g["pred"] == 1).sum())
        tp, fp, tn, fn = confusion(g.dropna(subset=["pred"]))

        print(f"=== {model} / {task} ===")
        print(f"n={n}  unparsed={n_unparsed}  predicted_malicious={n_pred_pos}"
              f"  ({n_pred_pos / max(n - n_unparsed, 1):.0%} of parsed)")
        print("confusion:            pred=malicious  pred=benign")
        print(f"  truth=malicious       {tp:>13}  {fn:>11}")
        print(f"  truth=benign          {fp:>13}  {tn:>11}")

        # ---- failure-mode checks -------------------------------------
        if n_unparsed == n:
            problems.append(
                f"{model}/{task}: ALL {n} rows unparsed -> plumbing bug "
                f"(check --base-url ends in /v1); do not report as a result")
            print("!! ALL ROWS UNPARSED - not a model finding.\n")
            continue

        parsed = n - n_unparsed
        if parsed and (n_pred_pos == parsed or n_pred_pos == 0):
            label = "malicious" if n_pred_pos else "benign"
            problems.append(
                f"{model}/{task}: constant prediction '{label}' on {parsed} "
                f"samples -> degenerate classifier, not detection")
            print(f"!! CONSTANT PREDICTION ('{label}') - degenerate.\n")
            continue

        if n_unparsed > 0.1 * n:
            problems.append(
                f"{model}/{task}: {n_unparsed}/{n} unparsed (>10%) -> inspect "
                f"the 'error' column; parse_verdict may be too strict")

        if g["error"].fillna("").astype(str).str.len().max() > 0:
            errs = g["error"].dropna()
            errs = errs[errs.astype(str).str.len() > 0]
            if len(errs):
                print(f"first error: {errs.iloc[0]}")
        print()

    print("=" * 60)
    if problems:
        print("ISSUES FOUND:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("No degenerate / plumbing issues detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
