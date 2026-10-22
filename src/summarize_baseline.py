"""
Build the paper's baseline table from results/clean_baseline_results.csv.

Recomputing from the saved CSV (instead of trusting numbers printed during a
run) means the table cannot drift from the data it claims to summarise, and
the per-class counts make a degenerate model visible at a glance.

    python src/summarize_baseline.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.common import RESULTS_DIR, binary_metrics  # noqa: E402

INPUT_CSV = os.path.join(RESULTS_DIR, "clean_baseline_results.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "baseline_table.csv")


def summarize(df):
    rows = []
    for (model, task), g in df.groupby(["model", "task"]):
        pred = g["pred"]
        tp = int(((g.ground_truth == 1) & (pred == 1)).sum())
        fp = int(((g.ground_truth == 0) & (pred == 1)).sum())
        tn = int(((g.ground_truth == 0) & (pred == 0)).sum())
        fn = int(((g.ground_truth == 1) & (pred == 0)).sum())
        row = binary_metrics(g.ground_truth, pred)
        row.update({"model": model, "task": task,
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn})
        rows.append(row)
    cols = ["model", "task", "accuracy", "precision", "recall", "f1",
            "n", "n_unparsed", "tp", "fp", "tn", "fn"]
    return pd.DataFrame(rows)[cols]


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"missing {INPUT_CSV} - run the baseline first")
        return 1
    df = pd.read_csv(INPUT_CSV)
    table = summarize(df)
    print(table.to_string(index=False))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    print(f"\nsaved -> {OUT_CSV}")

    degenerate = table[table["n_unparsed"] == table["n"]]
    if len(degenerate):
        print(f"!! {len(degenerate)} model/task rows are entirely unparsed - plumbing, not a result")
    constant = table[(table["f1"].notna()) &
                     ((table["recall"] == 1.0) & (table["precision"] <= 0.55) |
                      (table["recall"] == 0.0) & (table["precision"] == 0.0))]
    if len(constant):
        print("!! check for degenerate rows (recall 1.0 with low precision):")
        print(constant[["model", "task", "precision", "recall", "tp", "fp"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
