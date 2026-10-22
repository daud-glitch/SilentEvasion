"""
Re-read every saliency result through the PRIMARY verdict channel.

    python src/verify_saliency.py --base-url http://localhost:8000/v1
    python src/verify_saliency.py --model phi3.5-mini --limit 20

Why this exists
---------------
The char and word attacks decide success with a generated one-word answer
(`/v1/chat/completions` -> parse_verdict). The saliency attack needs a
probability at every candidate, so it decides success with
`/v1/verdict_probs` -> P(malicious) < 0.5 instead. Those are two different
readouts of the same model, and a table that compares their ASRs without
checking that they agree is comparing a ruler to a scale.

So every stored saliency row is re-scored through the generation route here. The
written CSV keeps both columns:

    pert_pred_gen   verdict from the same prompt the baseline used
    success_gen     orig was malicious and pert_pred_gen == benign

`src/summarize_attack.py` prefers `success_gen` when the column exists, which
makes the saliency ASR directly comparable to the other two modes.

Cost: one generation per stored saliency row (a few hundred per model), not per
candidate - the candidate search already happened.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.attack_common import chat_verdict, read_attack_csv  # noqa: E402
from src.common import RESULTS_DIR  # noqa: E402

OUT = os.path.join(RESULTS_DIR, "saliency_verification.csv")


def paths_for(model=None, results_dir=RESULTS_DIR):
    out = []
    for path in sorted(glob.glob(os.path.join(
            results_dir, "attack_results_saliency_*.csv"))):
        if model and not os.path.basename(path).endswith("%s.csv" % model):
            continue
        out.append(path)
    return out


def verify_file(path, base_url, limit=None):
    df = read_attack_csv(path)
    model = str(df["model"].iloc[0])
    counted = df["counted"] == True                        # noqa: E712
    idx = list(df.index[counted])
    if limit:
        idx = idx[:limit]
    print("verifying %s: %d counted rows" % (os.path.basename(path), len(idx)))

    for col in ("pert_pred_gen", "success_gen", "verdict_agree"):
        if col not in df.columns:
            df[col] = pd.NA

    done = 0
    for i in idx:
        text = df.at[i, "perturbed_text"]
        if not isinstance(text, str) or not text:
            continue
        pred, _raw, err = chat_verdict(base_url, model, text,
                                       task=str(df.at[i, "task"]))
        if err:
            print("!! %s failed: %s" % (df.at[i, "id"], err))
            continue
        df.at[i, "pert_pred_gen"] = -1 if pred is None else int(pred)
        df.at[i, "success_gen"] = bool(pred == 0)
        df.at[i, "verdict_agree"] = bool(pred == int(df.at[i, "pert_pred"]))
        done += 1
        if done % 25 == 0:
            print("  %d/%d verified" % (done, len(idx)), flush=True)

    df.to_csv(path, index=False)
    print("  %d rows verified, rewrote %s" % (done, os.path.basename(path)))
    return df


def summarize(dfs):
    out = []
    for df in dfs:
        model = str(df["model"].iloc[0])
        counted = df[df["counted"] == True]                # noqa: E712
        counted = counted[counted["success_gen"].notna()]
        for task, g in counted.groupby("task"):
            prob_only = int((g["success"] & ~g["success_gen"].astype(bool)).sum())
            gen_only = int((~g["success"].astype(bool)
                            & g["success_gen"].astype(bool)).sum())
            out.append({
                "model": model, "task": task, "n": len(g),
                "flips_prob": int(g["success"].sum()),
                "flips_gen": int(g["success_gen"].sum()),
                "agree": int(g["verdict_agree"].astype(bool).sum()),
                "prob_only": prob_only, "gen_only": gen_only,
            })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="one model key; default: every saliency CSV present")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--limit", default=None, type=int,
                    help="cap rows per file (smoke test)")
    ap.add_argument("--results-dir", default=RESULTS_DIR,
                    help="where the saliency CSVs live; point a smoke test at "
                         "a scratch copy so mock verdicts never land in "
                         "results/")
    args = ap.parse_args()

    paths = paths_for(args.model, args.results_dir)
    if not paths:
        print("no saliency result CSVs in %s" % args.results_dir)
        return 1
    dfs = [verify_file(p, args.base_url, args.limit) for p in paths]

    out_path = os.path.join(args.results_dir, os.path.basename(OUT))
    table = summarize(dfs)
    if os.path.exists(out_path):
        old = pd.read_csv(out_path)
        old = old[~old["model"].isin(table["model"])]
        table = pd.concat([old, table], ignore_index=True)
    table = table.sort_values(["model", "task"])
    table.to_csv(out_path, index=False)

    print("\n| model | task | n | flips (prob) | flips (gen) | agree | prob-only | gen-only |")
    print("|---|---|---|---|---|---|---|---|")
    for _, r in table.iterrows():
        print("| %s | %s | %d | %d | %d | %d | %d | %d |"
              % (r["model"], r["task"], r["n"], r["flips_prob"], r["flips_gen"],
                 r["agree"], r["prob_only"], r["gen_only"]))
    mismatched = int(table["prob_only"].sum() + table["gen_only"].sum())
    print("\nrows where the two channels disagree: %d" % mismatched)
    print("saved -> %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
