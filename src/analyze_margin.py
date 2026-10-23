"""
How far from the decision threshold were the messages the attacks were given?

    python src/analyze_margin.py

Section 5 used to say that the attack success rate orders by proximity to the
decision boundary, and the paper never measured the margin it was appealing to.
Every stored attack row carries `p_orig`, the probability the model gave the
ORIGINAL text on the two-token route the saliency attack uses, so the margin is
recoverable without touching a GPU: margin = p_orig - 0.5 on the messages the
probability route also called malicious.

Two things this script makes visible and the prose has to respect:

  * the route saturates on some models. Phi-3.5-mini returns exactly 1.0 on most
    rows, so no margin separates its messages and it cannot be placed on the
    scale at all -- which means its high ASR must be measured, not explained by
    a margin it does not have;
  * where the route does spread (Qwen2.5-3B, SmolLM2-1.7B) the model with the
    narrower mean margin is the one that flips more.

Writes results/margin_table.csv and prints the table the paper quotes.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import RESULTS_DIR  # noqa: E402

OUT = os.path.join(RESULTS_DIR, "margin_table.csv")
SATURATED = 0.9999          # p this close to 1.0 has no margin left to measure


def load_rows():
    """Every stored attack row that carries an original-text probability."""
    frames = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "defense_rows_*.csv"))):
        df = pd.read_csv(path)
        needed = {"id", "model", "task", "mode", "p_orig", "orig_malicious",
                  "success_baseline"}
        if not needed <= set(df.columns):
            print("skipped %s (missing %s)" % (os.path.basename(path),
                                               sorted(needed - set(df.columns))))
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True)
    rows["p_orig"] = pd.to_numeric(rows["p_orig"], errors="coerce")
    rows = rows[rows["p_orig"].notna()]
    # The denominator is the messages the probability route read as malicious:
    # the same rule the saliency attack applies before it attacks.
    rows = rows[rows["orig_malicious"].astype(bool)]
    # `flip_baseline` is the flip read on THIS channel (p_orig >= 0.5 and
    # p_pert < 0.5). `success_baseline` is not comparable across modes -- the
    # saliency rows carry the probability-route success there -- so the margin
    # comparison sticks to one channel.
    rows["flip"] = rows["flip_baseline"].astype(bool)
    rows["margin"] = rows["p_orig"] - 0.5
    return rows


def summarise(rows, model, task):
    sub = rows if task is None else rows[rows["task"] == task]
    if not len(sub):
        return None
    messages = sub.drop_duplicates(["id", "task"]) if task is None else \
        sub.drop_duplicates(["id"])
    flipped = sub[sub["flip"]]
    kept = sub[~sub["flip"]]
    return {
        "model": model,
        "task": task or "all",
        "n_rows": int(len(sub)),
        "n_messages": int(len(messages)),
        "n_prob_flips": int(len(flipped)),
        "mean_p_orig": round(float(sub["p_orig"].mean()), 4),
        "median_p_orig": round(float(messages["p_orig"].median()), 4),
        "mean_margin": round(float(sub["margin"].mean()), 4),
        "n_saturated": int((sub["p_orig"] >= SATURATED).sum()),
        "flip_mean_p": round(float(flipped["p_orig"].mean()), 4) if len(flipped) else None,
        "kept_mean_p": round(float(kept["p_orig"].mean()), 4) if len(kept) else None,
    }


def main():
    rows = load_rows()
    if not len(rows):
        print("no defense_rows_*.csv with p_orig found - run the defence probe first")
        return 1
    out = []
    for model in sorted(rows["model"].unique()):
        m = rows[rows["model"] == model]
        out.append(summarise(m, model, None))
        for task in sorted(m["task"].unique()):
            out.append(summarise(m, model, task))
    table = pd.DataFrame([r for r in out if r])
    table.to_csv(OUT, index=False)

    print("\n| model | task | rows | messages | mean p | median p | mean margin | "
          "saturated | prob flips | flip p | kept p |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in table.iterrows():
        print("| %s | %s | %d | %d | %.3f | %.3f | %.3f | %d | %d | %s | %s |"
              % (r["model"], r["task"], r["n_rows"], r["n_messages"],
                 r["mean_p_orig"], r["median_p_orig"], r["mean_margin"],
                 r["n_saturated"], r["n_prob_flips"],
                 "--" if r["flip_mean_p"] is None else "%.3f" % r["flip_mean_p"],
                 "--" if r["kept_mean_p"] is None else "%.3f" % r["kept_mean_p"]))

    print("\nMean margin on the caught messages, pooled over tasks "
          "(the generation-channel flip counts are in results/attack_table.csv):")
    per_model = table[table["task"] == "all"].sort_values("mean_margin")
    for _, r in per_model.iterrows():
        sat = 100.0 * r["n_saturated"] / r["n_rows"] if r["n_rows"] else 0.0
        print("  %-14s margin %.3f   prob flips %d of %d rows   saturated %d (%.0f%%)"
              % (r["model"], r["mean_margin"], r["n_prob_flips"], r["n_rows"],
                 r["n_saturated"], sat))

    flat = per_model[per_model["n_saturated"] < 0.5 * per_model["n_rows"]]
    if len(flat) > 1:
        lo = flat.loc[flat["mean_margin"].idxmin()]
        hi = flat.loc[flat["mean_margin"].idxmax()]
        print("\nInside the probability channel, between the models whose route does "
              "not saturate (%s): %s has the narrower margin (%.3f) and %d flips, "
              "%s the wider (%.3f) and %d."
              % (", ".join(flat["model"]), lo["model"], lo["mean_margin"],
                 lo["n_prob_flips"], hi["model"], hi["mean_margin"],
                 hi["n_prob_flips"]))
    sat = per_model[per_model["n_saturated"] >= 0.5 * per_model["n_rows"]]
    for _, r in sat.iterrows():
        print("\n%s cannot be placed on this scale: %d of its %d stored originals "
              "score at least %.4f, so the route returns no margin to compare."
              % (r["model"], r["n_saturated"], r["n_rows"], SATURATED))
    print("\nsaved -> %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
