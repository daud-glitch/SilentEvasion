"""
Seed sweep: how much of an attack's success rate is the attack, and how much
is the seed?

    python src/run_seed_sweep.py --model qwen2.5-3b --mode word
    python src/run_seed_sweep.py --model qwen2.5-3b --mode word --seeds 42,43,44,45,46

A single ASR from a single seed is a number about one draw. This runner
repeats the SAME attack on the SAME eligible samples with several seeds and
reports:

  * the ASR of every seed, and its mean, spread and range;
  * the per-sample matrix of which seed flipped which message, and from it
    the two numbers that matter for the paper: the samples flipped by EVERY
    seed (the attack works on them) and the samples flipped by SOME seed (the
    attack CAN work on them). The gap between the two is the part of the
    reported ASR that was luck on that particular run.

Writes results/seed_sweep_<mode>_<model>.csv (one row per sample, one column
per seed) and results/seed_sweep_<mode>_<model>.txt (the summary).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.attack_common import (asr_interval, chat_verdict,  # noqa: E402
                               load_eligible, run_attack)
from src.common import RESULTS_DIR, SEED  # noqa: E402

ELIGIBLE_PATH = os.path.join(RESULTS_DIR, "attack_eligible.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--task", default=None)
    ap.add_argument("--mode", default="word", choices=["char", "word"])
    ap.add_argument("--seeds", default="42,43,44,45,46",
                    help="comma-separated seeds (the first one is the study's "
                         "recorded seed when it is 42)")
    ap.add_argument("--attempts", default=1, type=int)
    ap.add_argument("--max-rate", default=0.15, type=float)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--eligible", default=ELIGIBLE_PATH)
    ap.add_argument("--limit", default=None, type=int)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds needs at least one seed")

    samples = load_eligible(args.eligible)
    if "model" in samples.columns:
        models = sorted(samples["model"].unique())
        model = args.model or (models[0] if len(models) == 1 else None)
        if model is None:
            raise SystemExit(f"{args.eligible} holds {models}; pass --model")
        samples = samples[samples["model"] == model]
    else:
        model = args.model or "unknown"
    if args.task:
        samples = samples[samples["task"] == args.task]
    if args.limit:
        samples = samples.head(args.limit)
    if samples.empty:
        raise SystemExit("no eligible samples")

    def chat_fn(text, task):
        return chat_verdict(args.base_url, model, text, task=task)[0]

    per_seed, frames = [], {}
    for seed in seeds:
        df, summary = run_attack(samples, mode=args.mode, model=model,
                                 attempts=args.attempts, max_rate=args.max_rate,
                                 chat_fn=chat_fn, seed=seed, quiet=True)
        counted = df[df["counted"] == True]              # noqa: E712
        flags = counted["success"].tolist()
        per_seed.append({
            "seed": seed,
            "n_attacked": summary["n_attacked"],
            "flips": summary["n_success"],
            "asr": summary["asr"],
            "mean_perturbation_rate": summary["mean_perturbation_rate"],
        })
        frames[seed] = df.set_index("id")["success"]
        print("seed %-4d attacked %-4d flips %-4d ASR %.1f%%"
              % (seed, summary["n_attacked"], summary["n_success"],
                 100 * summary["asr"]))

    # per-sample matrix
    matrix = pd.DataFrame({("seed%d" % s): frames[s] for s in seeds})
    matrix = matrix.reset_index()
    out_csv = os.path.join(RESULTS_DIR,
                           "seed_sweep_%s_%s.csv" % (args.mode, model))
    matrix.to_csv(out_csv, index=False)

    asrs = np.array([r["asr"] for r in per_seed], dtype=float)
    counted_flags = matrix[[("seed%d" % s) for s in seeds]].fillna(False)
    all_seeds = counted_flags.all(axis=1)
    some_seed = counted_flags.any(axis=1)
    n = int(len(counted_flags))
    pooled = asr_interval([bool(v) for v in counted_flags.values.ravel()])

    lines = []
    say = lines.append
    say("=== seed sweep: %s-level attack, model=%s ===" % (args.mode, model))
    say("seeds            : %s" % ", ".join(str(s) for s in seeds))
    say("eligible samples : %d" % n)
    say("")
    say("%-8s %-10s %-8s %-10s" % ("seed", "attacked", "flips", "ASR"))
    for r in per_seed:
        say("%-8d %-10d %-8d %-9.1f%%"
            % (r["seed"], r["n_attacked"], r["flips"], 100 * r["asr"]))
    say("")
    say("ASR across seeds : mean %.2f%%  sd %.2f%%  min %.2f%%  max %.2f%%"
        % (100 * asrs.mean(), 100 * asrs.std(ddof=1) if len(asrs) > 1 else 0.0,
           100 * asrs.min(), 100 * asrs.max()))
    say("")
    say("flipped by EVERY seed : %d/%d (%.1f%%)  <- the attack works here"
        % (int(all_seeds.sum()), n, 100 * all_seeds.mean() if n else 0.0))
    say("flipped by SOME seed  : %d/%d (%.1f%%)  <- the attack CAN work here"
        % (int(some_seed.sum()), n, 100 * some_seed.mean() if n else 0.0))
    say("flipped by ONE seed   : %d/%d (%.1f%%)  <- this part is seed luck"
        % (int((some_seed & ~all_seeds).sum()), n,
           100 * (some_seed & ~all_seeds).mean() if n else 0.0))
    say("")
    say("pooled over all seed x sample cells: %d/%d, %.1f%% (95%% Wilson "
        "[%.1f%%, %.1f%%])"
        % (pooled["n"], pooled["flips"], 100 * pooled["flips"] / max(pooled["n"], 1),
           100 * pooled["wilson_low"], 100 * pooled["wilson_high"]))
    say("")
    say("The pooled figure counts every seed x sample cell, so it is NOT the "
        "attack's ASR on a sample; it is a description of the sweep. The "
        "ASR per seed above is the comparable number.")

    out_txt = os.path.join(RESULTS_DIR,
                           "seed_sweep_%s_%s.txt" % (args.mode, model))
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nsaved -> %s" % out_csv)
    print("saved -> %s" % out_txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
