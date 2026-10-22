"""Honest ASR (Attack Success Rate) runner.

    python src/run_attack.py --model qwen2.5-3b --mode char --attempts 3
    python src/run_attack.py --model qwen2.5-3b --mode word --attempts 3

Runs the attack ONLY on samples the clean model classified malicious (the set
written by src/build_attack_set.py). The original is re-queried rather than
assumed malicious, the baseline prompt is reused per task, and a success is
strictly orig_pred == 1 -> pert_pred == 0.

Outputs (per model, so a three-model run does not overwrite itself):
    results/attack_results_<mode>_<model>.csv
    results/asr_summary_<mode>_<model>.txt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.common import RESULTS_DIR, SEED  # noqa: E402
from src.attack_common import load_eligible, run_attack, print_summary  # noqa: E402

ELIGIBLE_PATH = os.path.join(RESULTS_DIR, "attack_eligible.csv")


def main():
    ap = argparse.ArgumentParser(
        description="Run adversarial attack and report ASR."
    )
    ap.add_argument("--model", default=None,
                    help="model key (required if the CSV holds several models)")
    ap.add_argument("--task", default=None,
                    help="restrict to one task (default: every task present)")
    ap.add_argument("--mode", default="char", choices=["char", "word"],
                    help="perturbation mode: char or word")
    ap.add_argument("--attempts", default=1, type=int,
                    help="perturbation attempts per sample (stop at the first flip)")
    ap.add_argument("--max-rate", default=0.15, type=float,
                    help="max perturbation rate (fraction of chars/words)")
    ap.add_argument("--seed", default=SEED, type=int,
                    help="perturbation seed. The same attack with a different "
                         "seed is a DIFFERENT run: run src/run_seed_sweep.py to "
                         "measure the seed-to-seed spread of the ASR.")
    ap.add_argument("--base-url", default="http://localhost:8000/v1",
                    help="base URL of the /chat/completions shim")
    ap.add_argument("--eligible", default=ELIGIBLE_PATH,
                    help="attack-eligible CSV (default results/attack_eligible.csv)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-sample progress prints")
    args = ap.parse_args()

    samples = load_eligible(args.eligible)
    models = sorted(samples["model"].unique()) if "model" in samples.columns else []
    model = args.model
    if model is None:
        if len(models) != 1:
            raise SystemExit(f"{args.eligible} holds {models}; pass --model")
        model = models[0]
        samples = samples[samples["model"] == model]
    else:
        if "model" not in samples.columns:
            raise SystemExit(f"{args.eligible} has no model column; rebuild it "
                             f"with src/build_attack_set.py --all-models")
        samples = samples[samples["model"] == model]
    if args.task:
        if "task" not in samples.columns:
            raise SystemExit(f"{args.eligible} has no task column; rebuild it "
                             f"with src/build_attack_set.py --all-models")
        samples = samples[samples["task"] == args.task]
    if samples.empty:
        raise SystemExit(f"no eligible samples for model={model!r} "
                         f"task={args.task!r}")

    tasks = sorted(samples["task"].unique()) if "task" in samples.columns else ["?"]
    print(f"Running {args.mode}-level attack on {len(samples)} eligible samples")
    print(f"  model={model}, tasks={','.join(tasks)}, attempts={args.attempts}, "
          f"max_rate={args.max_rate}, seed={args.seed}")

    df, summary = run_attack(
        samples=samples,
        mode=args.mode,
        model=model,
        base_url=args.base_url,
        attempts=args.attempts,
        max_rate=args.max_rate,
        seed=args.seed,
        quiet=args.quiet,
    )

    print_summary(summary)

    suffix = args.mode + "_" + model + ("" if not args.task else "_" + args.task)
    if args.seed != SEED:
        # a different seed is a different run and must not overwrite the
        # recorded one
        suffix += "_seed%d" % args.seed
    out_path = os.path.join(RESULTS_DIR, f"attack_results_{suffix}.csv")
    df.to_csv(out_path, index=False)
    summary_path = os.path.join(RESULTS_DIR, f"asr_summary_{suffix}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        print_summary(summary, file=f)
    print(f"\nSaved per-sample results -> {out_path}")
    print(f"Saved summary -> {summary_path}")

    if summary["n_attacked"] == 0:
        print("\n!! nothing was attacked: the clean model did not reproduce "
              "malicious on any eligible sample. Check that the shim is serving "
              "the same model and prompt the baseline used.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
