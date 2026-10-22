"""
Score every stored perturbation against one target model.

    python src/run_transfer.py --target qwen2.5-3b --base-url http://localhost:8000/v1

The perturbations come from the finished attack runs (results/attack_results_*.csv),
so this needs no new attack: it is a re-play of the exact texts that were already
sent to the source models, evaluated on a different model with the same prompt.

Writes results/transfer_rows_<target>.csv (per sample, one row per perturbation).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.attack_common import chat_verdict  # noqa: E402
from src.common import RESULTS_DIR  # noqa: E402
from src.transfer_common import load_original_texts, load_perturbations, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    help="model key whose verdicts we measure")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--source", default=None,
                    help="only perturbations from this source model (default: all)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--include-self", action="store_true",
                    help="also replay the target's own perturbations; this must "
                         "reproduce its single-model ASR and is the faithfulness "
                         "check, at the cost of a second full pass")
    args = ap.parse_args()

    originals = load_original_texts()
    records = load_perturbations(originals, source_model=args.source)
    if not records:
        raise SystemExit("no perturbations to transfer: run the attacks first")
    n_all = len(records)
    if not args.include_self:
        records = [r for r in records if r["source"] != args.target]
    if not records:
        raise SystemExit("every stored perturbation comes from %s itself; "
                         "nothing to transfer" % args.target)
    print("replaying %d of %d stored perturbations on %s (%s)"
          % (len(records), n_all, args.target,
             "including its own, as the faithfulness check"
             if args.include_self else "self excluded"))

    seen = [0]

    def ask(text, task):
        if not args.quiet and seen[0] and seen[0] % 200 == 0:
            print("  %d queries" % seen[0], flush=True)
        seen[0] += 1
        return chat_verdict(args.base_url, args.target, text, task=task)[0]

    df = score(records, args.target, ask)
    out = os.path.join(RESULTS_DIR, "transfer_rows_%s.csv" % args.target)
    df.to_csv(out, index=False)
    counted = df[df["counted"] == True]          # noqa: E712
    print("target caught the original in %d/%d cases; %d transferred flips"
          % (len(counted), len(df), int(counted["success"].sum())))
    if args.include_self:
        # Loud, per model: this is an assertion about the pipeline, not a result.
        mine = df[df["source"] == df["target"]]
        for (mode, task), g in mine.groupby(["mode", "task"]):
            c = g[g["counted"] == True]          # noqa: E712
            print("  self %-9s %-15s stored %3d/%3d  replayed %3d/%3d  %s"
                  % (mode, task, int(c["flipped_source"].sum()), len(c),
                     int(c["success"].sum()), len(c),
                     "OK" if int(c["flipped_source"].sum()) == int(c["success"].sum())
                     else "MISMATCH"))
    print("saved -> %s" % out)
    return 0 if len(df) else 1


if __name__ == "__main__":
    sys.exit(main())
