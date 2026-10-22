"""
Run one arm of the matched-vocabulary / matched-query-budget ablation.

    python src/run_ablation.py --model qwen2.5-3b --arm random-prob
    python src/run_ablation.py --model qwen2.5-3b --arm saliency-prob
    python src/run_ablation.py --model qwen2.5-3b --arm random-verdict
    python src/run_ablation.py --model qwen2.5-3b --arm group-verdict

Every arm attacks the SAME messages (the eligible set of the model), uses the
SAME candidate vocabulary and the SAME query cap, and differs only in the
ordering rule and the oracle. That is what makes the contrasts in the paper
interpretable: random-prob vs saliency-prob is the ordering effect, and
random-prob vs random-verdict is the oracle effect, with the candidate set
held fixed on both sides.

Writes results/ablation_<arm>_<model>.csv, in the same per-sample schema the
other attacks use, plus a summary block.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.ablation_common import ARMS, run_arm  # noqa: E402
from src.attack_common import chat_verdict, load_eligible  # noqa: E402
from src.common import RESULTS_DIR  # noqa: E402
from src.saliency_common import verdict_probs  # noqa: E402


def make_ask(base_url, model, oracle):
    """A batched measurement function for the chosen oracle.

    probability -> P(malicious) from the shim's /v1/verdict_probs endpoint
    verdict    -> the generated one-word verdict, 1 malicious / 0 benign

    An unparseable verdict becomes NaN, not 0: "the model did not answer" is
    not "the model said benign", and counting it as a flip would be an attack
    success the model never granted. NaN is neither, so an original with an
    unparseable verdict is skipped (it was not confirmed malicious) and a
    perturbed one is not a flip.
    """
    if oracle == "probability":
        def ask(texts, task):
            return verdict_probs(base_url, model, texts, task=task)
        return ask

    def ask(texts, task):
        out = []
        for t in texts:
            pred, _raw, err = chat_verdict(base_url, model, t, task=task)
            if err or pred is None:
                out.append(float("nan"))
            else:
                out.append(float(pred))
        return out
    return ask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS),
                    help="which cell of the factorial to run")
    ap.add_argument("--task", default=None)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--eligible", default=os.path.join(RESULTS_DIR,
                                                       "attack_eligible.csv"))
    ap.add_argument("--vocab", default="lexicon_wordnet",
                    choices=["lexicon_wordnet", "small_table"])
    ap.add_argument("--seed", default=42, type=int,
                    help="perturbation seed; change it to measure seed-to-seed "
                         "variation of the same arm")
    ap.add_argument("--query-cap", default=30, type=int,
                    help="hard cap on model calls per message (the matched "
                         "budget across arms)")
    ap.add_argument("--max-positions", default=12, type=int)
    ap.add_argument("--max-replacements", default=2, type=int)
    ap.add_argument("--max-rounds", default=3, type=int)
    ap.add_argument("--max-rate", default=0.15, type=float)
    ap.add_argument("--alpha", default=0.5, type=float)
    ap.add_argument("--semantic-threshold", default=None, type=float,
                    help="reject a substitution whose meaning-preservation "
                         "score falls below this (default: no gate)")
    ap.add_argument("--limit", default=None, type=int)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    spec = ARMS[args.arm]
    samples = load_eligible(args.eligible)
    if "model" in samples.columns:
        samples = samples[samples["model"] == args.model]
    if args.task:
        samples = samples[samples["task"] == args.task]
    if args.limit:
        samples = samples.head(args.limit)
    if samples.empty:
        raise SystemExit("no eligible samples for model=%r task=%r"
                         % (args.model, args.task))

    if spec["oracle"] == "probability":
        probe = verdict_probs(args.base_url, args.model, ["hello"],
                              task="sms_smishing")
        if probe is None:
            raise SystemExit(
                "the shim has no working /v1/verdict_probs endpoint, so the "
                "%s arm cannot be run.\nStart the shim from the current "
                "src/shim_colab.py, or run a verdict arm instead." % args.arm)
        print("verdict_probs OK (p_malicious=%.3f on the probe text)" % probe[0])

    ask = make_ask(args.base_url, args.model, spec["oracle"])
    print("arm=%s  strategy=%s  order=%s  oracle=%s  vocab=%s  seed=%d  "
          "query_cap=%d  samples=%d"
          % (args.arm, spec["strategy"], spec["order"], spec["oracle"],
             args.vocab, args.seed, args.query_cap, len(samples)))

    df, info = run_arm(samples, args.model, ask, arm=args.arm,
                       vocab=args.vocab, seed=args.seed,
                       max_positions=args.max_positions,
                       max_replacements=args.max_replacements,
                       max_rate=args.max_rate, max_rounds=args.max_rounds,
                       query_cap=args.query_cap, alpha=args.alpha,
                       semantic_threshold=args.semantic_threshold,
                       quiet=args.quiet)

    counted = df[df["counted"] == True]                # noqa: E712
    flips = int(counted["success"].sum()) if len(counted) else 0
    print("\nattacked %d, flips %d, ASR %.1f%%"
          % (len(counted), flips, 100 * flips / max(len(counted), 1)))
    if len(counted):
        print("mean queries per message : %.1f"
              % counted["n_queries"].mean())
        print("mean perturbation rate   : %.4f"
              % counted["perturbation_rate"].mean())
        print("mean word Jaccard        : %.4f"
              % counted["similarity"].mean())
        for col in ("literal_retention", "paraphrase_coverage",
                    "semantic_similarity"):
            if col in counted.columns:
                print("mean %-20s: %.4f" % (col, counted[col].mean()))

    os.makedirs(args.out_dir, exist_ok=True)
    # The candidate set is part of what an arm IS: running random-verdict with
    # the small table and with the lexicon is the candidate-set contrast, and
    # two runs that differ only in --vocab must not overwrite each other.
    tag = "" if args.vocab == "lexicon_wordnet" else "_" + args.vocab
    out = os.path.join(args.out_dir,
                       "ablation_%s_%s%s%s.csv" % (args.arm, args.model, tag,
                                                  ("" if not args.task
                                                   else "_" + args.task)))
    df.to_csv(out, index=False)
    print("saved -> %s" % out)
    return 0 if len(counted) else 1


if __name__ == "__main__":
    sys.exit(main())
