"""Run the PWWS-style saliency word attack against one model.

    python src/run_saliency.py --model qwen2.5-3b --base-url http://localhost:8000/v1

Needs the shim's /v1/verdict_probs endpoint (src/shim_colab.py). If that
endpoint is missing the run stops with a clear message instead of falling back
to random search under a saliency label.

Writes results/attack_results_saliency_<model>.csv, the same schema the char and
word runs use, plus a summary block via attack_common.print_summary.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attack_common import load_eligible, print_summary, summarize_counts  # noqa: E402
from src.common import RESULTS_DIR  # noqa: E402
from src.saliency_common import run_saliency, verdict_probs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--task", default=None)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--eligible", default=os.path.join(RESULTS_DIR,
                                                       "attack_eligible.csv"))
    ap.add_argument("--alpha", default=0.5, type=float,
                    help="PWWS weight: alpha*saliency + (1-alpha)*P(replacement)")
    ap.add_argument("--max-positions", default=12, type=int,
                    help="content words probed per sample (query cost driver)")
    ap.add_argument("--max-rounds", default=3, type=int,
                    help="substitutions applied at most")
    ap.add_argument("--max-rate", default=0.15, type=float)
    ap.add_argument("--seed", default=None, type=int,
                    help="randomise the order of TIED candidates. The default "
                         "(no seed) is the deterministic run that produced the "
                         "recorded results; a seed sweep measures how much the "
                         "attack moves when the ties are broken differently.")
    ap.add_argument("--limit", default=None, type=int,
                    help="attack the first N eligible samples (smoke test)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out-dir", default=RESULTS_DIR,
                    help="where to write the CSVs (point a smoke test at a "
                         "scratch directory)")
    args = ap.parse_args()

    samples = load_eligible(args.eligible)
    models = sorted(samples["model"].unique()) if "model" in samples.columns else []
    model = args.model
    if model is None:
        if len(models) != 1:
            raise SystemExit(f"{args.eligible} holds {models}; pass --model")
        model = models[0]
    if "model" in samples.columns:
        samples = samples[samples["model"] == model]
    if args.task:
        samples = samples[samples["task"] == args.task]
    if args.limit:
        samples = samples.head(args.limit)
    if samples.empty:
        raise SystemExit(f"no eligible samples for model={model!r}")

    # fail fast and loudly: a run without probabilities is not this attack
    probe = verdict_probs(args.base_url, model, ["hello"], task="sms_smishing")
    if probe is None:
        raise SystemExit(
            "the shim has no working /v1/verdict_probs endpoint.\n"
            "Restart it from the current src/shim_colab.py, or skip the "
            "saliency mode - do NOT report random search under a saliency name.")
    print("verdict_probs OK (p_malicious={:.3f} on the probe text)".format(probe[0]))

    def ask_probs(texts, task):
        return verdict_probs(args.base_url, model, texts, task=task)

    print("saliency attack on %d eligible samples (model=%s, alpha=%.2f)"
          % (len(samples), model, args.alpha))
    df, info = run_saliency(samples, model, ask_probs, alpha=args.alpha,
                            max_positions=args.max_positions,
                            max_rounds=args.max_rounds,
                            max_rate=args.max_rate, seed=args.seed,
                            quiet=args.quiet)

    summary = summarize_counts(df, "saliency", model)
    if len(df):
        summary["mean_queries"] = round(float(df[df["counted"] == True]["n_queries"].mean()), 2) \
            if (df["counted"] == True).any() else 0.0   # noqa: E712
    print_summary(summary)

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "saliency_" + model + ("" if not args.task else "_" + args.task)
    if args.seed is not None:
        suffix += "_seed%d" % args.seed
    out = os.path.join(args.out_dir, "attack_results_%s.csv" % suffix)
    df.to_csv(out, index=False)
    txt = os.path.join(args.out_dir, "asr_summary_%s.txt" % suffix)
    with open(txt, "w", encoding="utf-8") as f:
        print_summary(summary, file=f)
        print("\nalpha=%.2f max_positions=%d max_rounds=%d" %
              (info["alpha"], info["max_positions"], info["max_rounds"]), file=f)
    print("\nsaved -> %s" % out)
    print("saved -> %s" % txt)
    return 0 if summary["n_attacked"] else 1


if __name__ == "__main__":
    sys.exit(main())
