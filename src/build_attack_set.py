"""
Write the attack-eligible set: samples the CLEAN model classified malicious.

ASR is only defined on those samples. Attacking a message the model already
called benign measures nothing, and attacking arbitrary malicious rows makes
the denominator meaningless. This is the single source of that set, used by the
runbook's C9 cell and by the Kaggle runner.

    python src/build_attack_set.py --all-models
    python src/build_attack_set.py --model qwen2.5-3b
    python src/build_attack_set.py --model qwen2.5-3b --task sms_smishing

`clean_baseline_results.csv` holds the verdicts but NOT the message text (it
was written to stay small). The text lives in `results/eval_subset.csv`, so
every row here is joined to it by `id`; a missing id is a hard error rather
than an empty prompt, because an empty prompt silently measures nothing.

`--all-models` writes results/attack_eligible.csv with every model x task in it
(that is what the Kaggle runner consumes, one file, filtered per model) plus one
results/attack_eligible_<model>_<task>.csv per pair for inspection.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.common import RESULTS_DIR  # noqa: E402

INPUT_CSV = os.path.join(RESULTS_DIR, "clean_baseline_results.csv")
TEXT_CSV = os.path.join(RESULTS_DIR, "eval_subset.csv")
CANONICAL = os.path.join(RESULTS_DIR, "attack_eligible.csv")


def load_text(path=TEXT_CSV):
    """id -> message text, from the frozen eval split."""
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} - rebuild the splits first "
                         f"(python src/build_splits.py)")
    ev = pd.read_csv(path)
    if "text" not in ev.columns or "id" not in ev.columns:
        raise SystemExit(f"{path} needs id and text columns")
    return dict(zip(ev["id"], ev["text"]))


def build(df, model=None, task="sms_smishing"):
    """Return (malicious_rows, caught_rows) for one model + task."""
    sub = df if model is None else df[df["model"] == model]
    if sub.empty:
        raise SystemExit(f"no rows for model={model!r} in {INPUT_CSV}; "
                         f"available: {sorted(df['model'].unique())}")
    mal = sub[(sub["task"] == task) & (sub["ground_truth"] == 1)]
    caught = mal[mal["pred"] == 1]
    return mal, caught


def attach_text(caught, texts, model, task):
    """Join the message text onto the caught rows; fail loudly on a gap."""
    out = caught.copy()
    out["text"] = out["id"].map(texts)
    missing = out[out["text"].isna()]
    if len(missing):
        raise SystemExit(
            f"{model}/{task}: {len(missing)} eligible ids have no text in "
            f"{TEXT_CSV} (e.g. {missing['id'].head(3).tolist()}) - the baseline "
            f"and the splits are from different runs; re-run the baseline")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="model key to build the set for (required if the CSV "
                         "holds several models and --all-models is not given)")
    ap.add_argument("--task", default=None,
                    help="task to build for (default: every task in the CSV, "
                         "or sms_smishing when --model is used alone)")
    ap.add_argument("--all-models", action="store_true",
                    help="build for every model x task in the input CSV")
    ap.add_argument("--input", default=INPUT_CSV)
    ap.add_argument("--text-csv", default=TEXT_CSV)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"missing {args.input} - run the baseline first "
                         f"(python src/run_baseline.py ...)")

    df = pd.read_csv(args.input)
    texts = load_text(args.text_csv)
    models = sorted(df["model"].unique())
    tasks = sorted(df["task"].unique())

    if args.all_models:
        pairs = [(m, t) for m in models for t in tasks]
    else:
        model = args.model
        if model is None:
            if len(models) != 1:
                raise SystemExit(f"{args.input} holds {models}; pass --model "
                                 f"or --all-models")
            model = models[0]
        pairs = [(model, args.task or "sms_smishing")]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    frames, rows = [], []
    for model, task in pairs:
        mal, caught = build(df, model, task)
        if caught.empty:
            print(f"{model}/{task}: no malicious samples caught by the clean "
                  f"model - nothing to attack")
            continue
        caught = attach_text(caught, texts, model, task)
        per_pair = os.path.join(RESULTS_DIR, f"attack_eligible_{model}_{task}.csv")
        caught.to_csv(per_pair, index=False)
        frames.append(caught)
        rate = len(caught) / len(mal) if len(mal) else 0.0
        rows.append((model, task, len(mal), len(caught), rate))
        print("model=%-12s task=%-14s malicious=%-4d caught=%-4d (%.0f%%) -> %s"
              % (model, task, len(mal), len(caught), 100 * rate,
                 os.path.basename(per_pair)))
        if len(caught) < 10:
            print("  !! fewer than 10 eligible samples - any ASR from this is "
                  "not reportable")

    if not frames:
        raise SystemExit("no attack-eligible rows were produced; the baseline "
                         "may have classified nothing as malicious")

    canonical = pd.concat(frames, ignore_index=True)
    canonical.to_csv(CANONICAL, index=False)
    print("\nattack-eligible set: %d rows (%d model x task pairs) -> %s"
          % (len(canonical), len(frames), CANONICAL))
    print("detection rate (caught/malicious) must be quoted next to every ASR:")
    for model, task, n_mal, n_caught, rate in rows:
        print("  %-12s %-14s %.1f%% (%d/%d)"
              % (model, task, 100 * rate, n_caught, n_mal))
    return 0


if __name__ == "__main__":
    sys.exit(main())
