"""
Turn the ablation's per-sample records into the two tables the paper needs.

    python src/summarize_ablation.py

Reads every results/ablation_*.csv written by src/run_ablation.py and writes

    results/ablation_table.csv       one row per (model, arm, vocab, task)
    results/ablation_contrasts.csv   one row per (contrast, model, task)

The second file is the point of the whole exercise. Each contrast holds the
candidate set, the probe phase and the query cap fixed and varies ONE thing, so
the difference it reports has one explanation:

    ordering   random-prob  vs saliency-prob    same candidates, same budget
    oracle     random-prob  vs random-verdict   same order, same candidates
    search     random-verdict vs group-verdict  same bit, better use of it
    candidates random-verdict(lexicon) vs random-verdict(small_table)
               (and, against the recorded runs, vs the word attack of
               results/attack_results_word_*.csv, which is the same order and
               the same oracle over the 14-entry table)

Every contrast is PAIRED on the message ids the two arms share, and reported
with the discordant counts and an exact McNemar p-value, because two arms over
the same messages are not two independent samples. Whatever is missing is
printed as missing rather than silently skipped: an arm that was never run must
not look like an arm that found nothing.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analyze_pairs import mcnemar_exact, paired_ci  # noqa: E402
from src.attack_common import asr_interval  # noqa: E402
from src.common import RESULTS_DIR  # noqa: E402

TABLE = os.path.join(RESULTS_DIR, "ablation_table.csv")
CONTRASTS = os.path.join(RESULTS_DIR, "ablation_contrasts.csv")

# (label, a_arm, a_vocab, b_arm, b_vocab, what it isolates)
CONTRASTS_SPEC = (
    ("ordering",   "saliency-prob",   "lexicon_wordnet",
     "random-prob",     "lexicon_wordnet", "ordering"),
    ("oracle",     "random-prob",     "lexicon_wordnet",
     "random-verdict",  "lexicon_wordnet", "oracle"),
    ("search",     "group-verdict",   "lexicon_wordnet",
     "random-verdict",  "lexicon_wordnet", "search under one bit"),
    ("candidates", "random-verdict",  "lexicon_wordnet",
     "random-verdict",  "small_table",      "candidate set"),
    ("stronger",   "textfooler-prob", "lexicon_wordnet",
     "saliency-prob",   "lexicon_wordnet", "search rule"),
)


def counted_mask(df):
    """The `counted` flag, whatever dtype the CSV round-trip gave it.

    Pandas parses a column of True/False as bool only while every row is a
    clean boolean. One skipped sample writes an empty cell, the column comes
    back as the strings 'True'/'False', and `df["counted"] == True` is then
    False for EVERY row -- so the arm disappears from the table and its
    contrast is reported as missing data. That is what happened to the
    candidate-set arm the first time it ran, and a missing contrast is much
    harder to notice than a wrong one.
    """
    col = df["counted"]
    if col.dtype == bool:
        return col
    return col.astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))


SMALL_TABLE_SUFFIX = "_small_table"


def parse_name(path):
    """(arm, vocab, model) from results/ablation_<arm>_<model>[_small_table].csv.

    The vocabulary is read from the SUFFIX, not from a split field: `small_table`
    itself contains the separator, so splitting on '_' yields 'small', the vocab
    falls back to the lexicon, and the small-table arm is silently merged into
    the lexicon arm it is supposed to be contrasted with. That merge made the
    candidate-set contrast report missing data while quietly inflating the
    lexicon arm's denominator with rows from a different candidate set.
    """
    base = os.path.basename(path)[len("ablation_"):-len(".csv")]
    if base.endswith(SMALL_TABLE_SUFFIX):
        vocab = "small_table"
        base = base[:-len(SMALL_TABLE_SUFFIX)]
    else:
        vocab = "lexicon_wordnet"
    arm, _, model = base.partition("_")
    if not arm or not model:
        return None
    return arm, vocab, model


def load_arms():
    """{(arm, vocab, model): {(task, id): success}} plus the per-row frames."""
    per_key, frames = {}, []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "ablation_*.csv"))):
        parsed = parse_name(path)
        if parsed is None:
            continue
        arm, vocab, model = parsed
        df = pd.read_csv(path)
        if "counted" not in df.columns:
            continue
        df["arm"], df["vocab"] = arm, vocab
        frames.append(df)
        counted = df[counted_mask(df)]
        key = (arm, vocab, model)
        bucket = per_key.setdefault(key, {})
        for r in counted.itertuples(index=False):
            # a sample attacked more than once counts as flipped if any attempt
            # flipped it - the same rule the other summaries use
            cell = (str(r.task), r.id)
            bucket[cell] = bucket.get(cell, False) or bool(r.success)
    return per_key, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def arm_rows(df):
    """One summary row per (model, arm, vocab, task), with intervals."""
    rows = []
    for (model, arm, vocab, task), g in df.groupby(["model", "arm", "vocab", "task"]):
        counted = g[counted_mask(g)]
        if not len(counted):
            continue
        # One message is one row, even if an arm wrote two records for it: a
        # denominator of 382 for a 191-message cell is not a bigger sample, it
        # is the same messages counted twice. Success is "any record flipped it",
        # which is the rule the per-sample bucket and the other summaries use.
        if "id" in counted.columns and counted["id"].duplicated().any():
            first = counted.drop_duplicates("id", keep="first").copy()
            first["success"] = (counted.groupby("id")["success"].any()
                                .reindex(first["id"]).to_numpy())
            counted = first
        ci = asr_interval(counted["success"].tolist())
        rows.append({
            "model": model, "arm": arm, "vocab": vocab, "task": task,
            "n_attacked": int(len(counted)),
            "n_flips": int(counted["success"].sum()),
            "asr": round(float(counted["success"].mean()), 4),
            "ci95_low": ci["wilson_low"], "ci95_high": ci["wilson_high"],
            "mean_queries": round(float(counted["n_queries"].mean()), 2)
            if "n_queries" in counted.columns else None,
            "mean_perturbation_rate": round(float(counted["perturbation_rate"].mean()), 4),
            "mean_semantic": round(float(counted["semantic_similarity"].mean()), 4)
            if "semantic_similarity" in counted.columns else None,
        })
    return rows


def contrast_rows(per_key):
    rows = []
    for label, a_arm, a_vocab, b_arm, b_vocab, isolates in CONTRASTS_SPEC:
        for model in sorted({k[2] for k in per_key if k[0] in (a_arm, b_arm)}):
            A = per_key.get((a_arm, a_vocab, model), {})
            B = per_key.get((b_arm, b_vocab, model), {})
            if not A or not B:
                print("  %-10s %-14s missing: A=%d rows, B=%d rows"
                      % (label, model, len(A), len(B)))
                continue
            for task in sorted({k[0] for k in A} & {k[0] for k in B}):
                keys = [k for k in A if k[0] == task and k in B]
                if not keys:
                    continue
                a_flips = [A[k] for k in keys]
                b_flips = [B[k] for k in keys]
                only_a = sum(1 for x, y in zip(a_flips, b_flips) if x and not y)
                only_b = sum(1 for x, y in zip(a_flips, b_flips) if (not x) and y)
                diffs = [float(x) - float(y) for x, y in zip(a_flips, b_flips)]
                lo, hi = paired_ci(diffs)
                rows.append({
                    "contrast": label, "isolates": isolates, "model": model,
                    "task": task, "a_arm": a_arm, "a_vocab": a_vocab,
                    "b_arm": b_arm, "b_vocab": b_vocab,
                    "n_paired": len(keys),
                    "a_flips": int(sum(a_flips)), "b_flips": int(sum(b_flips)),
                    "a_asr": round(float(np.mean(a_flips)), 4),
                    "b_asr": round(float(np.mean(b_flips)), 4),
                    "risk_diff": round(float(np.mean(diffs)), 4),
                    "diff_ci_low": round(lo, 4), "diff_ci_high": round(hi, 4),
                    "only_a": only_a, "only_b": only_b,
                    "mcnemar_p": round(mcnemar_exact(only_b, only_a), 6),
                })
    return rows


def main() -> int:
    per_key, df = load_arms()
    if df.empty:
        print("no results/ablation_*.csv yet - run src/run_ablation.py for an arm")
        return 1

    table = pd.DataFrame(arm_rows(df)).sort_values(["arm", "vocab", "task", "model"])
    table.to_csv(TABLE, index=False)

    print("\narm table (per model x task):")
    print("| arm | vocab | model | task | attacked | flips | ASR | 95% CI | queries |")
    print("|---|---|---|---|---|---|---|---|---|")
    for _, r in table.iterrows():
        print("| %s | %s | %s | %s | %d | %d | %.1f%% | %.1f-%.1f%% | %s |"
              % (r["arm"], r["vocab"], r["model"], r["task"], r["n_attacked"],
                 r["n_flips"], 100 * r["asr"], 100 * r["ci95_low"],
                 100 * r["ci95_high"],
                 "%.1f" % r["mean_queries"] if r["mean_queries"] == r["mean_queries"]
                 else "-"))

    contrasts = contrast_rows(per_key)
    if not contrasts:
        print("\nno contrast could be paired: at least two arms over the same "
              "messages are needed for any of them")
        return 1
    out = pd.DataFrame(contrasts).sort_values(["contrast", "task", "model"])
    out.to_csv(CONTRASTS, index=False)

    print("\ncontrasts (paired on shared message ids; positive = A flips more):")
    print("| contrast | isolates | model | task | paired | A | B | risk diff | "
          "95% CI | A-only/B-only | McNemar p |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in out.iterrows():
        print("| %s | %s | %s | %s | %d | %d | %d | %+.1fpp | %+.1f to %+.1f | "
              "%d/%d | %.4f |"
              % (r["contrast"], r["isolates"], r["model"], r["task"],
                 r["n_paired"], r["a_flips"], r["b_flips"],
                 100 * r["risk_diff"], 100 * r["diff_ci_low"],
                 100 * r["diff_ci_high"], r["only_a"], r["only_b"],
                 r["mcnemar_p"]))

    for label in ("ordering", "oracle", "search", "candidates"):
        sub = out[out["contrast"] == label]
        if not len(sub):
            continue
        print("\n%s: pooled A-only %d vs B-only %d, exact McNemar p=%.4f "
              "(cells reuse messages across models, so read this as the "
              "direction, not as a confirmatory test)"
              % (label, int(sub["only_a"].sum()), int(sub["only_b"].sum()),
                 mcnemar_exact(int(sub["only_b"].sum()), int(sub["only_a"].sum()))))
    print("\nsaved -> %s" % TABLE)
    print("saved -> %s" % CONTRASTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
