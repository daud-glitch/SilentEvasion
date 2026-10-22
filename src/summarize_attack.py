"""
Build the single ASR table the paper quotes, from the per-sample attack CSVs.

    python src/summarize_attack.py
    python src/summarize_attack.py --rescore   # recompute the metrics in place

Every row is recomputed from `results/attack_results_<mode>_<model>.csv`, so the
table cannot drift from the raw data. The catch rate from the clean baseline is
joined in, because an ASR without its denominator is not interpretable: "42% of
the emails this model catches flip" is a measurement, "42% ASR" on its own is
not.

Writes results/attack_table.csv and prints the same rows as markdown.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.attack_common import (  # noqa: E402
    asr_interval, edit_counts, jaccard, print_summary, read_attack_csv,
    summarize_counts,
)
from src.common import RESULTS_DIR  # noqa: E402

BASELINE = os.path.join(RESULTS_DIR, "clean_baseline_results.csv")
OUT = os.path.join(RESULTS_DIR, "attack_table.csv")


def catch_rates():
    """(model, task) -> (caught, malicious). The ASR denominator upstream."""
    rates = {}
    if not os.path.exists(BASELINE):
        return rates
    b = pd.read_csv(BASELINE)
    for (model, task), g in b.groupby(["model", "task"]):
        mal = g[g["ground_truth"] == 1]
        rates[(model, task)] = (int((mal["pred"] == 1).sum()), int(len(mal)))
    return rates


def rescore(path):
    """Recompute edits/rate/similarity for a saved run from its stored texts.

    Needed when the metric itself was wrong: the perturbed texts are in the
    file, so the honest number is recoverable without touching the GPU. Writes
    the CSV and its asr_summary_*.txt back in place.
    """
    df = read_attack_csv(path)
    if "perturbed_text" not in df.columns:
        print("skip %s: no stored perturbed_text" % os.path.basename(path))
        return df, None
    originals = {}
    for cand in (os.path.join(RESULTS_DIR, "attack_eligible.csv"),
                 os.path.join(RESULTS_DIR, "eval_subset.csv")):
        if os.path.exists(cand):
            src = pd.read_csv(cand)
            originals.update(dict(zip(src["id"], src["text"])))
    level = "char" if "char" in os.path.basename(path) else "word"
    for i, r in df.iterrows():
        if not isinstance(r.get("perturbed_text"), str) or pd.isna(r.get("orig_pred")):
            continue
        orig = originals.get(r["id"])
        if not isinstance(orig, str):
            continue
        stats = edit_counts(orig, r["perturbed_text"], level=level)
        df.at[i, "edits"] = stats["edits"]
        df.at[i, "perturbation_rate"] = stats["rate"]
        df.at[i, "similarity"] = round(jaccard(orig, r["perturbed_text"]), 4)
    for col in ("subs", "dels", "ins"):        # difflib artefacts, not measured
        if col in df.columns:
            df = df.drop(columns=[col])
    df.to_csv(path, index=False)
    model = df["model"].iloc[0] if "model" in df.columns and len(df) else "?"
    mode = df["mode"].iloc[0] if "mode" in df.columns and len(df) else level
    hdr_mode = mode
    summary = summarize_counts(df, hdr_mode, model)
    txt = os.path.join(RESULTS_DIR, "asr_summary_%s_%s.txt" % (hdr_mode, model))
    with open(txt, "w", encoding="utf-8") as f:
        print_summary(summary, file=f)
    print("rescored %-40s -> %s" % (os.path.basename(path), os.path.basename(txt)))
    return df, summary


def rows():
    rates = catch_rates()
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "attack_results_*.csv"))):
        a = read_attack_csv(path)
        if "counted" not in a.columns:
            print("skip %s: no 'counted' column (old-format file)"
                  % os.path.basename(path))
            continue
        counted = a[a["counted"] == True]        # noqa: E712
        # The saliency attack decides success on P(malicious) < 0.5 because it
        # needs a probability at every candidate; the other two modes decide on
        # the generated verdict. When src/verify_saliency.py has re-scored a
        # file through the generation route, that column is the comparable one.
        vcol = "success_gen" if "success_gen" in a.columns else "success"
        if vcol != "success":
            counted = counted[counted[vcol].notna()]
        for (model, mode, task), g in counted.groupby(["model", "mode", "task"]):
            flags = [bool(x) for x in g[vcol]]
            flips = int(sum(flags))
            ci = asr_interval(flags)
            caught, mal = rates.get((model, task), (None, None))
            out.append({
                "model": model,
                "mode": mode,
                "task": task,
                "n_attacked": int(len(g)),
                "n_flips": flips,
                "asr": round(flips / len(g), 4) if len(g) else 0.0,
                "ci95_low": ci["wilson_low"],
                "ci95_high": ci["wilson_high"],
                "boot95_low": ci["boot_low"],
                "boot95_high": ci["boot_high"],
                "caught": caught,
                "malicious": mal,
                "catch_rate": (round(caught / mal, 4) if caught and mal else None),
                "mean_perturbation_rate": round(float(g["perturbation_rate"].mean()), 4),
                "mean_jaccard": round(float(g["similarity"].mean()), 4),
                "verdict_channel": "generation" if vcol == "success_gen" else "probability",
                # only the saliency attack records its query count; the random
                # attacks spend one draw per sample and do not need to
                "mean_queries": (round(float(g["n_queries"].mean()), 1)
                                 if "n_queries" in g.columns else None),
            })
        excluded = a[a["counted"] != True]       # noqa: E712
        if len(excluded):
            print("note: %s excluded %d row(s) the model re-read as benign"
                  % (os.path.basename(path), len(excluded)))
    return out


def main():
    if "--rescore" in sys.argv:
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "attack_results_*.csv")))
        if not files:
            print("no attack results to rescore")
            return 1
        for path in files:
            rescore(path)
        print("")
    data = rows()
    if not data:
        print("no attack results found in %s" % RESULTS_DIR)
        print("run the attacks first: python src/run_attack.py --model ... --mode char")
        return 1
    table = pd.DataFrame(data).sort_values(["model", "task", "mode"])
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False))
    print("\nsaved -> %s" % OUT)

    print("\n| model | task | mode | attacked | flips | ASR | 95% CI (Wilson) | catch rate | mean perturb | mean Jaccard |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in table.iterrows():
        print("| %s | %s | %s | %d | %d | %.1f%% | %.1f-%.1f%% | %s | %.1f%% | %.3f |"
              % (r["model"], r["task"], r["mode"], r["n_attacked"], r["n_flips"],
                 100 * r["asr"], 100 * r["ci95_low"], 100 * r["ci95_high"],
                 ("%.1f%%" % (100 * r["catch_rate"])) if pd.notna(r["catch_rate"]) else "n/a",
                 100 * r["mean_perturbation_rate"], r["mean_jaccard"]))

    # The two intervals must agree in coverage; a large gap means the bootstrap
    # is degenerate (e.g. 0 flips) and the Wilson bound is the one to quote.
    degenerate = table[(table["n_flips"] == 0)]
    if len(degenerate):
        print("\nnote: %d row(s) with 0 flips -> bootstrap collapses to [0,0]; "
              "quote the Wilson upper bound (%.1f%%) as the ceiling"
              % (len(degenerate), 100 * degenerate["ci95_high"].max()))

    warn = []
    small = table[table["n_attacked"] < 10]
    if len(small):
        warn.append("%d row(s) attacked fewer than 10 samples - not reportable as an ASR"
                    % len(small))
    over = table[table["mean_perturbation_rate"] > 0.15]
    if len(over):
        warn.append("%d row(s) exceeded the 15%% perturbation budget" % len(over))
    if warn:
        print("\n!! " + "\n!! ".join(warn))
    return 0


if __name__ == "__main__":
    sys.exit(main())
