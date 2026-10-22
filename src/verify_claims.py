"""
Recompute every number the paper states, straight from the per-sample records.

Why this exists: the paper quotes a handful of numbers that are easy to get
subtly wrong because they mix three different denominators - attacked
instances, model queries, and eligible samples. A number that is correct for
one denominator is wrong for another, and the text does not say which one it
is using. This script prints each number with its denominator attached, so the
paper text can be checked against a record rather than against a memory.

    python src/verify_claims.py

Read-only: it touches no file in results/.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.common import RESULTS_DIR  # noqa: E402

RANDOM_MODES = ("char", "word")


def load_all():
    frames = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "attack_results_*.csv"))):
        df = pd.read_csv(path)
        df["_file"] = os.path.basename(path)
        frames.append(df)
    return frames


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    frames = load_all()
    if not frames:
        raise SystemExit("no attack_results_*.csv in %s" % RESULTS_DIR)
    allrows = pd.concat(frames, ignore_index=True)
    counted = allrows[allrows["counted"] == True]      # noqa: E712
    skipped = allrows[allrows["counted"] != True]      # noqa: E712

    # ── 1. excluded originals, by mode ────────────────────────────────
    rule("1. EXCLUDED ORIGINALS (counted == False), by mode")
    print(skipped.groupby("mode").size().to_string())
    print("\ntotal exclusions across all 9 files: %d" % len(skipped))
    print("\nby mode and skip_reason:")
    print(skipped.groupby(["mode", "skip_reason"]).size().to_string())
    sal_skipped = skipped[skipped["mode"] == "saliency"]
    print("\nsaliency exclusions: %d" % len(sal_skipped))
    print("  of which orig_pred != 1 (probability route read it benign): %d"
          % int((sal_skipped["orig_pred"] != 1).sum()))
    print("  by model (orig_pred != 1):")
    print(sal_skipped[sal_skipped["orig_pred"] != 1]
          .groupby("model").size().to_string())

    # ── 2. attacked instances vs model queries ───────────────────────
    rule("2. ATTACKED INSTANCES vs MODEL QUERIES")
    for mode in RANDOM_MODES:
        sub = allrows[allrows["mode"] == mode]
        n_elig = len(sub)
        n_att = int((sub["counted"] == True).sum())       # noqa: E712
        # one query for the original, one per attempt that actually ran
        attempts = sub["attempt"].fillna(0)
        n_queries = int((1 + attempts).sum())
        max_att = int(attempts.max()) if n_elig else 0
        print("\nmode=%s" % mode)
        print("  eligible rows                : %d" % n_elig)
        print("  attacked instances (counted)  : %d" % n_att)
        print("  max attempt index observed   : %d" % max_att)
        print("  model queries (orig + attempts): %d" % n_queries)
        print("  queries per attacked instance : %.2f"
              % (n_queries / max(n_att, 1)))
    rand_att = int(allrows[allrows["mode"].isin(RANDOM_MODES)]["counted"].sum())  # noqa: E712
    rand_q = int((1 + allrows[allrows["mode"].isin(RANDOM_MODES)]["attempt"].fillna(0)).sum())
    print("\nRANDOM ATTACKS COMBINED")
    print("  attacked instances : %d" % rand_att)
    print("  model queries      : %d" % rand_q)
    sal = allrows[allrows["mode"] == "saliency"]
    sal_att = int((sal["counted"] == True).sum())          # noqa: E712
    sal_q = int(sal["n_queries"].fillna(0).sum())
    print("\nSALIENCY")
    print("  attacked instances : %d" % sal_att)
    print("  candidate/probability queries (n_queries column): %d" % sal_q)
    print("  mean queries per attacked instance: %.1f"
          % (sal_q / max(sal_att, 1)))
    print("  + generation re-scoring pass (verify_saliency) : %d extra queries"
          % sal_att)

    # ── 3. mean perturbation rate, pooled by mode ────────────────────
    rule("3. MEAN PERTURBATION RATE (pooled over all counted rows)")
    for mode in ("char", "word", "saliency"):
        sub = counted[counted["mode"] == mode]
        if not len(sub):
            continue
        print("\nmode=%s  n=%d" % (mode, len(sub)))
        print("  all counted rows      : %.4f  (%.1f%%)"
              % (sub["perturbation_rate"].mean(), 100 * sub["perturbation_rate"].mean()))
        ok = sub[sub["success"] == True]                  # noqa: E712
        if len(ok):
            print("  successful rows only  : %.4f  (%.1f%%)  n=%d"
                  % (ok["perturbation_rate"].mean(),
                     100 * ok["perturbation_rate"].mean(), len(ok)))
        print("  successful rows, generation channel (success_gen):")
        if "success_gen" in sub.columns:
            okg = sub[sub["success_gen"] == True]         # noqa: E712
            if len(okg):
                print("    : %.4f  (%.1f%%)  n=%d"
                      % (okg["perturbation_rate"].mean(),
                         100 * okg["perturbation_rate"].mean(), len(okg)))
        print("  mean jaccard          : %.4f" % sub["similarity"].mean())
        print("  max perturbation rate : %.4f" % sub["perturbation_rate"].max())

    # saliency vs word, restricted to the SAME messages (paired)
    rule("3b. SALIENCY vs WORD on the messages both attacks counted")
    w = counted[counted["mode"] == "word"][["id", "task", "model", "success",
                                            "perturbation_rate", "similarity"]]
    s = counted[counted["mode"] == "saliency"][["id", "task", "model", "success_gen",
                                                "perturbation_rate", "similarity"]]
    s = s.rename(columns={"success_gen": "success"})
    j = w.merge(s, on=["id", "task", "model"], suffixes=("_w", "_s"))
    print("shared messages: %d" % len(j))
    print("  mean perturbation rate  word     : %.4f" % j["perturbation_rate_w"].mean())
    print("  mean perturbation rate  saliency : %.4f" % j["perturbation_rate_s"].mean())
    print("  flips word=%d  saliency=%d"
          % (int(j["success_w"].sum()), int(j["success_s"].sum())))

    # ── 4. denominators of the headline rates ───────────────────────
    rule("4. HEADLINE RATE DENOMINATORS")
    for mode in ("char", "word", "saliency"):
        sub = counted[counted["mode"] == mode]
        vcol = "success_gen" if mode == "saliency" and "success_gen" in sub.columns \
            else "success"
        flips = int(sub[vcol].sum())
        print("  %-9s %4d / %4d = %.1f%%"
              % (mode, flips, len(sub), 100 * flips / max(len(sub), 1)))
    print("\n  the three rates above do NOT share a denominator:")
    for mode in ("char", "word", "saliency"):
        sub = allrows[allrows["mode"] == mode]
        print("    %-9s eligible=%4d  attacked=%4d  excluded=%d"
              % (mode, len(sub), int((sub["counted"] == True).sum()),   # noqa: E712
                 int((sub["counted"] != True).sum())))                   # noqa: E712

    # ── 5. candidate sets: what each attack could actually substitute ─
    rule("5. CANDIDATE-SET SIZES (the vocabulary confound)")
    from src.attack_common import SYNONYMS
    print("  random word attack: %d hand-written entries, %d replacement strings"
          % (len(SYNONYMS), sum(len(v) for v in SYNONYMS.values())))
    try:
        from src.saliency_common import CANDIDATES
        print("  saliency attack: %d curated lexicon entries" % len(CANDIDATES))
    except Exception as exc:
        print("  saliency candidate table: %s" % exc)

    # ── 6. the vocabulary actually exercised, per attack ─────────────
    rule("6. HOW OFTEN EACH ATTACK COULD SUBSTITUTE A WORD AT ALL")
    for mode in ("word", "saliency"):
        sub = counted[counted["mode"] == mode]
        if not len(sub):
            continue
        types = sub["attack_types"].fillna("")
        has_syn = int(types.str.contains("synonym|substitute", case=False).sum())
        print("  %-9s rows with a substitution op: %4d / %4d (%.1f%%)"
              % (mode, has_syn, len(sub), 100 * has_syn / len(sub)))
        print("            distinct attack_types: %s"
              % sorted(set(t for t in types if t))[:8])

    print("\nDone. Every number above is read from results/attack_results_*.csv.")


if __name__ == "__main__":
    main()
