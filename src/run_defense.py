"""
Training-free defences against the measured evasions, evaluated on the SAME
stored perturbations the attacks produced.

Two defences, both of which a SOC can deploy without touching the weights:

  D1  input normalisation - NFKC, invisible/format characters removed, Cyrillic
      and Greek homoglyphs folded back to ASCII, whitespace collapsed. This
      targets the mechanism of the character attack: the model sees a different
      token, a human sees the same word.
  D2  confidence re-thresholding - the model's P(malicious) is compared against
      a threshold tau fitted on CLEAN benign mail to a stated false-positive
      budget (5 percent by default). The clean TPR and FPR at that operating
      point are
      reported next to every ASR, because an ASR that drops only because the
      filter stopped flagging anything is not a defence.

Two rules that the first version of this file got wrong, both of which made the
defence table meaningless, and both of which are now enforced:

  1. tau is the SMALLEST threshold that meets the FPR budget. Taking the largest
     one (the v4 code) always lands on "above every clean score", i.e. reject
     everything, which shows up as a 100% ASR for the defence.
  2. a sample counts as flipped only if the SAME rule called the ORIGINAL text
     malicious. Otherwise a threshold that is simply too high is credited with
     flips the attack never caused, and the attack's own effect is invisible.

Measured quantity is the same as everywhere else in this study: flips among the
samples the undefended model called malicious, with a Wilson interval.

One caveat that must stay in the paper: D1 only helps when the perturbation is
character-level noise it can undo. It is reported for the word and saliency
attacks too, where it should change almost nothing - that null result is the
evidence that the two attack families fail for different reasons.
"""
import argparse
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.attack_common import asr_interval, read_attack_csv  # noqa: E402
from src.common import RESULTS_DIR, TASKS  # noqa: E402
from src.saliency_common import verdict_probs  # noqa: E402

# Cyrillic/Greek letters that render close enough to Latin to be read as Latin.
HOMOGLYPH_TO_ASCII = {
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c",
    "\u0445": "x", "\u0443": "y", "\u0456": "i", "\u0455": "s", "\u0410": "A",
    "\u0412": "B", "\u0415": "E", "\u041a": "K", "\u041c": "M", "\u041d": "H",
    "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0422": "T", "\u0425": "X",
    "\u03b1": "a", "\u03bf": "o", "\u03bd": "v", "\u03b9": "i",
}

# Zero-width and bidirectional control characters: invisible in a UI, a different
# token to the model, and a classic way to split a keyword.
INVISIBLE = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
             "\u202a", "\u202b", "\u202c", "\u202d", "\u202e"}


def normalize_text(text):
    """Fold a text back to the form a human reader perceives.

    Order matters: NFKC first (it maps many compatibility forms, including some
    homoglyph look-alikes, in one step), then the explicit Cyrillic/Greek map,
    then invisible characters, then whitespace. The reverse order would leave
    NFKC with characters the map already rewrote.
    """
    if not isinstance(text, str):
        return text
    out = unicodedata.normalize("NFKC", text)
    out = "".join(HOMOGLYPH_TO_ASCII.get(ch, ch) for ch in out)
    out = "".join(ch for ch in out if ch not in INVISIBLE)
    out = "".join(ch for ch in out if unicodedata.category(ch) != "Cf")
    return " ".join(out.split())


def fpr_at(clean_benign_probs, tau):
    probs = [p for p in clean_benign_probs if p is not None]
    if not probs:
        return 0.0
    return sum(1 for p in probs if p >= tau) / len(probs)


def tpr_at(clean_malicious_probs, tau):
    probs = [p for p in clean_malicious_probs if p is not None]
    if not probs:
        return 0.0
    return sum(1 for p in probs if p >= tau) / len(probs)


def choose_threshold(clean_benign_probs, target_fpr, floor=0.5):
    """Smallest tau >= floor whose clean FPR stays inside the budget.

    FPR falls as tau rises, so the smallest feasible tau is the LEAST
    conservative operating point that still respects the budget - the one that
    keeps the most recall. The v4 implementation maximised tau instead, which is
    always the last candidate on the grid (clean FPR 0, nothing flagged) and
    produced a defence table reporting a 100% ASR.

    floor = 0.5 because a filter that flags more mail than the baseline is not a
    defence but an unmeasured trade; the caller compares clean TPR at both ends.
    """
    probs = [p for p in clean_benign_probs if p is not None]
    if not probs:
        return floor
    candidates = sorted({round(p, 4) for p in probs} | {floor, 1.0})
    for tau in candidates:
        if tau < floor:
            continue
        if fpr_at(probs, tau) <= target_fpr + 1e-12:
            return tau
    return 1.0


def attack_csvs_for(model):
    """Every stored attack run for one model, whatever its mode."""
    import glob
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "attack_results_*.csv"))):
        try:
            df = read_attack_csv(path)
        except Exception:
            continue
        if "counted" not in df.columns or not len(df):
            continue
        if "model" not in df.columns:                     # pragma: no cover
            continue
        if str(df["model"].iloc[0]) != model:
            continue
        out.append((path, df))
    return out


def evaluate(model, base_url, limit=None, target_fpr=0.05):
    """Score every stored perturbation under both defences."""
    clean_path = os.path.join(RESULTS_DIR, "eval_subset.csv")
    if not os.path.exists(clean_path):
        raise SystemExit("missing %s" % clean_path)
    clean = pd.read_csv(clean_path)
    benign = clean[clean["ground_truth"] == 0]
    malicious = clean[clean["ground_truth"] == 1]
    text_by_id = dict(zip(clean["id"], clean["text"]))

    rows = []
    thresholds = {}
    for task in TASKS:
        def ask(df):
            texts = [str(t) for t in df[df["task"] == task]["text"].tolist()]
            if limit:
                texts = texts[:limit]
            return texts, (verdict_probs(base_url, model, texts, task=task) or [])

        _bt, p_ben = ask(benign)
        _mt, p_mal = ask(malicious)
        tau = choose_threshold(p_ben, target_fpr)
        thresholds[task] = {
            "tau": round(tau, 4),
            "fpr_at_0.5": round(fpr_at(p_ben, 0.5), 4),
            "tpr_at_0.5": round(tpr_at(p_mal, 0.5), 4),
            "fpr_at_tau": round(fpr_at(p_ben, tau), 4),
            "tpr_at_tau": round(tpr_at(p_mal, tau), 4),
            "n_clean_benign": len([p for p in p_ben if p is not None]),
            "n_clean_malicious": len([p for p in p_mal if p is not None]),
            "target_fpr": target_fpr,
            # tau = 1.0 means no threshold inside [0.5, 1) met the budget: the
            # model flags so much clean mail that holding the FPR down requires
            # rejecting everything. That is a finding, not a number to print.
            "budget_met": bool(tau < 1.0),
        }
        t = thresholds[task]
        print("%-16s tau=%.4f  clean FPR %.3f -> %.3f   clean TPR %.3f -> %.3f"
              % (task, tau, t["fpr_at_0.5"], t["fpr_at_tau"],
                 t["tpr_at_0.5"], t["tpr_at_tau"]))

    for path, df in attack_csvs_for(model):
        counted = df[df["counted"] == True]               # noqa: E712
        if limit:
            counted = counted.head(limit)
        if not len(counted):
            continue
        mode = str(df["mode"].iloc[0])
        print("defence eval: %s rows of %s" % (len(counted), os.path.basename(path)))
        # Batched on purpose: every row needs two probabilities (perturbed and
        # normalised), and one-at-a-time would dominate the runtime of the
        # whole defence phase.
        for task in sorted(counted["task"].unique()):
            sub = counted[counted["task"] == task]
            kept = sub[[isinstance(p, str) and bool(p)
                        for p in sub["perturbed_text"]]]
            perts = [str(p) for p in kept["perturbed_text"].tolist()]
            norms = [normalize_text(p) for p in perts]
            p_pert = verdict_probs(base_url, model, perts, task=task)
            p_norm = verdict_probs(base_url, model, norms, task=task)
            if not p_pert or not p_norm:
                print("!! verdict_probs failed for %s/%s - skipped" % (mode, task))
                continue
            # The originals are scored under BOTH rules, so a flip is always
            # "the same filter caught it before and misses it now".
            orig_ids = list(kept["id"])
            origs = [str(text_by_id.get(i, "")) for i in orig_ids]
            p_orig = verdict_probs(base_url, model, origs, task=task)
            p_orig_norm = verdict_probs(
                base_url, model, [normalize_text(t) for t in origs], task=task)
            if not p_orig or not p_orig_norm:
                print("!! verdict_probs failed on originals (%s/%s) - skipped"
                      % (mode, task))
                continue
            tau = thresholds.get(task, {}).get("tau", 0.5)
            for r, a, b, po, pon in zip(kept.itertuples(index=False), p_pert,
                                        p_norm, p_orig, p_orig_norm):
                pert = str(r.perturbed_text)
                rows.append({
                    "id": r.id, "model": model, "mode": mode, "task": task,
                    "success_baseline": bool(getattr(r, "success", False)),
                    "p_orig": round(po, 6), "p_orig_norm": round(pon, 6),
                    "p_pert": round(a, 6), "p_norm": round(b, 6),
                    "changed_by_normalisation": normalize_text(pert) != pert,
                    "tau": tau,
                    "orig_malicious": po >= 0.5,
                    "orig_malicious_norm": pon >= 0.5,
                    "orig_malicious_tau": po >= tau,
                    "orig_malicious_norm_tau": pon >= tau,
                    "flip_baseline": po >= 0.5 and a < 0.5,
                    "flip_normalised": pon >= 0.5 and b < 0.5,
                    "flip_threshold": po >= tau and a < tau,
                    "flip_both": pon >= tau and b < tau,
                },)
    return pd.DataFrame(rows), thresholds


def summarize(df):
    """Per (mode, task, defence) flip counts and intervals.

    Each row carries its own denominator: only samples the SAME rule called
    malicious on the ORIGINAL text can be flipped by the attack, so `n` differs
    between defences whenever the probability route disagrees with the stored
    generated verdict on an original.
    """
    out = []
    keys = ["mode", "task"]
    cols = [("baseline", "flip_baseline", "orig_malicious"),
            ("normalise", "flip_normalised", "orig_malicious_norm"),
            ("threshold", "flip_threshold", "orig_malicious_tau"),
            ("both", "flip_both", "orig_malicious_norm_tau")]
    for (mode, task), g in df.groupby(keys):
        for name, col, elig in cols:
            sub = g[g[elig] == True]                      # noqa: E712
            ci = asr_interval([bool(x) for x in sub[col]])
            out.append({"mode": mode, "task": task, "defense": name,
                        "n_eligible": len(sub), "n_rows": len(g),
                        "n": ci["n"], "flips": ci["flips"],
                        # NaN, not 0.0: an empty denominator means the rule
                        # flagged no original at all, so nothing could flip.
                        "asr": round(ci["flips"] / ci["n"], 4) if ci["n"] else None,
                        "ci95_low": ci["wilson_low"], "ci95_high": ci["wilson_high"],
                        "tau": (round(float(g["tau"].iloc[0]), 4)
                                if name in ("threshold", "both") else 0.5)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--limit", default=None, type=int,
                    help="cap rows per attack CSV (smoke test)")
    ap.add_argument("--target-fpr", default=0.05, type=float,
                    help="clean false-positive budget the threshold is fitted "
                         "to (default 5%%)")
    ap.add_argument("--out-dir", default=RESULTS_DIR,
                    help="where to write the CSVs; point a smoke test at a "
                         "scratch directory so mock numbers never land in "
                         "results/")
    args = ap.parse_args()

    df, thresholds = evaluate(args.model, args.base_url, limit=args.limit,
                              target_fpr=args.target_fpr)
    if not len(df):
        print("no perturbations evaluated - run the attacks first")
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    rows_path = os.path.join(args.out_dir, "defense_rows_%s.csv" % args.model)
    df.to_csv(rows_path, index=False)

    table = summarize(df)
    table.insert(0, "model", args.model)
    out_path = os.path.join(args.out_dir, "defense_summary.csv")
    if os.path.exists(out_path):
        old = pd.read_csv(out_path)
        old = old[old["model"] != args.model]          # replace, never append twice
        table = pd.concat([old, table], ignore_index=True)
    table = table.sort_values(["model", "mode", "task", "defense"])
    table.to_csv(out_path, index=False)

    # The operating point has to travel with the table: an ASR quoted without
    # the clean TPR/FPR it was bought at is not a defence result.
    thr = pd.DataFrame([dict(model=args.model, task=t, **v)
                        for t, v in thresholds.items()])
    thr.insert(0, "model", thr.pop("model"))
    thr_path = os.path.join(args.out_dir, "defense_thresholds.csv")
    if os.path.exists(thr_path):
        old = pd.read_csv(thr_path)
        old = old[old["model"] != args.model]
        thr = pd.concat([old, thr], ignore_index=True)
    thr = thr.sort_values(["model", "task"])
    thr.to_csv(thr_path, index=False)

    print("\n| mode | task | defence | flips | n | ASR | 95% CI | tau |")
    print("|---|---|---|---|---|---|---|---|")
    for _, r in table.iterrows():
        print("| %s | %s | %s | %d | %d | %.1f%% | %.1f-%.1f%% | %.2f |"
              % (r["mode"], r["task"], r["defense"], r["flips"], r["n"],
                 100 * r["asr"], 100 * r["ci95_low"], 100 * r["ci95_high"],
                 r["tau"]))
    changed = int(df["changed_by_normalisation"].sum())
    print("\nnormalisation changed %d/%d perturbed texts "
          "(the char attack's mechanism; expect ~all for char, ~0 for word)"
          % (changed, len(df)))
    # Cross-check: the probability route must agree with the verdict stored by
    # the attack run. A mismatch means the shim is not serving the same model,
    # or that the two verdict channels disagree - which is itself a result, so
    # it is reported rather than asserted away.
    for mode, g in df.groupby("mode"):
        agree = int((g["flip_baseline"] == g["success_baseline"]).sum())
        print("re-derivation check %-9s %d/%d rows agree with the stored verdict"
              % (mode, agree, len(g)))
    inconsistent = int((~df["orig_malicious"]).sum())
    print("originals the probability route calls benign: %d/%d rows "
          "(excluded from the baseline denominator)"
          % (inconsistent, len(df)))
    print("saved -> %s" % rows_path)
    print("saved -> %s" % out_path)
    print("saved -> %s" % thr_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
