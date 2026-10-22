"""
Cross-model transfer: take perturbations found against one model and score them
on another.

The same accounting rules as the single-model attack apply, with one addition
that is specific to transfer:

  * the TARGET model decides the denominator. A perturbation transfers only if
    the target read the ORIGINAL text as malicious and the PERTURBED text as
    benign. A sample the target never caught cannot transfer, and counting it
    as a failure would punish the target for being wrong on the clean text.
  * the original verdict is queried once per (target, sample) and reused,
    because every mode and every source model reuses the same samples.
  * both subsets are reported: every perturbation the source actually applied
    (large n, the transfer rate of the perturbation family) and only the
    perturbations that fooled the source (small n, the classic transfer
    matrix). They answer different questions and the small one is noisy.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.attack_common import asr_interval, read_attack_csv  # noqa: E402
from src.common import RESULTS_DIR  # noqa: E402

SUBSETS = ("applied", "flipped_source")


def load_original_texts():
    """id -> original text, from the frozen evaluation split."""
    path = os.path.join(RESULTS_DIR, "eval_subset.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; rebuild the splits first")
    d = pd.read_csv(path)
    return dict(zip(d["id"], d["text"]))


def load_perturbations(originals, source_model=None):
    """Per-sample records worth transferring.

    Returns a list of dicts: id, task, source, mode, original, perturbed,
    flipped_source. Rows without a stored perturbed text are skipped, and rows
    whose original is missing from the split are skipped loudly - a silent drop
    would shrink the denominator.
    """
    import glob

    out, missing = [], 0
    pattern = os.path.join(RESULTS_DIR, "attack_results_*.csv")
    for path in sorted(glob.glob(pattern)):
        df = read_attack_csv(path)
        if "counted" not in df.columns:
            continue
        src = str(df["model"].iloc[0])
        if source_model and src != source_model:
            continue
        mode = str(df["mode"].iloc[0])
        counted = df[df["counted"] == True]                       # noqa: E712
        # The stored flag has to come from the SAME channel the replay uses.
        # The replay calls the generated verdict, and for the saliency rows the
        # stored `success` is the probability-channel verdict, which disagrees
        # with generation on 60 of 1,001 rows - using it here would make the
        # self-replay look unfaithful when it is the flag that is different.
        vcol = "success_gen" if "success_gen" in counted.columns else "success"
        counted = counted[counted[vcol].notna()]
        for r in counted.itertuples(index=False):
            text = getattr(r, "perturbed_text", None)
            if not isinstance(text, str) or not text:
                continue
            orig = originals.get(getattr(r, "id", None))
            if not isinstance(orig, str):
                missing += 1
                continue
            out.append({"id": r.id, "task": r.task, "source": src, "mode": mode,
                        "original": orig, "perturbed": text,
                        "flipped_source": bool(getattr(r, vcol))})
    if missing:
        print("!! %d perturbation(s) had no original text in eval_subset.csv "
              "and were dropped" % missing)
    return out


def score(records, target_model, ask):
    """Query the target on original and perturbed text; return per-sample rows.

    ask(text, task) -> 1 (malicious), 0 (benign) or None (unparseable).
    """
    verdicts = {}
    rows = []
    for rec in records:
        key = (rec["task"], rec["id"])
        if key not in verdicts:
            verdicts[key] = ask(rec["original"], rec["task"])
        orig = verdicts[key]
        caught = orig == 1
        pred = ask(rec["perturbed"], rec["task"]) if caught else None
        rows.append({
            "id": rec["id"], "task": rec["task"],
            "source": rec["source"], "target": target_model, "mode": rec["mode"],
            "flipped_source": rec["flipped_source"],
            "target_orig_pred": orig, "target_pert_pred": pred,
            "counted": caught,
            "success": bool(caught and pred == 0),
        })
    return pd.DataFrame(rows)


def matrix_rows(df):
    """Aggregate per-sample transfer rows into the table the paper quotes."""
    out = []
    for subset in SUBSETS:
        sub = df if subset == "applied" else df[df["flipped_source"] == True]  # noqa: E712
        for (source, target, mode, task), g in sub.groupby(
                ["source", "target", "mode", "task"]):
            counted = g[g["counted"] == True]      # noqa: E712
            if not len(counted):
                continue
            ci = asr_interval(counted["success"].tolist())
            out.append({
                "subset": subset, "source": source, "target": target,
                "mode": mode, "task": task,
                "n_candidates": int(len(g)),
                "n_target_caught": int(len(counted)),
                "n_flips": int(counted["success"].sum()),
                "transfer_asr": round(float(counted["success"].mean()), 4),
                "ci95_low": ci["wilson_low"], "ci95_high": ci["wilson_high"],
            })
    return out


def self_transfer(df):
    """source == target must reproduce the single-model ASR; a mismatch is a bug.

    Compared row by row, not cell by cell. A cell-level sum hides the worst case:
    one message that flipped on the attack and not on the replay, plus another
    that did the opposite, cancels out and the cell still looks faithful. Two
    things are checked instead:

      * every replayed row agrees with the flag its attack run recorded, and
      * the replay still reads every original as malicious. A dropped original
        would silently shrink the denominator, which is the failure mode that
        makes an ASR look better than it is.

    Returns a list of (source, mode, task, kind, count) tuples, kind being
    'row_mismatch' or 'uncaught_original'.
    """
    bad = []
    for (source, mode, task), g in df[df["source"] == df["target"]].groupby(
            ["source", "mode", "task"]):
        unc = int((g["counted"] != True).sum())    # noqa: E712
        if unc:
            bad.append((source, mode, task, "uncaught_original", unc))
        counted = g[g["counted"] == True]          # noqa: E712
        mism = int((counted["flipped_source"] != counted["success"]).sum())
        if mism:
            bad.append((source, mode, task, "row_mismatch", mism))
    return bad
