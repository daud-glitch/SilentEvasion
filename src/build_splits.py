"""
Build the three frozen input files the study depends on:

    results/eval_subset.csv        800 rows = 200 benign + 200 malicious per task
    results/attack_ch1_input.csv   200 malicious SMS rows, disjoint from eval
    results/calibration_subset.csv 800 rows, disjoint from BOTH of the above

All three are pure functions of data/processed/*.csv (see
common._disjoint_slices), so they can be regenerated on any machine. They
used to exist only as copy-paste snippets in GUIDE.md, which is how a fresh
Colab runtime ended up without them. Run this instead:

    python src/build_splits.py

The printed fingerprint is version-independent (sorted by id, text hashed),
so two machines can prove they built the SAME split even if pandas writes
different quoting. Colab bootstrap compares against the expected values.

The fingerprints are also written to results/SPLIT_FINGERPRINTS.json, so the
"did this machine build the same data?" question has one answer file instead
of a value copied by hand into a notebook.

The calibration slice is the third slice of the same shuffled pool, taken
from its TAIL, so the eval and attack slices are unchanged: every number
already recorded in this repo was computed on exactly those two files.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.common import (RESULTS_DIR, build_all_calibration_subsets,  # noqa: E402
                        build_all_splits, clean_subset)

EVAL_PATH = os.path.join(RESULTS_DIR, "eval_subset.csv")
ATTACK_PATH = os.path.join(RESULTS_DIR, "attack_ch1_input.csv")
CALIBRATION_PATH = os.path.join(RESULTS_DIR, "calibration_subset.csv")
FINGERPRINT_PATH = os.path.join(RESULTS_DIR, "SPLIT_FINGERPRINTS.json")


def fingerprint(df: pd.DataFrame) -> str:
    """Content hash of a split, stable across CSV quoting / pandas versions.

    Sorts by id and hashes id|task|label|text, so identical studies fingerprint
    identically while any sample swap or text change shows up.
    """
    rows = []
    for r in df.sort_values("id").itertuples(index=False):
        text_hash = hashlib.sha256(str(r.text).encode("utf-8")).hexdigest()[:16]
        rows.append(f"{r.id}\t{r.task}\t{int(r.ground_truth)}\t{text_hash}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (eval_subset, attack_input, calibration). Deterministic."""
    eval_df = build_all_splits()
    attack = clean_subset("sms_smishing")
    attack = attack[attack.ground_truth == 1].head(200).reset_index(drop=True)
    calibration = build_all_calibration_subsets()
    return eval_df, attack, calibration


def write(eval_df: pd.DataFrame, attack_df: pd.DataFrame,
          calibration_df: pd.DataFrame) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    eval_df.to_csv(EVAL_PATH, index=False)
    attack_df.to_csv(ATTACK_PATH, index=False)
    if len(calibration_df):
        calibration_df.to_csv(CALIBRATION_PATH, index=False)

    prints = {
        "eval_subset.csv": fingerprint(eval_df),
        "attack_ch1_input.csv": fingerprint(attack_df),
    }
    if len(calibration_df):
        prints["calibration_subset.csv"] = fingerprint(calibration_df)
    with open(FINGERPRINT_PATH, "w", encoding="utf-8") as fh:
        json.dump({
            "seed": 42,
            "n_per_class": 200,
            "splits": prints,
            "note": "eval, attack and calibration are disjoint slices of the "
                    "same shuffled pool; the calibration slice was added from "
                    "the pool's tail, so eval and attack are unchanged.",
        }, fh, indent=2, sort_keys=True)
    return prints


def main() -> int:
    eval_df, attack_df, calibration_df = build()
    prints = write(eval_df, attack_df, calibration_df)

    print(f"eval_subset.csv        {len(eval_df):>4} rows  "
          f"{dict(eval_df.groupby('task').size())}")
    print(f"attack_ch1_input.csv   {len(attack_df):>4} rows  (malicious SMS only)")
    if len(calibration_df):
        print(f"calibration_subset.csv {len(calibration_df):>4} rows  "
              f"{dict(calibration_df.groupby('task').size())}")
    for name, fp in prints.items():
        print(f"  {name:24s} sha256={fp}")
    print(f"fingerprints written -> {os.path.relpath(FINGERPRINT_PATH)}")

    # the whole point of the third slice: no message may appear twice
    if len(calibration_df):
        for other, name in ((eval_df, "eval"), (attack_df, "attack")):
            overlap = set(other["id"]) & set(calibration_df["id"])
            assert not overlap, "%s overlaps %s: %s" % (name, len(overlap),
                                                         sorted(overlap)[:5])
        print("disjointness: calibration shares no id with eval or attack")
    return 0


if __name__ == "__main__":
    sys.exit(main())
