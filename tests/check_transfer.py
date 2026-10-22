"""
Offline checks for cross-model transfer accounting - no model, no shim.

The dangerous mistakes in a transfer matrix are:
  1. counting a sample the target never caught as a "failed transfer"
     (that punishes the target for being wrong on clean text)
  2. crediting a flip when the target's original verdict was benign
     (there is nothing to flip)
  3. reporting the "applied" transfer rate as if it were the classic
     "successful attacks transfer" matrix, or vice versa
  4. a diagonal that does not reproduce the single-model ASR

    python tests/check_transfer.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.transfer_common import SUBSETS, matrix_rows, score, self_transfer  # noqa: E402

FAILS = []


def check(label, ok, detail=""):
    print(("  OK   " if ok else "  FAIL ") + label + ((" - " + detail) if detail and not ok else ""))
    if not ok:
        FAILS.append(label)


def records():
    """4 samples; source flipped 1 and 3."""
    return [
        {"id": "a", "task": "sms_smishing", "source": "m1", "mode": "char",
         "original": "ORIG-A", "perturbed": "PERT-A", "flipped_source": True},
        {"id": "b", "task": "sms_smishing", "source": "m1", "mode": "char",
         "original": "ORIG-B", "perturbed": "PERT-B", "flipped_source": False},
        {"id": "c", "task": "phishing_email", "source": "m1", "mode": "char",
         "original": "ORIG-C", "perturbed": "PERT-C", "flipped_source": True},
        {"id": "d", "task": "phishing_email", "source": "m1", "mode": "char",
         "original": "ORIG-D", "perturbed": "PERT-D", "flipped_source": False},
    ]


def test_accounting():
    print("\n[1] target decides the denominator")
    # target catches A and B only; flips neither
    seen_orig = []

    def ask(text, task):
        if text.startswith("ORIG"):
            seen_orig.append(text)
            return 1 if text in ("ORIG-A", "ORIG-B") else 0
        return 1                                  # never flips

    df = score(records(), "m2", ask)
    check("4 rows reported", len(df) == 4, str(len(df)))
    check("only caught originals are counted",
          int(df["counted"].sum()) == 2, str(df["counted"].tolist()))
    check("no transfer successes", int(df["success"].sum()) == 0)
    check("each original queried exactly once",
          sorted(seen_orig) == ["ORIG-A", "ORIG-B", "ORIG-C", "ORIG-D"], str(seen_orig))

    # a sample the target never caught must not be queried for its perturbation
    pert_queries = []

    def ask2(text, task):
        if text.startswith("PERT"):
            pert_queries.append(text)
        if text.startswith("ORIG"):
            return 1 if text == "ORIG-A" else 0
        return 0                                  # everything it sees flips

    df = score(records(), "m2", ask2)
    check("perturbation only queried when the original was caught",
          pert_queries == ["PERT-A"], str(pert_queries))
    check("exactly one success", int(df["success"].sum()) == 1, str(df["success"].tolist()))
    check("the success is sample a",
          df.loc[df["success"], "id"].tolist() == ["a"], str(df.loc[df["success"], "id"].tolist()))


def test_subsets():
    print("\n[2] the two subsets are different questions")

    def ask(text, task):
        if text.startswith("ORIG"):
            return 1                              # target caught everything
        return 0 if text in ("PERT-A", "PERT-C") else 1   # flips both source flips

    df = score(records(), "m2", ask)
    rows = matrix_rows(df)
    applied = [r for r in rows if r["subset"] == "applied"]
    classic = [r for r in rows if r["subset"] == "flipped_source"]
    check("both subsets present", {r["subset"] for r in rows} == set(SUBSETS), str(len(rows)))
    check("applied sees all 4 candidates",
          sum(r["n_candidates"] for r in applied) == 4,
          str([r["n_candidates"] for r in applied]))
    check("classic subset sees only the 2 source flips",
          sum(r["n_candidates"] for r in classic) == 2,
          str([r["n_candidates"] for r in classic]))
    check("applied ASR is 2/4", any(r["transfer_asr"] == 0.5 for r in applied),
          str([r["transfer_asr"] for r in applied]))
    check("classic ASR is 2/2", all(r["transfer_asr"] == 1.0 for r in classic),
          str([r["transfer_asr"] for r in classic]))
    check("every row carries an interval",
          all(r["ci95_high"] >= r["transfer_asr"] >= r["ci95_low"] for r in rows))


def test_diagonal():
    print("\n[3] source == target must reproduce the single-model ASR")

    def ask(text, task):
        if text.startswith("ORIG"):
            return 1
        return 0 if text in ("PERT-A", "PERT-C") else 1

    same = score(records(), "m1", ask)
    check("faithful replay has no diagonal mismatch", self_transfer(same) == [],
          str(self_transfer(same)))

    broken = same.copy()
    broken.loc[broken["id"] == "b", "success"] = True      # invent a flip
    bad = self_transfer(broken)
    check("an invented flip is caught on the diagonal", len(bad) == 1, str(bad))

    # The reason the comparison is per row: a flip lost on one message and a
    # flip gained on another leave the cell total unchanged.
    swap = same.copy()
    swap.loc[swap["id"] == "a", "success"] = False
    swap.loc[swap["id"] == "b", "success"] = True
    bad = self_transfer(swap)
    check("compensating flips inside one cell are caught", len(bad) == 1, str(bad))

    # A replay that re-reads a caught original as benign drops the row from the
    # denominator, which inflates the cell's ASR while the flip count still ties.
    lost = same.copy()
    lost.loc[lost["id"] == "d", "counted"] = False
    bad = self_transfer(lost)
    check("an original the replay no longer catches is caught", len(bad) == 1, str(bad))


def test_channel_consistency():
    """The stored flag must come from the channel the replay uses.

    Saliency rows carry both `success` (probability route) and `success_gen`
    (generated verdict). The replay calls the generated verdict, so using the
    probability flag here would make a faithful replay look unfaithful on the
    rows where the two channels disagree - which is 60 of 1,001 in our data.
    """
    import tempfile

    import pandas as pd

    import src.transfer_common as tc

    with tempfile.TemporaryDirectory() as tmp:
        pd.DataFrame({
            "id": ["a", "b", "c"], "text": ["ORIG-A", "ORIG-B", "ORIG-C"],
            "task": ["t"] * 3, "ground_truth": [1, 1, 1], "split": ["eval"] * 3,
        }).to_csv(os.path.join(tmp, "eval_subset.csv"), index=False)
        pd.DataFrame({
            "id": ["a", "b", "c"], "model": ["m1"] * 3, "mode": ["saliency"] * 3,
            "task": ["t"] * 3, "orig_pred": [1, 1, 1], "pert_pred": [0, 1, 0],
            "success": [True, False, True], "counted": [True, True, True],
            "skip_reason": [None] * 3, "perturbed_text": ["PERT-A", "PERT-B", "PERT-C"],
            "success_gen": [False, True, True],
        }).to_csv(os.path.join(tmp, "attack_results_saliency_m1.csv"), index=False)

        old = tc.RESULTS_DIR
        tc.RESULTS_DIR = tmp
        try:
            recs = tc.load_perturbations(tc.load_original_texts())
        finally:
            tc.RESULTS_DIR = old

    flags = {r["id"]: r["flipped_source"] for r in recs}
    check("flipped_source follows success_gen when it exists",
          flags == {"a": False, "b": True, "c": True}, str(flags))
    check("the probability flag is not used for the replay subset",
          flags.get("a") is False, str(flags))


def main():
    print("TRANSFER CHECK")
    test_accounting()
    test_subsets()
    test_diagonal()
    test_channel_consistency()
    print("\nRESULT: %s" % ("FAIL: %d problem(s)" % len(FAILS) if FAILS else "OK"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
