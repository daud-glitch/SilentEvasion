"""Tests for the saliency verdict-channel check and its use in the ASR table.

The saliency attack is the only mode that decides success from a probability
P(malicious) < 0.5; char and word decide from the generated one-word answer. Two
readouts of the same model are not automatically the same measurement, so:

  * the verification summary must count a flip per channel and separate the rows
    where only one channel fired (those rows are the finding);
  * the aggregated ASR table must use the generation column when it exists,
    because that is the channel the baseline and the other two modes use;
  * when the column is absent the table must fall back, not crash.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

import src.summarize_attack as sa  # noqa: E402
from src.verify_saliency import summarize  # noqa: E402


def fake_rows():
    return pd.DataFrame([
        # prob says flip, generation agrees
        {"model": "m", "mode": "saliency", "task": "t", "counted": True,
         "success": True, "success_gen": True},
        # prob says flip, generation says no: the ruler disagrees
        {"model": "m", "mode": "saliency", "task": "t", "counted": True,
         "success": True, "success_gen": False},
        # generation says flip, prob said no
        {"model": "m", "mode": "saliency", "task": "t", "counted": True,
         "success": False, "success_gen": True},
        # neither
        {"model": "m", "mode": "saliency", "task": "t", "counted": True,
         "success": False, "success_gen": False},
    ])


def test_summary_counts_both_channels_and_the_disagreements():
    df = fake_rows()
    df["verdict_agree"] = df["success"] == df["success_gen"]
    table = summarize([df])
    row = table.iloc[0]
    assert row["n"] == 4, row.to_dict()
    assert row["flips_prob"] == 2, row.to_dict()
    assert row["flips_gen"] == 2, row.to_dict()
    assert row["agree"] == 2, row.to_dict()
    assert row["prob_only"] == 1 and row["gen_only"] == 1, row.to_dict()


def _attack_file(path, gen_column):
    rows = fake_rows()
    if not gen_column:
        rows = rows.drop(columns=["success_gen"])
    rows["perturbation_rate"] = 0.1
    rows["similarity"] = 0.9
    rows.to_csv(path, index=False)


def _table_with(tmp, gen_column):
    _attack_file(os.path.join(tmp, "attack_results_saliency_m.csv"), gen_column)
    pd.DataFrame([
        {"id": "a", "task": "t", "ground_truth": 1, "pred": 1,
         "raw": "malicious", "error": "", "model": "m", "elapsed_s": 1.0},
    ]).to_csv(os.path.join(tmp, "clean_baseline_results.csv"), index=False)
    old = sa.RESULTS_DIR
    sa.RESULTS_DIR = tmp
    try:
        return pd.DataFrame(sa.rows())
    finally:
        sa.RESULTS_DIR = old


def test_the_table_uses_the_generation_channel_when_present():
    with tempfile.TemporaryDirectory() as tmp:
        table = _table_with(tmp, gen_column=True)
    row = table.iloc[0]
    assert row["n_flips"] == 2, row.to_dict()          # not 2 from the prob route
    assert row["verdict_channel"] == "generation", row.to_dict()


def test_the_table_falls_back_to_the_probability_channel():
    with tempfile.TemporaryDirectory() as tmp:
        table = _table_with(tmp, gen_column=False)
    row = table.iloc[0]
    assert row["n_flips"] == 2, row.to_dict()
    assert row["verdict_channel"] == "probability", row.to_dict()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("PASS %s" % fn.__name__)
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, exc))
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            print("ERROR %s: %s: %s" % (fn.__name__, type(exc).__name__, exc))
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
