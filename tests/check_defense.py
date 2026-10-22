"""Tests for the training-free defence probe.

What is checked here, and why each one matters:

  * normalisation actually restores homoglyphs and drops invisible characters,
    idempotently - if it is not idempotent the "defended" number is an artefact
    of running it twice;
  * the threshold is the SMALLEST one inside the clean false-positive budget,
    not the largest - the largest is always "flag nothing", which is what the
    first version of this file produced (a 100% ASR for a filter that had
    simply stopped flagging);
  * a flip needs the ORIGINAL to be malicious under the same rule, so a
    threshold that is too high cannot be credited with the attack's work;
  * summarising separates modes: normalisation must not claim credit for a word
    attack it cannot undo;
  * the Wilson interval is present on every row of the summary.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.run_defense import (choose_threshold, fpr_at,  # noqa: E402
                             normalize_text, summarize, tpr_at)


def test_normalize_restores_homoglyphs():
    attacked = "\u0430ccount \u0435xpired"          # Cyrillic a, Cyrillic e
    assert normalize_text(attacked) == "account expired"
    assert normalize_text("hello") == "hello"


def test_normalize_strips_invisible_and_collapses_space():
    assert normalize_text("cli\u200bck  here\n\nnow") == "click here now"
    assert normalize_text("a\u200db\u2060c") == "abc"


def test_normalize_is_idempotent():
    once = normalize_text("\u0430ccount  \u200bex\u0440ired")
    assert normalize_text(once) == once


def test_normalize_handles_non_string():
    assert normalize_text(None) is None


def test_threshold_holds_clean_fpr():
    clean = [0.05, 0.10, 0.20, 0.24, 0.30, 0.85]
    tau = choose_threshold(clean, 0.05)
    assert fpr_at(clean, tau) <= 0.05 + 1e-12, (tau, fpr_at(clean, tau))
    # and it must not be the degenerate "flag everything" answer
    assert tau >= 0.5


def test_threshold_never_exceeds_the_budget():
    clean = [0.1, 0.2, 0.3, 0.7, 0.8]
    tau = choose_threshold(clean, 0.0)
    assert fpr_at(clean, tau) == 0.0, tau


def test_threshold_is_the_least_conservative_feasible_point():
    """The v4 bug: maximising tau always returns "above every clean score".

    With clean probabilities that are all high (a conservative model), that
    threshold rejects every message, and because the v4 code counted a flip
    whenever the perturbed text fell below tau - without checking the original -
    it reported 100% ASR for the defence.
    """
    # 40 benign messages, 2 of them badly miscalibrated: a 5% budget (= 2 rows)
    # is only just reachable, and only by a threshold that still flags those 2.
    clean = [0.90 + 0.001 * i for i in range(38)] + [0.995, 0.999]
    tau = choose_threshold(clean, 0.05)
    assert tau < max(clean), tau                 # not the reject-all corner
    assert fpr_at(clean, tau) <= 0.05 + 1e-12, tau
    assert tau <= 0.995, tau                     # the feasible point is used
    assert fpr_at(clean, tau) == 0.05, (tau, fpr_at(clean, tau))


def test_flip_requires_the_original_to_be_malicious_under_the_same_rule():
    rows = [{"mode": "char", "task": "sms_smishing", "tau": 0.9,
             "orig_malicious": True, "orig_malicious_norm": True,
             "orig_malicious_tau": False, "orig_malicious_norm_tau": False,
             "flip_baseline": True, "flip_normalised": False,
             "flip_threshold": False, "flip_both": False}]
    table = summarize(pd.DataFrame(rows))
    thr = table[table["defense"] == "threshold"].iloc[0]
    # the original was never malicious at tau=0.9, so nothing can be "flipped"
    assert thr["n"] == 0 and thr["flips"] == 0, thr.to_dict()
    base = table[table["defense"] == "baseline"].iloc[0]
    assert base["n"] == 1 and base["flips"] == 1


def test_summary_counts_each_defence_separately():
    rows = []
    # word attack: normalisation changes nothing, threshold still rescues one
    for i in range(10):
        rows.append({"mode": "word", "task": "sms_smishing", "tau": 0.6,
                     "orig_malicious": True, "orig_malicious_norm": True,
                     "orig_malicious_tau": True, "orig_malicious_norm_tau": True,
                     "flip_baseline": True, "flip_normalised": True,
                     "flip_threshold": i >= 1, "flip_both": i >= 1})
    df = pd.DataFrame(rows)
    table = summarize(df)
    word = table[table["mode"] == "word"].set_index("defense")
    assert word.loc["baseline", "flips"] == 10
    assert word.loc["normalise", "flips"] == 10, "normalisation must not help here"
    assert word.loc["threshold", "flips"] == 9
    assert word.loc["both", "flips"] == 9
    assert word.loc["baseline", "n"] == 10
    for col in ("ci95_low", "ci95_high"):
        assert col in table.columns


def test_char_attack_is_what_normalisation_repairs():
    rows = [{"mode": "char", "task": "sms_smishing", "tau": 0.5,
             "orig_malicious": True, "orig_malicious_norm": True,
             "orig_malicious_tau": True, "orig_malicious_norm_tau": True,
             "flip_baseline": True, "flip_normalised": False,
             "flip_threshold": True, "flip_both": False}] * 5
    table = summarize(pd.DataFrame(rows))
    char = table[table["mode"] == "char"].set_index("defense")
    assert char.loc["normalise", "flips"] == 0
    assert char.loc["baseline", "flips"] == 5


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
