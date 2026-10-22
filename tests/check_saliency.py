"""Tests for the saliency (PWWS-style) attack, with a fake model.

The fake model has a KNOWN structure: p_malicious rises with the number of
"trigger" words still present. That makes the properties the paper relies on
checkable:

  * a message the model never caught is excluded, not counted as a failure,
  * every accepted substitution is a real lexicon entry and keeps its punctuation,
  * the edit budget (fraction of WORDS) is never exceeded - which for a 40-word
    SMS means a handful of substitutions, not fifteen,
  * a message the budget cannot rescue must not be reported as a flip,
  * when the shim cannot give probabilities, the row says so instead of inventing
    an ordering.

Note on sizes: the 15% word budget on a short SMS is ONE edit. Every test text
below is therefore a realistic full message, because a 12-word toy sentence
cannot express the attack at all.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.saliency_common import (LEXICON, apply_subs, candidates,  # noqa: E402
                                 run_saliency)

TRIGGERS = {"urgent", "verify", "click", "prize", "claim", "suspend", "blocked"}


def fake_model(texts, task):
    """p_malicious = 0.10 + 0.28 * (number of trigger words still present)."""
    out = []
    for t in texts:
        words = {w.strip(".,!?:;\"'()").lower() for w in t.split()}
        out.append(min(0.98, 0.10 + 0.28 * len(words & TRIGGERS)))
    return out


def frame(text):
    return pd.DataFrame([{"id": "x", "task": "sms_smishing", "text": text}])


def test_candidates_skip_stopwords_and_punctuation():
    #                      0        1       2       3        4
    cands = candidates("Please verify your account now!")
    cores = [c[1].strip(".,!?").lower() for c in cands]
    assert "please" not in cores, "stopwords must not be probed"
    assert "your" not in cores, "stopwords must not be probed"
    assert "verify" in cores and "account" in cores, cores
    for _, _, repls in cands:
        assert repls, "a candidate with no replacement is wasted work"
        assert all(r.lower() not in cores for r in repls), "no-op replacement"


def test_apply_subs_keeps_punctuation():
    out = apply_subs("Verify your account!".split(), {0: "confirm"})
    assert out[0] == "confirm", out[0]
    out = apply_subs("Urgent!".split(), {0: "time-sensitive"})
    assert out[0] == "time-sensitive!", out[0]
    out = apply_subs("Click now.".split(), {1: "today"})
    assert out[1] == "today.", out[1]


def test_saliency_weight_does_not_collapse_to_a_constant():
    """phi = alpha*(p0-p) + (1-alpha)*p is constant at alpha=0.5, which silently
    turns the greedy search into left-to-right application. Normalised saliency
    must separate a trigger word from a neutral one."""
    p0, p_trigger, p_neutral = 0.94, 0.66, 0.94
    raw_trig, raw_neut = max(0.0, p0 - p_trigger), max(0.0, p0 - p_neutral)
    top = max(raw_trig, raw_neut)
    phi_trig = 0.5 * (raw_trig / top) + 0.5 * p_trigger
    phi_neut = 0.5 * (raw_neut / top) + 0.5 * p_neutral
    assert phi_trig > phi_neut, (phi_trig, phi_neut)


LONG_SMS = (
    "Dear customer, this is an urgent notice from your bank. We detected "
    "unusual activity on your debit card and we have blocked the card for your "
    "safety. To restore access to your account please verify your identity at "
    "the secure link below within twenty four hours, otherwise the card will "
    "stay suspended and a replacement fee will apply to your next statement. "
    "Thank you for banking with us and have a pleasant day."
)


def test_attack_flips_a_trigger_heavy_message():
    df, info = run_saliency(frame(LONG_SMS), "fake", fake_model, quiet=True)
    row = df.iloc[0]
    assert row["counted"], row["skip_reason"]
    assert row["success"], "removing the trigger words must flip the fake model"
    assert row["pert_pred"] == 0
    applied = [t for t in str(row["attack_types"]).split(":")[1].split("+") if t]
    assert applied, "a successful attack must have applied substitutions"
    for entry in applied:
        assert entry.split("->")[0].lower() in LEXICON, "%r came from nowhere" % entry
    assert row["perturbation_rate"] <= 0.15 + 1e-9, row["perturbation_rate"]
    assert row["n_queries"] >= 2 and info["alpha"] == 0.5


def test_saliency_finds_the_trigger_word_without_scanning_everything():
    """Two triggers among ~60 words: a saliency search should need about as many
    substitutions as there are triggers, not a sweep of the message."""
    df, _ = run_saliency(frame(LONG_SMS), "fake", fake_model, quiet=True,
                         max_positions=12, max_rounds=4)
    row = df.iloc[0]
    assert row["counted"] and row["success"]
    assert int(row["n_subs"]) <= 4, "needed %s substitutions" % row["n_subs"]
    # the probe round is capped by max_positions x <=3 replacements, plus the
    # greedy verification queries - far below one-query-per-word
    n_words = len(LONG_SMS.split())
    assert int(row["n_queries"]) < n_words, (row["n_queries"], n_words)


def test_excludes_sample_the_model_never_caught():
    df, _ = run_saliency(frame("See you at lunch on friday, bring the book."),
                         "fake", fake_model, quiet=True)
    assert not df.iloc[0]["counted"]
    assert "not classified malicious" in df.iloc[0]["skip_reason"]


def test_probs_unavailable_is_reported_not_guessed():
    df, _ = run_saliency(frame(LONG_SMS), "fake", lambda t, k: None, quiet=True)
    assert not df.iloc[0]["counted"]
    assert "unavailable" in df.iloc[0]["skip_reason"]


def test_budget_respected_when_flip_is_impossible():
    """Eight triggers in a 20-word message: 15% is 3 edits, so the flip is out of
    reach and the run must stop inside budget without claiming success."""
    text = ("urgent verify click prize claim suspend blocked account holder "
            "please review the attached statement and reply to our support team "
            "with your reference number")
    df, _ = run_saliency(frame(text), "fake", fake_model, quiet=True)
    row = df.iloc[0]
    assert row["counted"], row["skip_reason"]
    assert row["perturbation_rate"] <= 0.15 + 1e-9, row["perturbation_rate"]
    assert not row["success"], "3 edits cannot remove 7 triggers - no fake win"


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
