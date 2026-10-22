"""
Tests for the meaning-preservation layer (src/semantic.py).

These run on no model at all, so they need no GPU and no network.

The properties that matter for the paper:

  * a replacement that changes the sense of a word is NOT a paraphrase
    ("now" -> "tone" is rejected, "now" -> "today" is accepted),
  * a DELETION is paid for by the gate (literal retention drops),
  * a PARAPHRASE is not paid for (coverage stays high, literal retention is
    the only thing that drops),
  * the gate is the MINIMUM of the two measures, never their average, so a
    single deleted word cannot be hidden by nineteen intact ones,
  * with no WordNet corpus the module says so instead of returning a number
    that looks like a measurement.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import semantic  # noqa: E402
from src.semantic import (is_paraphrase, literal_retention,  # noqa: E402
                          meaning_preserved, paraphrase_coverage,
                          semantic_similarity, wordnet_ready)

ORIGINAL = ("Your parcel is held at customs. Pay Rs 200 now to release the "
            "consignment.")


def test_wordnet_is_available_for_the_checks():
    """The checks are only meaningful with the corpus; the module must say
    plainly when it is missing rather than degrade silently."""
    assert wordnet_ready(), "nltk wordnet corpus is not available"


def test_a_real_synonym_is_admissible():
    assert is_paraphrase("now", "today"), "now/today are the same word"
    assert is_paraphrase("verify", "confirm"), "verify/confirm are the same claim"


def test_a_word_from_the_wrong_sense_is_rejected():
    """WordNet's third sense of a word is where synonyms stop being
    paraphrases. The gate exists to stop the attack from spending its
    budget there."""
    assert not is_paraphrase("now", "tone"), "tone is a different word"
    assert not is_paraphrase("click", "germany"), "nonsense replacement"
    assert not is_paraphrase("urgent", "bank"), "unrelated words"


def test_a_word_is_never_its_own_replacement():
    assert not is_paraphrase("now", "now")
    assert not is_paraphrase("now", "")


def test_deletion_is_paid_for_by_literal_retention():
    deleted = "Your parcel is held at customs."
    lit = literal_retention(ORIGINAL, deleted)
    assert lit < 1.0, lit
    assert lit < 0.9, "a deleted content word must show up: %s" % lit


def test_a_paraphrase_keeps_coverage_but_costs_literal_retention():
    paraphrased = ORIGINAL.replace("now", "today")
    lit = literal_retention(ORIGINAL, paraphrased)
    cov = paraphrase_coverage(ORIGINAL, paraphrased)
    assert cov >= 0.95, "the paraphrase still represents the original: %s" % cov
    assert lit < 1.0, "the word is not literally there any more: %s" % lit
    assert lit > 0.8, "one word out of eleven: %s" % lit


def test_the_gate_is_the_minimum_not_the_average():
    """One deleted word out of eleven must not be hidden by the ten intact
    ones: an average would score 0.95 and pass a 0.9 threshold."""
    deleted = ORIGINAL.replace("now", "").replace("Pay", "Pay")
    deleted = "Your parcel is held at customs. Pay Rs 200 to release the consignment."
    lit = literal_retention(ORIGINAL, deleted)
    cov = paraphrase_coverage(ORIGINAL, deleted)
    gate = semantic_similarity(ORIGINAL, deleted)
    assert gate == min(lit, cov), (gate, lit, cov)
    assert gate == lit, "the deletion is the binding measure here"


def test_an_unrelated_replacement_is_caught_even_when_everything_else_is_kept():
    """A bag-of-words measure over the whole text cannot catch this: one word
    of eleven replaced by an unrelated word leaves every set-overlap score
    near 1.0. The candidate-level gate is what catches it."""
    swapped = ORIGINAL.replace("consignment", "elephant")
    lit = literal_retention(ORIGINAL, swapped)
    assert lit > 0.8, "the literal measure alone would pass this: %s" % lit
    assert not is_paraphrase("consignment", "elephant")


def test_similarity_is_bounded_and_ordered():
    same = semantic_similarity(ORIGINAL, ORIGINAL)
    assert same == 1.0, same
    other = semantic_similarity(ORIGINAL, "Completely different words here now")
    assert 0.0 <= other <= same, (other, same)


def test_the_threshold_is_recorded_not_baked_in():
    assert meaning_preserved(ORIGINAL, ORIGINAL, threshold=0.6)
    assert not meaning_preserved(ORIGINAL, "", threshold=0.6)


def test_the_module_degrades_gradually_without_wordnet(monkey=None):
    """With no corpus the module must return no synsets rather than pretend."""
    saved = semantic._WN
    saved_tried = semantic._WN_TRIED
    try:
        semantic._WN = None
        semantic._WN_TRIED = True
        assert semantic.synsets("now") == []
        assert not semantic.same_synset("now", "today")
        # the hand-verified lexicon still works without WordNet
        assert is_paraphrase("verify", "confirm"), "lexicon fallback"
    finally:
        semantic._WN = saved
        semantic._WN_TRIED = saved_tried


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
