"""
Meaning-preservation checks for text perturbations.

Why this module exists: the study reports word-set Jaccard and calls it what
it is. That answers "is the message 92% word-identical?", which is a
statement about the EDIT, not about the MEANING. The two come apart in both
directions:

  * a homoglyph substitution collapses Jaccard (the token changes) while a
    human reads exactly the same message - Jaccard says "meaning destroyed",
    the reader says "nothing happened";
  * a synonym taken from the wrong WordNet sense keeps Jaccard at 1.0 (one word
    in, one word out) while changing the proposition - Jaccard says "meaning
    preserved", the reader says "that is a different claim".

A bag-of-words measure over the whole text cannot catch the second case: if
one word of twenty is replaced by an unrelated word, every set-overlap score
stays near 1.0. What catches it is a check on the REPLACEMENT ITSELF - is the
new word a paraphrase of the one it replaced? - and that has to be applied
when the candidate list is built, before the attack is allowed to spend a
query on it.

So this module provides two layers:

  * a candidate-level gate: `is_paraphrase(original_word, replacement)` -
    True when the two words share a WordNet synset, or when the pair is in
    the hand-verified security lexicon. The attack engines use this to decide
    which substitutions are admissible at all.
  * text-level measures for the record: `literal_retention` (fraction of the
    original's content words still literally present),
    `paraphrase_coverage` (fraction matched by a shared synset, whether the
    word was kept or paraphrased) and `semantic_similarity` (the minimum of
    the two, which is the number the acceptance gate uses).

Everything degrades gracefully: with no WordNet corpus the module says so
rather than returning a number that looks like a measurement.

    python src/semantic.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PUNCT and STOPWORDS are reused from the saliency module so the two agree on
# what a "content word" is.
from src.saliency_common import LEXICON, PUNCT, STOPWORDS  # noqa: E402

_WN = None
_WN_TRIED = False


def _wn():
    """The WordNet corpus reader, or None when it is not usable."""
    global _WN, _WN_TRIED
    if not _WN_TRIED:
        _WN_TRIED = True
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("test")
            _WN = wn
        except Exception:
            _WN = None
    return _WN


def wordnet_ready() -> bool:
    """True when the WordNet corpus can be used for the synset checks."""
    return _wn() is not None


def content_words(text) -> list:
    """Lower-cased content words: punctuation stripped, stopwords removed."""
    out = []
    for w in str(text).split():
        core = w.strip(PUNCT).lower()
        if not core or core in STOPWORDS or len(core) < 3 or not core.isalpha():
            continue
        out.append(core)
    return out


def synsets(word):
    """All WordNet synsets for a word (empty list if WordNet is unavailable)."""
    wn = _wn()
    if wn is None:
        return []
    try:
        return wn.synsets(word)
    except Exception:
        return []


def same_synset(a, b) -> bool:
    """True when two words share at least one WordNet synset."""
    sa, sb = synsets(str(a).lower()), synsets(str(b).lower())
    if not sa or not sb:
        return False
    ids_b = {s.name() for s in sb}
    return any(s.name() in ids_b for s in sa)


def is_paraphrase(original, replacement) -> bool:
    """
    Candidate-level admissibility: may `replacement` stand in for `original`?

    True when the two share a WordNet synset, or when the pair is one of the
    hand-verified entries of the security lexicon (those were checked by
    hand, so they are admissible even when WordNet's senses disagree).

    This is the check that keeps an attack from spending its budget on a
    substitution that changes the proposition: WordNet's third sense of a word
    is exactly where synonyms stop being paraphrases.
    """
    a = str(original).strip(PUNCT).lower()
    b = str(replacement).strip(PUNCT).lower()
    if not a or not b or a == b:
        return False
    if b in LEXICON.get(a, []):
        return True
    if a in LEXICON.get(b, []):
        return True
    return same_synset(a, b)


def literal_retention(original, perturbed) -> float:
    """
    Fraction of the original's content words that survive LITERALLY.

    This is the measure that catches deletion: an attack that removes the
    word carrying the urgency has changed the message, and this score says so
    even though the remaining text is perfectly fluent.
    """
    orig = content_words(original)
    if not orig:
        return 1.0
    pert = set(content_words(perturbed))
    return sum(1 for w in orig if w in pert) / len(orig)


def paraphrase_coverage(original, perturbed) -> float:
    """
    Fraction of the original's content words that are still represented, either
    literally or by a synonym.

    Stricter than literal_retention in the paraphrase direction and looser in
    the deletion direction, which is why the gate takes the minimum of the
    two: a deletion must be paid for, and a synonym substitution must not be.
    """
    orig = content_words(original)
    if not orig:
        return 1.0
    pert = content_words(perturbed)
    if not pert:
        return 0.0
    kept = 0
    for w in orig:
        if w in pert or any(same_synset(w, p) for p in pert):
            kept += 1
    return kept / len(orig)


def semantic_similarity(original, perturbed) -> float:
    """
    The single number the acceptance gate uses: the MINIMUM of literal
    retention and paraphrase coverage, not their average.

    The minimum is the conservative choice. An average would let a high
    literal-retention score hide a paraphrase gap (twenty words kept, one
    replaced by something unrelated) and would let a high paraphrase score
    hide a deletion (the urgent word removed, everything else paraphrased).
    """
    return min(literal_retention(original, perturbed),
               paraphrase_coverage(original, perturbed))


def meaning_preserved(original, perturbed, threshold=0.6) -> bool:
    """
    True when the perturbation keeps at least `threshold` of the original's
    content under the conservative measure above.

    The default threshold is a judgement, not a measurement. It is recorded
    with every row so the number can be re-derived at a different threshold
    without re-running the model.
    """
    return semantic_similarity(original, perturbed) >= threshold


def _selftest():
    pairs = [
        ("Your parcel is held at customs. Pay Rs 200 now",
         "Your parcel is held at customs. Pay Rs 200 today", "paraphrase"),
        ("Your parcel is held at customs. Pay Rs 200 now",
         "Your parcel is held at customs. Pay Rs 200 at once", "paraphrase"),
        ("Your parcel is held at customs. Pay Rs 200 now",
         "Your parcel is held at customs. Pay Rs 200 tone", "wrong sense"),
        ("Your parcel is held at customs. Pay Rs 200 now",
         "Your parcel is held at customs.", "deletion"),
    ]
    print("wordnet available: %s" % wordnet_ready())
    print("%-52s %-52s %6s %6s %6s %6s" %
          ("original", "perturbed", "lit", "cov", "gate", "kept"))
    for a, b, kind in pairs:
        lit = literal_retention(a, b)
        cov = paraphrase_coverage(a, b)
        print("%-52s %-52s %6.3f %6.3f %6.3f  %s" %
              (a[:51], b[:51], lit, cov, semantic_similarity(a, b), kind))
    print("\ncandidate-level admissibility:")
    for a, b in [("now", "today"), ("now", "tone"), ("verify", "confirm"),
                 ("account", "profile"), ("click", "tap"), ("click", "germany")]:
        print("  %-10s -> %-10s admissible=%s"
              % (a, b, is_paraphrase(a, b)))


if __name__ == "__main__":
    _selftest()
