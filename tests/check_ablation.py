"""
Tests for the matched-vocabulary / matched-query-budget ablation engine.

The fake model has a KNOWN structure: p_malicious rises with the number of
"trigger" words still present, and the binary verdict is 1 while p >= 0.5.
That makes the properties the paper relies on checkable:

  * every arm pays the SAME probe cost, which is what makes the query budget
    matched rather than merely similar;
  * the query cap is never exceeded, whatever the arm;
  * ordering actually matters: when a flip needs TWO edits, the saliency
    order finds it on every seed and the random order mostly does not;
  * the verdict-only group search finds a flipping SET in fewer queries than
    the sequential verdict search, which is the honest version of "the binary
    oracle cannot rank";
  * every applied substitution is a paraphrase of the word it replaced;
  * a message the model never caught is excluded, not counted as a failure.

Note on sizes: the 15% word budget on a short SMS is ONE or TWO edits, so the
messages below are realistic full messages, and the ordering test uses a
three-trigger message where a flip needs two edits - a single-edit flip is
found by the probe phase in every arm, so it cannot separate the orders.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.ablation_common import (ARMS, TEXTFOOLER_SEMANTIC_THRESHOLD,  # noqa: E402
                                 candidates_for, run_arm)
from src.semantic import is_paraphrase  # noqa: E402
from src.summarize_ablation import CONTRASTS_SPEC, arm_rows, contrast_rows  # noqa: E402

TRIGGERS = {"urgent", "verify", "click", "prize", "claim", "suspend", "blocked"}

# three triggers: one edit leaves two (still malicious), two edits leave one
# (benign), so a flip needs two edits and the ORDER of the edits matters
TWO_EDIT_MSG = (
    "Urgent: verify your account and message our support team because the "
    "card is blocked today"
)


def _p(text):
    words = {w.strip(".,!?;:\"'()[]{}<>").lower() for w in text.split()}
    return min(0.98, 0.10 + 0.28 * len(words & TRIGGERS))


def fake_probs(texts, task):
    """p_malicious = 0.10 + 0.28 * (number of trigger words still present)."""
    return [_p(t) for t in texts]


def fake_verdicts(texts, task):
    """The binary readout of the same model: 1 while p >= 0.5."""
    return [1.0 if _p(t) >= 0.5 else 0.0 for t in texts]


ARMS_BY_ASK = [
    (fake_probs, "random-prob"),
    (fake_probs, "saliency-prob"),
    (fake_verdicts, "random-verdict"),
    (fake_verdicts, "group-verdict"),
]


def frame(text, task="sms_smishing"):
    return pd.DataFrame([{"id": "x", "task": task, "text": text}])


def _probe_size(text, max_positions=12, max_replacements=2):
    """How many single-substitution candidates the probe phase will ask about."""
    cands = candidates_for(text, vocab="lexicon_wordnet",
                            max_replacements=max_replacements)[:max_positions]
    return sum(len(repls) for (_pos, _orig, repls) in cands)


def test_probe_phase_cost_is_identical_across_arms():
    """The budget is matched by construction: the probe phase asks the same
    questions in every arm, so a difference between arms is not a difference
    in how many questions they were allowed to ask."""
    costs = {}
    for ask, arm in ARMS_BY_ASK:
        batches = []

        def counting(texts, task, _ask=ask, _batches=batches):
            _batches.append(len(texts))
            return _ask(texts, task)

        df, _ = run_arm(frame(TWO_EDIT_MSG), "fake", counting, arm=arm, quiet=True)
        assert df.iloc[0]["counted"], df.iloc[0]["skip_reason"]
        assert batches[0] == 1, "the original is one query in every arm: %s" % batches
        assert len(batches) > 1, "the probe phase must have run: %s" % batches
        costs[arm] = batches[1]
    assert len(set(costs.values())) == 1, "probe cost differs by arm: %s" % costs
    assert costs["random-prob"] == _probe_size(TWO_EDIT_MSG), costs


def test_query_cap_is_never_exceeded():
    """The cap is a TOTAL budget: the probe phase has to fit inside it."""
    for ask, arm in ARMS_BY_ASK:
        df, _ = run_arm(frame(TWO_EDIT_MSG), "fake", ask, arm=arm, quiet=True,
                        query_cap=4)
        row = df.iloc[0]
        assert row["counted"], row["skip_reason"]
        assert int(row["n_queries"]) <= 4, (arm, row["n_queries"])


def test_ordering_matters_when_a_flip_needs_two_edits():
    """With two edits and three trigger words, the saliency order must find
    the flip on every seed and the random order must not find it on all of
    them. This is the mechanism the paper attributes the saliency advantage
    to, tested without a GPU."""
    sal_hits = rand_hits = 0
    seeds = list(range(1, 9))
    for seed in seeds:
        df, _ = run_arm(frame(TWO_EDIT_MSG), "fake", fake_probs,
                        arm="saliency-prob", quiet=True, seed=seed, max_rounds=2)
        sal_hits += int(bool(df.iloc[0]["success"]))
        df, _ = run_arm(frame(TWO_EDIT_MSG), "fake", fake_probs,
                        arm="random-prob", quiet=True, seed=seed, max_rounds=2)
        rand_hits += int(bool(df.iloc[0]["success"]))
    assert sal_hits == len(seeds), \
        "the saliency order failed on %d of %d seeds" % (len(seeds) - sal_hits,
                                                          len(seeds))
    assert rand_hits < len(seeds), \
        "the random order also hit every seed - the test is not discriminating"


def test_group_search_finds_a_flipping_set_in_fewer_queries():
    """The binary oracle cannot rank candidates, so its targeted strategy is a
    search over SETS. It must reach the flip in no more queries than the
    sequential search, which applies one edit at a time."""
    df_g, _ = run_arm(frame(TWO_EDIT_MSG), "fake", fake_verdicts,
                      arm="group-verdict", quiet=True, max_rounds=2)
    df_r, _ = run_arm(frame(TWO_EDIT_MSG), "fake", fake_verdicts,
                      arm="random-verdict", quiet=True, max_rounds=2)
    assert df_g.iloc[0]["success"], "the group search should find the flip"
    assert int(df_g.iloc[0]["n_queries"]) <= int(df_r.iloc[0]["n_queries"]), \
        (df_g.iloc[0]["n_queries"], df_r.iloc[0]["n_queries"])


def test_every_applied_substitution_is_a_paraphrase():
    for ask, arm in ARMS_BY_ASK:
        df, _ = run_arm(frame(TWO_EDIT_MSG), "fake", ask, arm=arm, quiet=True,
                        max_rounds=2)
        row = df.iloc[0]
        assert row["counted"], row["skip_reason"]
        for entry in [e for e in str(row["attack_types"]).split("+") if "->" in e]:
            a, b = entry.split("->")
            assert is_paraphrase(a, b), "%r is not a paraphrase" % entry


def test_meaning_measures_are_recorded_for_every_row():
    df, _ = run_arm(frame(TWO_EDIT_MSG), "fake", fake_probs, arm="saliency-prob",
                    quiet=True)
    row = df.iloc[0]
    for col in ("literal_retention", "paraphrase_coverage", "semantic_similarity"):
        assert col in row.index, col
        assert 0.0 <= float(row[col]) <= 1.0, (col, row[col])


def test_excludes_a_message_the_model_never_caught():
    benign = "See you at lunch on friday, bring the book."
    for ask, arm in ARMS_BY_ASK:
        df, _ = run_arm(frame(benign), "fake", ask, arm=arm, quiet=True)
        assert not df.iloc[0]["counted"]
        assert "not classified malicious" in df.iloc[0]["skip_reason"]


def test_the_two_vocabularies_are_different_candidate_sets():
    """The candidate-set effect is only measurable if the two vocabularies
    really differ."""
    big = candidates_for(TWO_EDIT_MSG, vocab="lexicon_wordnet")
    small = candidates_for(TWO_EDIT_MSG, vocab="small_table")
    n_big = sum(len(r) for (_p, _o, r) in big)
    n_small = sum(len(r) for (_p, _o, r) in small)
    assert n_big > n_small, (n_big, n_small)


def test_arm_table_is_complete():
    assert set(ARMS) == {"random-prob", "saliency-prob", "random-verdict",
                          "group-verdict", "textfooler-prob"}, sorted(ARMS)
    for name, spec in ARMS.items():
        assert spec["oracle"] in ("probability", "verdict"), name
        assert spec["strategy"] in ("rank", "group_test"), name


def test_the_textfooler_arm_turns_the_meaning_check_on():
    """TextFooler is defined with a semantic-similarity constraint. Without
    one it is not TextFooler, so the arm must carry the gate by default and
    record the threshold it used."""
    df, info = run_arm(frame(TWO_EDIT_MSG), "fake", fake_probs,
                       arm="textfooler-prob", quiet=True)
    assert info["semantic_threshold"] == TEXTFOOLER_SEMANTIC_THRESHOLD, info
    assert float(df.iloc[0]["semantic_threshold"]) == \
        TEXTFOOLER_SEMANTIC_THRESHOLD, df.iloc[0]["semantic_threshold"]


def test_the_textfooler_arm_finds_the_flip_and_respects_the_cap():
    df, _ = run_arm(frame(TWO_EDIT_MSG), "fake", fake_probs,
                    arm="textfooler-prob", quiet=True, max_rounds=2)
    row = df.iloc[0]
    assert row["counted"], row["skip_reason"]
    assert row["success"], "the importance round should find the trigger words"
    for entry in [e for e in str(row["attack_types"]).split("+") if "->" in e]:
        a, b = entry.split("->")
        assert is_paraphrase(a, b), "%r is not a paraphrase" % entry

    df, _ = run_arm(frame(TWO_EDIT_MSG), "fake", fake_probs,
                    arm="textfooler-prob", quiet=True, max_rounds=2,
                    query_cap=6)
    assert int(df.iloc[0]["n_queries"]) <= 6, df.iloc[0]["n_queries"]


def _fake_key(a_flip_ids, b_flip_ids, model="fake", task="sms_smishing"):
    """per_key in the shape summarize_ablation.load_arms() returns.

    Five messages; `a_flip_ids` flipped in arm A, `b_flip_ids` in arm B, and
    the rest in neither. The discordant counts are then known by construction,
    which is the only way to check that the summariser pairs the right rows.
    """
    ids = ["m1", "m2", "m3", "m4", "m5"]
    A, B = {}, {}
    for i in ids:
        A[(task, i)] = i in a_flip_ids
        B[(task, i)] = i in b_flip_ids
    spec = {s[0]: s for s in CONTRASTS_SPEC}
    _, a_arm, a_vocab, b_arm, b_vocab, _what = spec["ordering"]
    return {(a_arm, a_vocab, model): A, (b_arm, b_vocab, model): B}


def test_the_ordering_contrast_pairs_on_the_shared_messages():
    """A-only 2, B-only 1 must come out of the records, not out of a formula.

    This is the check that keeps the ablation's headline number honest: the
    contrast may only count messages BOTH arms attacked, because a message one
    arm skipped is not a message the other arm won.
    """
    key = _fake_key({"m1", "m2", "m3", "m4"}, {"m2", "m3", "m5"})
    rows = contrast_rows(key)
    ordering = [r for r in rows if r["contrast"] == "ordering"]
    assert len(ordering) == 1, rows
    r = ordering[0]
    # m2, m3 flipped in both; m1, m4 only in A; m5 only in B
    assert (r["only_a"], r["only_b"]) == (2, 1), r
    assert r["a_flips"] == 4 and r["b_flips"] == 3, r
    assert r["n_paired"] == 5, r
    assert r["risk_diff"] == 0.2, r


def test_a_missing_arm_is_reported_as_missing_not_as_zero():
    """Half an ablation is not a result: the contrast must be absent, not 0.0."""
    key = _fake_key({"m1"}, set())
    key = {k: v for k, v in key.items() if k[0] != "random-prob"}
    rows = contrast_rows(key)
    assert not rows, rows


def test_the_arm_table_carries_an_interval_at_zero_flips():
    """A 0-flip arm still needs its upper bound, or the ablation repeats the
    defect the rest of the paper exists to fix."""
    df = pd.DataFrame([
        {"model": "fake", "arm": "random-prob", "vocab": "lexicon_wordnet",
         "task": "sms_smishing", "counted": True, "success": False,
         "n_queries": 9, "perturbation_rate": 0.1, "semantic_similarity": 0.9},
        {"model": "fake", "arm": "random-prob", "vocab": "lexicon_wordnet",
         "task": "sms_smishing", "counted": True, "success": False,
         "n_queries": 11, "perturbation_rate": 0.1, "semantic_similarity": 0.9},
    ])
    rows = arm_rows(df)
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["n_flips"] == 0 and r["asr"] == 0.0, r
    assert r["ci95_high"] > 0.0, r
    assert r["mean_queries"] == 10.0, r


def test_counted_survives_the_csv_round_trip():
    """A skipped sample must not erase the whole arm from the table.

    Written to a CSV, a `counted` column with one empty cell comes back as the
    strings 'True'/'False'. Reading it with `== True` then drops every row, the
    arm vanishes from the table, and its contrast is reported as missing data --
    exactly what happened to the candidate-set arm on its first run.
    """
    import tempfile
    frame = pd.DataFrame([
        {"id": "m1", "task": "sms_smishing", "counted": True, "success": True,
         "n_queries": 9, "perturbation_rate": 0.1, "semantic_similarity": 0.9},
        {"id": "m2", "task": "sms_smishing", "counted": float("nan"),
         "success": False, "n_queries": 3, "perturbation_rate": 0.0,
         "semantic_similarity": 1.0},
    ])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ablation_random-verdict_fake_small_table.csv")
        frame.to_csv(path, index=False)
        back = pd.read_csv(path)
        assert back["counted"].dtype != bool, "the fixture no longer reproduces it"
        rows = arm_rows(back.assign(model="fake", arm="random-verdict",
                                    vocab="small_table"))
    assert len(rows) == 1, rows
    assert rows[0]["n_attacked"] == 1 and rows[0]["n_flips"] == 1, rows


def test_the_small_table_arm_is_not_merged_into_the_lexicon_arm():
    """`small_table` contains the filename separator, so a split-based parse
    turns it into 'small', falls back to the lexicon, and folds the small-table
    rows into the arm it is contrasted with -- which both hides the contrast and
    inflates the lexicon denominator.
    """
    from src.summarize_ablation import parse_name
    assert parse_name("ablation_random-verdict_qwen2.5-3b.csv") == \
        ("random-verdict", "lexicon_wordnet", "qwen2.5-3b")
    assert parse_name("ablation_random-verdict_qwen2.5-3b_small_table.csv") == \
        ("random-verdict", "small_table", "qwen2.5-3b")
    assert parse_name("ablation_saliency-prob_smollm2-1.7b.csv") == \
        ("saliency-prob", "lexicon_wordnet", "smollm2-1.7b")


def test_the_candidate_set_contrast_sees_both_vocabularies():
    """End to end from two files: the candidate-set contrast must exist.

    This is the check that would have caught the parse bug on the run that first
    produced the small-table arm, where the contrast was reported as missing
    data while its rows were folded into the lexicon arm.
    """
    import tempfile

    import src.summarize_ablation as S

    def rows(flipped):
        return [{"id": m, "task": "sms_smishing", "counted": True,
                 "success": m in flipped, "n_queries": 9,
                 "perturbation_rate": 0.1, "semantic_similarity": 0.9}
                for m in ("m1", "m2", "m3")]

    old = S.RESULTS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        pd.DataFrame(rows({"m1", "m2"})).to_csv(
            os.path.join(tmp, "ablation_random-verdict_fake.csv"), index=False)
        pd.DataFrame(rows({"m2"})).to_csv(
            os.path.join(tmp, "ablation_random-verdict_fake_small_table.csv"),
            index=False)
        S.RESULTS_DIR = tmp
        try:
            per_key, _df = S.load_arms()
            got = S.contrast_rows(per_key)
        finally:
            S.RESULTS_DIR = old
    cand = [r for r in got if r["contrast"] == "candidates"]
    assert len(cand) == 1, got
    r = cand[0]
    assert r["n_paired"] == 3, r
    assert (r["a_flips"], r["b_flips"]) == (2, 1), r
    assert (r["only_a"], r["only_b"]) == (1, 0), r


def test_a_duplicated_sample_is_counted_once():
    """One message is one message: two records must not double the denominator."""
    row = {"model": "fake", "arm": "random-verdict", "vocab": "lexicon_wordnet",
           "task": "sms_smishing", "counted": True, "success": False,
           "n_queries": 9, "perturbation_rate": 0.1, "semantic_similarity": 0.9}
    df = pd.DataFrame([dict(row, id="m1"), dict(row, id="m1", success=True),
                       dict(row, id="m2"), dict(row, id="m2")])
    rows = arm_rows(df)
    assert len(rows) == 1, rows
    # m1 flipped in one of its two records, m2 in neither
    assert rows[0]["n_attacked"] == 2, rows
    assert rows[0]["n_flips"] == 1, rows
    assert rows[0]["asr"] == 0.5, rows


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
