"""
The controlled word-substitution engine behind the ablation.

The paper's central comparison is "a random word draw flips 2.7% of caught
messages, the saliency-ordered word attack flips 4.6%". That comparison mixes
three different things and the paper says so only once, in a limitations
paragraph:

  1. ORDERING      - which word is changed: a uniform draw, or the word the
                     model is most sensitive to.
  2. CANDIDATE SET - what it is changed TO: a 14-entry hand-written table, or
                     the curated security lexicon plus WordNet.
  3. ORACLE        - what the attacker can measure: the binary verdict, or
                     P(malicious) for a continuous ranking.

The existing runs hold 1 and 3 fixed and vary 2 as well, so the reported
difference is an upper bound on the ordering effect and says nothing about the
oracle effect. This module is the engine that separates them.

THE DESIGN: one candidate set, one query budget, four arms.

Every arm runs the same probe phase on the same messages:

    probe   query the ORIGINAL once, then query every single-substitution
            candidate once. |candidates| is capped identically for every arm,
            so this phase costs 1 + |candidates| calls in EVERY arm. The
            measurements are kept; what the arm does with them is what differs.

Then the arms differ in what they do next:

    rank        sort the candidates by a rule and substitute them in that
                order, re-querying after every accepted edit, stopping at the
                first flip, at the shared 15% word budget, or at the query cap.
    group_test  a binary search over SUBSETS of the candidates: apply half,
                query, and keep the half that flipped (or the half that did
                not, if neither did). This is the strongest search a
                one-bit oracle can do.

Because the probe phase is mandatory and identical, the QUERY BUDGET is
matched by construction rather than by an assumption: a random-order arm pays
exactly the same 1 + |candidates| calls as the saliency arm before either of
them starts editing. The realised cost is still recorded per row, because a
cap that is matched on paper and exhausted early in practice would be the
same kind of error as an unmatched denominator.

The arms:

    random-prob     rank/random,     probability oracle
    saliency-prob   rank/saliency,   probability oracle   (the existing attack)
    random-verdict  rank/random,     verdict oracle
    group-verdict   group_test,      verdict oracle
    textfooler-prob rank/textfooler, probability oracle   (a stronger,
                   established attack - see _textfooler_apply)

Reading the contrasts:

    random-prob   vs saliency-prob  -> the ORDERING effect, probability oracle
    random-verdict vs group-verdict -> the SEARCH effect, verdict oracle
    random-prob   vs random-verdict -> the ORACLE effect, random order
    saliency-prob vs group-verdict  -> the ORACLE effect, targeted order

and, against the already-recorded runs:

    random-prob   vs the random word attack of results/attack_results_word_*.csv
                  -> the CANDIDATE-SET effect, because that attack is the same
                     random-order, verdict-only attack with 14 replacements
                     instead of the lexicon+WordNet set.

The textfooler arm is NOT part of the factorial: its leave-one-out importance
round costs an extra query per candidate position, so it is matched to the
others by the runner's --query-cap rather than by construction, and the
realised query count is reported for every arm.

A NOTE ON WHAT A BINARY ORACLE CANNOT DO: PWWS's ordering needs a scalar per
candidate. A binary verdict has one bit, and that bit is exhausted by the
first flip - so a verdict-only arm cannot rank candidates by saliency at all.
The verdict arm's targeted rule is therefore a SEARCH, not a ranking, and the
oracle contrast is reported as "how much of the advantage survives when the
attacker only has the bit" rather than as a like-for-like ranking comparison.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.attack_common import SYNONYMS, edit_counts, jaccard  # noqa: E402
from src.saliency_common import (LEXICON, PUNCT, STOPWORDS,  # noqa: E402
                                 _wordnet_synonyms)
from src.semantic import (is_paraphrase, literal_retention,  # noqa: E402
                          paraphrase_coverage, semantic_similarity)

STRATEGIES = ("rank", "group_test")
ORDERS = ("random", "saliency", "textfooler")
ORACLES = ("probability", "verdict")
VOCABS = ("lexicon_wordnet", "small_table")

# TextFooler REQUIRES a meaning-preservation constraint on the replacement
# (it uses a USE sentence-embedding similarity). This repo ships no USE model,
# so the arm uses the WordNet-based gate of src.semantic instead, at this
# threshold. It is a judgement, not a measurement, so it is recorded in every
# row of the run and can be re-derived at another threshold.
TEXTFOOLER_SEMANTIC_THRESHOLD = 0.6

# The placeholder TextFooler's released code substitutes for a word when it
# measures the word's importance (leave-one-out). The paper says "delete the
# word"; the code says "replace it with <oov>". The code is what actually
# runs, so the code's token is the one used here.
TEXTFOOLER_PLACEHOLDER = "<oov>"

# The arm table, so the runner and the paper cannot drift apart.
ARMS = {
    "random-prob":     {"strategy": "rank", "order": "random",
                        "oracle": "probability"},
    "saliency-prob":   {"strategy": "rank", "order": "saliency",
                        "oracle": "probability"},
    "random-verdict":  {"strategy": "rank", "order": "random",
                        "oracle": "verdict"},
    "group-verdict":   {"strategy": "group_test", "order": "random",
                        "oracle": "verdict"},
    "textfooler-prob": {"strategy": "rank", "order": "textfooler",
                        "oracle": "probability"},
}


# ──────────────────────── candidates ────────────────────────

def candidates_for(text, vocab="lexicon_wordnet", max_replacements=2):
    """
    [(position, original_word, [replacement, ...]), ...] for one text.

    Two vocabularies, and that difference IS one of the things the ablation
    exists to measure:

      lexicon_wordnet  the curated security lexicon plus WordNet synonyms
                      (the saliency attack's candidate set)
      small_table     the 14-entry hand-written table of the random word
                      attack

    Every candidate passes the paraphrase gate (src.semantic.is_paraphrase).
    That gate is what stops the attack from spending a query on a replacement
    that changes the proposition: WordNet's third sense of a word is exactly
    where synonyms stop being paraphrases, and a budget spent there buys a
    change of the message, not an evasion of the detector.
    """
    words = text.split()
    present = {w.strip(PUNCT).lower() for w in words}
    out = []
    for i, w in enumerate(words):
        core = w.strip(PUNCT)
        low = core.lower()
        if not core or low in STOPWORDS or len(low) < 3 or not low.isalpha():
            continue
        if vocab == "small_table":
            repl = list(SYNONYMS.get(low, []))
        else:
            repl = list(LEXICON.get(low, []))
            for cand in _wordnet_synonyms(low):
                if cand not in repl:
                    repl.append(cand)
        repl = [r for r in repl if " " not in r and r.lower() not in present]
        repl = [r for r in repl if is_paraphrase(core, r)][:max_replacements]
        if repl:
            out.append((i, w, repl))
    return out


def apply_subs(words, subs):
    """
    Apply {position: replacement} to a word list, keeping punctuation.

    Returns the word LIST, like src.saliency_common.apply_subs, so the two
    modules cannot disagree about what a substitution does to a message.
    """
    out = list(words)
    for i, repl in subs.items():
        original = out[i]
        lead = original[:len(original) - len(original.lstrip(PUNCT))]
        trail = original[len(original.rstrip(PUNCT)):]
        out[i] = lead + repl + trail
    return out


# ──────────────────────── the run ────────────────────────

def run_arm(samples, model, ask, *, arm="random-prob", vocab="lexicon_wordnet",
            seed=42, max_positions=12, max_replacements=2, max_rate=0.15,
            max_rounds=3, query_cap=30, alpha=0.5, semantic_threshold=None,
            placeholder=TEXTFOOLER_PLACEHOLDER, quiet=False):
    """
    Run one arm of the ablation.

    ask(texts, task) -> list of measurements, one per text: a probability in
    [0,1] for the probability oracle, a 0/1 verdict for the verdict oracle.
    It may return None for a batch the endpoint could not serve; the sample
    is then recorded as unusable rather than counted as a failure.

    Returns (per-sample DataFrame, info dict).
    """
    if arm not in ARMS:
        raise ValueError("arm must be one of %s" % (sorted(ARMS),))
    spec = ARMS[arm]
    strategy, order, oracle = spec["strategy"], spec["order"], spec["oracle"]
    if order == "textfooler" and semantic_threshold is None:
        # TextFooler is defined with a meaning constraint; without one it is
        # not TextFooler, so the arm turns the gate on by default and the
        # threshold it used travels with every row.
        semantic_threshold = TEXTFOOLER_SEMANTIC_THRESHOLD

    rows = []
    total_queries = 0

    for idx, sample in enumerate(samples.itertuples(index=False)):
        text = str(sample.text)
        task = getattr(sample, "task", "sms_smishing") or "sms_smishing"
        words = text.split()
        rng = random.Random(seed * 100003 + idx)
        counter = {"n": 0}

        def query(texts, task=task):
            """One batched call. Counts every message the endpoint is asked about."""
            if not texts:
                return []
            out = ask(texts, task)
            if out is None:
                return None
            counter["n"] += len(texts)
            return out

        base = query([text])
        if base is None:
            rows.append(_skip(sample.id, model, task, "endpoint unavailable"))
            continue
        p0 = float(base[0])
        malicious = (p0 >= 0.5) if oracle == "probability" else (p0 == 1)
        if not malicious:
            rows.append(_skip(sample.id, model, task,
                              "original not classified malicious"))
            continue

        # ── probe phase: identical cost in every arm ─────────────────
        cands = candidates_for(text, vocab=vocab,
                               max_replacements=max_replacements)[:max_positions]
        probe_texts, probe_meta = [], []
        for (pos, orig, repls) in cands:
            for repl in repls:
                probe_texts.append(" ".join(apply_subs(words, {pos: repl})))
                probe_meta.append((pos, orig, repl))
        # the cap is a TOTAL budget, not an apply-phase budget: the probe has
        # to fit inside it, and it is truncated in exactly the same way in
        # every arm, so the arms stay matched even when the cap binds
        room = max(0, query_cap - counter["n"])
        if len(probe_texts) > room:
            probe_texts = probe_texts[:room]
            probe_meta = probe_meta[:room]
        probe = query(probe_texts) if probe_texts else []
        if probe is None:
            rows.append(_skip(sample.id, model, task, "probe batch unavailable"))
            continue
        probe = [float(v) for v in probe]

        # a candidate that already flips the message is a success on its own
        flips = [(v < 0.5) if oracle == "probability" else (v == 0)
                 for v in probe]
        if any(flips):
            j = flips.index(True)
            pos, orig, repl = probe_meta[j]
            pert = " ".join(apply_subs(words, {pos: repl}))
            rows.append(_row(sample.id, model, task, text, pert,
                              "%s->%s" % (orig.strip(PUNCT), repl), True,
                              counter["n"], 1, semantic_threshold))
            total_queries += counter["n"]
            continue

        if strategy == "rank" and order == "textfooler":
            pert, used, success, n_sub = _textfooler_apply(
                words, text, probe_meta, probe, p0, oracle, rng,
                budget=max(1, int(len(words) * max_rate)), max_rounds=max_rounds,
                query_cap=query_cap, counter=counter, query=query,
                semantic_threshold=semantic_threshold, placeholder=placeholder)
        elif strategy == "rank":
            pert, used, success, n_sub = _rank_apply(
                words, text, probe_meta, probe, p0, order, oracle, rng, alpha,
                budget=max(1, int(len(words) * max_rate)), max_rounds=max_rounds,
                query_cap=query_cap, counter=counter, query=query,
                semantic_threshold=semantic_threshold)
        else:
            pert, used, success, n_sub = _group_test(
                words, text, probe_meta, oracle, budget=max(1, int(len(words) * max_rate)),
                max_rounds=max_rounds, query_cap=query_cap, counter=counter,
                query=query, semantic_threshold=semantic_threshold)

        rows.append(_row(sample.id, model, task, text, pert, "+".join(used[:6]),
                          success, counter["n"], n_sub, semantic_threshold))
        total_queries += counter["n"]
        if not quiet and (idx + 1) % 25 == 0:
            print("  %d/%d (%s)" % (idx + 1, len(samples), arm), flush=True)

    df = pd.DataFrame(rows)
    info = {"arm": arm, "strategy": strategy, "order": order, "oracle": oracle,
            "vocab": vocab, "seed": seed, "max_positions": max_positions,
            "max_replacements": max_replacements, "max_rate": max_rate,
            "max_rounds": max_rounds, "query_cap": query_cap, "alpha": alpha,
            "semantic_threshold": semantic_threshold,
            "placeholder": placeholder if order == "textfooler" else None,
            "n_queries": total_queries}
    return df, info


def _rank_apply(words, text, probe_meta, probe, p0, order, oracle, rng, alpha,
                budget, max_rounds, query_cap, counter, query,
                semantic_threshold):
    """Sort the candidates by the arm's rule, then substitute them in order."""
    if order == "random":
        items = list(zip(probe_meta, probe))
        rng.shuffle(items)
        ranked = [(pos, orig, repl) for (pos, orig, repl), _v in items]
    else:
        raw = [(max(0.0, p0 - v), pos, orig, repl, v)
               for (pos, orig, repl), v in zip(probe_meta, probe)]
        top = max([r[0] for r in raw], default=0.0)
        scored = []
        for sal, pos, orig, repl, v in raw:
            s_norm = (sal / top) if top > 0 else 0.0
            scored.append((alpha * s_norm + (1 - alpha) * v, pos, orig, repl))
        scored.sort(key=lambda r: -r[0])
        ranked = [(pos, orig, repl) for _s, pos, orig, repl in scored]

    applied, used, success = {}, [], False
    for (pos, orig, repl) in ranked:
        if pos in applied or len(applied) >= budget or len(applied) >= max_rounds:
            continue
        if counter["n"] >= query_cap:
            break
        trial = dict(applied)
        trial[pos] = repl
        cand_text = " ".join(apply_subs(words, trial))
        if cand_text == text:
            continue
        if semantic_threshold is not None and \
                semantic_similarity(text, cand_text) < semantic_threshold:
            continue
        got = query([cand_text])
        if got is None:
            continue
        applied = trial
        used.append("%s->%s" % (orig.strip(PUNCT), repl))
        v = float(got[0])
        if (v < 0.5) if oracle == "probability" else (v == 0):
            success = True
            break
    pert = " ".join(apply_subs(words, applied)) if applied else text
    return pert, used, success, len(applied)


def _textfooler_apply(words, text, probe_meta, probe, p0, oracle, rng, budget,
                      max_rounds, query_cap, counter, query,
                      semantic_threshold, placeholder=TEXTFOOLER_PLACEHOLDER):
    """
    TextFooler (Jin et al., AAAI 2020), adapted to a black-box probability
    endpoint and to this repo's meaning check.

    Faithful to the published method in the three points that matter:

      1. WORD IMPORTANCE IS LEAVE-ONE-OUT. Each candidate word is replaced by
         a placeholder (the authors' released code uses '<oov>') and the
         importance is the drop in P(malicious). When the leave-one-out text
         has already flipped the prediction, the drop is DOUBLED - that is
         the correction in Eq. 2 of the paper, and it is what stops a word
         that the model has already lost from looking unimportant.
      2. THE GREEDY LOOP replaces words in descending importance, and for
         each word it predicts ALL of that word's candidates in ONE batch.
      3. ACCEPTANCE: a candidate that flips the prediction AND passes the
         meaning check wins; if none of them flips, the valid candidate that
         lowers the malicious confidence most is kept and the loop moves on
         to the next word.

    Deviations, all of them stated in the paper: the meaning check is
    WordNet-based (this repo ships no USE model), and the substitute
    vocabulary is the security lexicon rather than a GloVe neighbourhood.

    The importance round costs one extra query per candidate position, so
    this arm is NOT query-matched to the four factorial arms by construction -
    it is matched by the runner's --query-cap, and the realised query count
    is reported for every arm so the comparison can be made honestly.
    """
    per_pos = {}
    for (pos, orig, repl) in probe_meta:
        # probe_meta holds ONE replacement per entry, not a list of them
        entry = per_pos.setdefault(pos, [orig, []])
        if repl not in entry[1]:
            entry[1].append(repl)
    if not per_pos:
        return text, [], False, 0

    # ── 1. leave-one-out importance ──
    importance = {}
    for pos in sorted(per_pos):
        if counter["n"] >= query_cap:
            break
        cand_text = " ".join(apply_subs(words, {pos: placeholder}))
        if cand_text == text:
            continue
        got = query([cand_text])
        if got is None:
            continue
        p_i = float(got[0])
        drop = p0 - p_i
        flipped = (p_i < 0.5) if oracle == "probability" else (p_i == 0)
        if flipped:
            drop = 2.0 * drop
        importance[pos] = drop
    if not importance:
        return text, [], False, 0

    # ── 2. greedy loop over the words, in descending importance ──
    applied, used, success = {}, [], False
    for pos in sorted(importance, key=lambda p: -importance[p]):
        if pos in applied or len(applied) >= budget or len(applied) >= max_rounds:
            continue
        if counter["n"] >= query_cap:
            break
        orig, repls = per_pos[pos]
        trials = []
        for repl in repls:
            trial = dict(applied)
            trial[pos] = repl
            cand_text = " ".join(apply_subs(words, trial))
            if cand_text == text:
                continue
            if semantic_threshold is not None and \
                    semantic_similarity(text, cand_text) < semantic_threshold:
                continue
            trials.append((repl, cand_text))
        if not trials:
            continue
        room = max(0, query_cap - counter["n"])
        if len(trials) > room:
            trials = trials[:room]
            if not trials:
                break
        got = query([t for _r, t in trials])
        if got is None:
            continue
        scored = []
        for (repl, cand_text), v in zip(trials, got):
            v = float(v)
            flipped = (v < 0.5) if oracle == "probability" else (v == 0)
            scored.append((repl, cand_text, v, flipped,
                           semantic_similarity(text, cand_text)))
        winners = [s for s in scored if s[3]]
        if winners:
            # the winning candidate with the highest meaning similarity
            best = max(winners, key=lambda s: s[4])
            applied[pos] = best[0]
            used.append("%s->%s" % (orig.strip(PUNCT), best[0]))
            success = True
            break
        # no flip: keep the candidate that lowers the malicious confidence most
        # (TextFooler only keeps it if the confidence actually dropped)
        best = min(scored, key=lambda s: s[2])
        if best[2] < p0:
            applied[pos] = best[0]
            used.append("%s->%s" % (orig.strip(PUNCT), best[0]))
    pert = " ".join(apply_subs(words, applied)) if applied else text
    return pert, used, success, len(applied)


def _group_test(words, text, probe_meta, oracle, budget, max_rounds,
                query_cap, counter, query, semantic_threshold):
    """
    Binary search over SUBSETS of the candidates - the strongest search a
    one-bit oracle can do.

    Apply the first half of the candidate set, query once. If the verdict
    flipped, the useful candidates are in that half and the search recurses
    into it; if it did not, try the other half. A set that flips is found in
    O(log n) queries per level, which is what makes a verdict-only attacker
    competitive with a probability-guided one on the same probe budget.
    """
    # one candidate per position, in probe order
    seen, picks = set(), []
    for (pos, orig, repl) in probe_meta:
        if pos in seen:
            continue
        seen.add(pos)
        picks.append((pos, orig, repl))
    if not picks:
        return text, [], False, 0

    def flips_with(subset):
        """Apply the subset, query once, and report the verdict."""
        if counter["n"] >= query_cap:
            return None, None
        trial = {pos: repl for (pos, _o, repl) in subset}
        cand_text = " ".join(apply_subs(words, trial))
        if cand_text == text:
            return False, cand_text
        if semantic_threshold is not None and \
                semantic_similarity(text, cand_text) < semantic_threshold:
            return False, cand_text
        got = query([cand_text])
        if got is None:
            return None, cand_text
        v = float(got[0])
        return (v < 0.5) if oracle == "probability" else (v == 0), cand_text

    subset = picks[:budget]
    found, cand_text = flips_with(subset)
    if found is None:
        return text, [], False, 0
    if found:
        used = ["%s->%s" % (o.strip(PUNCT), r) for (_p, o, r) in subset]
        return cand_text, used, True, len(subset)

    # no flip with everything: bisect towards the smallest flipping set
    lo, hi = 0, len(picks)
    best_text, best_set = text, None
    while hi - lo > 1 and counter["n"] < query_cap:
        mid = (lo + hi) // 2
        trial = picks[lo:mid]
        found, cand_text = flips_with(trial)
        if found is None:
            break
        if found:
            best_text, best_set = cand_text, trial
            hi = mid
        else:
            lo = mid
            trial = picks[lo:hi]
            found, cand_text = flips_with(trial)
            if found is None:
                break
            if found:
                best_text, best_set = cand_text, trial
    if best_set is None:
        return text, [], False, 0
    used = ["%s->%s" % (o.strip(PUNCT), r) for (_p, o, r) in best_set]
    return best_text, used, True, len(best_set)


def _skip(sid, model, task, reason):
    return {"id": sid, "model": model, "mode": "ablation", "task": task,
            "counted": False, "skip_reason": reason, "success": False,
            "orig_pred": None, "pert_pred": None}


def _row(sid, model, task, text, pert, attack_types, success, n_queries,
         n_subs, semantic_threshold):
    """One per-sample record, in the schema the other attacks use."""
    stats = edit_counts(text, pert, level="word")
    return {
        "id": sid, "model": model, "mode": "ablation", "task": task,
        "orig_pred": 1, "pert_pred": 0 if success else 1,
        "success": success, "counted": True, "skip_reason": "",
        "attempt": 1, "perturbed_text": pert,
        "attack_types": attack_types or "",
        "similarity": round(jaccard(text, pert), 4),
        "edits": stats["edits"], "perturbation_rate": stats["rate"],
        "n_queries": n_queries, "n_subs": n_subs,
        "literal_retention": round(literal_retention(text, pert), 4),
        "paraphrase_coverage": round(paraphrase_coverage(text, pert), 4),
        "semantic_similarity": round(semantic_similarity(text, pert), 4),
        "semantic_threshold": semantic_threshold,
    }
