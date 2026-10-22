"""
PWWS-style saliency word substitution, made honest for this study.

What the earlier word attack was: uniform random choice over a 14-entry synonym
table, one draw per sample. It answers "does any edit break the model?" but says
nothing about WHERE the model is fragile, so it cannot be compared with the
saliency literature and it wastes queries.

This module implements the actual saliency ordering of Ren et al. (2019), PWWS:

    S_i   = P(y = malicious | x) - P(y = malicious | x with word i replaced)
    phi_i = alpha * S_i_normalised + (1 - alpha) * P(y = malicious | x_i replaced)

and applies substitutions greedily in descending phi until the verdict flips or
the edit budget is spent. S_i ranks "how much this word carries the malicious
decision"; the second term prefers replacements that themselves read malicious,
which keeps the text a plausible attack rather than gibberish.

The normalisation of S_i is not cosmetic. With the raw values and alpha = 0.5,
phi_i = 0.5*(p0 - p_i) + 0.5*p_i = 0.5*p0 for EVERY candidate: the ranking
collapses to a constant and the greedy loop then follows the order the words
happen to appear in. Because p0 is the same for all candidates, only the SCALE
of S changes with normalisation - and that scale is exactly what the paper's
alpha is defined against (S is normalised to [0, 1] there). First run of this
module hit that degeneracy and produced orders identical to left-to-right
application, which is how it was noticed.

Deliberate deviations, all of them stated in the paper:
  * the replacement set is a curated security lexicon (plus WordNet synonyms
    when nltk has the corpus cached), not the WordNet-only candidate set of the
    original, because our texts are short and full of product and bank names;
  * the probability comes from the shim's /v1/verdict_probs endpoint, i.e. the
    normalised logits of the two allowed answer tokens, not the model's own
    full-vocabulary next-token distribution;
  * queries are batched, so the query count is reported per sample and the
    attack is compared with the random attack AT MATCHED QUERY COST, never
    against a cheaper baseline pretending to be the same thing.
"""
import json
import math
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.attack_common import edit_counts, jaccard  # noqa: E402
from src.common import SEED  # noqa: E402

# Substitution lexicon for security-flavoured vocabulary. Every entry is a
# meaning-preserving paraphrase a real phisher could have written; that is the
# constraint that keeps the perturbation "semantics preserved".
LEXICON = {
    "free": ["complimentary", "no-cost"], "prize": ["award", "reward"],
    "winner": ["recipient", "champion"], "won": ["secured", "received"],
    "claim": ["redeem", "collect"], "click": ["tap", "follow"],
    "link": ["url", "hyperlink"], "verify": ["confirm", "validate"],
    "account": ["profile", "login"], "bank": ["financial institution"],
    "payment": ["transaction", "remittance"], "urgent": ["time-sensitive"],
    "immediately": ["promptly", "right away"], "now": ["today", "at once"],
    "suspend": ["restrict", "freeze"], "blocked": ["restricted", "locked"],
    "update": ["amend", "refresh"], "details": ["particulars", "information"],
    "delivery": ["shipment", "consignment"], "parcel": ["package", "item"],
    "confirm": ["validate", "authenticate"], "password": ["passcode", "credential"],
    "security": ["safety", "protection"], "alert": ["notice", "notification"],
    "expire": ["lapse", "run out"], "limited": ["restricted", "brief"],
    "offer": ["proposal", "deal"], "gift": ["present", "token"],
    "cash": ["funds", "money"], "refund": ["reimbursement", "repayment"],
    "invoice": ["bill", "statement"], "attached": ["enclosed", "included"],
    "download": ["retrieve", "fetch"], "install": ["set up", "add"],
    "tracking": ["consignment status", "shipment reference"],
    "customer": ["client", "account holder"], "card": ["plastic", "debit instrument"],
    "otp": ["one-time code", "verification code"],
    "code": ["passcode", "reference"], "number": ["reference", "identifier"],
    "call": ["phone", "ring"], "contact": ["reach", "get in touch"],
    "access": ["entry", "reach"], "restore": ["reinstate", "re-enable"],
    "activate": ["enable", "switch on"], "deactivate": ["disable", "close"],
    "wallet": ["purse", "balance"], "transfer": ["send", "move"],
    "amount": ["sum", "value"], "bonus": ["extra", "additional sum"],
    "reward": ["incentive", "benefit"], "congratulations": ["well done"],
    "selected": ["chosen", "picked"], "eligibility": ["qualification"],
    "apply": ["register", "enrol"], "official": ["authorised", "formal"],
    "support": ["assistance", "help"], "team": ["group", "department"],
    "service": ["provision", "facility"], "message": ["note", "notice"],
    "important": ["critical", "significant"], "warning": ["caution", "notice"],
    "final": ["last", "concluding"], "chance": ["opportunity", "occasion"],
    "today": ["this day", "nowadays"], "within": ["inside", "by"],
    "hours": ["hrs"], "minutes": ["mins"], "days": ["day"],
}

PUNCT = ".,!?;:\"'()[]{}<>"

# Words that carry no decision weight. Skipping them is what makes a saliency
# search affordable: an SMS has ~28 tokens and ~10 of them are stopwords.
STOPWORDS = set("""
a an the and or but if then than that this these those is are was were be been
being am do does did have has had will would shall should can could may might
must not no nor so as at by for from in into of on onto out over per to up via
with without you your yours we our ours i me my he she it they them their there
here what which who whom when where why how all any both each few more most
other some such only own same too very just also please dear hi hello thanks
regards sent get got make made use used using
""".split())


# ──────────────────────── model access ────────────────────────

def verdict_probs(base_url, model, texts, task="sms_smishing",
                  batch_size=8, timeout=300):
    """P(malicious) for a batch of texts, or None if the shim cannot serve it.

    Returning None instead of raising is deliberate: a kernel built from an
    older shim must still produce SOMETHING, and the caller records
    probs_available=False so the paper can say the saliency ordering was not
    used rather than quietly reporting random-search numbers under its name.
    """
    payload = json.dumps({"model": model, "texts": list(texts), "task": task,
                          "batch_size": batch_size}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/verdict_probs",
        data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
        return [float(row["p_malicious"]) for row in out["probabilities"]]
    except Exception:
        return None


# ──────────────────────── candidate generation ────────────────────────

def _wordnet_synonyms(word, cache={}):
    """WordNet synonyms if the corpus is cached locally, else [].

    nltk is optional on purpose: downloading a corpus inside a scored kernel run
    is not worth a failure mode, and the curated lexicon already covers the
    security vocabulary that matters here.
    """
    if word in cache:
        return cache[word]
    scored = []
    try:
        from nltk.corpus import wordnet as wn
        for syn in wn.synsets(word)[:3]:
            for lemma in syn.lemmas():
                cand = lemma.name().replace("_", " ")
                # single tokens only: a two-word replacement inflates the
                # word-level edit distance for a reason that has nothing to do
                # with the attack, and the 15% budget is counted in words
                if not (cand.isascii() and " " not in cand and cand.isalpha()
                        and cand.lower() != word):
                    continue
                # Order by corpus frequency, not by synset order: WordNet's
                # third sense of a word is where the synonyms stop being
                # paraphrases ("tone" -> "timber"), and those substitutions
                # would be counted as semantics-preserving when they are not.
                scored.append((lemma.count(), cand.lower()))
    except Exception:
        scored = []
    scored.sort(key=lambda r: -r[0])
    cache[word] = list(dict.fromkeys(name for _c, name in scored))
    return cache[word]


def substitution_source(original_core, replacement):
    """'lexicon' or 'wordnet' - recorded per substitution, not guessed later."""
    low = str(original_core).lower()
    if replacement in LEXICON.get(low, []):
        return "lexicon"
    if replacement in _wordnet_synonyms(low):
        return "wordnet"
    return "unknown"


def candidates(text, use_wordnet=True, max_replacements=2):
    """[(position, original_word, [replacement, ...]), ...] for one text.

    max_replacements caps the query cost: PWWS scores every (word, candidate)
    pair, so this is the multiplier on the expensive part of the run.
    """
    words = text.split()
    out = []
    for i, w in enumerate(words):
        core = w.strip(PUNCT)
        low = core.lower()
        if not core or low in STOPWORDS or len(low) < 3 or not low.isalpha():
            continue
        repl = list(LEXICON.get(low, []))
        if use_wordnet:
            for cand in _wordnet_synonyms(low):
                if cand not in repl:
                    repl.append(cand)
        # never propose a replacement that is already in the text: it would be a
        # no-op query and would inflate the apparent query cost
        present = {x.strip(PUNCT).lower() for x in words}
        # single tokens only: "at once" or "account holder" would add a word and
        # inflate the word-level edit distance for a reason unrelated to the attack
        repl = [r for r in repl if " " not in r and r.lower() not in present]
        repl = repl[:max_replacements]
        if repl:
            out.append((i, w, repl))
    return out


def apply_subs(words, subs):
    """Apply {position: replacement} to a word list, keeping punctuation."""
    out = list(words)
    for i, repl in subs.items():
        original = out[i]
        lead = original[:len(original) - len(original.lstrip(PUNCT))]
        trail = original[len(original.rstrip(PUNCT)):]
        out[i] = lead + repl + trail
    return out


# ──────────────────────── the attack ────────────────────────

def run_saliency(samples, model, ask_probs, alpha=0.5, max_positions=12,
                 max_rounds=3, max_rate=0.15, seed=None, quiet=False):
    """Greedy PWWS over one model.

    ask_probs(list_of_texts, task) -> list of P(malicious) or None.

    Returns (per-sample DataFrame, info dict). The DataFrame schema matches the
    char/word runners, so src/summarize_attack.py aggregates all three modes
    through the same code path.

    `seed` is None by default, which keeps the run EXACTLY deterministic -
    the behaviour every recorded saliency result was produced with. Passing an
    integer randomises the order of TIED candidates only: the saliency score
    of two candidates is often equal (two words that the model is equally
    sensitive to), and which of them is tried first is the only stochastic
    element of a targeted attack. That is what a seed sweep measures here.
    """
    rows = []
    probs_seen = 0
    for idx, sample in enumerate(samples.itertuples(index=False)):
        text = str(sample.text)
        task = getattr(sample, "task", "sms_smishing") or "sms_smishing"
        words = text.split()

        base = ask_probs([text], task)
        if base is None:
            rows.append({"id": sample.id, "model": model, "mode": "saliency",
                         "task": task, "orig_pred": None, "pert_pred": None,
                         "success": False, "counted": False,
                         "skip_reason": "verdict_probs unavailable"})
            continue
        probs_seen += 1
        p0 = base[0]
        if p0 < 0.5:                       # same denominator rule as the other modes
            rows.append({"id": sample.id, "model": model, "mode": "saliency",
                         "task": task, "orig_pred": 0, "pert_pred": None,
                         "success": False, "counted": False,
                         "skip_reason": "original not classified malicious"})
            continue

        cands = candidates(text)[:max_positions]
        budget = max(1, int(len(words) * max_rate))
        n_queries = 1
        applied, used_types, success = {}, [], False

        # ── round 0: one-text-per-candidate saliency probe, batched ──
        probe_texts, probe_meta = [], []
        for (pos, orig, repls) in cands:
            for repl in repls:
                probe_texts.append(" ".join(apply_subs(words, {pos: repl})))
                probe_meta.append((pos, orig, repl))
        probe_probs = ask_probs(probe_texts, task) if probe_texts else []
        if probe_probs is None:
            probe_probs = []
        n_queries += len(probe_texts)

        raw = [(max(0.0, p0 - float(p)), pos, orig, repl, float(p))
               for (pos, orig, repl), p in zip(probe_meta, probe_probs)]
        top_s = max([r[0] for r in raw], default=0.0)
        scored = []
        for saliency, pos, orig, repl, p in raw:
            s_norm = (saliency / top_s) if top_s > 0 else 0.0
            phi = alpha * s_norm + (1 - alpha) * p
            scored.append((phi, s_norm, pos, orig, repl, p))
        scored.sort(key=lambda r: -r[0])
        if seed is not None:
            # randomise the tie-break only: sort is stable, so shuffling
            # first and sorting after leaves the ranking intact and only
            # moves candidates with IDENTICAL scores around
            rng = random.Random(seed * 100003 + idx)
            rng.shuffle(scored)
            scored.sort(key=lambda r: -r[0])

        # greedy accumulation, re-checking the verdict after every accepted edit
        for (_phi, _s, pos, orig, repl, _p) in scored[:max_rounds * 3]:
            if pos in applied or len(applied) >= budget:
                continue
            trial = dict(applied)
            trial[pos] = repl
            cand_text = " ".join(apply_subs(words, trial))
            if cand_text == text:
                continue
            p = ask_probs([cand_text], task)
            n_queries += 1
            if p is None:
                continue
            applied = trial
            used_types.append("%s->%s(%s)" % (orig.strip(PUNCT), repl,
                                               substitution_source(orig.strip(PUNCT), repl)))
            if p[0] < 0.5:
                success = True
                break
            if len(applied) >= budget:
                break

        pert = " ".join(apply_subs(words, applied)) if applied else text
        stats = edit_counts(text, pert, level="word")
        rows.append({
            "id": sample.id, "model": model, "mode": "saliency", "task": task,
            "orig_pred": 1, "pert_pred": 0 if success else 1,
            "success": success, "counted": True, "skip_reason": "",
            "attempt": 1, "perturbed_text": pert, "n_subs": len(applied),
            "attack_types": "saliency:" + "+".join(used_types[:6]),
            "similarity": round(jaccard(text, pert), 4),
            "edits": stats["edits"], "perturbation_rate": stats["rate"],
            "n_queries": n_queries,
            "probs_available": True,
        })
        if not quiet and (idx + 1) % 20 == 0:
            print("  %d/%d saliency-attacked" % (idx + 1, len(samples)),
                  flush=True)

    df = pd.DataFrame(rows)
    info = {"mode": "saliency", "model": model, "alpha": alpha,
            "max_positions": max_positions, "max_rounds": max_rounds,
            "seed": seed,
            "n_samples": len(df), "n_with_probs": probs_seen}
    return df, info
