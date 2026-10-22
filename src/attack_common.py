"""
Honest attack runner: shared primitives for char-level and word-level ASR.

Every rule below fixes a specific defect of the earlier scripts, each of which
made the reported ASR meaningless:

  * the attack set comes from results/attack_eligible.csv - samples the CLEAN
    model actually classified malicious. Attacking a message the model already
    called benign cannot produce a flip (the old scripts hard-coded
    `is_malicious_original = 1` for every sample).
  * the prompt is build_messages(...), the same one the baseline used. The old
    scripts sent the raw text with no system prompt, so they were measuring a
    different task than the baseline they wanted to compare against.
  * the ORIGINAL text is re-queried rather than assumed malicious.
  * attack_success requires orig_pred == 1 AND pert_pred == 0.
  * edits are the exact Levenshtein distance, not estimated as
    len(text) * 0.15 (and not a difflib opcode tally, which is not minimal).
  * similarity is Jaccard word overlap, reported under that name - not
    "BERTScore". If a real BERTScore is wanted it must be computed with the
    model, which this repo does not ship.
  * every ASR carries a 95% interval (Wilson, plus percentile bootstrap),
    because flip counts of 0-6 cannot be quoted as a bare percentage.
  * each perturbation is seeded per sample, so a run is reproducible and a
    reported ASR can be re-derived.
"""
import difflib
import json
import math
import os
import random
import string
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import RESULTS_DIR, SEED  # noqa: E402
from src.prompting import build_messages, parse_verdict  # noqa: E402

ELIGIBLE_PATH = os.path.join(RESULTS_DIR, "attack_eligible.csv")

# Characters that look like ASCII letters but are different codepoints. The
# model tokenises these differently while a human reads the same text.
HOMOGLYPHS = {
    "a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440",
    "c": "\u0441", "x": "\u0445", "y": "\u0443", "i": "\u0456",
    "s": "\u0455", "A": "\u0410", "B": "\u0412", "E": "\u0415",
}


# ──────────────────────── model access ────────────────────────

def chat_verdict(base_url, model, text, task="sms_smishing", timeout=180):
    """One verdict from the shim. Returns (pred, raw, error)."""
    payload = json.dumps({
        "model": model,
        "messages": build_messages(task, text),   # same prompt as the baseline
        "temperature": 0.0,
        "max_tokens": 8,
        "seed": SEED,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
        raw = out["choices"][0]["message"]["content"]
        return parse_verdict(raw), raw, ""
    except Exception as exc:                      # a dead shim must be visible, not silent
        return None, "", f"{type(exc).__name__}: {exc}"


# ──────────────────────── measurement ────────────────────────

def jaccard(a, b):
    """Word-set Jaccard overlap. Named honestly: this is NOT BERTScore."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def levenshtein(seq_a, seq_b):
    """Exact edit distance (insert/delete/substitute = 1 each).

    difflib is NOT usable for this: its SequenceMatcher finds the longest
    matching block recursively, which is not a minimal edit script. On a 749
    character message where the attack deleted 56 characters it reported 204
    edits, and the "15% budget" was then read as 27% - a fabricated-looking
    number produced by the measurement, not by the attack.
    """
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)
    # Myers' bit-vector algorithm: same exact distance as the O(n*m) table,
    # but word-parallel. A 1000-char pair takes ~16k integer ops instead of
    # 10^6, which is what makes re-scoring a finished run practical.
    if len(seq_b) > len(seq_a):
        seq_a, seq_b = seq_b, seq_a
    m = len(seq_b)
    mask = (1 << m) - 1
    peq = {}
    for i, ch in enumerate(seq_b):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    pv, mv = mask, 0
    score = m
    last = 1 << (m - 1)
    for ch in seq_a:
        eq = peq.get(ch, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = (mv | ~(xh | pv)) & mask
        mh = pv & xh
        if ph & last:
            score += 1
        elif mh & last:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = (mh | ~(xv | ph)) & mask
        mv = ph & xv
    return score


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a binomial proportion.

    Needed because the flip counts are tiny: with 0/188 flips a percentile
    bootstrap collapses to [0, 0], which claims a precision the data cannot
    support. Wilson stays honest at the boundaries (0 and n).
    """
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_interval(flags, n_boot=10000, seed=SEED, alpha=0.05):
    """Percentile bootstrap of the ASR over samples.

    Resampling samples (not flips) keeps the unit of analysis right: the seed
    is the message, and a resampled set can only contain messages that were
    actually attacked. Deterministic for a fixed seed.
    """
    arr = np.asarray([1.0 if f else 0.0 for f in flags], dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    asrs = arr[idx].mean(axis=1)
    lo, hi = np.percentile(asrs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def asr_interval(flags):
    """Both intervals for one ASR row, plus the denominator they apply to."""
    flags = list(flags)
    n = len(flags)
    k = int(sum(1 for f in flags if f))
    w_lo, w_hi = wilson_interval(k, n)
    b_lo, b_hi = bootstrap_interval(flags)
    return {"n": n, "flips": k,
            "wilson_low": round(w_lo, 4), "wilson_high": round(w_hi, 4),
            "boot_low": round(b_lo, 4), "boot_high": round(b_hi, 4)}


def edit_counts(original, perturbed, level="char"):
    """Exact edits between original and perturbed, and the budgeted rate."""
    seq_a = list(original) if level == "char" else original.split()
    seq_b = list(perturbed) if level == "char" else perturbed.split()
    edits = levenshtein(seq_a, seq_b)
    units = len(seq_a) if seq_a else 1
    return {"edits": edits, "rate": round(edits / units, 4),
            "units": units}


# ──────────────────────── perturbations ────────────────────────

def char_perturb(text, rng, max_rate=0.15):
    """Character-level noise: homoglyph swap, delete, insert, swap.

    ONE edit budget for the whole perturbation, not one per operation. The
    earlier version gave each op its own budget, so a sample could take ~1.8x
    the nominal 15% and the "imperceptible" claim in the paper was false.
    """
    if not text:
        return text, []
    out = list(text)
    budget = max(1, int(len(text) * max_rate))
    used, spent = [], 0

    # substitution: visibly the same character, so it is the cheapest edit
    if rng.random() < 0.7:
        usable = [i for i, ch in enumerate(out) if ch in HOMOGLYPHS]
        k = min(budget - spent, len(usable))
        if k > 0:
            for i in rng.sample(usable, k):
                out[i] = HOMOGLYPHS[out[i]]
            spent += k
            used.append("homoglyph")

    if spent < budget and rng.random() < 0.6 and len(out) > 1:
        k = min(max(1, (budget - spent) // 2), len(out) - 1)
        if k > 0:
            for i in sorted(rng.sample(range(len(out)), k), reverse=True):
                del out[i]
            spent += k
            used.append("delete")

    if spent < budget and rng.random() < 0.4:
        k = min(budget - spent, budget)
        if k > 0:
            for _ in range(k):
                out.insert(rng.randint(0, len(out)), rng.choice(string.ascii_lowercase))
            spent += k
            used.append("insert")

    # a swap changes two positions, so it only fits when 2 edits remain
    if len(out) >= 2 and spent + 2 <= budget and rng.random() < 0.4:
        i, j = rng.sample(range(len(out)), 2)
        out[i], out[j] = out[j], out[i]
        spent += 2
        used.append("swap")
    return "".join(out), used


SYNONYMS = {
    "free": ["costless", "complimentary"], "prize": ["award", "reward"],
    "winner": ["champion"], "claim": ["collect", "redeem"],
    "now": ["today", "immediately"], "urgent": ["immediate"],
    "account": ["profile"], "verify": ["confirm"], "click": ["tap"],
    "link": ["url"], "bank": ["finance"], "payment": ["transaction"],
    "delivery": ["shipment"], "parcel": ["package"],
}


def word_perturb(text, rng, max_rate=0.15):
    """Word-level noise: synonym swap, delete, duplicate, swap.

    Same single-budget rule as char_perturb.
    """
    words = text.split()
    if len(words) < 2:
        return text, []
    used = []
    budget = max(1, int(len(words) * max_rate))
    out = list(words)
    spent = 0

    if rng.random() < 0.7:
        pool = [i for i, w in enumerate(out) if w.lower().strip(".,!") in SYNONYMS]
        k = min(budget - spent, len(pool))
        if k > 0:
            for i in rng.sample(pool, k):
                key = out[i].lower().strip(".,!")
                out[i] = rng.choice(SYNONYMS[key])
            spent += k
            used.append("synonym")

    if spent < budget and rng.random() < 0.5 and len(out) > 1:
        k = min(max(1, (budget - spent) // 2), len(out) - 1)
        if k > 0:
            for i in sorted(rng.sample(range(len(out)), k), reverse=True):
                del out[i]
            spent += k
            used.append("delete")

    if spent < budget and rng.random() < 0.3 and out:
        i = rng.randrange(len(out))
        out.insert(i, out[i])
        spent += 1
        used.append("duplicate")

    if len(out) >= 2 and spent + 2 <= budget and rng.random() < 0.3:
        i, j = rng.sample(range(len(out)), 2)
        out[i], out[j] = out[j], out[i]
        spent += 2
        used.append("swap")
    return " ".join(out), used


PERTURBERS = {"char": char_perturb, "word": word_perturb}


# ──────────────────────── the run ────────────────────────

def read_attack_csv(path):
    """Read a per-sample attack CSV, restoring `model` if the file predates it.

    Files written before `model` was a row column only carry it in the
    filename (`attack_results_<mode>_<model>[_<task>].csv`). Every aggregator
    groups by model, so an old file must not be silently skipped - or worse,
    crash the report.
    """
    df = pd.read_csv(path)
    if "model" not in df.columns:
        base = os.path.basename(path)[len("attack_results_"):-len(".csv")]
        parts = base.split("_")
        if len(parts) >= 2:
            df["model"] = parts[1]
            df.attrs["model_from_filename"] = True
    return df


def load_eligible(path=ELIGIBLE_PATH):
    """Samples the clean model classified malicious. The only valid attack set."""
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\nThe attack set must come from the baseline, not from "
            "the dataset:\n  python src/analyze_baseline.py   # then re-save eligible rows\n"
            "or run the C9 cell of colab_runbook.ipynb.")
    df = pd.read_csv(path)
    if "text" not in df.columns or "id" not in df.columns:
        raise SystemExit(f"{path} needs at least id and text columns")
    return df


def run_attack(samples, mode="char", model="qwen2.5-3b",
               base_url="http://localhost:8000/v1", attempts=1,
               max_rate=0.15, chat_fn=None, seed=SEED, quiet=False):
    """
    Attack every sample, then report a defensible ASR.

    Returns (per-sample DataFrame, summary dict). chat_fn is injectable so the
    logic can be tested without a model.

    `seed` decides which perturbation each sample gets. It is a PARAMETER,
    not a constant, because the variance of the attack across seeds is a
    result in its own right: a random word attack that reports one number
    from one seed is reporting the seed, not the attack.
    """
    if mode not in PERTURBERS:
        raise ValueError(f"mode must be one of {sorted(PERTURBERS)}")
    perturb = PERTURBERS[mode]
    # the task travels with the sample: an email attacked with the SMS prompt
    # would score a different question than the baseline it is compared to
    ask = chat_fn or (lambda text, task: chat_verdict(base_url, model, text,
                                                     task=task)[0])
    level = "char" if mode == "char" else "word"
    default_task = "sms_smishing"

    rows, skipped_benign = [], 0
    for i, sample in enumerate(samples.itertuples(index=False)):
        text = str(sample.text)
        task = getattr(sample, "task", default_task) or default_task

        # the original is re-queried, never assumed malicious
        orig_pred = ask(text, task)
        if orig_pred != 1:
            skipped_benign += 1
            rows.append({"id": sample.id, "model": model, "mode": mode,
                         "task": task, "orig_pred": orig_pred,
                         "pert_pred": None, "success": False, "counted": False,
                         "skip_reason": "original not classified malicious"})
            continue

        # model is part of every row, not just the summary: the paper's table
        # groups on it, and a CSV without it cannot be aggregated at all.
        row = {"id": sample.id, "model": model, "mode": mode, "task": task,
               "orig_pred": orig_pred,
               "pert_pred": None, "success": False, "counted": False,
               "skip_reason": ""}
        for attempt in range(attempts):
            rng = random.Random(seed + 1000 * i + attempt)
            perturbed, used = perturb(text, rng, max_rate)
            if perturbed == text:
                row["skip_reason"] = "perturbation produced identical text"
                continue
            stats = edit_counts(text, perturbed, level=level)
            pred = ask(perturbed, task)
            success = pred == 0                      # 1 -> 0 is the flip that matters
            row.update({
                "attempt": attempt + 1,
                "perturbed_text": perturbed,
                "pert_pred": pred,
                "success": success,
                "counted": True,
                "skip_reason": "",
                "attack_types": "+".join(used),
                "similarity": round(jaccard(text, perturbed), 4),
                "edits": stats["edits"],
                "perturbation_rate": stats["rate"],
            })
            if success:
                break
        rows.append(row)
        if not quiet and (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(samples)} attacked", flush=True)

    df = pd.DataFrame(rows)
    summary = summarize_counts(df, mode, model, skipped_benign)
    summary["seed"] = seed
    summary["attempts"] = attempts
    return df, summary


def summarize_counts(df, mode, model, skipped_benign=None):
    """The summary dict, from a per-sample frame.

    Separated out so a saved CSV can be re-summarised (and re-scored) without
    re-running the model, which is what makes a metric fix publishable instead
    of requiring a fresh GPU run.
    """
    counted = df[df["counted"] == True]              # noqa: E712 (pandas object dtype)
    n_success = int(counted["success"].sum())
    if skipped_benign is None:
        skipped_benign = int(len(df) - len(counted))
    per_task = {}
    if len(counted):
        for task, g in counted.groupby("task"):
            n_ok = int(g["success"].sum())
            per_task[str(task)] = {"n_attacked": int(len(g)),
                                   "n_success": n_ok,
                                   "asr": round(n_ok / len(g), 4)}
    summary = {
        "mode": mode,
        "model": model,
        "per_task": per_task,
        "n_eligible": len(df),
        "n_skipped_original_benign": skipped_benign,
        "n_attacked": len(counted),
        "n_success": n_success,
        "asr": round(n_success / len(counted), 4) if len(counted) else 0.0,
        "asr_of_eligible": round(n_success / len(df), 4) if len(df) else 0.0,
        "mean_perturbation_rate": (round(float(counted["perturbation_rate"].mean()), 4)
                                   if len(counted) else 0.0),
        "mean_jaccard": (round(float(counted["similarity"].mean()), 4)
                         if len(counted) else 0.0),
    }
    return summary


def print_summary(summary, file=None):
    """Human-readable ASR block. Also written to a file, so vagueness in the
    report is impossible: the numbers on disk come from the same code path."""
    out = file or sys.stdout

    def say(msg=""):
        print(msg, file=out)

    say(f"\n=== {summary['mode']}-level attack ({summary.get('model', '?')}) ===")
    say(f"eligible (clean-correct) samples : {summary['n_eligible']}")
    say(f"excluded (original not malicious): {summary['n_skipped_original_benign']}")
    say(f"actually attacked                : {summary['n_attacked']}")
    say(f"flips (malicious -> benign)      : {summary['n_success']}")
    say(f"ASR (of attacked)                : {summary['asr']:.1%}")
    say(f"mean perturbation rate           : {summary['mean_perturbation_rate']:.2%}")
    say(f"mean word Jaccard                : {summary['mean_jaccard']:.3f}")
    for task, d in sorted(summary.get("per_task", {}).items()):
        say(f"  {task:<16} n={d['n_attacked']:<4} flips={d['n_success']:<4} "
            f"ASR={d['asr']:.1%}")
    if summary["mean_perturbation_rate"] > 0.15:
        say("!! perturbation rate above the 15% budget - not imperceptible")
    if summary["n_attacked"] < 10:
        say("!! fewer than 10 attacked samples - too small to report as an ASR")
