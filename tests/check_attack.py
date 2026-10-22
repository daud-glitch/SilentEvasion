"""
Offline checks for the attack runner - no model, no shim.

The ASR number is only meaningful if the accounting is right, so these assert
the accounting directly:

  1. a flippable model (1 on the ORIGINAL text, 0 on any perturbation) must
     score ASR 1.0, not 0.5 or "estimated"
  2. a model that never flips must score ASR 0.0
  3. a model that calls an eligible sample benign is excluded from the
     denominator instead of being counted as a failure
  4. the per-sample task is what reaches the prompt (an email must not be
     attacked with the SMS prompt)
  5. the perturbation budget is a real edit rate and the similarity is Jaccard
  6. every ASR interval is honest at 0 flips and repeats for a fixed seed

    python tests/check_attack.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from src.attack_common import (  # noqa: E402
    asr_interval, bootstrap_interval, edit_counts, jaccard, levenshtein,
    print_summary, run_attack, wilson_interval,
)

FAILS = []


def check(label, ok, detail=""):
    print(("  OK   " if ok else "  FAIL ") + label + ((" - " + detail) if detail and not ok else ""))
    if not ok:
        FAILS.append(label)


def samples(n=12):
    return pd.DataFrame({
        "id": [f"s{i}" for i in range(n)],
        "task": ["phishing_email" if i % 2 else "sms_smishing" for i in range(n)],
        "model": ["m"] * n,
        # unique per row: the fake chat_fn below matches on exact text
        "text": [("Your account is locked, verify at http://x.example/pay now"
                  if i % 3 else "Congratulations winner claim your free prize today")
                 + f" ref{i}" for i in range(n)],
    })


def test_flip():
    print("\n[1] flippable model -> ASR 1.0")
    originals = set(samples()["text"])
    seen_tasks = set()

    def chat_fn(text, task):
        seen_tasks.add(task)
        return 1 if text in originals else 0

    df, summ = run_attack(samples(), mode="char", model="m", attempts=3,
                          chat_fn=chat_fn, quiet=True)
    check("ASR is 1.0", summ["asr"] == 1.0, f"got {summ['asr']}")
    check("successes == attacked",
          summ["n_success"] == summ["n_attacked"], str(summ))
    check("nothing skipped", summ["n_skipped_original_benign"] == 0)
    check("both prompts' tasks reached the model",
          seen_tasks == {"sms_smishing", "phishing_email"}, str(seen_tasks))
    check("per-task breakdown present", len(summ["per_task"]) == 2, str(summ["per_task"]))
    check("edit rate within the 15% budget",
          float(df[df.counted == True].perturbation_rate.max()) <= 0.15,  # noqa: E712
          str(df.perturbation_rate.max()))
    return df, summ


def test_no_flip():
    print("\n[2] model that never flips -> ASR 0.0")
    df, summ = run_attack(samples(), mode="word", model="m", attempts=2,
                          chat_fn=lambda text, task: 1, quiet=True)
    check("ASR is 0.0", summ["asr"] == 0.0, f"got {summ['asr']}")
    check("all samples were attacked",
          summ["n_attacked"] == summ["n_eligible"], str(summ))


def test_excluded():
    print("\n[3] original re-read as benign -> excluded, not counted")
    # first half reads benign, second half flips
    texts = list(samples()["text"])

    def chat_fn(text, task):
        if text in texts[:6]:
            return 0                      # original not malicious -> skip
        return 1 if text in texts else 0

    df, summ = run_attack(samples(), mode="char", model="m", attempts=1,
                          chat_fn=chat_fn, quiet=True)
    check("6 excluded", summ["n_skipped_original_benign"] == 6, str(summ))
    check("6 attacked", summ["n_attacked"] == 6, str(summ))
    check("ASR is over attacked only", summ["asr"] == 1.0, str(summ))


def test_metrics():
    print("\n[4] edit distance and Jaccard are real measurements")
    c = edit_counts("abcdef", "abXdef", level="char")
    check("one substitution detected", c["edits"] == 1, str(c))
    check("rate is edits/length", abs(c["rate"] - 1 / 6) < 1e-3, str(c))
    check("identical text -> 0 edits", edit_counts("a b", "a b", "word")["edits"] == 0)
    check("Jaccard identical -> 1.0", jaccard("a b c", "a b c") == 1.0)
    check("Jaccard disjoint -> 0.0", jaccard("a b", "c d") == 0.0)

    # difflib reported 204 edits for 56 deletions on a repetitive 749-char
    # message; the count has to be the exact distance, not an opcode tally.
    def brute(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i] + [0] * len(b)
            for j, cb in enumerate(b, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            prev = cur
        return prev[-1]

    import random
    rng = random.Random(7)
    bad = 0
    for _ in range(120):
        a = "".join(rng.choice("abc") for _ in range(rng.randint(0, 30)))
        b = "".join(rng.choice("abc") for _ in range(rng.randint(0, 30)))
        if levenshtein(list(a), list(b)) != brute(a, b):
            bad += 1
    check("bit-vector distance matches the DP table", bad == 0, "%d mismatch" % bad)

    # a pure deletion costs exactly one edit per character, no matter how
    # repetitive the message is (the case difflib inflated)
    text = "x" * 200
    cut = text[:150]
    check("56-of-749 style deletion counts exactly",
          edit_counts(text, cut, "char")["edits"] == 50,
          str(edit_counts(text, cut, "char")))


def test_summary_file():
    print("\n[5] summary can be written to a file (the old code crashed here)")
    _, summ = run_attack(samples(), mode="char", model="m", attempts=1,
                         chat_fn=lambda text, task: 0, quiet=True)
    path = os.path.join(ROOT, "results", "_asr_smoke.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        print_summary(summ, file=f)
    body = open(path, encoding="utf-8").read()
    os.remove(path)
    check("file got the ASR line", "ASR (of attacked)" in body, body[:200])


def test_intervals():
    print("\n[6] ASR intervals are honest at the boundaries")
    lo, hi = wilson_interval(0, 188)
    check("0/188 does not claim a zero upper bound", hi > 0, "%.4f" % hi)
    check("0/188 lower is 0", lo < 1e-6, "%.10f" % lo)
    lo, hi = wilson_interval(188, 188)
    check("188/188 lower is below 1", lo < 1.0, "%.4f" % lo)

    # the interval must contain the point estimate it describes
    bad = []
    for k, n in [(0, 10), (1, 50), (6, 120), (25, 40), (40, 40), (3, 161)]:
        lo, hi = wilson_interval(k, n)
        if not (lo <= k / n <= hi):
            bad.append((k, n, lo, hi))
    check("Wilson covers the point estimate in every case", not bad, str(bad))

    # more samples at the same rate -> a narrower interval
    wide = wilson_interval(5, 100)
    narrow = wilson_interval(50, 1000)
    check("interval narrows with n",
          (narrow[1] - narrow[0]) < (wide[1] - wide[0]),
          "%s vs %s" % (wide, narrow))

    # bootstrap must be reproducible: a paper's CI cannot change between runs
    flags = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    check("bootstrap is deterministic for a fixed seed",
          bootstrap_interval(flags) == bootstrap_interval(flags), "")
    b_lo, b_hi = bootstrap_interval(flags)
    check("bootstrap contains the observed rate",
          b_lo <= 0.3 <= b_hi, "%.3f-%.3f" % (b_lo, b_hi))

    d = asr_interval([0] * 188)
    check("0-flip row reports both bounds",
          d["wilson_high"] > 0 and d["boot_low"] == 0.0 and d["flips"] == 0, str(d))
    d = asr_interval([1, 0, 0, 0])
    check("mixed row: n and flips are the real counts",
          d["n"] == 4 and d["flips"] == 1, str(d))


def main():
    test_flip()
    test_no_flip()
    test_excluded()
    test_metrics()
    test_summary_file()
    test_intervals()
    print("\n%s: %d check(s) failed" % ("FAIL" if FAILS else "PASS", len(FAILS)))
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
