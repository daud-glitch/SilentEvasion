"""
Check that the paper's numbers come from the results CSVs, not from typing.

Three checks, all offline:

  1. every generated table under paper/tables matches results/*.csv
  2. every percentage written in paper/sections/*.tex is one the CSVs can
     produce (ASR, interval bounds, catch rate, perturbation rate, Jaccard
     scaled, an aggregate of those, or an explicitly justified constant)
  3. the paper's LaTeX inputs all exist, so a missing section fails here
     rather than in a compile

    python tests/check_paper.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

PAPER = os.path.join(ROOT, "paper")
SECTIONS = os.path.join(PAPER, "sections")
TABLES = os.path.join(PAPER, "tables")
RESULTS = os.path.join(ROOT, "results")

FAILS = []


def check(label, ok, detail=""):
    print(("  OK   " if ok else "  FAIL ") + label + ((" - " + detail) if detail and not ok else ""))
    if not ok:
        FAILS.append(label)


def load(name):
    return pd.read_csv(os.path.join(RESULTS, name))


# ── the set of numbers the data can legitimately produce ──────────────────
def measured_percentages():
    """One and two decimal renderings of every measured quantity."""
    out = set()

    def add(v, scale=100.0):
        for nd in (1, 2):
            out.add("%.*f" % (nd, scale * float(v)))

    for _, r in load("attack_table.csv").iterrows():
        for col in ("asr", "ci95_low", "ci95_high", "boot95_low", "boot95_high",
                    "catch_rate", "mean_perturbation_rate"):
            add(r[col])
        add(r["mean_jaccard"], scale=1.0)          # Jaccard is quoted 0-1
        add(r["mean_jaccard"], scale=100.0)        # and sometimes as a percent
        add(r["asr"], scale=100.0)
        # ratios the text is allowed to quote from the same row
        add(r["n_flips"] / r["n_attacked"]) if r["n_attacked"] else None

    base = load("baseline_table.csv")
    for _, r in base.iterrows():
        for col in ("accuracy", "precision", "recall", "f1"):
            add(r[col], scale=1.0)                 # quoted as 0.920 style
            add(r[col])

    atk = load("attack_table.csv")
    for mode in list(atk["mode"].unique()) + ["all"]:
        sub = atk if mode == "all" else atk[atk["mode"] == mode]
        n, k = int(sub["n_attacked"].sum()), int(sub["n_flips"].sum())
        if n:
            add(k / n)
    # mode-to-mode comparison the results section makes explicitly
    for mode in atk["mode"].unique():
        sub = atk[atk["mode"] == mode]
        add(sub["mean_perturbation_rate"].mean())

    # the three later experiments, once their CSVs exist: their generated
    # result blocks quote these values, so they must be in the allowed set
    for name, cols in (("defense_summary.csv", ("asr", "ci95_low", "ci95_high")),
                       ("transfer_matrix.csv", ("transfer_asr", "ci95_low",
                                                "ci95_high"))):
        path = os.path.join(RESULTS, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df):
                for col in cols:
                    if col in df.columns:
                        for v in df[col].tolist():
                            add(v)
    # pooled defence rates: the generated block quotes flips/n aggregated over
    # models AND over tasks, which is none of the per-row values above
    dpath = os.path.join(RESULTS, "defense_summary.csv")
    if os.path.exists(dpath):
        d = pd.read_csv(dpath)
        for col in ("n", "flips"):
            d[col] = pd.to_numeric(d[col], errors="coerce")
        for mode in d["mode"].unique():
            for defence in d["defense"].unique():
                sub = d[(d["mode"] == mode) & (d["defense"] == defence)]
                n = int(sub["n"].fillna(0).sum())
                if n:
                    add(float(sub["flips"].fillna(0).sum()) / n)
    tpath = os.path.join(RESULTS, "defense_thresholds.csv")
    if os.path.exists(tpath):
        t = pd.read_csv(tpath)
        for col in ("fpr_at_0.5", "tpr_at_0.5", "fpr_at_tau", "tpr_at_tau"):
            if col in t.columns:
                for v in t[col].tolist():
                    add(v)

    # pooled transfer: the off-diagonal `applied` subset, which the transfer
    # section quotes as a rate rather than cell by cell
    trpath = os.path.join(RESULTS, "transfer_matrix.csv")
    if os.path.exists(trpath):
        t = pd.read_csv(trpath)
        off = t[(t["subset"] == "applied") & (t["source"] != t["target"])]
        if len(off) and int(off["n_target_caught"].sum()):
            add(float(off["n_flips"].sum()) / int(off["n_target_caught"].sum()))

    # interval-overlap ratios quoted in the "interval discipline" paragraph
    for (_task, _mode), g in atk.groupby(["task", "mode"]):
        rows = g.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                ov = min(a["ci95_high"], b["ci95_high"]) - max(a["ci95_low"], b["ci95_low"])
                narrow = min(a["ci95_high"] - a["ci95_low"], b["ci95_high"] - b["ci95_low"])
                add(ov / narrow)
    return out


# Numbers that are not percentages of a measurement of THIS study. Each entry
# carries the reason it is allowed and the only files it may appear in, so a
# number copied into the wrong section fails instead of passing.
#   value -> (reason, allowed file prefixes)
JUSTIFIED = {
    "15": ("the edit budget fixed by the protocol", None),
    "95": ("confidence level, and a quoted F1 from a cited paper", None),
    "90": ("F1 quoted from the Flan-T5 result in a cited paper", {"02_related"}),
    "99.70": ("accuracy quoted from the ChatSpamDetector paper", {"02_related"}),
    "98.8": ("accuracy quoted from the SmishX paper", {"02_related"}),
    "75": ("false-positive rate quoted from the CTI attack paper", {"02_related"}),
    "0": ("zero, as in 'not 0%'", None),
    "22": ("perturbation rate measured in our pre-fix pipeline (see section 3); "
           "deliberately not present in the released CSVs",
           {"03_protocol", "10_ethics_availability"}),
    "5": ("the clean false-positive budget the defence threshold is fitted to; "
          "it is an input to the experiment, not a measurement",
          {"00_abstract", "09_discussion", "08_defense", "gen_defense_results"}),
    "100": ("either a defect in our own earlier code, or a clean recall of "
            "1.00 written as a percentage (both are identified in place)",
            {"09_discussion", "10_ethics_availability", "08_defense"}),
}


def tex_percentages():
    """Every '12.3' that is written immediately before a percent sign."""
    found = []
    rx = re.compile(r"(\d+(?:\.\d+)?)\s*\\%")
    for name in sorted(os.listdir(SECTIONS)):
        if not name.endswith(".tex"):
            continue
        text = open(os.path.join(SECTIONS, name), encoding="utf-8").read()
        for m in rx.finditer(text):
            start = max(0, m.start() - 70)
            found.append((name, m.group(1), text[start:m.end()].replace("\n", " ")))
    return found


def check_tables_vs_csv():
    print("\n[1] generated tables match the CSVs")
    for name in ("t1_baseline.tex", "t2_char.tex", "t3_word.tex", "t4_budget.tex"):
        path = os.path.join(TABLES, name)
        check("%s exists" % name, os.path.exists(path))
        if os.path.exists(path):
            body = open(path, encoding="utf-8").read()
            check("%s has a bottom rule" % name, "\\bottomrule" in body)

    # A row ending in four backslashes is `\\` twice: LaTeX reads it as the end
    # of the row plus an empty row, which silently adds vertical space to every
    # table in the paper. Cheap to check, invisible to the eye.
    doubled = []
    for name in sorted(os.listdir(TABLES)):
        if not name.endswith(".tex"):
            continue
        for line in open(os.path.join(TABLES, name), encoding="utf-8"):
            if line.rstrip().endswith("\\\\\\\\"):
                doubled.append("%s: %s" % (name, line.strip()[:40]))
    check("no table row is terminated twice", not doubled, "; ".join(doubled[:3]))

    # t2: every ASR cell must equal flips/attacked for that row
    atk = load("attack_table.csv")
    for mode, fname in (("char", "t2_char.tex"), ("word", "t3_word.tex")):
        body = open(os.path.join(TABLES, fname), encoding="utf-8").read()
        for _, r in atk[atk["mode"] == mode].iterrows():
            cell = "%d & %d & %.1f\\%% & %.1f--%.1f" % (
                int(r["n_attacked"]), int(r["n_flips"]), 100 * r["asr"],
                100 * r["ci95_low"], 100 * r["ci95_high"])
            check("%s row %s/%s present with the CSV numbers"
                  % (fname, r["model"], r["task"]), cell in body,
                  "looked for: " + cell)

    base = load("baseline_table.csv")
    body = open(os.path.join(TABLES, "t1_baseline.tex"), encoding="utf-8").read()
    for _, r in base.iterrows():
        cell = "%.3f & %.3f & %.3f & %.3f" % (r["accuracy"], r["precision"],
                                               r["recall"], r["f1"])
        check("t1 row %s/%s present with the CSV numbers" % (r["model"], r["task"]),
              cell in body, "looked for: " + cell)

    # the budget table's "no row above 15%" claim
    check("no row exceeds the 15% budget in the data",
          float(atk["mean_perturbation_rate"].max()) <= 0.15,
          str(atk["mean_perturbation_rate"].max()))
    body = open(os.path.join(TABLES, "t4_budget.tex"), encoding="utf-8").read()
    for mode, label in (("char", "Character"), ("word", "Word")):
        sub = atk[atk["mode"] == mode]
        cell = "%d & %d" % (int(sub["n_attacked"].sum()), int(sub["n_flips"].sum()))
        check("t4 %s totals match the CSV" % label, cell in body, "looked for: " + cell)


def check_prose_numbers():
    print("\n[2] every percentage in the prose is a measured value")
    allowed = measured_percentages()
    # tightest cross-row aggregate the text is allowed to use
    allowed |= {("%.1f" % v) for v in (max(load("attack_table.csv")["ci95_high"]) * 100,
                                       max(load("attack_table.csv")["mean_perturbation_rate"]) * 100)}
    bad = []
    for name, value, ctx in tex_percentages():
        if value in allowed:
            continue
        if value in JUSTIFIED:
            _, files = JUSTIFIED[value]
            stem = name.split(".")[0]
            if files is None or stem in files:
                continue
            bad.append("%s: %s%% is justified only for %s ... %s"
                       % (name, value, sorted(files), ctx.strip()))
            continue
        bad.append("%s: %s%% ... %s" % (name, value, ctx.strip()))
    check("no unexplained percentage in the sections", not bad,
          "\n         ".join(bad[:8]))


def check_inputs_exist():
    print("\n[3] every \\input in main.tex resolves")
    main = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    missing = []
    for m in re.finditer(r"^\\input\{([^}]+)\}", main, re.M):
        target = m.group(1)
        for cand in (os.path.join(PAPER, target), os.path.join(PAPER, target + ".tex")):
            if os.path.exists(cand):
                break
        else:
            missing.append(target)
    check("all \\input targets exist", not missing, str(missing))
    # a commented \input is fine, an enabled missing one is not (same loop)


def prose_text():
    """Every section's text, keyed by file stem, plus a concatenation."""
    out = {}
    for name in sorted(os.listdir(SECTIONS)):
        if name.endswith(".tex"):
            out[name.split(".")[0]] = open(os.path.join(SECTIONS, name),
                                           encoding="utf-8").read()
    out["__all__"] = "\n".join(out.values())
    return out


def num(n):
    """Count as the paper writes it: 3{,}080 so it never line-breaks."""
    return "{:,}".format(int(n)).replace(",", "{,}")


def check_claims():
    print("\n[4] the specific claims in the text are true of the data")
    atk = load("attack_table.csv")
    base = load("baseline_table.csv")
    tex = prose_text()

    def expect(label, needle, where="__all__"):
        """A quoted count may be written with or without the {,} grouping."""
        forms = [needle]
        try:
            plain = str(int(needle.replace("{,}", "")))
            if plain != needle:
                forms.append(plain)
        except ValueError:
            pass
        check(label, any(f in tex[where] for f in forms),
              "not in %s: %r" % (where, needle))

    # Every count the abstract or the results section quotes, rebuilt from the
    # CSVs and then required to appear in the prose. This is the check that
    # stops a rerun from leaving last week's totals in the text.
    for mode, stem in (("char", "Character"), ("word", "Word"),
                       ("saliency", "Saliency")):
        sub = atk[atk["mode"] == mode]
        if not len(sub):
            continue
        expect("%s attack total (%d flips of %d) is quoted"
               % (stem, int(sub["n_flips"].sum()), int(sub["n_attacked"].sum())),
               num(sub["n_flips"].sum()))
        expect("%s attacked count (%s) is quoted"
               % (stem, num(sub["n_attacked"].sum())),
               num(sub["n_attacked"].sum()))
    # The abstract's headline total is the two RANDOM attacks: one draw per
    # sample. The saliency attack costs a measured number of queries per sample,
    # so adding it to that total would make the count mean two different things.
    # Its own totals are quoted in section 10 and checked above.
    two = atk[atk["mode"].isin(["char", "word"])]
    expect("aggregate attacks (%s) is quoted" % num(two["n_attacked"].sum()),
           num(two["n_attacked"].sum()))
    expect("aggregate flips (%s) is quoted" % num(two["n_flips"].sum()),
           num(two["n_flips"].sum()))

    # The strongest single cell: the prose names it, so the name has to be the
    # argmax and not a memory of an earlier run.
    top = atk.loc[atk["asr"].idxmax()]
    model_tex = {"qwen2.5-3b": "Qwen", "smollm2-1.7b": "SmolLM2",
                 "phi3.5-mini": "Phi-3.5"}.get(top["model"], top["model"])
    expect("the strongest cell is named in the text", model_tex,
           "05_results" if "05_results" in tex else "__all__")
    check("the strongest single-cell ASR is above every char-level cell",
          float(top["asr"]) > float(atk[atk["mode"] == "char"]["asr"].max()),
          "%s %.3f" % (top["mode"], top["asr"]))

    # The saliency-vs-random pairing only means something if both modes exist on
    # the same messages; the generated block states both totals, so they must
    # also be consistent with the table.
    if "saliency" in set(atk["mode"]):
        s = atk[atk["mode"] == "saliency"]
        w = atk[atk["mode"] == "word"]
        check("saliency flips more messages than the random word attack",
              int(s["n_flips"].sum()) > int(w["n_flips"].sum()),
              "saliency %d vs word %d" % (int(s["n_flips"].sum()),
                                          int(w["n_flips"].sum())))

    # "N of M pairs of intervals overlap by more than half the narrower width":
    # both numbers are recomputed and then required in the text, so neither can
    # be a leftover from the two-mode version of this study.
    overlap = 0
    pairs = 0
    two_mode = atk[atk["mode"].isin(["char", "word"])]
    for (task, mode), g in two_mode.groupby(["task", "mode"]):
        rows = g.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                ov = min(a["ci95_high"], b["ci95_high"]) - max(a["ci95_low"], b["ci95_low"])
                narrow = min(a["ci95_high"] - a["ci95_low"], b["ci95_high"] - b["ci95_low"])
                pairs += 1
                if ov / narrow > 0.5:
                    overlap += 1
    check("interval-overlap count is quoted",
          ("%d of %d" % (overlap, pairs)) in tex["__all__"],
          "computed %d of %d" % (overlap, pairs))

    # "the model that catches the most also flips the least" - stated in the
    # text as a comparison, so the comparison is what is checked.
    smol = base[(base["model"] == "smollm2-1.7b")]
    check("smollm2 catch rates are 0.985 and 0.925",
          sorted(smol["recall"].tolist()) == [0.925, 0.985], str(smol["recall"].tolist()))
    smol_flips = int(atk[atk["model"] == "smollm2-1.7b"]["n_flips"].sum())
    phi_flips = int(atk[atk["model"] == "phi3.5-mini"]["n_flips"].sum())
    check("the highest catch rate does not belong to the most-flipped model",
          smol_flips <= phi_flips, "smol %d phi %d" % (smol_flips, phi_flips))

    # the paired claim in the abstract and in section 5.7
    pairs_path = os.path.join(RESULTS, "paired_tests.csv")
    if os.path.exists(pairs_path):
        pairs = pd.read_csv(pairs_path)
        wc = pairs[(pairs["a_mode"] == "word") & (pairs["b_mode"] == "char")]
        check("paired: word-only flips are 26",
              int(wc["only_a_mode"].sum()) == 26, str(wc["only_a_mode"].sum()))
        check("paired: char-only flips are 10",
              int(wc["only_b_mode"].sum()) == 10, str(wc["only_b_mode"].sum()))
        check("paired: five of six cells favour word, none favour char",
              (int((wc["a_flips"] > wc["b_flips"]).sum()),
               int((wc["a_flips"] < wc["b_flips"]).sum())) == (5, 0),
              str(((wc["a_flips"] > wc["b_flips"]).sum(),
                   (wc["a_flips"] < wc["b_flips"]).sum())))
        sig = int((wc["mcnemar_p"] < 0.05).sum())
        check("paired: exactly one cell is significant on its own", sig == 1, str(sig))
        # the abstract quotes p=0.011, recomputed from the pooled counts with
        # the same function the analysis uses (not read off the table)
        from src.analyze_pairs import mcnemar_exact
        pooled = mcnemar_exact(int(wc["only_b_mode"].sum()),
                               int(wc["only_a_mode"].sum()))
        check("paired: pooled p rounds to 0.011",
              round(pooled, 3) == 0.011, "%.4f" % pooled)

    # Self-replay: a model's own perturbations replayed must give the same flip
    # count the attack recorded. Missing is acceptable (the run did not include
    # it and the paper says so); wrong is never acceptable.
    tr_path = os.path.join(RESULTS, "transfer_matrix.csv")
    if os.path.exists(tr_path):
        tr = pd.read_csv(tr_path)
        diag = tr[(tr["subset"] == "applied") & (tr["source"] == tr["target"])]
        if len(diag):
            mism = []
            for _, r in diag.iterrows():
                m = atk[(atk["model"] == r["source"])
                        & (atk["mode"] == r["mode"])
                        & (atk["task"] == r["task"])]
                if len(m) and int(m["n_flips"].iloc[0]) != int(r["n_flips"]):
                    mism.append((r["source"], r["mode"], r["task"]))
            check("the self-replay reproduces the single-model ASR in every cell",
                  not mism, str(mism[:4]))
            # Every mode the transfer table reports must have been replayed on
            # its own diagonal; a transferred mode with no self-replay is
            # unvalidated, and the table would not say so.
            measured = set(tr[tr["subset"] == "applied"]["mode"])
            check("the self-replay covers every transferred mode",
                  set(diag["mode"]) >= measured,
                  "%s vs %s" % (set(diag["mode"]), measured))
            # A mode that was attacked but never transferred is a hole in the
            # section: it must be declared in prose, because a missing row reads
            # to a reader as "no effect" rather than "not measured".
            untransferred = sorted(set(atk["mode"]) - measured)
            if untransferred:
                body = prose_text()
                check("untransferred mode(s) %s are declared in the text"
                      % ",".join(untransferred),
                      any("Not every attack is in this table" in v for v in body.values()),
                      "looked for the 'not every attack is in this table' note")
            # The section used to describe itself as unvalidated; that sentence
            # must not survive the run that validated it.
            check("no section still calls the self-replay unrun or unvalidated",
                  all("NOT RUN" not in v for v in prose_text().values()),
                  "a section still prints NOT RUN")
            expect("the self-replay row count is quoted", num(diag["n_candidates"].sum()))

            # The discussion quotes the pooled off-diagonal transfer as a count
            # with its denominator. It is recomputed here because that sentence
            # survived a whole run: when the saliency attack joined the matrix
            # the total changed from 27 of 3,707 to 51 of 5,514 and the prose
            # kept the old pair, which no percentage check can see.
            off = tr[(tr["subset"] == "applied") & (tr["source"] != tr["target"])]
            n_off = int(off["n_flips"].sum())
            n_caught = int(off["n_target_caught"].sum())
            check("the pooled off-diagonal transfer is quoted with its denominator",
                  ("%d of %s" % (n_off, num(n_caught))) in tex["__all__"],
                  "looked for %d of %s" % (n_off, num(n_caught)))
            # And the section that owns the matrix must name every mode in it: a
            # section that still describes two attack modes next to a three-mode
            # table is the same defect as a stale total.
            mode_word = {"char": "character", "word": "word",
                         "saliency": "saliency"}
            section = (tex.get("07_transfer", "")
                       + tex.get("gen_transfer_results", "")).lower()
            missing = [m for m in sorted(measured)
                       if mode_word.get(m, m) not in section]
            check("the transfer section names every mode it tabulates",
                  not missing, "missing: %s" % missing)

    # Originals the model re-read as benign: the number of shots that were never
    # fired. Quoted in the protocol section, so it is recomputed and required.
    excluded = 0
    import glob
    for path in glob.glob(os.path.join(RESULTS, "attack_results_*.csv")):
        d = pd.read_csv(path)
        if "counted" in d.columns:
            excluded += int((d["counted"] != True).sum())   # noqa: E712
    expect("the excluded-shot count (%d) is quoted" % excluded, num(excluded))

    # The ablation's table is in the PDF only while its CSV exists, and the prose
    # is held to the same state in BOTH directions: while the cells are missing
    # the paper has to say so, and once they exist no section may still call the
    # experiment unrun. A stale "not run" sentence sitting next to a table of
    # results is exactly the failure this paper is about.
    abl_csv = os.path.join(RESULTS, "ablation_contrasts.csv")
    abl_tab = os.path.join(TABLES, "t9_ablation.tex")
    ran_ablation = os.path.exists(abl_csv) and os.path.exists(abl_tab)
    # The phrases below are the ones earlier drafts actually used, including the
    # near-misses ("not yet run", "never replayed") that a narrower list let
    # through while the table was already in the PDF.
    stale = [s for s in ("cells are not part of this version",
                         "no number in this paper is taken from it",
                         "implemented and not run",
                         "its cells are not run",
                         "not yet run",
                         "never replayed",
                         "scheduled before the saliency attack",
                         "absent from that matrix",
                         "absent from this matrix")
             if s in tex["__all__"]]
    if ran_ablation:
        check("no section still calls the ablation unrun", not stale, str(stale))
        con = pd.read_csv(abl_csv)
        # Every contrast the CSV holds must reach the section that discusses it:
        # a contrast computed and never written up is a number the paper cannot
        # be asked about.
        gen = os.path.join(SECTIONS, "gen_ablation_results.tex")
        gtext = open(gen, encoding="utf-8").read() if os.path.exists(gen) else ""
        # One sentence and one exact McNemar p-value per contrast: if a contrast
        # silently stopped reaching the prose, the counts stop matching.
        n_contrasts = int(con["contrast"].nunique())
        check("the ablation prose reports every contrast it tabulates",
              gtext.count("$p=") >= n_contrasts,
              "%d contrasts but %d p-values in the prose"
              % (n_contrasts, gtext.count("$p=")))
        # Section 6 explains the one contrast whose sign is a budget effect:
        # TextFooler's ranking is paid for out of the same 24-call cap, so the
        # paragraph quotes both arms' realised query counts. They are recomputed
        # from the per-arm table and required in the text, so the explanation
        # cannot drift away from the data it explains.
        arm_path = os.path.join(RESULTS, "ablation_table.csv")
        if os.path.exists(arm_path):
            arm = pd.read_csv(arm_path)
            def pooled_queries(name):
                s = arm[arm["arm"] == name]
                if not len(s) or not s["n_attacked"].sum():
                    return None
                return float((s["mean_queries"] * s["n_attacked"]).sum()
                             / s["n_attacked"].sum())
            tf = pooled_queries("textfooler-prob")
            sal = pooled_queries("saliency-prob")
            if tf is not None and sal is not None:
                body = tex.get("06_saliency", "")
                check("the budget explanation quotes the realised query counts",
                      ("%.1f of the 24" % tf) in body and ("%.1f" % sal) in body,
                      "looked for %.1f and %.1f in 06_saliency" % (tf, sal))
    else:
        check("the missing ablation is declared, not silently omitted",
              bool(stale), "no table and nothing in the text says why")

    # The defences quote clean TPR at their operating point; if the threshold
    # sits at 1.0 the filter rejects everything and the row is not a defence.
    thr_path = os.path.join(RESULTS, "defense_thresholds.csv")
    if os.path.exists(thr_path):
        thr = pd.read_csv(thr_path)
        # tau = 1.0 is legitimate ONLY when the row says the budget could not be
        # met (the model flags so much clean mail that holding the FPR down
        # means rejecting everything). Silently using that threshold as a
        # defence is the bug this check exists for.
        corner = thr[thr["tau"] >= 1.0]
        check("every reject-all threshold is reported as budget-not-met",
              bool((corner["budget_met"] == False).all()),   # noqa: E712
              str(corner[["model", "task", "tau", "budget_met"]].to_dict("records")))
        check("the fitted thresholds are otherwise usable",
              int((thr["tau"] < 1.0).sum()) >= 1,
              str(thr[["model", "task", "tau"]].to_dict("records")))


GATED = (("t5_transfer.tex", "transfer_matrix.csv"),
         ("t6_defense.tex", "defense_summary.csv"),
         ("t7_saliency.tex", None),
         ("t9_ablation.tex", "ablation_contrasts.csv"))


def check_gating():
    """A generated table may exist only while its CSV does.

    The three results sections after the random attacks (saliency, transfer,
    defence) are included through \IfFileExists, so a table left over from an
    earlier run would silently put a deleted result back into the PDF. This
    check is what makes 'delete the CSV and the section disappears' true.
    """
    print("\n[5] gated sections match the data that exists")
    # The gate may live in main.tex (a whole section) or in the section that
    # owns the table (a table inside an already-gated section). Searching every
    # source for it is what keeps this check honest as sections nest.
    sources = {}
    for path in [os.path.join(PAPER, "main.tex")] + sorted(
            os.path.join(PAPER, "sections", f)
            for f in os.listdir(os.path.join(PAPER, "sections"))
            if f.endswith(".tex")):
        with open(path, encoding="utf-8") as fh:
            sources[os.path.basename(path)] = fh.read()
    for table, csv in GATED:
        gate = "\\IfFileExists{tables/%s}" % table
        holders = [n for n, body in sources.items() if gate in body]
        check("%s is gated" % table, bool(holders),
              r"no \IfFileExists{tables/%s} line anywhere" % (table,))
        table_exists = os.path.exists(os.path.join(TABLES, table))
        if csv is None:
            ok = True
            detail = ""
            if table_exists:
                atk = load("attack_table.csv")
                ok = "saliency" in set(atk["mode"])
                detail = "saliency table without saliency rows"
        else:
            ok = table_exists == os.path.exists(os.path.join(RESULTS, csv))
            detail = ("%s exists=%s but %s exists=%s"
                      % (table, table_exists, csv,
                         os.path.exists(os.path.join(RESULTS, csv))))
        check("%s is consistent with its CSV" % table, ok, detail)


def check_citations():
    """Bibliography and cross-references resolve, and every float is pointed at.

    Three silent failures in LaTeX, all of which survive a clean compile:
    a \\cite key that is not in refs.bib (prints [?]), a \\ref to a label
    nobody defined (prints ??), and a float no sentence refers to (the reader
    meets a table with no reason to read it). The bibliography is also checked
    against BIB_VERIFICATION.md, because an entry with no row there was written
    from memory rather than read on a page.
    """
    print("\n[6] citations, labels and floats")
    section_texts = {}
    for name in sorted(os.listdir(SECTIONS)):
        if name.endswith(".tex"):
            section_texts[name] = open(os.path.join(SECTIONS, name),
                                       encoding="utf-8").read()
    table_texts = {}
    for name in sorted(os.listdir(TABLES)):
        if name.endswith(".tex"):
            table_texts[name] = open(os.path.join(TABLES, name),
                                     encoding="utf-8").read()
    figure_texts = {}
    FIGURES = os.path.join(PAPER, "figures")
    if os.path.exists(FIGURES):
        for name in sorted(os.listdir(FIGURES)):
            if name.endswith(".tex"):
                figure_texts[name] = open(os.path.join(FIGURES, name),
                                          encoding="utf-8").read()
    everything = "\n".join(list(section_texts.values()) + list(table_texts.values()) + list(figure_texts.values()))

    bib_path = os.path.join(PAPER, "refs.bib")
    bib = open(bib_path, encoding="utf-8").read() if os.path.exists(bib_path) else ""
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set()
    for text in section_texts.values():
        for m in re.finditer(r"\\cite\{([^}]+)\}", text):
            cited |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    check("every cited key resolves in refs.bib", cited <= keys, str(sorted(cited - keys)))
    check("no bibliography entry is uncited", keys <= cited, str(sorted(keys - cited)))

    ver_path = os.path.join(PAPER, "BIB_VERIFICATION.md")
    ver = open(ver_path, encoding="utf-8").read() if os.path.exists(ver_path) else ""
    unverified = sorted(k for k in keys if "`%s`" % k not in ver)
    check("every entry has a BIB_VERIFICATION row", not unverified, str(unverified))

    labels = set(re.findall(r"\\label\{([^}]+)\}", everything))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", everything))
    check("every \\ref has a matching \\label", refs <= labels, str(sorted(refs - labels)))
    unused = sorted(l for l in labels - refs if l.startswith(("tab:", "fig:")))
    check("every table and figure is referred to in the text", not unused, str(unused))


def main():
    print("PAPER CHECK")
    check_tables_vs_csv()
    check_prose_numbers()
    check_inputs_exist()
    check_claims()
    check_gating()
    check_citations()
    print("\nRESULT: %s" % ("FAIL: %d problem(s)" % len(FAILS) if FAILS else "OK"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
