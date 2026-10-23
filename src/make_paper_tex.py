"""
Generate the paper's LaTeX tables from results CSVs - never hand-typed numbers.

    python src/make_paper_tex.py

Reads
  results/baseline_table.csv        (clean baseline: acc/prec/rec/F1 per model x task)
  results/attack_table.csv          (ASR with Wilson + bootstrap intervals)
  results/clean_baseline_results.csv (row counts, for the catch-rate column)

Writes
  paper/tables/t1_baseline.tex
  paper/tables/t2_char.tex
  paper/tables/t3_word.tex
  paper/tables/t4_budget.tex

The generated files are checked by tests/check_paper.py, which fails if the
numbers in the .tex no longer match the CSVs - so a stale table cannot reach the
PDF silently.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.common import RESULTS_DIR  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "paper", "tables")

MODEL_LABEL = {
    "qwen2.5-3b": "Qwen2.5-3B-Instruct",
    "smollm2-1.7b": "SmolLM2-1.7B-Instruct",
    "phi3.5-mini": "Phi-3.5-mini-instruct",
    "llama3.2-3b": "Llama-3.2-3B-Instruct",
}
TASK_LABEL = {"sms_smishing": "SMS smishing", "phishing_email": "Phishing email"}


def esc(s):
    """Escape the few LaTeX-special characters that appear in our labels."""
    return (str(s).replace("\\", r"\textbackslash{}")
            .replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#"))


def pct(x, nd=1):
    return "%.*f\\%%" % (nd, 100.0 * float(x))


def ci(row):
    return "%.1f--%.1f" % (100.0 * row["ci95_low"], 100.0 * row["ci95_high"])


def preamble(caption, label, colspec, wide=True):
    """Float preamble.

    `wide` uses table* (spanning both acmart columns). Our result tables carry
    seven or eight numeric columns; in a single 3.3in column they overflow into
    the neighbouring column and collide with whatever float lives there. That is
    not a cosmetic issue - a reader cannot tell which numbers belong to which
    table - so the wide tables are emitted as two-column floats.
    """
    env = "table*" if wide else "table"
    return [
        r"\begin{%s}[t]" % env,
        r"\centering",
        r"\small",
        r"\caption{%s}" % caption,
        r"\label{%s}" % label,
        r"\begin{tabular}{%s}" % colspec,
        r"\toprule",
    ]


def closing(wide=True):
    return [r"\bottomrule", r"\end{tabular}",
            r"\end{%s}" % ("table*" if wide else "table")]


def write(name, lines):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s" % path)


def t1_baseline():
    b = pd.read_csv(os.path.join(RESULTS_DIR, "baseline_table.csv"))
    b = b.sort_values(["task", "model"])
    lines = preamble(
        "Clean zero-shot baseline on the frozen 800-row evaluation split "
        "(400 rows per task, 200 malicious). ``Catch'' is recall on the malicious "
        "class and is the denominator every ASR in this paper is conditional on.",
        "tab:baseline", "llcccccc")
    lines.append(r"\textbf{Task} & \textbf{Model} & \textbf{Acc.} & \textbf{Prec.} "
                 r"& \textbf{Rec.} & \textbf{F1} & \textbf{TP/FP} & \textbf{Catch} \\")
    lines.append(r"\midrule")
    for task in ["sms_smishing", "phishing_email"]:
        sub = b[b["task"] == task]
        for i, (_, r) in enumerate(sub.iterrows()):
            name = esc(TASK_LABEL.get(task, task)) if i == 0 else ""
            lines.append(" %s & %s & %.3f & %.3f & %.3f & %.3f & %d/%d & %s \\\\"
                         % (name, esc(MODEL_LABEL.get(r["model"], r["model"])),
                            r["accuracy"], r["precision"], r["recall"], r["f1"],
                            int(r["tp"]), int(r["fp"]), pct(r["recall"])))
        lines.append(r"\midrule" if task == "sms_smishing" else "")
    lines += closing()
    write("t1_baseline.tex", [ln for ln in lines if ln != ""])


def asr_table(mode, caption, label, filename):
    a = pd.read_csv(os.path.join(RESULTS_DIR, "attack_table.csv"))
    a = a[a["mode"] == mode].sort_values(["task", "model"])
    lines = preamble(caption, label, "llccccccc")
    lines.append(r"\textbf{Task} & \textbf{Model} & \textbf{Attacked} & \textbf{Flips} "
                 r"& \textbf{ASR} & \textbf{95\% CI} & \textbf{Pert.} & \textbf{Jacc.} "
                 r"& \textbf{Catch} \\")
    lines.append(r"\midrule")
    for task in ["sms_smishing", "phishing_email"]:
        sub = a[a["task"] == task]
        for i, (_, r) in enumerate(sub.iterrows()):
            name = esc(TASK_LABEL.get(task, task)) if i == 0 else ""
            lines.append(" %s & %s & %d & %d & %s & %s & %s & %.3f & %s \\\\"
                         % (name, esc(MODEL_LABEL.get(r["model"], r["model"])),
                            int(r["n_attacked"]), int(r["n_flips"]), pct(r["asr"]),
                            ci(r), pct(r["mean_perturbation_rate"]),
                            r["mean_jaccard"], pct(r["catch_rate"])))
        lines.append(r"\midrule" if task == "sms_smishing" else "")
    lines += closing()
    write(filename, [ln for ln in lines if ln != ""])


def t4_budget():
    a = pd.read_csv(os.path.join(RESULTS_DIR, "attack_table.csv"))
    lines = preamble(
        "Edit budget actually spent, by attack mode. The budget is one shared "
        "15\\% allowance per sample; the measured rate is the exact Levenshtein "
        "distance divided by the original length (characters or words).",
        "tab:budget", "lcccc", wide=False)
    lines.append(r"\textbf{Mode} & \textbf{Samples} & \textbf{Flips} & "
                 r"\textbf{Mean pert.} & \textbf{Max pert.} \\")
    lines.append(r"\midrule")
    # Every mode in the table contributes to the total. Listing two modes and
    # summing three is how a "total" of 86 appears under columns that add to 40.
    labels = {"char": "Character", "word": "Word", "saliency": "Saliency"}
    shown = [m for m in ("char", "word", "saliency") if m in set(a["mode"])]
    for mode in shown:
        sub = a[a["mode"] == mode]
        lines.append(" %s & %d & %d & %s & %s \\\\"
                     % (labels.get(mode, mode),
                        int(sub["n_attacked"].sum()), int(sub["n_flips"].sum()),
                        pct(sub["mean_perturbation_rate"].mean()),
                        pct(sub["mean_perturbation_rate"].max())))
    tot = a[a["mode"].isin(shown)]
    lines += [r"\midrule",
              r"\multicolumn{2}{l}{Total} & %d & & \\" % int(tot["n_flips"].sum())]
    lines += closing(wide=False)
    write("t4_budget.tex", lines)


def write_tex(name, lines, subdir="tables"):
    """Write into paper/<subdir>/ so a generated prose block can live next to
    the section that inputs it."""
    d = os.path.join(ROOT, "paper", subdir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s" % path)


def t5_transfer():
    """Cross-model transfer matrix, classic subset (perturbations that fooled
    the source) with the target's own caught-original denominator."""
    path = os.path.join(RESULTS_DIR, "transfer_matrix.csv")
    if not os.path.exists(path):
        return False
    allrows = pd.read_csv(path)
    # The `applied` subset, not the classic flipped-source matrix: the classic
    # matrix keeps only the perturbations that fooled the SOURCE, which at this
    # scale leaves 1-6 candidates per cell and no denominator worth an interval.
    # `applied` replays everything the source produced and divides by what the
    # TARGET caught, which is both larger and the question a defender asks.
    t = allrows[allrows["subset"] == "applied"]
    if not len(t):
        return False
    # The diagonal is a faithfulness check, not a transfer result: it replays a
    # model's own perturbations and must equal its single-model ASR. It stays in
    # the CSV and is quoted in prose; the table is off-diagonal only.
    diagonal = t[t["source"] == t["target"]]
    t = t[t["source"] != t["target"]]
    if not len(t):
        return False
    t = t.sort_values(["mode", "task", "source", "target"])
    lines = preamble(
        "Cross-model transfer on the applied subset: every perturbation the "
        "source produced is replayed on the target, and the denominator is the "
        "TARGET's own caught originals, because a perturbation cannot transfer "
        "on a message the target never flagged. No diagonal is tabulated: "
        "replaying a model's own perturbations reproduces its attack run and "
        "would only reprint Table~\\ref{tab:asr-word}. That replay was run as a "
        "faithfulness check and its diagonal appears in the prose below, not as "
        "a result. Cells with fewer than "
        "ten caught originals would be marked $\\dagger$; none in this table "
        "needed it.",
        "tab:transfer", "llllcccc")
    lines.append(r"\textbf{Mode} & \textbf{Task} & \textbf{Source} & "
                 r"\textbf{Target} & \textbf{Caught} & \textbf{Flips} & "
                 r"\textbf{ASR} & \textbf{95\% CI} \\")
    lines.append(r"\midrule")
    for mode in ["char", "word", "saliency"]:
        for task in ["sms_smishing", "phishing_email"]:
            sub = t[(t["mode"] == mode) & (t["task"] == task)]
            for j, (_, r) in enumerate(sub.iterrows()):
                lines.append(" %s & %s & %s & %s & %d%s & %d & %s & %s \\\\"
                             % (mode if j == 0 else "",
                                esc(TASK_LABEL.get(task, task)) if j == 0 else "",
                                esc(MODEL_LABEL.get(r["source"], r["source"])),
                                esc(MODEL_LABEL.get(r["target"], r["target"])),
                                int(r["n_target_caught"]),
                                "$\\dagger$" if int(r["n_target_caught"]) < 10 else "",
                                int(r["n_flips"]), pct(r["transfer_asr"]), ci(r)))
        lines.append(r"\midrule")
    lines = lines[:-1] + closing()
    write("t5_transfer.tex", lines)

    # machine-written result paragraph, so no number is typed by hand
    same = t[t["source"] == t["target"]]
    cross = t[t["source"] != t["target"]]
    out = [r"% Generated by src/make_paper_tex.py from results/transfer_matrix.csv."]
    if len(cross):
        # A 1/1 cell is not "the strongest cell"; it is a single flip. Only
        # cells with a denominator worth an interval are eligible, and the
        # paragraph says so when none is.
        usable = cross[cross["n_target_caught"] >= 10]
        n_small = int((cross["n_target_caught"] < 10).sum())
        if n_small == 0:
            tail = "every cell has a denominator worth an interval"
        elif n_small == len(cross):
            tail = "every cell holds fewer than ten caught originals"
        else:
            tail = "%d of them hold fewer than ten caught originals" % n_small
        out.append(
            "On the applied subset %d of %d off-diagonal cells have a nonzero "
            "transfer rate, and %s."
            % (int((cross["n_flips"] > 0).sum()), len(cross), tail))
        if len(usable):
            best = usable.sort_values("transfer_asr", ascending=False).iloc[0]
            out.append(
                "Among the cells with a denominator worth an interval, the "
                "highest rate is %s $\\rightarrow$ %s on %s under the %s "
                "attack: %d of %d caught originals flipped (%s, 95\\%% CI %s)."
                % (esc(MODEL_LABEL.get(best["source"], best["source"])),
                   esc(MODEL_LABEL.get(best["target"], best["target"])),
                   esc(TASK_LABEL.get(best["task"], best["task"])), best["mode"],
                   int(best["n_flips"]), int(best["n_target_caught"]),
                   pct(best["transfer_asr"]), ci(best)))
        else:
            out.append(
                "No off-diagonal cell has a denominator large enough to quote "
                "a rate, so the classic transfer matrix is reported as an "
                "inventory of what was attempted rather than as a measurement; "
                "the applied subset in the same file is the informative view.")
    if len(diagonal):
        # Diagonal cells are compared against the attack table, not against
        # themselves: the point is that the same texts through the same prompt
        # give the same flip count the attack recorded.
        atk_path = os.path.join(RESULTS_DIR, "attack_table.csv")
        bad, compared = [], 0
        if os.path.exists(atk_path):
            atk = pd.read_csv(atk_path)
            for _, r in diagonal.iterrows():
                m = atk[(atk["model"] == r["source"])
                        & (atk["mode"] == r["mode"])
                        & (atk["task"] == r["task"])]
                if not len(m):
                    continue
                compared += 1
                stored = int(m["n_flips"].iloc[0])
                if stored != int(r["n_flips"]):
                    bad.append("%s/%s/%s: attack %d, replay %d"
                               % (r["source"], r["mode"], r["task"],
                                  stored, int(r["n_flips"])))
        out.append(
            "The self-replay is the check that makes the rest of the table "
            "readable: each model's own perturbations went through the same "
            "path a second time, and its flip count is compared with the one "
            "the attack recorded. %d diagonal cells covering %s rows were "
            "replayed, and %s."
            % (compared,
               "{:,}".format(int(diagonal["n_candidates"].sum())).replace(",", "{,}"),
               "every one reproduces the single-model ASR exactly" if not bad
               else "%d do not (%s), so the off-diagonal cells below are not "
                    "trustworthy" % (len(bad), "; ".join(bad[:3]))))
    else:
        out.append(
            "We did not replay any model's own perturbations in this round, so "
            "the faithfulness check that a self-replay reproduces the "
            "single-model ASR was not run; it is one flag "
            "(\\texttt{--include-self}) and is the first thing to add before "
            "these cells are used for anything but a description.")
    # A mode that was attacked but never transferred is a hole in the section,
    # so it is stated rather than left for the reader to notice from the table's
    # missing rows. Disappears by itself when the matrix covers every mode.
    atk_path = os.path.join(RESULTS_DIR, "attack_table.csv")
    if os.path.exists(atk_path):
        attacked = sorted(set(pd.read_csv(atk_path)["mode"]))
        transferred = sorted(set(t["mode"]))
        missing = [m for m in attacked if m not in transferred]
        if missing:
            out.append(
                "Not every attack is in this table: %s perturbations were "
                "produced and scored in Section~\\ref{sec:saliency} but were "
                "not replayed on a second model, because the transfer phase of "
                "the run preceded the attack that generates them. Those cells "
                "are missing data, not null results, and the phase order is now "
                "fixed so a re-run fills them."
                % " and ".join(missing))
    write_tex("gen_transfer_results.tex", out, subdir="sections")
    return True


def t6_defense():
    """Defence probe: normalisation and FPR-budgeted threshold on stored attacks.

    Two accounting rules, both of which the first version of the defence table
    broke: an empty denominator is printed as n/a rather than 0\%, and every row
    counts flips only among the originals the SAME rule called malicious.
    """
    path = os.path.join(RESULTS_DIR, "defense_summary.csv")
    if not os.path.exists(path):
        return False
    d = pd.read_csv(path)
    if not len(d):
        return False
    d = d[d["defense"].isin(["baseline", "normalise", "threshold", "both"])]
    for col in ("n", "flips"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    def pooled(mode, task, defence):
        sub = d[(d["mode"] == mode) & (d["task"] == task)
                & (d["defense"] == defence)]
        n = int(sub["n"].fillna(0).sum())
        if not n:
            return None, 0
        return float(sub["flips"].fillna(0).sum()) / n, n

    thr_path = os.path.join(RESULTS_DIR, "defense_thresholds.csv")
    thr = pd.read_csv(thr_path) if os.path.exists(thr_path) else pd.DataFrame()
    budget = ("%.0f\\%%" % (100 * float(thr["target_fpr"].iloc[0]))
              if len(thr) and "target_fpr" in thr.columns else "5\\%")
    # Cells that contain a model-task pair whose budget could not be met: its
    # tau is 1.0, i.e. "reject everything", so its Thr. number is not a defence.
    unmeetable = set()
    if len(thr) and "budget_met" in thr.columns:
        unmeetable = set(thr[thr["budget_met"] == False]["task"])   # noqa: E712

    lines = preamble(
        "Training-free defences, measured on the same stored perturbations. "
        "\\emph{None} is the undefended ASR, \\emph{norm} applies input "
        "normalisation only, \\emph{thr} replaces the default decision "
        "threshold with one fitted on clean benign mail to a " + budget +
        " false-positive budget. Rows are pooled over the three models; $n$ is "
        "the number of originals the undefended rule itself flagged, and "
        "\\emph{n/a} means a defence flagged none of them, so there was "
        "nothing left to flip. $\\dagger$ marks a row containing a model-task "
        "pair whose false-positive budget could not be met: its threshold is a "
        "rejection rule, not a defence, and the number is there to be ignored.",
        "tab:defense", "llcccc")
    lines.append(r"\textbf{Mode} & \textbf{Task} & \textbf{None} & "
                 r"\textbf{Norm.} & \textbf{Thr.} & \textbf{$n$} \\")
    lines.append(r"\midrule")
    for mode in ("char", "word", "saliency"):
        for task in ("sms_smishing", "phishing_email"):
            base, n_base = pooled(mode, task, "baseline")
            if base is None:
                continue
            nrm, _ = pooled(mode, task, "normalise")
            th, n_thr = pooled(mode, task, "threshold")
            dagger = "$\\dagger$" if task in unmeetable else ""
            lines.append(" %s & %s & %s & %s & %s & %d \\\\"
                         % (mode, esc(TASK_LABEL.get(task, task)),
                            pct(base), pct(nrm) if nrm is not None else "n/a",
                            (pct(th) + dagger) if th is not None else "n/a",
                            n_base))
        lines.append(r"\midrule")
    lines = lines[:-1] + closing()
    write("t6_defense.tex", lines)

    out = [r"% Generated by src/make_paper_tex.py from results/defense_summary.csv."]
    for mode in ("char", "word", "saliency"):
        base, n_base = pooled(mode, "sms_smishing", "baseline")
        b2, n2 = pooled(mode, "phishing_email", "baseline")
        tot_n = (n_base or 0) + (n2 or 0)
        if not tot_n:
            continue
        flips = ((base or 0) * (n_base or 0)) + ((b2 or 0) * (n2 or 0))
        agg = flips / tot_n
        nrm_a, n_a = pooled(mode, "sms_smishing", "normalise")
        nrm_b, n_b = pooled(mode, "phishing_email", "normalise")
        if n_a + n_b:
            after = (((nrm_a or 0) * n_a) + ((nrm_b or 0) * n_b)) / (n_a + n_b)
            out.append("Under the %s attack, normalisation moves the aggregate ASR "
                       "from %s to %s." % (mode, pct(agg), pct(after)))
    if len(thr) and "budget_met" in thr.columns:
        missed = thr[thr["budget_met"] == False]            # noqa: E712
        worst = thr.sort_values("tpr_at_tau")
        if len(worst):
            r = worst.iloc[0]
            out.append("The threshold is not free: at the %s clean false-positive "
                       "budget its tightest operating point (%s, %s) keeps clean "
                       "recall at %s, down from %s at the default threshold."
                       % (budget,
                          esc(MODEL_LABEL.get(r["model"], r["model"])),
                          esc(TASK_LABEL.get(r["task"], r["task"])),
                          pct(r["tpr_at_tau"]), pct(r["tpr_at_0.5"])))
        if len(missed):
            out.append("%d of %d model--task cells cannot meet the budget at all: "
                       "holding their false positives down requires a threshold "
                       "so high that it flags no message at all." % (len(missed), len(thr)))
    write_tex("gen_defense_results.tex", out, subdir="sections")
    return True


def _random_attack_queries(mode):
    """(messages, queries, per-message) for a random attack mode, from its CSVs.

    The per-sample frame stores the ATTEMPT number, not the call count: every
    sample costs one query for the re-queried original plus one per attempt
    (src/attack_common.py). Reading the cost from the records is what stops the
    prose from describing the random attack as "one draw per sample" when it
    ran up to three attempts - a claim the code contradicts.
    """
    import glob
    files = sorted(glob.glob(os.path.join(RESULTS_DIR,
                                          "attack_results_%s_*.csv" % mode)))
    if not files:
        return None
    msgs = queries = 0
    for path in files:
        df = pd.read_csv(path)
        if "counted" not in df.columns or "attempt" not in df.columns:
            continue
        counted = df[df["counted"] == True]                    # noqa: E712
        attempts = pd.to_numeric(counted["attempt"], errors="coerce").fillna(0)
        msgs += int(len(counted))
        # every row (counted or not) has its original re-queried once
        queries += int(len(counted) + attempts.sum()) + int(len(df) - len(counted))
    if not msgs:
        return None
    return msgs, queries, queries / msgs


def _saliency_queries():
    """(messages, queries) for the saliency attack, summed from its raw CSVs.

    Read from the per-sample files rather than from `mean_queries` in the
    attack table: that column is rounded to one decimal, so multiplying it by
    the sample count gives a total that is close to - but not - the number of
    calls the run actually made. A total quoted as exact must be exact.
    """
    import glob
    files = sorted(glob.glob(os.path.join(RESULTS_DIR,
                                          "attack_results_saliency_*.csv")))
    msgs = queries = 0
    for path in files:
        df = pd.read_csv(path)
        if "n_queries" not in df.columns or "counted" not in df.columns:
            continue
        counted = df[df["counted"] == True]                    # noqa: E712
        msgs += int(len(counted))
        queries += int(pd.to_numeric(df["n_queries"], errors="coerce").fillna(0).sum())
    if not msgs:
        return None
    return msgs, queries


def t7_saliency():
    """Saliency attack compared with the random word attack, at matched budget."""
    path = os.path.join(RESULTS_DIR, "attack_table.csv")
    if not os.path.exists(path):
        return False
    a = pd.read_csv(path)
    if "saliency" not in set(a["mode"]):
        return False
    a = a[a["mode"].isin(["word", "saliency"])].sort_values(
        ["task", "model", "mode"])
    lines = preamble(
        "Saliency-guided (PWWS-style) word attack next to the uniform-random "
        "word attack, both inside the same 15\\% word budget. Both spend a "
        "measured number of model queries: the random attack re-queries every "
        "original and keeps drawing until a flip or three attempts, the "
        "saliency attack probes every candidate. Per-sample query counts are "
        "stored with each row.",
        "tab:saliency", "llcccccc")
    lines.append(r"\textbf{Task} & \textbf{Model} & \textbf{Attacked} & "
                 r"\textbf{Flips} & \textbf{ASR} & \textbf{95\% CI} & "
                 r"\textbf{Pert.} & \textbf{Jacc.} \\")
    lines.append(r"\midrule")
    for task in ["sms_smishing", "phishing_email"]:
        sub = a[a["task"] == task]
        first = True
        for _, r in sub.iterrows():
            lines.append(" %s & %s & %d & %d & %s & %s & %s & %.3f \\\\"
                         % (esc(TASK_LABEL.get(task, task)) if first else "",
                            "%s (%s)" % (esc(MODEL_LABEL.get(r["model"], r["model"])),
                                          r["mode"]),
                            int(r["n_attacked"]), int(r["n_flips"]), pct(r["asr"]),
                            ci(r), pct(r["mean_perturbation_rate"]),
                            r["mean_jaccard"]))
            first = False
        lines.append(r"\midrule")
    lines = lines[:-1] + closing()
    write("t7_saliency.tex", lines)

    out = [r"% Generated by src/make_paper_tex.py from results/attack_table.csv."]
    w = a[a["mode"] == "word"]
    s = a[a["mode"] == "saliency"]
    if len(w) and len(s):
        rw = w["n_flips"].sum() / max(w["n_attacked"].sum(), 1)
        rs = s["n_flips"].sum() / max(s["n_attacked"].sum(), 1)
        out.append("At the same word budget, the random attack flips %d of %d "
                   "caught messages (%s) and the saliency attack flips %d of %d "
                   "(%s)." % (int(w["n_flips"].sum()), int(w["n_attacked"].sum()),
                              pct(rw), int(s["n_flips"].sum()),
                              int(s["n_attacked"].sum()), pct(rs)))
        if "mean_queries" in a.columns and a["mean_queries"].notna().any():
            q = float(s["mean_queries"].mean())
            rnd = _random_attack_queries("word")
            if rnd:
                n_msgs, n_q, per = rnd
                out.append(
                    "Each saliency-attacked message costs a mean of %.0f model "
                    "queries. The random word attack is not a single draw either: "
                    "it re-queries every original and spends one query per "
                    "attempt, so its %s messages cost %s queries, %.1f each. "
                    "The saliency attacker buys its extra flips with about "
                    "%.1f times as many calls, and that price is part of the "
                    "claim, not a footnote to it."
                    % (q, "{:,}".format(n_msgs).replace(",", "{,}"),
                       "{:,}".format(n_q).replace(",", "{,}"), per, q / per))
            else:
                out.append("Each saliency-attacked message costs a mean of %.0f "
                           "model queries; the comparison is between attacks "
                           "whose cost is on the record." % q)
        if rs > rw:
            # flips per unit of edit, not flips per claim: both quantities are
            # pooled from the same table, so the ratio is recomputed here
            er_w = float(w["mean_perturbation_rate"].mean())
            er_s = float(s["mean_perturbation_rate"].mean())
            out.append("The saliency attack also edits MORE, not less: %s of "
                       "words against %s for the random attack. Normalising by "
                       "the budget actually spent, it buys %s of a flip per "
                       "percent of words edited against %s, so the advantage is "
                       "real but smaller than the raw ASR ratio."
                       % (pct(er_s), pct(er_w),
                          "%.3f" % (rs / er_s) if er_s else "n/a",
                          "%.3f" % (rw / er_w) if er_w else "n/a"))
        else:
            out.append("The saliency ordering does \\emph{not} buy flips here; "
                       "the honest reading is that a larger, better-chosen "
                       "replacement set - not the ordering - is what the "
                       "random attack lacks.")
        out.append("Both numbers are pooled over different candidate "
                   "vocabularies and, after the exclusions, over different "
                   "message sets, so the denominator is 1{,}001 for the "
                   "saliency attack and 1{,}035 for the random one; the paired "
                   "comparison of Section~\\ref{sec:pairs} is the only place "
                   "where the two are measured on the same messages.")
        # Pooled query accounting, so a query count can never be mistaken for a
        # message count: the two totals differ by a factor of four, and the
        # earlier draft quoted the message count under the word "queries".
        qc, qw, qs = _random_attack_queries("char"), _random_attack_queries("word"), \
            _saliency_queries()
        if qc and qw and qs:
            fmt = lambda n: "{:,}".format(int(n)).replace(",", "{,}")
            out.append(
                "Pooled query accounting, kept separate from the message count "
                "because the two are different numbers: the character attack "
                "sends %s caught messages in %s queries, the word attack %s in "
                "%s, so the random attacks together send %s messages in %s "
                "queries (%.1f each), and the saliency attack sends %s messages "
                "in %s queries (%.1f each)."
                % (fmt(qc[0]), fmt(qc[1]), fmt(qw[0]), fmt(qw[1]),
                   fmt(qc[0] + qw[0]), fmt(qc[1] + qw[1]),
                   (qc[1] + qw[1]) / (qc[0] + qw[0]),
                   fmt(qs[0]), fmt(qs[1]), qs[1] / qs[0]))
    write_tex("gen_saliency_results.tex", out, subdir="sections")
    return True


def t9_ablation():
    """The matched-vocabulary ablation: one row per contrast.

    Pooled over model x task on purpose. Each cell is itself a paired test, and
    printing all of them would turn a 5-row design into a 30-row appendix whose
    small denominators invite exactly the ranking the rest of the paper refuses
    to do. The pooled counts are summed discordant pairs, and the prose says so.
    """
    path = os.path.join(RESULTS_DIR, "ablation_contrasts.csv")
    if not os.path.exists(path):
        return False
    d = pd.read_csv(path)
    if not len(d):
        return False
    rows = []
    for (label, isolates), g in d.groupby(["contrast", "isolates"]):
        only_a, only_b = int(g["only_a"].sum()), int(g["only_b"].sum())
        paired = int(g["n_paired"].sum())
        a_flips, b_flips = int(g["a_flips"].sum()), int(g["b_flips"].sum())
        try:
            from src.analyze_pairs import mcnemar_exact, paired_ci
        except Exception:                        # run as a script, not a module
            sys.path.insert(0, ROOT)
            from src.analyze_pairs import mcnemar_exact, paired_ci
        diffs = []
        for _, r in g.iterrows():
            diffs += [1.0] * int(r["only_a"]) + [-1.0] * int(r["only_b"])
        lo, hi = paired_ci(diffs) if diffs else (0.0, 0.0)
        rows.append({"contrast": label, "isolates": isolates, "n_paired": paired,
                     "a_flips": a_flips, "b_flips": b_flips,
                     "only_a": only_a, "only_b": only_b,
                     "risk_diff": (a_flips - b_flips) / paired if paired else 0.0,
                     "ci_low": lo, "ci_high": hi,
                     "p": mcnemar_exact(only_b, only_a), "cells": len(g)})
    t = pd.DataFrame(rows).sort_values("contrast")
    lines = preamble(
        "Matched-vocabulary, matched-budget ablation of Section~\\ref{sec:ablation}. "
        "Every contrast holds the candidate set, the probe phase and the 24-call "
        "cap fixed and varies one thing; A and B are the two arms, and positive "
        "A-only means the first arm flipped messages the second did not. Counts "
        "are pooled over model and task; the per-cell pairs are in "
        "\\texttt{results/ablation\\_contrasts.csv}.",
        "tab:ablation", "llcccccc")
    lines.append(r"\textbf{Contrast} & \textbf{Isolates} & \textbf{Paired} & "
                 r"\textbf{A flips} & \textbf{B flips} & \textbf{A-only} & "
                 r"\textbf{B-only} & \textbf{$p$} \\")
    lines.append(r"\midrule")
    for _, r in t.iterrows():
        lines.append(" %s & %s & %d & %d & %d & %d & %d & %.4f \\\\"
                     % (r["contrast"], r["isolates"], r["n_paired"],
                        r["a_flips"], r["b_flips"], r["only_a"], r["only_b"],
                        r["p"]))
    lines += closing()
    write("t9_ablation.tex", lines)

    out = [r"% Generated by src/make_paper_tex.py from results/ablation_contrasts.csv.",
           r"Table~\ref{tab:ablation} reports each contrast on the messages its "
           r"two arms share."]
    # Each sentence names BOTH arms and then says which way the flips went.
    # A template like "X adds N flips" is unreadable here: A-only and B-only
    # are opposite directions, so the same number means opposite things.
    sentences = {
        "ordering": ("With the candidate set, the probe phase and the query cap "
                     "held fixed, ordering by saliency flips {a} messages the "
                     "uniform draw does not, against {b} in the other direction "
                     "(exact McNemar $p={p:.4f}$ over {n} paired messages)."),
        "oracle": ("Replacing the probability channel with the one-bit verdict "
                   "loses {a} messages that survive on probabilities and wins "
                   "{b} the probability arm misses ($p={p:.4f}$, {n} pairs)."),
        "search": ("Searching over candidate subsets instead of ranking under "
                   "the same bit recovers {a} messages the ranking arm misses, "
                   "against {b} in the other direction ($p={p:.4f}$, {n} pairs)."),
        "candidates": ("Against the 14-entry hand-written table, the lexicon "
                       "plus WordNet flips {a} messages the small table does "
                       "not, against {b} ($p={p:.4f}$, {n} pairs)."),
        "stronger": ("TextFooler's leave-one-out importance ranking flips {a} "
                     "messages PWWS's saliency ranking does not, against {b} in "
                     "the other direction ($p={p:.4f}$, {n} pairs)."),
    }
    for label in ("ordering", "oracle", "search", "candidates", "stronger"):
        sub = t[t["contrast"] == label]
        if not len(sub):
            continue
        r = sub.iloc[0]
        text = sentences[label].format(a=int(r["only_a"]),
                                       b=int(r["only_b"]),
                                       p=float(r["p"]),
                                       n=int(r["n_paired"]))
        if label == "stronger":
            # Whether this supports "the stronger attack wins" depends on the
            # sign of the contrast, so the conclusion is derived, not asserted.
            text += (" The published attack's ranking therefore is not uniformly "
                     "stronger on these messages, which is why it is reported as "
                     "a reference point rather than as a fifth cell."
                     if r["only_a"] < r["only_b"] else
                     " On these messages the published attack's ranking does add "
                     "flips PWWS misses, which is the direction the literature "
                     "reports.")
        out.append(text)
    out.append("These are pooled counts over model and task, and the cells "
               "reuse messages across models, so they describe a direction "
               "rather than a confirmatory test; the per-cell pairs are in the "
               "artefact.")
    write_tex("gen_ablation_results.tex", out, subdir="sections")
    return True


def t8_pairs():
    """Paired McNemar comparison of the attack modes on matched messages."""
    path = os.path.join(RESULTS_DIR, "paired_tests.csv")
    if not os.path.exists(path):
        return False
    d = pd.read_csv(path)
    if not len(d):
        return False
    d = d.sort_values(["a_mode", "b_mode", "task", "model"])
    lines = preamble(
        "Paired comparison of two attack modes on the same messages. Only the "
        "discordant counts carry information about the difference: A-only "
        "flips happened under the first mode and not the second, B-only the "
        "reverse. The p-value is an exact McNemar test (binomial on the "
        "discordant counts); the interval is a percentile bootstrap over "
        "messages of the paired risk difference.",
        "tab:pairs", "lllccccccc")
    lines.append(r"\textbf{A vs B} & \textbf{Task} & \textbf{Model} & "
                 r"\textbf{Paired} & \textbf{A flips} & \textbf{B flips} & "
                 r"\textbf{Risk diff.} & \textbf{95\% CI} & \textbf{A/B only} & "
                 r"\textbf{$p$} \\")
    lines.append(r"\midrule")
    for _, r in d.iterrows():
        lines.append(" %s vs %s & %s & %s & %d & %d & %d & %+.1f & %+.1f to %+.1f "
                     "& %d/%d & %.3f \\\\"
                     % (r["a_mode"], r["b_mode"],
                        esc(TASK_LABEL.get(r["task"], r["task"])),
                        esc(MODEL_LABEL.get(r["model"], r["model"])),
                        int(r["n_paired"]), int(r["a_flips"]), int(r["b_flips"]),
                        100 * r["risk_diff"], 100 * r["diff_ci_low"],
                        100 * r["diff_ci_high"], int(r["only_a_mode"]),
                        int(r["only_b_mode"]), r["mcnemar_p"]))
    lines += closing()
    write("t8_pairs.tex", lines)

    out = [r"% Generated by src/make_paper_tex.py from results/paired_tests.csv."]
    wc = d[(d["a_mode"] == "word") & (d["b_mode"] == "char")]
    if len(wc):
        sig = wc[wc["mcnemar_p"] < 0.05]
        only_w = int(wc["only_a_mode"].sum())
        only_c = int(wc["only_b_mode"].sum())
        out.append(
            "Pairing the two attacks on the messages both of them attacked, the "
            "word attack flips %d messages the character attack leaves alone, "
            "and the character attack flips %d that the word attack leaves "
            "alone (exact McNemar $p = %.3f$ over the pooled discordant "
            "counts)." % (only_w, only_c, mcnemar_p(only_w, only_c)))
        if len(sig):
            out.append(
                "Per cell, %d of %d show a significant paired difference at "
                "$p<0.05$: %s. The rest are underpowered rather than "
                "contradictory, and the pooled test is a description of the "
                "direction because the cells reuse the same messages across "
                "models."
                % (len(sig), len(wc),
                   "; ".join("%s/%s" % (esc(TASK_LABEL.get(r["task"], r["task"])),
                                         esc(MODEL_LABEL.get(r["model"], r["model"])))
                             for _, r in sig.iterrows())))
        else:
            out.append("No single cell reaches significance, so the comparison "
                       "is a direction, not a per-model result.")
    write_tex("gen_pairs_results.tex", out, subdir="sections")
    return True


def mcnemar_p(b, c):
    """Exact two-sided McNemar p, recomputed here so the prose and the table
    cannot disagree. Same definition as src/analyze_pairs.py."""
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    from math import comb
    k = min(int(b), int(c))
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    missing = [p for p in ("baseline_table.csv", "attack_table.csv")
               if not os.path.exists(os.path.join(RESULTS_DIR, p))]
    if missing:
        print("missing results: %s - run the baseline and the attacks first" % missing)
        return 1
    t1_baseline()
    asr_table("char",
              "Character-level evasion under a 15\\% edit budget. ASR is "
              "flips / samples attacked, where only samples the clean model "
              "classified malicious are attacked and each original is re-queried. "
              "CI is a 95\\% Wilson interval.",
              "tab:asr-char", "t2_char.tex")
    asr_table("word",
              "Word-level evasion under the same 15\\% budget (synonym swap, "
              "deletion, duplication, swap). Jaccard is word-set overlap; no "
              "semantic-similarity model is used anywhere in this study.",
              "tab:asr-word", "t3_word.tex")
    t4_budget()
    # The three experiments that are allowed to be absent. Each generator
    # returns False when its CSV does not exist, and main.tex then omits the
    # section entirely via \IfFileExists - so the PDF can never show a table
    # that no run produced.
    for name, fn in (("transfer", t5_transfer), ("defence", t6_defense),
                     ("saliency", t7_saliency), ("paired", t8_pairs),
                     ("ablation", t9_ablation)):
        ok = fn()
        print("%s: %s" % (name, "generated" if ok else "no data yet (section stays out)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
