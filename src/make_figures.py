"""
Paper figures, generated from the same CSVs as the tables.

    python src/make_figures.py

Writes paper/figures/fig_asr.pdf and paper/figures/fig_budget.pdf.

Deliberate choices, because a figure is where an over-claim usually enters a
paper:

  * the y axis is the ASR with its 95% Wilson interval drawn as an error bar, so
    a 0/188 cell cannot look like "no effect, exactly" - it looks like "at most
    about 2%";
  * bars are drawn per (task, model, mode) and never averaged across models.
    The paper's claim is about individual deployments, and an average over
    models with different catch rates would hide exactly the effect it reports;
  * the denominator is printed in the tick label (n=...), because an ASR of 5%
    on 120 samples and on 300 samples are different measurements;
  * no colour-only encoding: each attack mode also has its own hatch, so the
    figure survives greyscale printing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import pandas as pd                             # noqa: E402

from src.common import RESULTS_DIR              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "paper", "figures")

MODEL_LABEL = {
    "qwen2.5-3b": "Qwen2.5-3B",
    "smollm2-1.7b": "SmolLM2-1.7B",
    "phi3.5-mini": "Phi-3.5-mini",
    "llama3.2-3b": "Llama-3.2-3B",
}
TASK_LABEL = {"sms_smishing": "SMS smishing", "phishing_email": "Phishing e-mail"}
MODE_STYLE = {"char": ("#4C72B0", ""), "word": ("#C44E52", "//"),
              "saliency": ("#55A868", "xx")}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def fig_asr(a):
    """ASR with its interval, per task and model, one group per attack mode."""
    modes = [m for m in ("char", "word", "saliency") if m in set(a["mode"])]
    tasks = [t for t in ("sms_smishing", "phishing_email") if t in set(a["task"])]
    models = sorted(a["model"].unique())

    fig, axes = plt.subplots(1, len(tasks), figsize=(3.4 * len(tasks), 2.2),
                             squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = a[a["task"] == task]
        width = 0.8 / max(len(modes), 1)
        for k, mode in enumerate(modes):
            xs, ys, los, his, ns = [], [], [], [], []
            for i, model in enumerate(models):
                row = sub[(sub["model"] == model) & (sub["mode"] == mode)]
                if not len(row):
                    continue
                r = row.iloc[0]
                xs.append(i + (k - (len(modes) - 1) / 2) * width)
                ys.append(100 * r["asr"])
                los.append(100 * (r["asr"] - r["ci95_low"]))
                his.append(100 * (r["ci95_high"] - r["asr"]))
                ns.append(int(r["n_attacked"]))
            colour, hatch = MODE_STYLE.get(mode, ("#777777", ""))
            # no in-plot legend: it collides with the tallest bar's label in the
            # first panel. The caption states the encoding (colour AND hatch),
            # which is also what keeps the figure readable in greyscale.
            ax.bar(xs, ys, width=width * 0.92, color=colour, hatch=hatch,
                   edgecolor="black", linewidth=0.4)
            ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor="black",
                        elinewidth=0.6, capsize=1.5)
            # the denominator above the upper CI, horizontally: a rotated label
            # inside a 3%-tall bar is unreadable in print
            for x, y, hi, n in zip(xs, ys, his, ns):
                ax.text(x, y + hi + 0.25, "%d" % n, ha="center",
                        va="bottom", fontsize=5.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models], rotation=20,
                           ha="right")
        ax.set_title(TASK_LABEL.get(task, task))
        ax.set_ylabel("ASR (%)" if task == tasks[0] else "")
        ax.set_ylim(0, max([100 * r["ci95_high"] for _, r in sub.iterrows()]
                           or [1.0]) * 1.35)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)
        ax.set_axisbelow(True)
    path = os.path.join(FIG_DIR, "fig_asr.pdf")
    fig.savefig(path)
    # PNG as well: the .docx build cannot embed a PDF, and the two renderings
    # come from the same figure object here so they cannot drift apart.
    fig.savefig(path.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print("wrote %s (+ .png)" % path)


def fig_budget(a):
    """ASR against the perturbation actually spent: the trade an attacker faces."""
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    for mode in [m for m in ("char", "word", "saliency") if m in set(a["mode"])]:
        sub = a[a["mode"] == mode]
        colour, hatch = MODE_STYLE.get(mode, ("#777777", ""))
        first = True
        for _, r in sub.iterrows():
            ax.errorbar(100 * r["mean_perturbation_rate"], 100 * r["asr"],
                        yerr=[[100 * (r["asr"] - r["ci95_low"])],
                              [100 * (r["ci95_high"] - r["asr"])]],
                        fmt="s" if mode == "char" else ("o" if mode == "word"
                                                         else "^"),
                        color=colour, ecolor="black", elinewidth=0.6,
                        capsize=1.5, markersize=3.5, markeredgecolor="black",
                        markeredgewidth=0.4,
                        label=mode if first else None)
            first = False
    ax.axvline(15.0, color="black", linestyle=":", linewidth=0.8)
    ax.text(15.2, ax.get_ylim()[1] * 0.92, "15% budget", fontsize=6)
    ax.set_xlabel("mean perturbation spent (% of units)")
    ax.set_ylabel("ASR (%), 95% Wilson CI")
    ax.grid(linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    path = os.path.join(FIG_DIR, "fig_budget.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print("wrote %s (+ .png)" % path)


def main():
    path = os.path.join(RESULTS_DIR, "attack_table.csv")
    if not os.path.exists(path):
        print("missing %s - run the attacks first" % path)
        return 1
    os.makedirs(FIG_DIR, exist_ok=True)
    a = pd.read_csv(path)
    if not len(a):
        print("attack_table.csv is empty")
        return 1
    fig_asr(a)
    fig_budget(a)
    # the number of rows drawn, so a stale figure is visible
    print("%d (task, model, mode) cells drawn" % len(a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
