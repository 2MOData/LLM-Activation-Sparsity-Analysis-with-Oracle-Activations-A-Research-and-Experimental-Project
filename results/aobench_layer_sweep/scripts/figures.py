"""Figures for the analysis report (static PNGs for the paper)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "figures"
o = pd.read_csv(HERE.parent / "stats_per_window.csv")
INK, MUTED, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
SERIES = {  # fixed categorical order: slot 1, 2, 3
    "aggregate": ("#257de0", "All 9 evals (AObench aggregate)"),
    "objective_only": ("#f47230", "4 judge-free evals"),
    "judged_only": ("#00b894", "5 LLM-judged evals"),
}
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)
    ax.axvline(21, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xticks(range(13, 32, 2))
    ax.set_xticklabels([f"{s}-{s + 4}" for s in range(13, 32, 2)], rotation=45, ha="right")
    ax.set_xlabel("layers fed to the oracle (trained: 21-25)")


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for ax, (col, lo, hi, title) in zip(
    axes,
    [
        ("score", "lo", "hi", "A. Normalized score (95% CI over problems)"),
        ("diff", "diff_lo", "diff_hi", "B. Change vs the trained window (paired 95% CI)"),
    ],
    strict=False,
):
    for key, (color, label) in SERIES.items():
        t = o[o["metric"] == key].sort_values("start")
        ax.fill_between(t["start"], t[lo], t[hi], color=color, alpha=0.13, linewidth=0)
        ax.plot(
            t["start"],
            t[col],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
        ax.annotate(
            label.split(" (")[0],
            (31, t[col].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=8,
        )
    style(ax)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.set_xlim(12.5, 35.5)
axes[1].axhline(0, color=AXIS, linewidth=1)
axes[1].legend(
    loc="lower left", bbox_to_anchor=(0.39, 0.0), frameon=False, fontsize=8, labelcolor=INK
)
fig.suptitle(
    "AObench score of the layer 21-25 oracle when fed another 5-layer window",
    x=0.01,
    ha="left",
    color=INK,
    fontsize=11,
)
fig.savefig(OUT / "exp1_aggregate.png", dpi=200)

# Figure 2: hallucination, judge score vs judge corrected by the LLM re-grading
h = o[o["metric"] == "hallucination"].sort_values("start")
corr = pd.read_csv(HERE.parent / "hallucination_correction.csv")
fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
ax.fill_between(h["start"], h["lo"], h["hi"], color="#257de0", alpha=0.13, linewidth=0)
ax.plot(
    h["start"],
    h["score"],
    color="#257de0",
    linewidth=2,
    marker="o",
    markersize=4,
    markeredgecolor="white",
    markeredgewidth=0.8,
    label="Qwen3-32B judge (all items)",
)
ax.errorbar(
    corr["start"],
    corr["corrected"],
    yerr=[corr["corrected"] - corr["lo"], corr["hi"] - corr["corrected"]],
    fmt="D",
    color="#f47230",
    markersize=6,
    capsize=3,
    linewidth=1.5,
    label="corrected by blind re-grading with a second LLM (24 verdicts/window)",
)
for _, r in corr.iterrows():
    ax.annotate(
        f"{r['corrected']:.2f}",
        (r["start"], r["corrected"]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=INK,
    )
style(ax)
ax.set_ylim(0.3, 1.0)
ax.set_title("Hallucination ('not obviously wrong' rate)", loc="left", color=INK, fontsize=10)
ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
fig.savefig(OUT / "exp1_hallucination_correction.png", dpi=200)
print("saved")
