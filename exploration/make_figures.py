"""Generate clean summary figures in exploration/figures/ from confirmation results."""

import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "exploration" / "results"
FIG = ROOT / "exploration" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

C_OLD = "#8c96a4"
C_SEP = "#d4732a"
C_NEW = "#1b5e8a"
C_ORA = "#2a8f5e"


def fig_separation(p1):
    """Compare Pure cPCA, 6-stage separate.py, and Unified cpca_v2 across A-F, Stress G-M, and Waking."""
    # Published fresh-seed confirmation numbers from docs/unmixing.md & results/pure/summary.md
    af = p1["planted_AF_confirm"]
    st = p1["planted_stress_confirm"]
    wl = p1["planted_laps_confirm"]
    rw = p1["real_waking"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), gridspec_kw={"width_ratios": [1.1, 1.35, 1.25]})

    # Panel A: Planted Reactivations A-F (fresh seeds 9000)
    ax = axes[0]
    labels = ["A\ndiff\n10+10", "B\ndiff\n13+7", "C\nsame\nfwd", "D\nsame\nboth", "E\n1 map\n20%", "F\nno rep"]
    old_pure = [0, 0, 0, 0, 0, 0]  # clean fits out of 10 (0/5 -> 0/10)
    sep_6 = [10, 10, 9, 10, 10, 10]  # variance-share / corrected criterion (docs/unmixing.md)
    new_v2 = [r["clean_s"] for r in af]
    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, [v * 10 for v in old_pure], w, label="Pure cPCA (existing)", color=C_OLD)
    ax.bar(x, [v * 10 for v in sep_6], w, label="6-stage separate.py", color=C_SEP)
    ax.bar(x + w, [v * 10 for v in new_v2], w, label="Unified cPCA v2", color=C_NEW)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 124); ax.set_ylabel("Clean fits (%)")
    ax.axhline(80, ls="--", color="#aaaaaa", lw=0.9)
    ax.set_title("A. Planted reactivations A–F\n(fresh seeds 9000)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor="#e0e0e0", fontsize=8, loc="upper left", ncol=1)

    # Panel B: Stress G-M (fresh seeds 15000)
    ax = axes[1]
    labels_s = ["G\n3 diff\n7%", "H\ndiff\n5+5%", "I\ndiff\n3+3%", "J\n25 cells\n10+10", "K\nsame\n5+5%", "L\n3 same\n7%", "M\n1 map\n5%"]
    sep_s = [9, 8, 1, 10, 4, 5, 10]  # docs/unmixing.md stress confirmation (corrected criterion)
    new_s = [r["clean_s"] for r in st]
    x = np.arange(len(labels_s))
    w = 0.36
    ax.bar(x - w / 2, [v * 10 for v in sep_s], w, label="6-stage separate.py (64% avg)", color=C_SEP)
    ax.bar(x + w / 2, [v * 10 for v in new_s], w, label=f"Unified cPCA v2 ({100*sum(new_s)/70:.0f}% avg)", color=C_NEW)
    for i, (a, b) in enumerate(zip(sep_s, new_s)):
        if b - a >= 3:
            ax.text(i + w / 2, b * 10 + 1.5, f"{b}/10", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=C_NEW)
    ax.set_xticks(x); ax.set_xticklabels(labels_s, fontsize=8.5)
    ax.set_ylim(0, 124)
    ax.axhline(80, ls="--", color="#aaaaaa", lw=0.9)
    ax.set_title("B. Stress conditions G–M\n(fresh seeds 15000)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor="#e0e0e0", fontsize=8, loc="upper left")

    # Panel C: Waking benchmarks (planted W1-W5 + real hc-11, hc-5, 001695)
    ax = axes[2]
    labels_w = ["W1–W5\nplanted\n(50 fits)", "hc-11\nlinear\n(10)", "hc-11\ncirc.\n(6)", "hc-5\naligned\n(12)", "hc-5\nunalign\n(12)", "001695\nlaps\n(30)"]
    sep_w = [
        100 * (10 + 10 + 10 + 10 + 10) / 50,
        100 * 5 / 10,
        100 * 6 / 6,
        100 * 9 / 12,
        np.nan,
        100 * 14 / 30,
    ]
    new_w = [
        100 * sum(r["clean_s"] for r in wl) / 50,
        100 * rw["hc11_linear"]["clean_s"] / rw["hc11_linear"]["n_fits"],
        100 * rw["hc11_circular"]["clean_s"] / rw["hc11_circular"]["n_fits"],
        100 * rw["hc5_aligned"]["clean"] / rw["hc5_aligned"]["n_fits"],
        100 * rw["hc5_unaligned"]["clean"] / rw["hc5_unaligned"]["n_fits"],
        100 * rw["dandi001695_laps"]["clean_s"] / rw["dandi001695_laps"]["n_fits"],
    ]
    x = np.arange(len(labels_w))
    w = 0.36
    for i in range(len(labels_w)):
        if np.isfinite(sep_w[i]):
            ax.bar(i - w / 2, sep_w[i], w, color=C_SEP, label="6-stage separate.py" if i == 0 else None)
        ax.bar(i + (w / 2 if np.isfinite(sep_w[i]) else 0), new_w[i], w, color=C_NEW, label="Unified cPCA v2" if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(labels_w, fontsize=8.5)
    ax.set_ylim(0, 124)
    ax.set_title("C. Waking traversals\n(zero separate.py, zero held-out split)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor="#e0e0e0", fontsize=8, loc="upper right")

    fig.suptitle("Per-component route separation: Unified cPCA v2 vs 6-stage separate.py", fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "separation_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_real_reactivations(p1, p2):
    """hc-11 spike-in power curve (2 largest & 6 smaller), maze rest timing control, and 000552 fig-8."""
    sp = p2["hc11_spikein"]
    hr = p2["hc11_real"]
    f8 = p1["real_waking"]["dandi000552_fig8"]

    fracs = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
    pcts = [f * 100 for f in fracs]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # Panel A: hc-11 Spike-in power (2 largest sessions)
    ax = axes[0]
    big = [r for r in sp if r["session"].startswith("Achilles")]
    v2_big = [100 * sum(r["hits"][f] for r in big) / 10 for f in fracs]
    old_big = [20, 10, 0, 0, 10, 90, 100]        # docs/segments.md Check C (standard balanced)
    ora_c = [10, 0, 0, 20, 10, 100, 100]         # centred direction oracle
    ora_u = [0, 20, 50, 60, 100, 100, 100]       # uncentred direction oracle

    ax.plot(pcts, old_big, "o--", color=C_OLD, lw=2, label="Existing cPCA (centred)")
    ax.plot(pcts, ora_c, "s:", color=C_SEP, lw=1.8, label="Centred direction oracle (cheating)")
    ax.plot(pcts, ora_u, "^:", color=C_ORA, lw=1.8, label="Uncentred direction oracle (cheating)")
    ax.plot(pcts, v2_big, "o-", color=C_NEW, lw=2.8, ms=7, label="Unified cPCA v2 (label-free!)")
    ax.set_xlabel("Ideal replays spiked into real PRE sleep (%)")
    ax.set_ylabel("Draws recovering map (%)")
    ax.set_ylim(-5, 108); ax.set_xlim(-0.5, 21)
    ax.set_title("A. hc-11 spike-in power\n(2 largest sessions, 10 draws/pt)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # Panel B: 6 smaller sessions + Maze rest co-firing FP drop
    ax = axes[1]
    sml = [r for r in sp if not r["session"].startswith("Achilles")]
    v2_sml = [100 * sum(r["hits"][f] for r in sml) / 30 for f in fracs]
    old_sml = [0, 0, 0, 0, 0, 100 * 5 / 30, 100 * 10 / 30]  # docs/power.md addendum
    ax.plot(pcts, old_sml, "o--", color=C_OLD, lw=2, label="Existing cPCA (6 smaller sess.)")
    ax.plot(pcts, v2_sml, "o-", color=C_NEW, lw=2.8, ms=7, label="Unified cPCA v2 (6 smaller sess.)")
    ax.set_xlabel("Ideal replays spiked into real PRE sleep (%)")
    ax.set_ylabel("Draws recovering map (%)")
    ax.set_ylim(-5, 108); ax.set_xlim(-0.5, 21)
    ax.set_title("B. hc-11 spike-in power\n(6 smaller sessions, 30 draws/pt)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    # Panel C: DANDI 000552 figure-eight unsupervised L/R trial accuracy across 65 sessions
    ax = axes[2]
    accs = [100 * x["acc"] for x in f8 if np.isfinite(x["acc"])]
    sims = [x["oracle_sim"] for x in f8 if np.isfinite(x["oracle_sim"])]
    bins = np.linspace(50, 100, 16)
    ax.hist(accs, bins=bins, color=C_NEW, edgecolor="white", alpha=0.9)
    ax.axvline(np.median(accs), color=C_SEP, ls="--", lw=2, label=f"Median = {np.median(accs):.1f}%\nMean = {np.mean(accs):.1f}%\nOracle sim = {np.median(sims):.2f}")
    ax.set_xlabel("Unsupervised Left vs Right trial accuracy (%)")
    ax.set_ylabel(f"Sessions (k ≥ 2 in {len(accs)}/{len(f8)})")
    ax.set_title("C. DANDI 000552 figure-eight\n(65 real mouse sessions, label-free)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.suptitle("Real hippocampal data: Sensitivity, co-firing immunity, and unsupervised trajectory separation", fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "real_data_breakthrough.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_ablation(p1, p2):
    """Pillar ablation + Maze rest timing-shuffled FP comparison."""
    af = p1["planted_AF_confirm"]
    abl = p1["ablation"]
    hr = p2["hc11_real"]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    # Panel A: Ablation on fresh confirmation A-F (clean fits / 60)
    ax = axes[0]
    names = ["Full\ncPCA v2", "Without\nharm. fold", "alpha = pi\n(half-circle)", "Without\nsym. merge", "Without\nICA", "Existing\npure cPCA"]
    vals = [
        sum(r["clean_s"] for r in af),
        sum(r["clean_s"] for r in abl["no_harmonic_fold"]),
        sum(r["clean_s"] for r in abl["alpha_pi"]),
        sum(r["clean_s"] for r in abl["no_symmetry"]),
        sum(r["clean_s"] for r in abl["no_ica"]),
        0,
    ]
    colors = [C_NEW, "#3d7ea6", "#5e9bc2", C_SEP, "#c2593f", C_OLD]
    bars = ax.bar(range(len(names)), [100 * v / 60 for v in vals], color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.8, f"{v}/60", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylim(0, 108); ax.set_ylabel("Clean fits across A–F (%)")
    ax.set_title("A. Ablation of each mathematical pillar\n(fresh confirmation seeds 9000)")

    # Panel B: Timing-shuffled false positives on hc-11 maze rest (Achilles 10/25, Achilles 11/01, all 8 pooled)
    ax = axes[1]
    m_rows = {r["session"]: r for r in hr["maze"]}
    cats = ["Achilles 10/25\nmaze rest", "Achilles 11/01\nmaze rest", "All 8 hc-11\nmaze rest pooled"]
    old_fp = [98.0, 87.0, 45.0]  # docs/maze_rest.md: Achilles 98/100, 87/100
    v2_a1 = 100 * (m_rows["Achilles_10252013"]["shuf_fp"] / 50)
    v2_a2 = 100 * (m_rows["Achilles_11012013"]["shuf_fp"] / 50)
    v2_all = 100 * sum(r["shuf_fp"] for r in hr["maze"]) / (50 * len(hr["maze"]))
    new_fp = [v2_a1, v2_a2, v2_all]
    x = np.arange(len(cats)); w = 0.35
    ax.bar(x[:2] - w / 2, old_fp[:2], w, color=C_OLD, label="Existing cPCA (global centring)")
    ax.bar(x[:2] + w / 2, new_fp[:2], w, color=C_NEW, label="Unified cPCA v2 (active-centring)")
    ax.bar(x[2], new_fp[2], w, color=C_NEW)
    for i in range(2):
        ax.text(i - w / 2, old_fp[i] + 2, f"{old_fp[i]:.0f}%", ha="center", va="bottom", fontsize=9, color="#555")
        ax.text(i + w / 2, max(new_fp[i], 5.0) + 2.5, f"{new_fp[i]:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_NEW)
    ax.text(2, max(new_fp[2], 5.0) + 2.5, f"{new_fp[2]:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_NEW)
    ax.axhline(5.0, ls="--", color=C_ORA, lw=1.2, label="Nominal 5% level")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylim(0, 110); ax.set_ylabel("Timing-shuffled fits passing order test (%)")
    ax.set_title("B. Co-firing artifact elimination on real bursts\n(timing-shuffled control, 50 shuffles/session)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG / "ablation_and_cofiring.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p1 = pickle.load(open(RES / "confirm_part1.pkl", "rb"))
    p2 = pickle.load(open(RES / "confirm_part2.pkl", "rb"))
    fig_separation(p1)
    fig_real_reactivations(p1, p2)
    fig_ablation(p1, p2)
    print("Saved figures to", FIG)
