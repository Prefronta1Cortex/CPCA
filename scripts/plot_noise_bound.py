"""Figure: how many components are real? The Marchenko-Pastur bound against the shuffle null.

    figures/noise_bound.png / .pdf

Panels A-D: eigenvalue spectra (share of total variance, log scale) for pure independent noise
(synthetic) and for real sleep events (hc-11, Achilles 10/25 POST), uncentred and centred, with
the shuffle null's spread, the ceiling used to keep components, and the textbook bound.
Panels E-F: components kept by each rule in every hc-11 session and sleep epoch.
Reads results/hc11/*.pkl (scripts/run_hc11.py); the synthetic noise is generated here.
"""

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator, PercentFormatter  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpca import fit  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results" / "hc11"
ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
         "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]
EXAMPLE = "Achilles_10252013"
VAR = {"uncentred": "uncentred, T = event duration", "centred": "centred, T = event duration"}
COL = {"uncentred": "#1f4e79", "centred": "#d4652f"}
INK, SOFT, GRID, NULL = "#1d2430", "#6b7380", "#e9ecf1", "#c9ced6"
N_SHOW = 20

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10.5, "axes.titlesize": 11.5,
    "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9aa1ab", "axes.linewidth": 0.8,
    "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": INK, "text.color": INK, "axes.titlecolor": INK,
    "xtick.major.size": 3, "ytick.major.size": 3, "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def synthetic_noise(M=3000, N=80, seed=11):
    """Independent cells: each fires with its own probability (5-50% of events) at a uniform time on the half-circle."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.05, 0.5, N)
    return np.where(rng.random((M, N)) < q, rng.uniform(0, 100, (M, N)), np.nan), 100.0


def spectrum_from_fit(res):
    total = res["eigenvalues"].sum()
    return {"eigen": res["eigenvalues"][:N_SHOW] / total, "null": {q: v[:N_SHOW] / total for q, v in res["null_eigenvalues"].items()},
            "ceiling": res["noise_bound"] / total, "mp": res["mp_bound"] / total, "k": res["n_kept"], "k_mp": res["n_mp"]}


def spectrum_from_store(block):
    return {"eigen": np.array(block["eigen_share"]), "null": {q: np.array(v) for q, v in block["null_share"].items()},
            "ceiling": block["noise_bound_share"], "mp": block["mp_bound_share"], "k": block["n_kept"], "k_mp": block["n_mp"]}


def spectrum_panel(ax, s, colour, title, subtitle, centred):
    """Uncentred spectra span an order of magnitude (log axis); centred ones a narrow band (linear axis)."""
    x = np.arange(1, len(s["eigen"]) + 1)
    lo, med, hi = s["null"][5], s["null"][50], s["null"][95]
    ax.fill_between(x, 100 * lo, 100 * hi, color=NULL, alpha=0.55, lw=0, zorder=1)
    ax.plot(x, 100 * med, color="#9aa1ab", lw=1.0, zorder=2)
    ax.axhline(100 * s["mp"], color=INK, lw=1.1, ls=(0, (3, 2)), zorder=3)
    ax.axhline(100 * s["ceiling"], color=colour, lw=1.6, zorder=3)
    kept = x <= s["k"]
    ax.plot(x, 100 * s["eigen"], color=colour, lw=1.0, alpha=0.5, zorder=4)
    ax.scatter(x[kept], 100 * s["eigen"][kept], s=34, color=colour, edgecolor="white", linewidth=0.7, zorder=5)
    ax.scatter(x[~kept], 100 * s["eigen"][~kept], s=30, facecolor="white", edgecolor=colour, linewidth=1.2, zorder=5)
    if centred:
        top = max(100 * s["eigen"].max(), 100 * s["mp"], 100 * hi.max())
        bottom = min(100 * s["eigen"].min(), 100 * lo.min())
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=2 if top - bottom < 1 else 1))
        pad = 0.08 * (top - bottom)
        ax.set_ylim(bottom - pad, top + 3.6 * pad)
        badge_xy, ha, va = (0.97, 0.97), "right", "top"
    else:
        ax.set_yscale("log")
        ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50) if 0.6 * 100 * s["eigen"].min() <= t <= 1.6 * 100 * s["eigen"].max()]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.annotate("mean pattern: present in\nevery shuffle as well", xy=(1.15, 100 * s["eigen"][0]), xytext=(3.0, 100 * s["eigen"][0] * 0.62),
                    fontsize=8.8, color=SOFT, va="center", arrowprops=dict(arrowstyle="-", color=SOFT, lw=0.7, shrinkB=4))
        badge_xy, ha, va = (0.97, 0.96), "right", "top"
    ax.set_xlim(0.3, len(x) + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.yaxis.grid(True, which="major", color=GRID, zorder=0)
    ax.set_title(title, pad=24)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9.2, color=SOFT, va="bottom")
    badge = f"textbook bound keeps {s['k_mp']}\nshuffle null keeps {s['k']}"
    ax.text(*badge_xy, badge, transform=ax.transAxes, ha=ha, va=va, fontsize=9.4, color=INK, zorder=6,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d5d9e0", lw=0.8))


def spectrum_legend(fig, x, y):
    handles = [Line2D([], [], marker="o", color=INK, ls="", markersize=6, markeredgecolor="white"),
               Line2D([], [], marker="o", color=INK, ls="", markersize=6, markerfacecolor="white"),
               plt.Rectangle((0, 0), 1, 1, color=NULL, alpha=0.55), Line2D([], [], color=INK, lw=1.6),
               Line2D([], [], color=INK, lw=1.1, ls=(0, (3, 2)))]
    fig.legend(handles, ["component kept", "component not kept", "shuffled data (5-95% at each rank)",
                         "shuffle ceiling: 95% of the largest noise eigenvalue", "textbook (Marchenko-Pastur) bound"],
               loc="upper left", bbox_to_anchor=(x, y), ncol=3, frameon=False, fontsize=9.2, handlelength=1.8, columnspacing=1.6)


def counts_panel(ax, sessions, variant, title):
    labels, y = [], 0
    rows = []
    for name in ORDER:
        s = sessions.get(name)
        if s is None:
            continue
        for epoch in ("post", "pre"):
            b = s["epochs"][epoch]["variants"][VAR[variant]]
            rows.append((name, epoch, b["n_mp"], b["n_kept"]))
    ys = np.arange(len(rows))[::-1]
    for i, ((name, epoch, k_mp, k), yy) in enumerate(zip(rows, ys)):
        if (i // 2) % 2 == 0 and epoch == "post":
            ax.axhspan(yy - 1.5, yy + 0.5, color="#f4f5f8", zorder=0, lw=0)
        ax.plot([k_mp, k], [yy, yy], color="#cdd2da", lw=2.4, zorder=1, solid_capstyle="round")
        ax.scatter([k_mp], [yy], s=78, facecolor="white", edgecolor=INK, linewidth=1.1, zorder=2)
        ax.scatter([k], [yy], s=34, color=COL[variant], edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{n.split('_')[0]} {n.split('_')[1][:2]}/{n.split('_')[1][2:4]}  {e.upper()}" for n, e, _, _ in rows], fontsize=8.8)
    ax.set_xlim(-0.8, max(max(r[2] for r in rows), max(r[3] for r in rows)) + 1.2)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("components kept")
    tot_mp, tot = sum(r[2] for r in rows), sum(r[3] for r in rows)
    ax.set_title(title, pad=24)
    ax.text(0, 1.03, f"total: ring {tot_mp}, dot {tot}", transform=ax.transAxes, fontsize=9.2, color=SOFT, va="bottom")


def main():
    sessions = {}
    for name in ORDER:
        p = RES / f"{name}.pkl"
        if p.exists():
            sessions[name] = pickle.load(open(p, "rb"))
    times, T = synthetic_noise()
    noise = {v: spectrum_from_fit(fit(times, T, rng=np.random.default_rng(5), ica=False, center=(v == "centred"), scale=(v == "centred")))
             for v in ("uncentred", "centred")}
    real = {v: spectrum_from_store(sessions[EXAMPLE]["epochs"]["post"]["variants"][VAR[v]]) for v in ("uncentred", "centred")}

    fig = plt.figure(figsize=(16, 10.6))
    left = fig.add_gridspec(2, 2, left=0.055, right=0.585, top=0.775, bottom=0.1, wspace=0.26, hspace=0.5)
    right = fig.add_gridspec(1, 2, left=0.705, right=0.985, top=0.775, bottom=0.1, wspace=0.1)
    fig.text(0.035, 0.962, "How many components are real? The textbook noise bound against a shuffle null", fontsize=16, fontweight="bold")
    fig.text(0.035, 0.928, "Each dot is one eigenvalue: the share of the variance a component carries. Noise alone spreads the eigenvalues out, so a component "
             "counts only if it is larger than the largest one noise can produce.", fontsize=10.5, color=SOFT)
    fig.text(0.035, 0.905, "Shuffle null: each cell's arrows are permuted across events. That keeps everything about each cell (how often it fires, its timing, "
             "its mean arrow) and destroys only the coordination between cells.", fontsize=10.5, color=SOFT)
    fig.text(0.035, 0.882, "The textbook (Marchenko-Pastur) formula assumes every cell has the same variance and zero mean: true after centring and scaling, "
             "false for uncentred arrows, where cells fire at very different rates.", fontsize=10.5, color=SOFT)
    spectrum_legend(fig, 0.05, 0.862)

    ax = fig.add_subplot(left[0, 0])
    spectrum_panel(ax, noise["uncentred"], COL["uncentred"], "A   Pure noise, uncentred", "80 independent cells, 3000 events: nothing to find", centred=False)
    ax.set_ylabel("share of total variance (log scale)")
    spectrum_panel(fig.add_subplot(left[0, 1]), noise["centred"], COL["centred"], "B   Pure noise, centred and scaled",
                   "the same events: both rules correctly keep nothing", centred=True)
    ax = fig.add_subplot(left[1, 0])
    spectrum_panel(ax, real["uncentred"], COL["uncentred"], "C   Sleep events, uncentred", "hc-11 Achilles 10/25, POST sleep, 3260 events, 120 cells", centred=False)
    ax.set_xlabel("component (largest first)")
    ax.set_ylabel("share of total variance (log scale)")
    ax = fig.add_subplot(left[1, 1])
    spectrum_panel(ax, real["centred"], COL["centred"], "D   Sleep events, centred and scaled", "the same events: the two rules agree", centred=True)
    ax.set_xlabel("component (largest first)")

    counts_panel(fig.add_subplot(right[0, 0]), sessions, "uncentred", "E   Kept, uncentred")
    ax = fig.add_subplot(right[0, 1])
    counts_panel(ax, sessions, "centred", "F   Kept, centred")
    ax.set_yticklabels([])
    fig.legend([Line2D([], [], marker="o", color=INK, ls="", markersize=8.5, markerfacecolor="white"),
                Line2D([], [], marker="o", color=SOFT, ls="", markersize=6, markeredgecolor="white")],
               ["textbook bound (ring)", "shuffle null (dot)"], loc="upper left", bbox_to_anchor=(0.70, 0.862), ncol=2, frameon=False, fontsize=9.2)
    fig.text(0.035, 0.018, "E-F: hc-11, all 8 sessions, POST and PRE sleep, T = event duration; a dot inside its ring means both rules keep the same number. "
             "The shuffle null is the rule now used throughout.", fontsize=9, color=SOFT)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"noise_bound.{ext}", dpi=200)
    plt.close(fig)
    print("figures/noise_bound.png")


if __name__ == "__main__":
    main()
