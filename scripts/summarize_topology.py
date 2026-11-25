"""Figures and table for the topology test (results of scripts/run_topology.py).

    figures/topology_hc11.png / .pdf        the reduced space against the waking map on hc-11, POST vs PRE
    results/topology/summary.md             all numbers, including the checks on planted data and hc-5
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topology import HC11_ORDER, N_DRAWS, synthetic_conditions  # noqa: E402

RES = ROOT / "results" / "topology"
FIG = ROOT / "figures"
EXAMPLE = "Achilles_10252013"        # fixed in advance: the session with the most place cells
INK, SOFT, GRID = "#1d2430", "#6b7380", "#e9ecf1"
POST_C, PRE_C = "#1f4e79", "#9fb3c8"
REPS = [("cPCA reduced", "complex PCA,\nreduced space", "#1f4e79"),
        ("cPCA magnitude", "same components,\ntiming ignored", "#6aa7d8"),
        ("cPCA full", "complex PCA,\nall components", "#7a8aa0"),
        ("cPCA uncentred", "complex PCA,\nuncentred", "#3b8bba"),
        ("count PCA", "count PCA\n(no timing)", "#1f8a65")]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10.5, "axes.titlesize": 11.5,
    "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9aa1ab", "axes.linewidth": 0.8, "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": INK,
    "text.color": INK, "axes.titlecolor": INK, "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(family):
    return [pickle.load(open(p, "rb")) for p in sorted((RES / family).glob("*.pkl"))]


def short(name):
    rat, day = name.split("_")
    return f"{rat} {day[:2]}/{day[2:4]}"


def significant(s):
    return np.isfinite(s["p"]) and s["p"] < 0.05


def count_p(k, n=8, alpha=0.05):
    """Chance of k or more of n independent sessions reaching p < alpha when nothing is there."""
    return float(binomtest(k, n, alpha, alternative="greater").pvalue) if k > 0 else 1.0


# ----------------------------------------------------------------------------- figure 1
def cell_order(r):
    """Cells sorted by the running direction of their strongest field, then by that field's centre."""
    pd = r["peak_direction"]
    centre = np.where(pd == 0, r["centres"][1.0], r["centres"][-1.0])
    centre = np.where(np.isfinite(centre), centre, np.nan_to_num(np.where(pd == 0, r["centres"][-1.0], r["centres"][1.0]), nan=0.0))
    return np.lexsort((centre, pd)), pd


def pair_percentiles(S):
    """Each off-diagonal entry as its percentile among all pairs (the statistic is rank-based, so this is what it sees)."""
    from scipy.stats import rankdata
    A = np.array(S, float)
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    P = np.full((n, n), np.nan)
    P[iu] = (rankdata(A[iu]) - 0.5) / iu[0].size
    P[(iu[1], iu[0])] = P[iu]
    return P


def matrix_panel(ax, S, order, pd, title, subtitle, cmap):
    A = pair_percentiles(np.array(S, float)[np.ix_(order, order)])
    ax.imshow(A, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    n_plus = int(np.sum(pd[order] == 0))
    if 0 < n_plus < len(order):
        for a in (ax.axhline, ax.axvline):
            a(n_plus - 0.5, color=INK, lw=0.8, alpha=0.55)
        ax.set_xticks([n_plus / 2 - 0.5, (n_plus + len(order)) / 2 - 0.5])
        ax.set_xticklabels(["field in direction +", "direction −"], fontsize=8.5)
        ax.tick_params(axis="x", length=0, colors=SOFT)
    else:
        ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, pad=24)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=SOFT, va="bottom")
    return n_plus


def spread_labels(values, min_gap):
    """Positions for labels next to sorted values, at least min_gap apart."""
    order = np.argsort(values)
    pos = np.array(values, float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    shift = (pos - np.array(values, float)[order]).mean()
    out = np.empty(len(pos))
    out[order] = pos - max(shift, 0) / 2
    return out


MAP_CMAP = LinearSegmentedColormap.from_list("m", ["#ffffff", "#e9d9a6", "#b8901a", "#5c4708"])
SLEEP_CMAP = LinearSegmentedColormap.from_list("r", ["#ffffff", "#c2d4e6", "#1f4e79", "#0b1d33"])


def sessions_panel(ax, H, rep):
    """Every session, POST against PRE, for one representation of all events."""
    post_rho = [H[(n, "post")]["scores"][rep]["rho"] for n in HC11_ORDER]
    label_y = spread_labels(post_rho, 0.017)
    for i, name in enumerate(HC11_ORDER):
        a, b = H[(name, "post")]["scores"][rep], H[(name, "pre")]["scores"][rep]
        ax.plot([0, 1], [a["rho"], b["rho"]], color="#cdd2da", lw=1, zorder=1)
        for x, sc, col in ((0, a, POST_C), (1, b, PRE_C)):
            ax.scatter([x], [sc["rho"]], s=46, color=col if significant(sc) else "white", edgecolor=col, linewidth=1.5, zorder=3)
            ax.plot([x - 0.06, x + 0.06], [sc["null95"]] * 2, color=INK, lw=0.9, alpha=0.5, zorder=2)
        ax.plot([-0.22, -0.05], [label_y[i], a["rho"]], color="#d5d9e0", lw=0.7, zorder=1)
        ax.text(-0.24, label_y[i], short(name), ha="right", va="center", fontsize=8.3, color=SOFT)
    n_post = sum(significant(H[(n, "post")]["scores"][rep]) for n in HC11_ORDER)
    n_pre = sum(significant(H[(n, "pre")]["scores"][rep]) for n in HC11_ORDER)
    higher = sum(H[(n, "post")]["scores"][rep]["rho"] > H[(n, "pre")]["scores"][rep]["rho"] for n in HC11_ORDER)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"POST: {n_post}/8 sessions\nabove chance\n(binomial p = {count_p(n_post):.1g})",
                        f"PRE: {n_pre}/8\n\n(binomial p = {count_p(n_pre):.2f})"], fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-1.0, 1.3)
    ax.axhline(0, color="#b9bfc8", lw=0.8)
    ax.yaxis.grid(True, color=GRID)
    ax.set_ylabel("map topology kept (Spearman ρ over pairs)")
    return higher


def figure_hc11(H):
    post, pre = H[(EXAMPLE, "post")], H[(EXAMPLE, "pre")]
    order, pd = cell_order(post)
    M = post["map_similarity"]
    fig = plt.figure(figsize=(16, 6.9))
    gs = fig.add_gridspec(1, 5, left=0.035, right=0.985, top=0.70, bottom=0.17, wspace=0.2, width_ratios=[1, 1, 1, 0.08, 1.15])
    fig.text(0.035, 0.94, "Does the reduced space keep the shape of the waking map?", fontsize=16, fontweight="bold")
    fig.text(0.035, 0.885, "For every pair of place cells: how close they are in complex PCA's reduced space (sleep spikes only), against how similar their "
             "waking rate maps are (position x running direction).", fontsize=10.5, color=SOFT)
    fig.text(0.035, 0.85, "Closeness in the reduced space: the kept components' covariance between the two cells, normalised; high when they load on the same "
             "components at similar phases.", fontsize=10.5, color=SOFT)
    fig.text(0.035, 0.815, "Score: Spearman correlation over all pairs; chance from shuffling which cell has which map (Mantel test, 2000 shuffles).",
             fontsize=10.5, color=SOFT)
    ax = fig.add_subplot(gs[0, 0])
    matrix_panel(ax, M, order, pd, "A   Waking map", f"{short(EXAMPLE)}: {len(order)} place cells, rate-map correlation", MAP_CMAP)
    ax.set_ylabel("cells, sorted by direction and field position", color=SOFT)
    s_post, s_pre = post["scores"]["cPCA reduced"], pre["scores"]["cPCA reduced"]
    matrix_panel(fig.add_subplot(gs[0, 1]), post["similarities"]["cPCA reduced"], order, pd, "B   Sleep after the maze",
                 f"reduced space, POST · ρ = {s_post['rho']:.2f}, p = {s_post['p']:.4f}", SLEEP_CMAP)
    matrix_panel(fig.add_subplot(gs[0, 2]), pre["similarities"]["cPCA reduced"], order, pd, "C   Sleep before the maze",
                 f"reduced space, PRE · ρ = {s_pre['rho']:.2f}, p = {s_pre['p']:.2f}", SLEEP_CMAP)
    fig.text(0.035, 0.075, "A–C: same cell order; colour = the pair's percentile among all pairs (dark = closest). "
             "A map-like pattern in B, and not in C, is what a kept topology looks like.", fontsize=9, color=SOFT)

    ax = fig.add_subplot(gs[0, 4])
    higher = sessions_panel(ax, H, "cPCA reduced")
    ax.set_title("D   Every session: sleep after vs before the maze", pad=38)
    ax.text(0, 1.03, f"filled: p < 0.05; dash: that session's 95% chance level\nPOST > PRE in {higher}/8 sessions "
            f"(sign test p = {binomtest(higher, 8, 0.5, alternative='greater').pvalue:.3f})", transform=ax.transAxes, fontsize=9, color=SOFT,
            va="bottom", linespacing=1.3)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"topology_hc11.{ext}", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- summary
def summary(H, S, Q):
    rep_names = [r[0] for r in REPS]
    lines = ["# Topology test: does closeness in a reduced space match closeness on the map?", "",
             "Generated by `scripts/summarize_topology.py` from `results/topology/` (`scripts/run_topology.py`). Design: `docs/topology.md`.", "",
             "Per session and sleep epoch: Spearman ρ between closeness of every pair of place cells in the representation and the correlation of "
             "their direction-resolved waking rate maps; p from 2000 cell shuffles of the map (Mantel test). Cells: ρ (p).", "",
             "## hc-11, all events", "", "| session | epoch | place cells | k (centred) | " + " | ".join(rep_names) + " |", "|---" * (len(rep_names) + 4) + "|"]
    for name in HC11_ORDER:
        for ep in ("post", "pre"):
            r = H[(name, ep)]
            cells = [f"{r['scores'][rep]['rho']:+.3f} ({r['scores'][rep]['p']:.4f})" if np.isfinite(r["scores"][rep]["rho"]) else "—" for rep in rep_names]
            lines.append(f"| {short(name)} | {ep.upper()} | {len(r['cells'])} | {r['k']['centred']} | " + " | ".join(cells) + " |")
    for ep in ("post", "pre"):
        cells = []
        for rep in rep_names:
            k = sum(significant(H[(n, ep)]["scores"][rep]) for n in HC11_ORDER)
            cells.append(f"{k}/8 (binomial p = {count_p(k):.2g})")
        lines.append(f"| **sessions with p < 0.05** | {ep.upper()} | | | " + " | ".join(cells) + " |")
    higher = sum(H[(n, "post")]["scores"]["cPCA reduced"]["rho"] > H[(n, "pre")]["scores"]["cPCA reduced"]["rho"] for n in HC11_ORDER)
    lines += ["", f"cPCA reduced: POST ρ > PRE ρ in {higher}/8 sessions (one-sided sign test p = {binomtest(higher, 8, 0.5, alternative='greater').pvalue:.3f})."]
    lines += ["", "## Planted data (fraction of draws with p < 0.05; median ρ)", "", "| condition | draws | " + " | ".join(rep_names) + " |", "|---" * (len(rep_names) + 2) + "|"]
    for c, (kind, *_r) in synthetic_conditions().items():
        cells = []
        for rep in rep_names:
            vals = S[c][rep]
            k = sum(significant(v) for v in vals)
            rho = np.nanmedian([v["rho"] for v in vals]) if any(np.isfinite(v["rho"]) for v in vals) else np.nan
            cells.append(f"{k}/{len(vals)}; ρ {rho:.2f}" if np.isfinite(rho) else f"{k}/{len(vals)}; no components")
        lines.append(f"| {c} | {N_DRAWS[kind]} | " + " | ".join(cells) + " |")
    lines += ["", "## hc-5 wheel runs (fits with p < 0.05; median ρ)", "", "| windows | " + " | ".join(rep_names) + " |", "|---" * (len(rep_names) + 1) + "|"]
    for cond in ("aligned", "unaligned"):
        cells = []
        for rep in rep_names:
            vals = Q[cond][rep]
            k = sum(significant(v) for v in vals)
            fin = [v["rho"] for v in vals if np.isfinite(v["rho"])]
            cells.append(f"{k}/{len(vals)}; ρ {np.median(fin):.2f}" if fin else f"{k}/{len(vals)}")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines += ["", "A fit with no component above the noise bound has nothing to test and counts as not significant (centred cPCA on the aligned hc-5 "
              "runs, where the sequence is in every run, keeps none in most fits)."]
    (RES / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    H = {(r["name"], r["epoch"]): r for r in load("hc11")}
    S = defaultdict(lambda: defaultdict(list))
    for r in load("synthetic"):
        for rep, s in r["scores"].items():
            S[r["condition"]][rep].append(s)
    Q = defaultdict(lambda: defaultdict(list))
    for r in load("hc5"):
        for rep, s in r["scores"].items():
            Q[r["condition"]][rep].append(s)
    FIG.mkdir(exist_ok=True)
    figure_hc11(H)
    summary(H, S, Q)
    print("figures/topology_hc11.png; results/topology/summary.md")


if __name__ == "__main__":
    main()
