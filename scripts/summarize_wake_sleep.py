"""Figure and table for the waking-against-sleep check (results of scripts/run_wake_sleep.py; docs/wake_sleep.md).

    figures/wake_sleep.png / .pdf
    results/wake_sleep/summary.md
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
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_wake_sleep import HC11_ORDER, HEADLINE, N_DRAWS, SYNTHETIC  # noqa: E402
from scripts.summarize_topology import GRID, INK, POST_C, PRE_C, SOFT, count_p, pair_percentiles, short, spread_labels  # noqa: E402

RES = ROOT / "results" / "wake_sleep"
FIG = ROOT / "figures"
EXAMPLE = "Achilles_10252013"        # fixed in advance: the session with the most place cells, as in every figure
WAKE_C = "#5c6f3a"
WAKE_CMAP = LinearSegmentedColormap.from_list("w", ["#ffffff", "#d7dfc4", "#7f9448", "#2f3a17"])
SLEEP_CMAP = LinearSegmentedColormap.from_list("s", ["#ffffff", "#c2d4e6", "#1f4e79", "#0b1d33"])
SYN_LABEL = {"same maps, sleep as L2": "same maps,\nwhole replays", "same maps, sleep as H (partial replays, assemblies)": "same maps,\nrealistic sleep (H)",
             "other maps": "other maps\non the same cells", "no replays in sleep": "no replays\nin sleep"}


def significant(s):
    return np.isfinite(s["p"]) and s["p"] < 0.05


def seriate(S):
    """Cell order that puts similar cells next to each other (optimal leaf ordering of average linkage on 1 - S)."""
    D = np.clip(1 - np.asarray(S, float), 0, None)
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), "average")
    return leaves_list(optimal_leaf_ordering(Z, squareform(D, checks=False)))


def matrix(ax, S, order, cmap, title, subtitle, colour=INK):
    ax.imshow(pair_percentiles(np.asarray(S, float)[np.ix_(order, order)]), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, pad=24, color=colour)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=SOFT, va="bottom")


def figure(H, S):
    r = H[EXAMPLE]
    post, pre = r["epochs"]["post"], r["epochs"]["pre"]
    cells = post["cells"]
    W = r["wake_similarity"][HEADLINE[0]][np.ix_(cells, cells)]
    order = seriate(W)
    fig = plt.figure(figsize=(16, 11.2))
    top = fig.add_gridspec(1, 3, left=0.035, right=0.70, top=0.775, bottom=0.465, wspace=0.12)
    bot = fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.335, bottom=0.075, wspace=0.42, width_ratios=[1.35, 1, 1.25])
    fig.text(0.035, 0.962, "Does the model fitted to sleep agree with the same model fitted to waking?", fontsize=16, fontweight="bold")
    fig.text(0.035, 0.928, "Complex PCA is fitted twice, without any map: once to the laps on the maze (every lap one event), once to the population bursts of "
             "sleep. For every pair of cells: how close they are in the waking fit's", fontsize=10.5, color=SOFT)
    fig.text(0.035, 0.905, "reduced space against how close they are in the sleep fit's. Score: Spearman over all pairs; chance from shuffling which cell is "
             "which (Mantel test, 2000 shuffles). Waking fit uncentred (the lap", fontsize=10.5, color=SOFT)
    fig.text(0.035, 0.882, "sequence is in every lap), sleep fit centred (replays are a minority), as the regime rule says.", fontsize=10.5, color=SOFT)

    k_w = r["wake_k"][HEADLINE[0]]
    s_post, s_pre = post["scores"][HEADLINE], pre["scores"][HEADLINE]
    matrix(fig.add_subplot(top[0, 0]), W, order, WAKE_CMAP, "A   Waking fit",
           f"{short(EXAMPLE)}: {r['n_wake_events']} laps, {k_w} components · {len(cells)} cells", colour=WAKE_C)
    fig.axes[-1].set_ylabel("cells, grouped by the waking fit", color=SOFT)
    matrix(fig.add_subplot(top[0, 1]), post["sleep_similarity"][HEADLINE[1]][np.ix_(cells, cells)], order, SLEEP_CMAP, "B   Sleep after the maze",
           f"{post['k'][HEADLINE[1]]} components · ρ with A = {s_post['rho']:.2f}, p = {s_post['p']:.4f}", colour=POST_C)
    cells_pre = pre["cells"]
    same = np.array_equal(cells_pre, cells)
    pre_S = pre["sleep_similarity"][HEADLINE[1]][np.ix_(cells, cells)]
    matrix(fig.add_subplot(top[0, 2]), pre_S, order, SLEEP_CMAP, "C   Sleep before the maze",
           f"{pre['k'][HEADLINE[1]]} components · ρ with A = {s_pre['rho']:.2f}, p = {s_pre['p']:.2f}" + ("" if same else " (own cells)"), colour=POST_C)
    fig.text(0.035, 0.43, "A–C: the same cells in the same order (grouped by the waking fit); colour = the pair's percentile among all pairs "
             "(dark = closest). The waking blocks reappearing in B, and not in C, is what agreement looks like.", fontsize=9, color=SOFT)

    # right of the matrices: how to read, in one sentence per outcome
    ax = fig.add_axes([0.725, 0.465, 0.26, 0.31])
    ax.axis("off")
    ax.text(0, 1.0, "What the check can and cannot say", fontsize=11.5, fontweight="bold", va="top")
    ax.text(0, 0.86,
            "Agreement does not prove that the sleep fit\nextracts useful information: the two fits\ncould share something trivial, such as which\n"
            "cells fire a lot.\n\nNo agreement where replay is expected would\nalmost certainly mean an implementation\nproblem.\n\n"
            "The two checks below are what rule that out:\nthe waking fit must reflect the waking map,\nand on planted data the two fits must agree\n"
            "with the same maps and not with other maps.", fontsize=9.6, va="top", color=INK, linespacing=1.35)

    # D: every session
    ax = fig.add_subplot(bot[0, 0])
    post_rho = [H[n]["epochs"]["post"]["scores"][HEADLINE]["rho"] for n in HC11_ORDER]
    label_y = spread_labels(post_rho, 0.017)
    for i, n in enumerate(HC11_ORDER):
        a, b = H[n]["epochs"]["post"]["scores"][HEADLINE], H[n]["epochs"]["pre"]["scores"][HEADLINE]
        ax.plot([0, 1], [a["rho"], b["rho"]], color="#cdd2da", lw=1, zorder=1)
        for x, s, col in ((0, a, POST_C), (1, b, PRE_C)):
            ax.scatter([x], [s["rho"]], s=46, color=col if significant(s) else "white", edgecolor=col, linewidth=1.5, zorder=3,
                       marker="o" if not H[n]["circular"] else "s")
            ax.plot([x - 0.06, x + 0.06], [s["null95"]] * 2, color=INK, lw=0.9, alpha=0.5, zorder=2)
        ax.plot([-0.22, -0.05], [label_y[i], a["rho"]], color="#d5d9e0", lw=0.7, zorder=1)
        ax.text(-0.24, label_y[i], short(n) + (" (circ.)" if H[n]["circular"] else ""), ha="right", va="center", fontsize=8.3, color=SOFT)
    n_post = sum(significant(H[n]["epochs"]["post"]["scores"][HEADLINE]) for n in HC11_ORDER)
    n_pre = sum(significant(H[n]["epochs"]["pre"]["scores"][HEADLINE]) for n in HC11_ORDER)
    higher = sum(H[n]["epochs"]["post"]["scores"][HEADLINE]["rho"] > H[n]["epochs"]["pre"]["scores"][HEADLINE]["rho"] for n in HC11_ORDER)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"POST: {n_post}/8 sessions\nabove chance\n(binomial p = {count_p(n_post):.1g})",
                        f"PRE: {n_pre}/8\n\n(binomial p = {count_p(n_pre):.2f})"], fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-1.05, 1.3)
    ax.axhline(0, color="#b9bfc8", lw=0.8)
    ax.yaxis.grid(True, color=GRID)
    ax.set_ylabel("waking and sleep fits agree\n(Spearman ρ over pairs)")
    ax.set_title("D   Every session: sleep after vs before the maze", pad=38)
    ax.text(0, 1.03, f"filled: p < 0.05; dash: that session's 95% chance level; square: circular track\nPOST > PRE in {higher}/8 sessions "
            f"(sign test p = {binomtest(higher, 8, 0.5, alternative='greater').pvalue:.3f})", transform=ax.transAxes, fontsize=9, color=SOFT,
            va="bottom", linespacing=1.3)

    # E: the waking fit reflects the waking map
    ax = fig.add_subplot(bot[0, 1])
    vals = [H[n]["wake_vs_map"][HEADLINE[0]] for n in HC11_ORDER]
    y = np.arange(len(HC11_ORDER))[::-1]
    for yi, n, v in zip(y, HC11_ORDER, vals):
        ax.plot([0, v["rho"]], [yi, yi], color="#d5d9e0", lw=1.0, zorder=1)
        ax.scatter([v["rho"]], [yi], s=46, color=WAKE_C if significant(v) else "white", edgecolor=WAKE_C, linewidth=1.5, zorder=3,
                   marker="o" if not H[n]["circular"] else "s")
        ax.text(v["rho"] + 0.02, yi, f"{len(H[n]['place'])} place cells", fontsize=7.8, color=SOFT, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([short(n) for n in HC11_ORDER], fontsize=8.6)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlim(0, 0.8)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlabel("waking fit vs waking place map (Spearman ρ)")
    ax.set_title("E   Check: the waking fit reflects the map", pad=38)
    ax.text(0, 1.03, f"place cells only; filled: p < 0.05 ({sum(significant(v) for v in vals)}/8)\nsquare: circular track", transform=ax.transAxes,
            fontsize=9, color=SOFT, va="bottom", linespacing=1.3)

    # F: planted data
    ax = fig.add_subplot(bot[0, 2])
    rng = np.random.default_rng(0)
    for j, cond in enumerate(SYNTHETIC):
        v = S[cond]
        rho = np.array([x["rho"] for x in v], float)
        sig = np.array([significant(x) for x in v])
        col = POST_C if j < 2 else "#8a919c"
        xs = j + rng.uniform(-0.12, 0.12, len(v))
        fin = np.isfinite(rho)
        ax.scatter(xs[fin & sig], rho[fin & sig], s=22, color=col, edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(xs[fin & ~sig], rho[fin & ~sig], s=20, facecolor="white", edgecolor=col, linewidth=1.1, zorder=3)
        label = f"{int(sig.sum())}/{len(v)}" if fin.any() else "no sleep\ncomponent"
        ax.text(j, 1.1, label, ha="center", va="top", fontsize=8.6, color=col, fontweight="bold")
        if not fin.any():
            ax.text(j, 0.05, "nothing\nto test", ha="center", va="bottom", fontsize=8, color=SOFT)
    ax.set_xticks(range(len(SYNTHETIC)))
    ax.set_xticklabels([SYN_LABEL[c] for c in SYNTHETIC], fontsize=8.4)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(SYNTHETIC) - 0.45)
    ax.set_ylim(-0.2, 1.12)
    ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axhline(0, color="#b9bfc8", lw=0.8)
    ax.yaxis.grid(True, color=GRID)
    ax.set_ylabel("waking and sleep fits agree (ρ)")
    ax.set_title("F   Check: planted data", pad=38)
    ax.text(0, 1.03, f"{N_DRAWS} draws each; 60 waking laps, 3000 sleep events;\nfilled: p < 0.05; numbers: draws above chance",
            transform=ax.transAxes, fontsize=9, color=SOFT, va="bottom", linespacing=1.3)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"wake_sleep.{ext}", dpi=200)
    plt.close(fig)


def summary(H, S):
    conv = [("uncentred", "centred"), ("uncentred", "uncentred"), ("centred", "centred"), ("centred", "uncentred")]
    lines = ["# Waking fit against sleep fit", "", "Generated by `scripts/summarize_wake_sleep.py` from `results/wake_sleep/` "
             "(`scripts/run_wake_sleep.py`). Design and expectations: `docs/wake_sleep.md`.", "",
             "Spearman ρ over cell pairs between closeness in the waking fit and in the sleep fit (p from 2000 cell shuffles). Headline: waking "
             "uncentred, sleep centred.", "", "## hc-11", "",
             "| session | track | waking events | waking fit vs map | epoch | cells | " + " | ".join(f"wake {a} / sleep {b}" for a, b in conv) + " |",
             "|---" * (6 + len(conv)) + "|"]
    for n in HC11_ORDER:
        r = H[n]
        m = r["wake_vs_map"][HEADLINE[0]]
        for e in ("post", "pre"):
            ep = r["epochs"][e]
            cells = " | ".join(f"{ep['scores'][c]['rho']:+.3f} ({ep['scores'][c]['p']:.4f})" if np.isfinite(ep["scores"][c]["rho"]) else "–" for c in conv)
            lines.append(f"| {short(n)} | {'circular' if r['circular'] else 'linear'} | {r['n_wake_events']} | {m['rho']:.2f} ({m['p']:.4f}) | "
                         f"{e.upper()} | {len(ep['cells'])} | {cells} |")
    for e in ("post", "pre"):
        lines.append(f"| **sessions with p < 0.05** | | | {sum(significant(H[n]['wake_vs_map'][HEADLINE[0]]) for n in HC11_ORDER)}/8 | {e.upper()} | | " +
                     " | ".join(f"{sum(significant(H[n]['epochs'][e]['scores'][c]) for n in HC11_ORDER)}/8" for c in conv) + " |")
    higher = sum(H[n]["epochs"]["post"]["scores"][HEADLINE]["rho"] > H[n]["epochs"]["pre"]["scores"][HEADLINE]["rho"] for n in HC11_ORDER)
    k = sum(significant(H[n]["epochs"]["post"]["scores"][HEADLINE]) for n in HC11_ORDER)
    lines += ["", f"Headline: POST {k}/8 sessions (binomial p = {count_p(k):.2g}); POST ρ > PRE ρ in {higher}/8 "
              f"(sign test p = {binomtest(higher, 8, 0.5, alternative='greater').pvalue:.3f}).", "", "## Planted data", "",
              "| condition | draws with p < 0.05 | median ρ | sleep components (median) |", "|---|---|---|---|"]
    for cond in SYNTHETIC:
        v = S[cond]
        rho = np.array([x["rho"] for x in v], float)
        lines.append(f"| {cond} | {sum(significant(x) for x in v)}/{len(v)} | {np.nanmedian(rho):.3f} |"
                     f" {int(np.median([x['k'] for x in v]))} |" if np.isfinite(rho).any() else
                     f"| {cond} | 0/{len(v)} | – (no sleep component) | 0 |")
    (RES / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    H = {r["name"]: r for r in (pickle.load(open(p, "rb")) for p in sorted((RES / "hc11").glob("*.pkl")))}
    S = defaultdict(list)
    for p in sorted((RES / "synthetic").glob("*.pkl")):
        r = pickle.load(open(p, "rb"))
        S[r["condition"]].append(dict(r["scores"][HEADLINE], k=r["sleep_k"][HEADLINE[1]]))
    FIG.mkdir(exist_ok=True)
    figure(H, S)
    summary(H, S)
    print("figures/wake_sleep.png; results/wake_sleep/summary.md")


if __name__ == "__main__":
    main()
