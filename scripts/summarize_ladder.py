"""Figure and table for the difficulty ladder (results of scripts/run_ladder.py).

    figures/synthetic_ladder.png / .pdf
    results/ladder/summary.md

Left: what each rung contains (one difficulty added at a time). Middle: recovery of each planted
map by each method on the same events. Right: how often each method recovers a map above its own
chance level (a null that takes the best of the method's outputs).
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
from scipy.stats import beta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.routes import LADDER  # noqa: E402

RES = ROOT / "results" / "ladder"
FIG = ROOT / "figures"
INK, SOFT, GRID = "#1d2430", "#6b7380", "#e9ecf1"
# (label, method file, variant, colour, marker)
METHODS = [("Complex PCA, centred", "cpca", "centred", "#1f4e79", "o"),
           ("Complex PCA, uncentred", "cpca", "uncentred", "#6aa7d8", "o"),
           ("seqNMF (K = 15, L = 120 ms)", "seqnmf", "K15_L120", "#d4652f", "s"),
           ("PP-Seq", "ppseq", "type_rule", "#b8901a", "D")]
FEATURES = [("second\nmap", lambda g, r: r.get("direction_maps", False)),
            ("backward\nplays", lambda g, r: g.get("reverse_frac", 0) > 0),
            ("partial\nreplays", lambda g, r: g.get("segment", 1.0) < 1.0),
            ("assem-\nblies", lambda g, r: g.get("assemblies", 0) > 0),
            ("variable\npacing", lambda g, r: g.get("span_range_ms") is not None)]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10.5, "axes.titlesize": 11.5,
    "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9aa1ab", "axes.linewidth": 0.8, "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": INK,
    "text.color": INK, "axes.titlecolor": INK, "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load():
    """rec[(rung code, method, variant)] -> list of per-map dicts over draws."""
    rec = defaultdict(list)
    n_files = 0
    for path in sorted(RES.glob("*.pkl")):
        r = pickle.load(open(path, "rb"))
        n_files += 1
        code = r["rung"].split(" ", 1)[0]
        for variant, v in r["variants"].items():
            for m in v["maps"]:
                rec[(code, r["method"], variant)].append(m | {"draw": r["draw"], "runtime_s": v.get("runtime_s", np.nan)})
    return rec, n_files


def rho_of(m):
    return m["rho"] if np.isfinite(m["rho"]) else 0.0


def clopper(k, n, a=0.05):
    if n == 0:
        return np.nan, np.nan
    return (beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0), (beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0)


def figure(rec):
    rungs = list(LADDER)
    codes = [r.split(" ", 1)[0] for r in rungs]
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(1, 3, left=0.215, right=0.985, top=0.765, bottom=0.11, wspace=0.07, width_ratios=[1.15, 2.6, 1.05])
    axf, ax, axh = (fig.add_subplot(gs[0, i]) for i in range(3))
    fig.text(0.02, 0.962, "What each difficulty costs each method: condition H taken apart one difficulty at a time", fontsize=16, fontweight="bold")
    fig.text(0.02, 0.928, "Every rung: 80 cells (60 on the route), 1500 events, 40% of them replays. Every method gets exactly the same events in every draw "
             "(3 draws per rung); each planted map is scored separately.", fontsize=10.5, color=SOFT)
    fig.text(0.02, 0.905, "Score: |Spearman| between the order of the best output's members and the planted positions. A map counts as recovered "
             "when that beats a null taking the best of the method's outputs (p < 0.05).", fontsize=10.5, color=SOFT)
    ys = np.arange(len(rungs))[::-1].astype(float)

    # ingredients matrix
    for i, name in enumerate(rungs):
        g, r = LADDER[name]
        if i % 2 == 0:
            for a in (axf, ax, axh):
                a.axhspan(ys[i] - 0.5, ys[i] + 0.5, color="#f6f7f9", zorder=0, lw=0)
        for j, (_, has) in enumerate(FEATURES):
            on = has(g, r)
            axf.scatter(j, ys[i], s=120 if on else 40, color=INK if on else "white", edgecolor=INK if on else "#c3c8d0", linewidth=1.0, zorder=3)
    axf.set_xticks(range(len(FEATURES)))
    axf.set_xticklabels([f[0] for f in FEATURES], fontsize=8.8)
    axf.xaxis.tick_top()
    axf.tick_params(axis="x", length=0, pad=4)
    axf.set_yticks(ys)
    axf.set_yticklabels([f"{c}   {rungs[i].split(' ', 1)[1]}" for i, c in enumerate(codes)], fontsize=9.3)
    axf.tick_params(axis="y", length=0)
    axf.set_xlim(-0.6, len(FEATURES) - 0.4)
    axf.spines[["left", "bottom"]].set_visible(False)
    axf.set_title("What the rung contains", pad=46)

    # recovery strip
    n_m = len(METHODS)
    offs = np.linspace(0.3, -0.3, n_m)
    jit = np.random.default_rng(1)
    hit = np.full((len(rungs), n_m), np.nan)
    counts = {}
    for i, c in enumerate(codes):
        for k, (label, f, var, col, mk) in enumerate(METHODS):
            ms = rec.get((c, f, var), [])
            if not ms:
                continue
            rho = np.array([rho_of(m) for m in ms])
            y = ys[i] + offs[k]
            ax.scatter(rho, y + jit.uniform(-0.04, 0.04, rho.size), s=26, color=col, marker=mk, alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3)
            med = np.median(rho)
            ax.plot([med, med], [y - 0.1, y + 0.1], color=col, lw=3.2, solid_capstyle="butt", zorder=4)
            kk = sum(np.isfinite(m["p"]) and m["p"] < 0.05 for m in ms)
            hit[i, k] = kk / len(ms)
            counts[(i, k)] = (kk, len(ms))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(ys.min() - 0.6, ys.max() + 0.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_xlabel("|Spearman|, best output vs planted map (dots: maps x draws; bar: median)")
    ax.set_title("Recovery of each planted map", pad=46)

    # recovered-rate grid
    cmap = LinearSegmentedColormap.from_list("hit", ["#ffffff", "#c8d3e0", "#1f4e79"])
    for i in range(len(rungs)):
        for k in range(n_m):
            if not np.isfinite(hit[i, k]):
                continue
            h = hit[i, k]
            axh.add_patch(plt.Rectangle((k - 0.42, ys[i] - 0.36), 0.84, 0.72, color=cmap(h), lw=0, zorder=2))
            kk, n = counts[(i, k)]
            axh.text(k, ys[i], f"{kk}/{n}", ha="center", va="center", fontsize=9.2, color="white" if h > 0.6 else INK, zorder=3)
    axh.set_xlim(-0.6, n_m - 0.4)
    axh.set_ylim(ax.get_ylim())
    axh.set_yticks([])
    axh.set_xticks(range(n_m))
    axh.set_xticklabels(["cPCA\ncentred", "cPCA\nuncentred", "seqNMF", "PP-Seq"], fontsize=8.8)
    axh.xaxis.tick_top()
    axh.tick_params(axis="x", length=0, pad=4)
    for k, (_, _, _, col, _) in enumerate(METHODS):
        axh.get_xticklabels()[k].set_color(col)
    axh.spines[["left", "bottom"]].set_visible(False)
    axh.set_title("Maps recovered", pad=46)

    fig.legend([Line2D([], [], color=m[3], marker=m[4], ls="", markersize=7, markeredgecolor="white") for m in METHODS], [m[0] for m in METHODS],
               loc="upper left", bbox_to_anchor=(0.015, 0.888), ncol=4, frameon=False, fontsize=9.5, columnspacing=2.2)
    fig.text(0.02, 0.025, "Settings as in the matched rerun (docs/baselines.md): cPCA components above the shuffle null; seqNMF lambda by the authors' "
             "crossover rule; PP-Seq 20-candidate held-out sweep, refit of the next-best candidate if a refit has no latent events.",
             fontsize=9, color=SOFT)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"synthetic_ladder.{ext}", dpi=200)
    plt.close(fig)


def summary(rec, n_files):
    lines = ["# Difficulty ladder: cPCA, seqNMF and PP-Seq on condition H taken apart", "",
             "Generated by `scripts/summarize_ladder.py` from `results/ladder/` (`scripts/run_ladder.py`, " + f"{n_files} result files). "
             "Every rung: 80 cells (60 on the route), 1500 events, 40% replay, same events for every method, 3 draws. Per planted map: best "
             "output's |Spearman| with the planted positions; recovered if p < 0.05 against a null that takes the best of the method's outputs. "
             "Cells: median |ρ| over maps and draws (recovered / maps).", "",
             "| rung | " + " | ".join(m[0] for m in METHODS) + " | seqNMF best of 9 (K, L) | PP-Seq package rule |", "|---" * (len(METHODS) + 3) + "|"]
    for name in LADDER:
        c = name.split(" ", 1)[0]
        cells = []
        for _, f, var, _, _ in METHODS:
            ms = rec.get((c, f, var), [])
            k = sum(np.isfinite(m["p"]) and m["p"] < 0.05 for m in ms)
            cells.append(f"{np.median([rho_of(m) for m in ms]):.2f} ({k}/{len(ms)})" if ms else "—")
        grid = [(v, rec[(c, "seqnmf", v)]) for (cc, f, v) in list(rec) if cc == c and f == "seqnmf"]
        if grid:
            best_v, best = max(grid, key=lambda t: np.median([rho_of(m) for m in t[1]]))
            k = sum(np.isfinite(m["p"]) and m["p"] < 0.05 for m in best)
            cells.append(f"{np.median([rho_of(m) for m in best]):.2f} ({k}/{len(best)}, {best_v})")
        else:
            cells.append("—")
        ms = rec.get((c, "ppseq", "package_rule"), [])
        k = sum(np.isfinite(m["p"]) and m["p"] < 0.05 for m in ms)
        cells.append(f"{np.median([rho_of(m) for m in ms]):.2f} ({k}/{len(ms)})" if ms else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (RES / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    rec, n_files = load()
    if not rec:
        print("no ladder results yet")
        return
    FIG.mkdir(exist_ok=True)
    figure(rec)
    summary(rec, n_files)
    print(f"{n_files} files -> figures/synthetic_ladder.png, results/ladder/summary.md")


if __name__ == "__main__":
    main()
