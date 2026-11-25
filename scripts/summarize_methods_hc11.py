"""Figure and table: six methods on the same hc-11 sleep events (results of scripts/run_methods_hc11.py).

Usage: python scripts/summarize_methods_hc11.py
Output: figures/methods_hc11.png/.pdf, results/methods_hc11/summary.md
"""

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_methods_hc11 import HC11_ORDER, OUT  # noqa: E402

FIG = ROOT / "figures"
EXAMPLE = "Achilles_10252013"
METHODS = [("cpca", "Complex PCA", "#1f4e79"), ("seqnmf", "seqNMF", "#d4652f"), ("ppseq", "PP-Seq", "#b8901a"),
           ("spotdis", "SPOTDis", "#8e5ea2"), ("rastermap", "Rastermap", "#2a8c82"), ("count_ica", "Count ICA", "#5b8a3c")]
CHANCE = ("chance", "Random order", "#9aa1ab")
UNIT = {"cpca": "component", "seqnmf": "factor", "ppseq": "sequence type", "spotdis": "cluster", "count_ica": "assembly"}
MIN_MEMBERS = 6


def significant(s):
    return s is not None and s["index"] >= 0 and np.isfinite(s["p"]) and s["p"] < 0.05


def count(R, epoch, kind, m):
    return sum(significant(R[(n, epoch)][kind].get(m)) for n in HC11_ORDER)


def figure(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from scipy.stats import binom

    INK, SOFT, RULE = "#1d2430", "#6b7380", "#d5d9e0"
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10, "axes.titlesize": 11.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9aa1ab", "axes.linewidth": 0.8,
        "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": INK, "text.color": INK, "axes.titlecolor": INK,
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    })
    cmap = LinearSegmentedColormap.from_list("fields", ["#ffffff", "#c9d1dc", "#5d6b80", "#1d2430"])
    ex = R[(EXAMPLE, "post")]
    maps, length = ex["maps"], ex["length"]
    n_bins, gap = maps.shape[2], 5

    fig = plt.figure(figsize=(17.0, 11.6))
    fig.text(0.03, 0.968, "Six methods on the same hc-11 sleep events, scored against the waking map the same way", fontsize=16, fontweight="bold")
    fig.text(0.03, 0.940, "Every method gets the same 1500 sleep bursts per session and epoch (8 sessions); no method sees the waking session. "
             "Scores take the best of a method's outputs against a null that takes the same best, so returning more outputs gains nothing.",
             fontsize=10.3, color=SOFT)

    # ------------------------------------------------------------------ top: what each method recovers, one session
    top = fig.add_gridspec(1, 6, left=0.03, right=0.975, top=0.790, bottom=0.535, wspace=0.10)
    sess = f"{EXAMPLE.split('_')[0]} {EXAMPLE.split('_')[1][:2]}/{EXAMPLE.split('_')[1][2:4]}"
    fig.text(0.03, 0.902, f"A   What each method recovers: session {sess}, sleep after the maze", fontsize=12.5, fontweight="bold")
    fig.text(0.03, 0.880, "Rows: place cells' waking firing rates along the track in both running directions (each scaled to its peak), in the order of the "
             "method's best output; dot: the field centre in the direction scored. A falling diagonal of dots means the sleep order is the waking map.",
             fontsize=9.6, color=SOFT)
    fig.add_artist(Line2D([0.03, 0.975], [0.872] * 2, color=RULE, lw=0.8))

    def strip(cells):
        m = maps[cells]
        m = m / np.maximum(m.max(axis=(1, 2), keepdims=True), 1e-9)
        out = np.full((len(cells), 2 * n_bins + gap), np.nan)
        out[:, :n_bins], out[:, n_bins + gap:] = m[:, 0], m[:, 1]
        return out

    def x_of(position, direction):
        return position / length * n_bins + (0 if direction > 0 else n_bins + gap)

    def heat(ax, cells, direction, colour, highlight=None):
        img = strip(cells)
        if highlight is not None:
            img[~highlight] *= 0.22
        ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=1.15, interpolation="nearest", extent=[0, 2 * n_bins + gap, len(cells), 0])
        dots = np.arange(len(cells)) if highlight is None else np.flatnonzero(highlight)
        ax.scatter([x_of(ex["centres"][direction][cells[i]], direction) for i in dots], dots + 0.5, s=11 if highlight is None else 22,
                   color="white" if highlight is None else colour, edgecolor=colour, linewidth=1.0, zorder=3, clip_on=False)
        ax.set_xlim(0, 2 * n_bins + gap)
        ax.set_ylim(len(cells), 0)
        ax.set_yticks([])
        ax.set_xticks([0, n_bins, n_bins + gap, 2 * n_bins + gap])
        labels = ax.set_xticklabels(["0", f"{length:.1f} m", "0", f"{length:.1f} m"], fontsize=8)
        for lab, ha in zip(labels, ("left", "right", "left", "right")):
            lab.set_horizontalalignment(ha)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.spines[["left", "bottom"]].set_visible(False)
        for x, lab in ((n_bins / 2, "direction +"), (n_bins + gap + n_bins / 2, "direction −")):
            ax.text(x, -0.075, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color=SOFT)

    def p_text(p):
        return f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"

    for c, (m, label, colour) in enumerate(METHODS):
        ax = fig.add_subplot(top[0, c])
        ax.set_title(label, color=colour, pad=31)
        if m == "count_ica":
            s = ex["compact"].get(m)
            if s is None or s["index"] < 0:
                ax.axis("off")
                continue
            d = s["map"]
            cen = ex["centres"][d]
            use = np.flatnonzero(np.isfinite(cen))
            use = use[np.argsort(cen[use], kind="stable")]
            members = ex["sets"][m][s["index"]][use]
            heat(ax, use, d, colour, highlight=members)
            ax.text(0, 1.015, f"no order: all cells by their own field, the most\nlocal assembly's {int(members.sum())} cells bright · {p_text(s['p'])}",
                    transform=ax.transAxes, fontsize=8.3, color=SOFT, va="bottom", linespacing=1.25)
            continue
        s = ex["order"].get(m)
        if s is None or s["index"] < 0:
            ax.axis("off")
            n_out = len(ex["orders"][m])
            why = (f"no {UNIT[m]} contains {MIN_MEMBERS} or more\nplace cells with a field" if n_out else
                   "no clusters: every event is\nnoise to HDBSCAN" if m == "spotdis" else f"no {UNIT.get(m, 'output')}")
            ax.text(0.5, 0.5, why, ha="center", va="center", transform=ax.transAxes, color=SOFT, fontsize=9.5, linespacing=1.4)
            continue
        o = ex["orders"][m][s["index"]]
        cen = ex["centres"][s["direction"]]
        use = np.flatnonzero(np.isfinite(o) & np.isfinite(cen))
        use = use[np.argsort(o[use] * np.sign(s["rho"]), kind="stable")]
        heat(ax, use, s["direction"], colour)
        ax.text(0, 1.015, f"|ρ| = {s['abs']:.2f} · {p_text(s['p'])}\n{s['n']} of {s['n_key']} cells, direction {'+' if s['direction'] > 0 else '−'}"
                f"{', reversed' if s['rho'] < 0 else ''}", transform=ax.transAxes, fontsize=8.3, color=SOFT, va="bottom", linespacing=1.25)

    # ------------------------------------------------------------------ bottom: all sessions
    fig.text(0.03, 0.452, "All 8 sessions: sleep after the maze against sleep before it, the control", fontsize=12.5, fontweight="bold")
    lx = 0.455
    for marker, face, text, dx in (("o", METHODS[0][2], "sleep after the maze (POST)", 0.0), ("o", "none", "sleep before (PRE)", 0.135),
                                   ("x", METHODS[0][2], "no output with 6 or more place cells", 0.235)):
        fig.add_artist(Line2D([lx + dx], [0.4585], marker=marker, markersize=6, markerfacecolor=face, markeredgecolor=METHODS[0][2],
                              color=METHODS[0][2], lw=0, transform=fig.transFigure))
        fig.text(lx + dx + 0.008, 0.4585, text, fontsize=9.5, color=SOFT, va="center")
    fig.text(0.975, 0.4585, "k/8: sessions with p < 0.05", fontsize=9.5, color=SOFT, va="center", ha="right")
    fig.add_artist(Line2D([0.03, 0.975], [0.444] * 2, color=RULE, lw=0.8))
    bot = fig.add_gridspec(1, 3, left=0.095, right=0.948, top=0.352, bottom=0.095, wspace=0.40, width_ratios=[1.3, 1.0, 0.85])
    by_chance = binom.sf(2, 8, 0.05)

    def lanes(ax, kind, rows, xlabel, title, sub):
        ax.set_title(title, pad=35)
        ax.text(0, 1.025, sub, transform=ax.transAxes, fontsize=8.6, color=SOFT, va="bottom", linespacing=1.25)
        ax.set_xscale("log")
        ax.set_xlim(1.5, 1 / 1500)
        floor = 1.0
        ax.axvspan(0.05, 1 / 1500, color="#eef1f5", zorder=0)
        ax.axvline(0.05, color="#9aa1ab", lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.text(0.05, 1.0, "p = 0.05 ", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=7.8, color=SOFT)
        yt, yl = [], []
        for i, (m, label, colour) in enumerate(rows):
            for k, (epoch, filled) in enumerate((("post", True), ("pre", False))):
                y = i + (k - 0.5) * 0.34
                jitter = np.linspace(-0.07, 0.07, len(HC11_ORDER))
                ps = []
                for j, n in enumerate(HC11_ORDER):
                    s = R[(n, epoch)][kind].get(m)
                    p = s["p"] if (s is not None and s["index"] >= 0 and np.isfinite(s["p"])) else np.nan
                    ps.append(p)
                    x = p if np.isfinite(p) else floor * 1.25
                    ax.scatter(x, y + jitter[j], s=26, marker="o" if np.isfinite(p) else "x",
                               facecolor=colour if (filled and np.isfinite(p)) else ("none" if np.isfinite(p) else colour),
                               edgecolor=colour, linewidth=1.1, zorder=3, alpha=0.9 if np.isfinite(p) else 0.6)
                k_sig = count(R, epoch, kind, m)
                ax.text(1.01, y, f"{k_sig}/8", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8.8,
                        color=colour if epoch == "post" else SOFT, fontweight="bold" if (epoch == "post" and k_sig >= 3) else "normal")
            yt.append(i)
            yl.append(label)
            if i < len(rows) - 1:
                ax.axhline(i + 0.5, color="#edf0f3", lw=0.8, zorder=0)
        ax.set_yticks(yt)
        ax.set_yticklabels(yl, fontsize=9.5)
        for tick, (_, _, colour) in zip(ax.get_yticklabels(), rows):
            tick.set_color(colour)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xticks([1, 0.1, 0.01, 0.001])
        ax.set_xticklabels(["1", "0.1", "0.01", "0.001"])
        ax.set_xlabel(xlabel)
        ax.text(0.985, 0.985, "stronger →", transform=ax.transAxes, ha="right", va="top", fontsize=8.3, color=SOFT)
        ax.spines["left"].set_visible(False)

    order_rows = [r for r in METHODS if r[0] != "count_ica"] + [CHANCE]
    ax = fig.add_subplot(bot[0, 0])
    lanes(ax, "order", order_rows, "p of the best output (permuted field centres)",
          "B   Is a sleep order the waking map?",
          "one dot per session: its best output's |Spearman| with the field centres.\nCount ICA gives no order. "
          f"By chance 5% pass; 3 or more of 8 has p = {by_chance:.3f}.")

    comp_rows = [r for r in METHODS if r[0] != "rastermap"]
    ax = fig.add_subplot(bot[0, 1])
    lanes(ax, "compact", comp_rows, "p of the most local output (permuted field centres)",
          "C   Do an output's cells share a track segment?",
          "descriptive only: a whole-route sequence covers the track,\nso it is not expected to pass. Rastermap gives no groups.")

    ax = fig.add_subplot(bot[0, 2])
    ax.set_title("D   Time for one fit", pad=35)
    ax.text(0, 1.025, "per session and epoch, one CPU core; seqNMF one\nsetting of its grid, PP-Seq including its sweep",
            transform=ax.transAxes, fontsize=8.6, color=SOFT, va="bottom", linespacing=1.25)
    for i, (m, label, colour) in enumerate(METHODS):
        t = np.array([R[(n, e)]["runtime_s"][m] for n in HC11_ORDER for e in ("post", "pre")])
        ax.scatter(t, i + np.linspace(-0.18, 0.18, t.size), s=16, color=colour, alpha=0.75, linewidth=0, zorder=3)
        med = np.median(t)
        ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, lw=1.6, zorder=4)
        ax.text(1.01, i, f"{med:.0f} s" if med >= 10 else f"{med:.1f} s", transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=8.8, color=colour)
        if i < len(METHODS) - 1:
            ax.axhline(i + 0.5, color="#edf0f3", lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([lab for _, lab, _ in METHODS], fontsize=9.5)
    for tick, (_, _, colour) in zip(ax.get_yticklabels(), METHODS):
        tick.set_color(colour)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(METHODS) - 0.5, -0.5)
    ax.set_xlabel("seconds (log scale); bar: median")

    fig.text(0.03, 0.034, "Count ICA: Lopes-dos-Santos et al. 2013 (25 ms bins, Marchenko–Pastur bound, FastICA). SPOTDis: Grossberger et al. 2018, the authors' "
             "code, pooled metric, HDBSCAN at the paper's settings. Rastermap: Stringer et al. 2024, the authors' settings for few neurons, 10 ms bins.",
             fontsize=8.6, color=SOFT)
    fig.text(0.03, 0.017, "All three new methods recover planted sequences or assemblies and report none without replays (results/methods_hc11/validation.md). "
             "Complex PCA, seqNMF and PP-Seq: the outputs and scores of figures/hc11_sequences.png.", fontsize=8.6, color=SOFT)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"methods_hc11.{ext}", dpi=200)
    plt.close(fig)


def summary(R):
    def cell(s):
        if s is None or s["index"] < 0:
            return "no output"
        return f"{s['abs']:.2f} (p {s['p']:.3f}, {s['n']}/{s['n_key']} cells)"

    lines = ["# hc-11: six methods on the same sleep events", "",
             "Produced by `scripts/run_methods_hc11.py` and `scripts/summarize_methods_hc11.py` (figure: `figures/methods_hc11.png`). "
             "Matched 1500-event subsets of the rerun. Complex PCA, seqNMF and PP-Seq scores are those of `results/hc11_sequences` (unchanged).", "",
             "## Order test: |ρ| of the best output against the waking map (p, best-of-outputs null)", "",
             "| session | epoch | " + " | ".join(lab for m, lab, _ in METHODS if m != "count_ica") + " | random order |",
             "|---" * (len(METHODS) + 2) + "|"]
    order_methods = [m for m, *_ in METHODS if m != "count_ica"] + ["chance"]
    for n in HC11_ORDER:
        for e in ("post", "pre"):
            lines.append(f"| {n} | {e.upper()} | " + " | ".join(cell(R[(n, e)]["order"].get(m)) for m in order_methods) + " |")
    for e in ("post", "pre"):
        lines.append(f"| **sessions with p < 0.05** | {e.upper()} | " + " | ".join(f"{count(R, e, 'order', m)}/8" for m in order_methods) + " |")
    comp_methods = [m for m, *_ in METHODS if m != "rastermap"]
    lines += ["", "## Assembly locality (descriptive): mean field distance of the most local output over its null mean (p)", "",
              "A route-wide sequence covers the whole track and is not expected to pass; this describes grouping, it does not rank sequence methods.", "",
              "| session | epoch | " + " | ".join(lab for m, lab, _ in METHODS if m != "rastermap") + " |", "|---" * (len(comp_methods) + 2) + "|"]
    for n in HC11_ORDER:
        for e in ("post", "pre"):
            lines.append(f"| {n} | {e.upper()} | " + " | ".join(
                (lambda s: "no output" if s is None or s["index"] < 0 else f"{s['ratio']:.2f} (p {s['p']:.3f}, {s['n']} cells)")(R[(n, e)]["compact"].get(m))
                for m in comp_methods) + " |")
    for e in ("post", "pre"):
        lines.append(f"| **sessions with p < 0.05** | {e.upper()} | " + " | ".join(f"{count(R, e, 'compact', m)}/8" for m in comp_methods) + " |")
    lines += ["", "## Outputs and runtime", "",
              "| session | epoch | complex PCA components | seqNMF factors | PP-Seq types | SPOTDis clusters (noise share) | count ICA assemblies | "
              "Rastermap cells | " + " | ".join(f"{lab} s" for _, lab, _ in METHODS) + " |", "|---" * (8 + len(METHODS)) + "|"]
    for n in HC11_ORDER:
        for e in ("post", "pre"):
            r = R[(n, e)]
            d = r["details"]
            lines.append(f"| {n} | {e.upper()} | {len(r['orders']['cpca'])} | {len(r['orders']['seqnmf'])} | {len(r['orders']['ppseq'])} | "
                         f"{d['spotdis']['n_clusters']} ({d['spotdis']['noise_fraction']:.2f}) | {d['count_ica']['k']} | {d['rastermap']['n_active']} | " +
                         " | ".join(f"{r['runtime_s'][m]:.1f}" for m, *_ in METHODS) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    R = {(n, e): pickle.load(open(OUT / f"{n}_{e}.pkl", "rb")) for n in HC11_ORDER for e in ("post", "pre")}
    FIG.mkdir(exist_ok=True)
    figure(R)
    summary(R)
    print("figures/methods_hc11.png; results/methods_hc11/summary.md")


if __name__ == "__main__":
    main()
