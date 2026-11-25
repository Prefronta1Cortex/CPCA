"""Figure and table: is there replay in hc-11 sleep, and is there enough for complex PCA?

Reads results/replay/ (scripts/run_replay_hc11.py) and results/spikein/ (scripts/run_spikein_hc11.py).
Output: figures/replay_hc11.png/.pdf, results/replay/summary.md
"""

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_replay_hc11 import HC11_ORDER, OUT  # noqa: E402
from scripts.run_spikein_hc11 import FRACTIONS, OUT as SPIKE  # noqa: E402

FIG = ROOT / "figures"
EPOCHS = [("pre", "sleep before (PRE)", "#9aa1ab"), ("maze", "rest on the maze", "#5b8a3c"), ("post", "sleep after (POST)", "#1f4e79")]


def label(n):
    a, d = n.split("_")
    return f"{a} {d[:2]}/{d[2:4]}"


def strict(res):
    return (res["p"] < 0.05) & (res["p_time"] < 0.05)


def figure(R, S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import mannwhitneyu

    INK, SOFT, RULE = "#1d2430", "#6b7380", "#d5d9e0"
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10, "axes.titlesize": 11.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9aa1ab", "axes.linewidth": 0.8,
        "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": INK, "text.color": INK, "axes.titlecolor": INK,
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    })
    fig = plt.figure(figsize=(16.5, 7.9))
    fig.text(0.03, 0.955, "hc-11 sleep does replay the maze, but in too few events for complex PCA to find it", fontsize=16, fontweight="bold")
    fig.text(0.03, 0.918, "A–B: the original paper's Bayesian replay analysis (Grosmark & Buzsáki 2016), reproduced on the 8 sessions; the decoder is given the waking map.",
             fontsize=10.3, color=SOFT)
    fig.text(0.03, 0.892, "C: real sleep events from before the maze, a known share of them turned into replays, and complex PCA run on them without the map.",
             fontsize=10.3, color=SOFT)
    gs = fig.add_gridspec(1, 3, left=0.105, right=0.985, top=0.765, bottom=0.235, wspace=0.42, width_ratios=[1.05, 1.0, 1.15])
    y = np.arange(len(HC11_ORDER))

    # A: mean sequence score per epoch
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A   Sequence content per event (rZ)", pad=34)
    ax.text(0, 1.025, "mean ± SE per session; higher = decoded positions run along\nthe track more than with shuffled place fields",
            transform=ax.transAxes, fontsize=8.6, color=SOFT, va="bottom", linespacing=1.25)
    for k, (ep, lab, colour) in enumerate(EPOCHS):
        m = [np.nanmean(R[n]["paper"][ep]["rZ"]) for n in HC11_ORDER]
        se = [np.nanstd(R[n]["paper"][ep]["rZ"]) / np.sqrt(np.isfinite(R[n]["paper"][ep]["rZ"]).sum()) for n in HC11_ORDER]
        off = (k - 1) * 0.22
        ax.errorbar(m, y + off, xerr=se, fmt="o", ms=5.5, color=colour, mfc=colour if ep != "pre" else "white", mec=colour, lw=1.1,
                    capsize=0, zorder=3, label=lab)
    for n_i in y[:-1]:
        ax.axhline(n_i + 0.5, color="#edf0f3", lw=0.8, zorder=0)
    ax.axvline(0, color="#9aa1ab", lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels([label(n) for n in HC11_ORDER], fontsize=9.3)
    ax.set_ylim(len(y) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("mean rZ")
    pre = np.concatenate([R[n]["paper"]["pre"]["rZ"] for n in HC11_ORDER])
    post = np.concatenate([R[n]["paper"]["post"]["rZ"] for n in HC11_ORDER])
    p = mannwhitneyu(post[np.isfinite(post)], pre[np.isfinite(pre)], alternative="greater").pvalue
    n_up = sum(np.nanmean(R[n]["paper"]["post"]["rZ"]) > np.nanmean(R[n]["paper"]["pre"]["rZ"]) for n in HC11_ORDER)
    ax.text(0, -0.15, f"POST > PRE in {n_up} of 8 sessions; all events pooled, p = {p:.3f}\n(one-sided rank-sum), as in the paper",
            transform=ax.transAxes, fontsize=8.6, color=SOFT, va="top", linespacing=1.3)
    ax.legend(loc="lower right", fontsize=8.3, frameon=False, handletextpad=0.3, borderaxespad=0.2)

    # B: share of clear replays
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("B   Clear replays: significant under both shuffles", pad=34)
    ax.text(0, 1.025, "share of events, p < 0.05 against shuffled place fields\nand against shuffled time bins",
            transform=ax.transAxes, fontsize=8.6, color=SOFT, va="bottom", linespacing=1.25)
    for k, (ep, lab, colour) in enumerate(EPOCHS):
        v = [100 * strict(R[n]["paper"][ep]).mean() for n in HC11_ORDER]
        ax.scatter(v, y + (k - 1) * 0.22, s=34, color=colour if ep != "pre" else "white", edgecolor=colour, linewidth=1.2, zorder=3)
    for n_i in y[:-1]:
        ax.axhline(n_i + 0.5, color="#edf0f3", lw=0.8, zorder=0)
    ex = [100 * (strict(R[n]["paper"]["post"]).mean() - strict(R[n]["paper"]["pre"]).mean()) for n in HC11_ORDER]
    for i, n in enumerate(HC11_ORDER):
        ax.text(1.01, i, f"{ex[i]:+.1f}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8.8,
                color=EPOCHS[2][2] if ex[i] > 0 else SOFT)
    ax.text(1.01, -0.75, "POST − PRE\n(points)", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=8, color=SOFT, linespacing=1.15)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(len(y) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("events (%)")
    ax.set_xlim(0, None)
    ax.text(0, -0.15, f"sleep after the maze adds a median {np.median(ex):.1f} and at most {max(ex):.1f} percentage\npoints of clear replays to the level before it",
            transform=ax.transAxes, fontsize=8.6, color=SOFT, va="top", linespacing=1.3)

    # C: power curve
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("C   How much replay complex PCA needs", pad=34)
    ax.text(0, 1.025, "share of draws (5 per session and level) in which complex PCA's\nbest component matches the map (p < 0.05), PRE events with replays added",
            transform=ax.transAxes, fontsize=8.6, color=SOFT, va="bottom", linespacing=1.25)
    xs = np.arange(len(FRACTIONS))
    lo, hi = 0.0, max(ex) / 100
    def xpos(f):
        return np.interp(f, FRACTIONS, xs)
    ax.axvspan(xpos(max(lo, 0)), xpos(hi), color="#e6edf5", zorder=0)
    ax.text(xpos(lo) + 0.08, 0.62, "real excess of\nclear replays\nin POST:\n0–3.4%", fontsize=8.2, color="#1f4e79", va="top", ha="left", linespacing=1.2)
    big = {"Achilles_10252013", "Achilles_11012013"}
    for n in HC11_ORDER:
        rows = S[n]["rows"]
        rate = [np.mean([np.isfinite(r["p"]) and r["p"] < 0.05 for r in rows if r["kind"] == "ideal" and r["f"] == f]) for f in FRACTIONS]
        ax.plot(xs, rate, color="#1f4e79" if n in big else "#9aa1ab", lw=2.0 if n in big else 1.0, alpha=0.95 if n in big else 0.8,
                marker="o", ms=4 if n in big else 2.5, zorder=3 if n in big else 2)
        aw = [(f, np.mean([np.isfinite(r["p"]) and r["p"] < 0.05 for r in rows if r["kind"] == "awake" and r["f"] == f]))
              for f in FRACTIONS if any(r["kind"] == "awake" and r["f"] == f for r in rows)]
        if aw:
            ax.scatter([xpos(f) for f, _ in aw], [v for _, v in aw], marker="D", s=26, color="#5b8a3c", edgecolor="white", linewidth=0.6, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{100 * f:.0f}%" for f in FRACTIONS])
    ax.set_xlabel("share of events turned into replays")
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "50%", "100%"])
    handles = [Line2D([], [], color="#1f4e79", lw=2, marker="o", ms=4, label="ideal replays, the two Achilles sessions (most place cells)"),
               Line2D([], [], color="#9aa1ab", lw=1, marker="o", ms=2.5, label="ideal replays, the other six sessions"),
               Line2D([], [], color="#5b8a3c", lw=0, marker="D", ms=5, label="real awake replays from the maze (all sessions with enough)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.15), fontsize=8.3, frameon=False, handletextpad=0.5, borderaxespad=0)
    fig.text(ax.get_position().x0, 0.03, "at 0%: sleep before the maze as it is (Achilles 10/25 borderline: 2 of 5 fits pass)",
             fontsize=8.3, color=SOFT, va="bottom")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"replay_hc11.{ext}", dpi=200)
    plt.close(fig)


def summary(R, S):
    lines = ["# hc-11: replay in sleep (the original paper's analysis) and how much complex PCA needs", "",
             "Produced by `scripts/run_replay_hc11.py`, `scripts/run_spikein_hc11.py`, `scripts/summarize_replay.py`. Design: `docs/replay.md`.", "",
             "## Bayesian replay analysis (paper's events)", "",
             "| session | place cells (paper's criterion, per direction) | running decode error (m; shuffled) | events PRE / MAZE / POST | mean rZ PRE / MAZE / POST | "
             "clear replays PRE / MAZE / POST | POST − PRE (points) |", "|---" * 7 + "|"]
    for n in HC11_ORDER:
        r = R[n]
        pp = r["paper"]
        err = "; ".join(f"{c['median_error_m']:.2f} ({c['median_error_shuffled_m']:.2f})" for c in r["running_check"].values())
        s = {ep: strict(pp[ep]).mean() for ep in ("pre", "maze", "post")}
        lines.append(f"| {n} | {', '.join(str(v) for v in r['n_place'].values())} | {err} | {pp['pre']['rZ'].size} / {pp['maze']['rZ'].size} / {pp['post']['rZ'].size} | "
                     f"{np.nanmean(pp['pre']['rZ']):.2f} / {np.nanmean(pp['maze']['rZ']):.2f} / {np.nanmean(pp['post']['rZ']):.2f} | "
                     f"{s['pre']:.1%} / {s['maze']:.1%} / {s['post']:.1%} | {100 * (s['post'] - s['pre']):+.1f} |")
    lines += ["", "Our population-burst windows (the events every method was given), scored the same way:", "",
              "| session | mean rZ PRE / POST | clear replays PRE / POST |", "|---|---|---|"]
    for n in HC11_ORDER:
        o = R[n]["ours"]
        lines.append(f"| {n} | {np.nanmean(o['pre']['rZ']):.2f} / {np.nanmean(o['post']['rZ']):.2f} | {strict(o['pre']).mean():.1%} / {strict(o['post']).mean():.1%} |")
    lines += ["", "## Complex PCA on PRE events with replays added (share of 5 draws with p < 0.05)", "",
              "| session | " + " | ".join(f"ideal {100 * f:.0f}%" for f in FRACTIONS) + " | awake 1% | awake 2% | awake 3% |", "|---" * (len(FRACTIONS) + 4) + "|"]
    for n in HC11_ORDER:
        rows = S[n]["rows"]
        def cell(kind, f):
            rs = [r for r in rows if r["kind"] == kind and r["f"] == f]
            return f"{sum(np.isfinite(r['p']) and r['p'] < 0.05 for r in rs)}/{len(rs)}" if rs else "–"
        lines.append(f"| {n} | " + " | ".join(cell("ideal", f) for f in FRACTIONS) + " | " + " | ".join(cell("awake", f) for f in (0.01, 0.02, 0.03)) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    R = {n: pickle.load(open(OUT / f"{n}.pkl", "rb")) for n in HC11_ORDER}
    S = {n: pickle.load(open(SPIKE / f"{n}.pkl", "rb")) for n in HC11_ORDER}
    FIG.mkdir(exist_ok=True)
    figure(R, S)
    summary(R, S)
    print("figures/replay_hc11.png; results/replay/summary.md")


if __name__ == "__main__":
    main()
