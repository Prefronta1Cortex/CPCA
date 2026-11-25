"""Figure and table for the supervised ceiling (docs/ceiling.md): results of scripts/run_ceiling.py and run_ceiling_events.py.

    figures/ceiling.png / .pdf
    results/ceiling/summary.md
"""

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import beta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_topology import GRID, INK, SOFT  # noqa: E402

RES = ROOT / "results" / "ceiling"
GROUPS = [("synthetic", "replays_25", "planted:\n25% replays"), ("synthetic", "replays_10", "planted:\n10% replays"),
          ("synthetic", "weak_replays_10", "planted:\n10%, weak"), ("synthetic", "segment_replays_10_busy_events", "planted:\n10% segments,\nbusy events"),
          ("synthetic", "no_replays", "planted:\nno replays"), ("hc11", "post", "hc-11\nPOST"), ("hc11", "pre", "hc-11\nPRE"),
          ("huszar", "post", "000552\nPOST"), ("huszar", "pre", "000552\nPRE")]
READOUTS = [("centred", "complex PCA, centred", "#1f4e79"), ("uncentred", "complex PCA, uncentred", "#3b8bba"),
            ("aligned", "complex PCA, directions aligned", "#1f8a65"), ("averaging", "averaging (no PCA)", "#8a919c")]
PLANTED_C, REAL_C = "#7a8aa0", "#b8901a"


def load():
    R = [pickle.load(open(p, "rb")) for p in sorted(RES.glob("*/*.pkl")) if p.parent.name != "events"]
    out = {}
    for ds, g, _ in GROUPS:
        rows = [r for r in R if r["dataset"] == ds and (r["name"].startswith(g + "_d") if ds == "synthetic" else r["epoch"] == g)]
        out[(ds, g)] = [x for r in rows for m in r["maps"].values() for x in m["splits"]]
    return out


def clopper(k, n):
    if n == 0:
        return np.nan, np.nan
    return (beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0), (beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


def passed(x, kind, r):
    return bool(np.isfinite(x[kind][r]["p"]) and x[kind][r]["p"] < 0.05)


def figure(S):
    fig = plt.figure(figsize=(15.5, 9.6))
    gs = fig.add_gridspec(2, 1, left=0.06, right=0.985, top=0.81, bottom=0.1, hspace=0.68)
    fig.text(0.035, 0.962, "Where is the order lost? Replays chosen with half of the place cells, order scored on the other half", fontsize=15.5, fontweight="bold")
    fig.text(0.035, 0.928, "Supervised diagnostic (not the method): an event counts as a replay if its A cells fire in the map's order beyond a shuffle "
             "of which A cell has which field; every read-out is fitted and scored on the B cells only.", fontsize=10.3, color=SOFT)
    fig.text(0.035, 0.905, "Planted data show the diagnostic's power; on both real datasets the share of events chosen as replays is at the chance level "
             "measured on the same events with the map shuffled.", fontsize=10.3, color=SOFT)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(GROUPS))
    for i, (ds, g, label) in enumerate(GROUPS):
        sp = S[(ds, g)]
        exc = np.array([(r["n_selected"] - r["n_selected_shuffled"]) / max(r["n_testable"], 1) for r in sp]) * 100
        col = PLANTED_C if ds == "synthetic" else REAL_C
        ax.scatter(i + np.random.default_rng(i).uniform(-0.18, 0.18, exc.size), exc, s=8, color=col, alpha=0.35, lw=0, zorder=2)
        m = np.median(exc)
        ax.plot([i - 0.28, i + 0.28], [m, m], color=col, lw=3, zorder=3)
        ax.text(i, max(m, 0) + 2.2, f"{m:+.1f}", ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(4.5, color=GRID, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([g[2] for g in GROUPS], fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(-6, 48)
    ax.set_ylabel("events chosen as replays,\nexcess over chance (percentage points)")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("A   How many sleep events replay the map, by half of the place cells", pad=24)
    ax.text(0, 1.03, "one dot per cell split (and map); bar and number: median; chance = the same selection with the A cells' fields shuffled",
            transform=ax.transAxes, fontsize=9, color=SOFT, va="bottom")

    ax = fig.add_subplot(gs[1, 0])
    show = [g for g in GROUPS if g[1] not in ("no_replays",) and (g[0], g[1]) in S]
    w = 0.1
    for i, (ds, g, label) in enumerate(show):
        sp = [r for r in S[(ds, g)] if "selected" in r]
        for j, (r, rlabel, col) in enumerate(READOUTS):
            for k, (kind, alpha) in enumerate((("selected", 1.0), ("random", 0.35))):
                n = len(sp)
                hits = sum(passed(x, kind, r) for x in sp)
                xpos = i + (j * 2 + k - 3.5) * w
                ax.bar(xpos, hits / n if n else 0, w * 0.9, color=col, alpha=alpha, lw=0)
                lo, hi = clopper(hits, n)
                ax.plot([xpos, xpos], [lo, hi], color=INK, lw=0.6, alpha=0.5)
    ax.axhline(0.05, color=SOFT, lw=0.8, ls=(0, (3, 3)))
    ax.axvline(3.5, color=GRID, lw=1.2)
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels([g[2] for g in show], fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("splits recovering the B cells' order\n(p < 0.05)")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("B   Can the order of the other half be recovered from those events?", pad=24)
    ax.text(0, 1.03, "solid: the chosen events; faded: the same number of random events; 95% intervals; dashed: 5% chance", transform=ax.transAxes,
            fontsize=9, color=SOFT, va="bottom")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l) for _, l, c in READOUTS], loc="lower right", ncol=4, frameon=False,
              fontsize=8.8, bbox_to_anchor=(1.0, 1.02))
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"ceiling.{ext}", dpi=200)
    plt.close(fig)


def summary(S):
    lines = ["# Supervised ceiling", "", "Generated by `scripts/summarize_ceiling.py`. Design: `docs/ceiling.md`. Event definitions: "
             "`results/ceiling/events_summary.md`.", "",
             "| data | splits | events chosen | chance | excess | " + " | ".join(f"{l}: chosen / random" for _, l, _ in READOUTS) + " | standard pipeline, all events |",
             "|---" * (5 + len(READOUTS) + 1) + "|"]
    for ds, g, label in GROUPS:
        sp = S[(ds, g)]
        sel = np.median([r["n_selected"] / max(r["n_testable"], 1) for r in sp])
        shf = np.median([r["n_selected_shuffled"] / max(r["n_testable"], 1) for r in sp])
        exc = np.median([(r["n_selected"] - r["n_selected_shuffled"]) / max(r["n_testable"], 1) for r in sp])
        sc = [r for r in sp if "selected" in r]
        cells = [f"{np.mean([passed(x, 'selected', r) for x in sc]):.0%} / {np.mean([passed(x, 'random', r) for x in sc]):.0%}" if sc else "–"
                 for r, _, _ in READOUTS]
        base = np.mean([np.isfinite(x["baseline"]["p"]) and x["baseline"]["p"] < 0.05 for x in sp])
        lines.append(f"| {label.replace(chr(10), ' ')} | {len(sp)} | {sel:.1%} | {shf:.1%} | {exc:+.1%} | " + " | ".join(cells) + f" | {base:.0%} |")
    (RES / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    S = load()
    figure(S)
    summary(S)
