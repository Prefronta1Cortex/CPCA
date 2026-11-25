"""Figure and table: complex PCA on DANDI 000552 against hc-11, identical measures (results of scripts/run_huszar.py).

    figures/huszar_vs_hc11.png / .pdf
    results/huszar/summary.md
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import beta, binomtest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_topology import GRID, INK, SOFT  # noqa: E402  (also sets the shared rcParams)

RES = ROOT / "results" / "huszar"
FIG = ROOT / "figures"
DATASETS = [("hc11", "hc-11 (rats, novel track)", "#1f4e79"), ("huszar", "DANDI 000552 (mice, familiar figure-eight)", "#b8901a")]
MEASURES = [("best", "best component matches the map"), ("topology", "reduced space keeps the map's topology"),
            ("wake_sleep", "sleep fit agrees with waking fit"), ("direction", "one template holds up on held-out events"),
            ("templates", "a validated template matches the map")]


def load(family):
    return [pickle.load(open(p, "rb")) for p in sorted((RES / family).glob("*.pkl"))]


def clopper(k, n):
    if n == 0:
        return np.nan, np.nan
    return (beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0), (beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


def passes(e, measure):
    if e is None:
        return None
    if measure == "best":
        return bool(np.isfinite(e["best"]["p"]) and e["best"]["p"] < 0.05)
    if measure in ("topology", "wake_sleep"):
        return bool(np.isfinite(e[measure]["p"]) and e[measure]["p"] < 0.05)
    if measure == "direction":
        d = e.get("direction")
        return bool(d is not None and np.isfinite(d["heldout_p"]) and d["heldout_p"] < 0.05)
    if measure == "templates":
        return bool(e["templates"]["valid_on_map"] > 0)


def table(R):
    """dataset -> epoch -> measure -> list of (session, subject, passes)."""
    T = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for ds, rows in R.items():
        for r in rows:
            for ep in ("post", "pre"):
                e = r["epochs"].get(ep)
                for m, _ in MEASURES:
                    v = passes(e, m)
                    if v is not None:
                        T[ds][ep][m].append((r["name"], r["subject"], v))
    return T


def per_mouse(rows):
    """Share of sessions passing, averaged over mice (each mouse weighs the same)."""
    by = defaultdict(list)
    for _, subj, v in rows:
        by[subj].append(v)
    return float(np.mean([np.mean(v) for v in by.values()])) if by else np.nan, len(by)


def figure(R):
    T = table(R)
    fig = plt.figure(figsize=(16, 10.4))
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.985, top=0.79, bottom=0.07, hspace=0.62, wspace=0.32, width_ratios=[1, 1, 1.25])
    fig.text(0.035, 0.962, "Complex PCA on a second dataset: DANDI 000552 against hc-11, same code, same measures", fontsize=16, fontweight="bold")
    fig.text(0.035, 0.928, "Centred complex PCA on all NREM population bursts of the sleep before (PRE) and after (POST) the task; maps from the waking "
             "session (hc-11: two running directions; 000552: left-arm and right-arm trials).", fontsize=10.5, color=SOFT)
    fig.text(0.035, 0.905, f"hc-11: {len(R['hc11'])} sessions, 4 rats, novel tracks. 000552: {len(R['huszar'])} sessions, "
             f"{len({r['subject'] for r in R['huszar']})} mice, a familiar figure-eight, so PRE is a weaker control there than on hc-11.",
             fontsize=10.5, color=SOFT)
    rng = np.random.default_rng(0)

    # A: place cells per session
    ax = fig.add_subplot(gs[0, 0])
    for i, (ds, label, col) in enumerate(DATASETS):
        v = np.array([max(r["n_place"].values()) for r in R[ds]], float)
        ax.scatter(i + rng.uniform(-0.15, 0.15, v.size), v, s=22, color=col, alpha=0.75, lw=0)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=col, lw=2.6)
        ax.text(i + 0.3, np.median(v), f"median {np.median(v):.0f}", va="center", fontsize=8.8, color=col)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["hc-11", "DANDI 000552"])
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(0, None)
    ax.set_ylabel("place cells in the larger of the two maps")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("A   How much data per session", pad=38)
    ax.text(0, 1.03, "one dot per session; same place-field criteria in both", transform=ax.transAxes, fontsize=9, color=SOFT, va="bottom")

    # B: component tests, pooled
    ax = fig.add_subplot(gs[0, 1])
    for i, (ds, label, col) in enumerate(DATASETS):
        for j, (ep, face) in enumerate((("post", col), ("pre", "white"))):
            k = sum(r["epochs"][ep]["component_hits"] for r in R[ds] if r["epochs"].get(ep))
            n = sum(int(r["epochs"][ep]["component_testable"]) for r in R[ds] if r["epochs"].get(ep))
            lo, hi = clopper(k, n)
            x = i + (j - 0.5) * 0.32
            ax.plot([x, x], [lo, hi], color=col, lw=1.6)
            ax.plot(x, k / n, "o", ms=8, mfc=face, mec=col, mew=1.6)
            ax.text(x + 0.05, k / n, f"{k}/{n}", fontsize=8.3, color=col, va="center")
    ax.axhline(0.05, color=SOFT, lw=0.8, ls=(0, (3, 3)))
    ax.text(1.75, 0.052, "chance", fontsize=8.3, color=SOFT, va="bottom", ha="right")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["hc-11", "DANDI 000552"])
    ax.set_xlim(-0.5, 1.8)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("component tests matching the map")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("B   Components on the map", pad=38)
    ax.text(0, 1.03, "every component against each map, pooled; 95% interval;\nfilled: POST, open: PRE", transform=ax.transAxes, fontsize=9,
            color=SOFT, va="bottom", linespacing=1.3)

    # C: power: best-output significance against place cells
    ax = fig.add_subplot(gs[0, 2])
    for ds, label, col in DATASETS:
        for ep, face in (("post", col), ("pre", "white")):
            xs, ys = [], []
            for r in R[ds]:
                e = r["epochs"].get(ep)
                if e and np.isfinite(e["best"]["p"]):
                    xs.append(max(r["n_place"].values()))
                    ys.append(-np.log10(e["best"]["p"]))
            ax.scatter(xs, ys, s=26, facecolor=face, edgecolor=col, lw=1.1, alpha=0.85)
    ax.axhline(-np.log10(0.05), color=SOFT, lw=0.8, ls=(0, (3, 3)))
    ax.text(0.99, -np.log10(0.05) + 0.05, "p = 0.05", fontsize=8.3, color=SOFT, va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("place cells in the larger map (log)")
    ax.set_ylabel("best component vs map, −log10 p")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("C   More cells, more power?", pad=38)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=6, label=l.split(" (")[0]) for _, l, c in DATASETS] +
              [Line2D([], [], marker="o", ls="", mfc="white", mec=SOFT, ms=6, label="PRE (open)")], loc="upper left", frameon=False, fontsize=8.6)
    ax.text(0, 1.03, "one dot per session and sleep; p from the best-of-outputs null (smallest 0.001)", transform=ax.transAxes, fontsize=9,
            color=SOFT, va="bottom")

    # D: share of sessions above chance per measure
    ax = fig.add_subplot(gs[1, :])
    w = 0.18
    for mi, (m, mlabel) in enumerate(MEASURES):
        for di, (ds, label, col) in enumerate(DATASETS):
            for ei, (ep, alpha) in enumerate((("post", 1.0), ("pre", 0.35))):
                rows = T[ds][ep][m]
                k, n = sum(v for *_, v in rows), len(rows)
                x = mi + (di * 2 + ei - 1.5) * w
                ax.bar(x, k / n, w * 0.9, color=col, alpha=alpha, edgecolor=col, lw=0.8)
                lo, hi = clopper(k, n)
                ax.plot([x, x], [lo, hi], color=INK, lw=0.8, alpha=0.6)
                ax.text(x, hi + 0.015, f"{k}/{n}", ha="center", va="bottom", fontsize=7.6, color=col)
                if ds == "huszar":
                    pm, nm = per_mouse(rows)
                    ax.plot([x - w * 0.42, x + w * 0.42], [pm, pm], color=INK, lw=1.4)
    ax.axhline(0.05, color=SOFT, lw=0.8, ls=(0, (3, 3)))
    ax.set_xticks(range(len(MEASURES)))
    ax.set_xticklabels([l for _, l in MEASURES], fontsize=9.6)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(MEASURES) - 0.4)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("sessions above chance (p < 0.05)")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("D   Every measure: share of sessions above chance, after (dark) and before (light) the task", pad=24)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l) for _, l, c in DATASETS] +
              [plt.Rectangle((0, 0), 1, 1, color=SOFT, alpha=0.35, label="PRE (light)"),
               Line2D([], [], color=INK, lw=1.4, label="000552 averaged over mice")],
              loc="upper right", ncol=4, frameon=False, fontsize=8.8)
    ax.text(0, 1.03, "95% intervals; dashed: 5% chance; measures as defined in docs/huszar.md", transform=ax.transAxes, fontsize=9,
            color=SOFT, va="bottom")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"huszar_vs_hc11.{ext}", dpi=200)
    plt.close(fig)


def summary(R):
    T = table(R)
    lines = ["# Complex PCA on DANDI 000552 against hc-11", "",
             "Generated by `scripts/summarize_huszar.py` from `results/huszar/` (`scripts/run_huszar.py`). Design: `docs/huszar.md`.", "",
             "## Data", "", "| dataset | sessions | subjects | putative pyramidal (median) | place cells, larger map (median, range) | sleep events PRE / POST (median) |",
             "|---|---|---|---|---|---|"]
    for ds, label, _ in DATASETS:
        rows = R[ds]
        pc = [max(r["n_place"].values()) for r in rows]
        ev = {ep: [r["epochs"][ep]["n_events"] for r in rows if r["epochs"].get(ep)] for ep in ("pre", "post")}
        lines.append(f"| {label} | {len(rows)} | {len({r['subject'] for r in rows})} | {np.median([r['n_pyramidal'] for r in rows]):.0f} | "
                     f"{np.median(pc):.0f} ({min(pc)}–{max(pc)}) | {np.median(ev['pre']):.0f} / {np.median(ev['post']):.0f} |")
    lines += ["", "## Measures: sessions above chance (p < 0.05)", "",
              "| measure | hc-11 POST | hc-11 PRE | 000552 POST | 000552 PRE | 000552 POST, mean over mice | 000552 PRE, mean over mice |", "|---|---|---|---|---|---|---|"]
    for m, mlabel in MEASURES:
        cells = []
        for ds in ("hc11", "huszar"):
            for ep in ("post", "pre"):
                rows = T[ds][ep][m]
                k, n = sum(v for *_, v in rows), len(rows)
                cells.append(f"{k}/{n} ({k / n:.0%}; binomial p = {binomtest(k, n, 0.05, alternative='greater').pvalue:.1g})" if n else "–")
        cells += [f"{per_mouse(T['huszar'][ep][m])[0]:.0%}" for ep in ("post", "pre")]
        lines.append(f"| {mlabel} | " + " | ".join(cells) + " |")
    lines += ["", "## Component tests (pooled)", "", "| dataset | POST | PRE |", "|---|---|---|"]
    for ds, label, _ in DATASETS:
        c = []
        for ep in ("post", "pre"):
            k = sum(r["epochs"][ep]["component_hits"] for r in R[ds] if r["epochs"].get(ep))
            n = sum(int(r["epochs"][ep]["component_testable"]) for r in R[ds] if r["epochs"].get(ep))
            c.append(f"{k}/{n} ({k / n:.1%})")
        lines.append(f"| {label} | " + " | ".join(c) + " |")
    lines += ["", "## POST against PRE, paired by session", "", "| measure | dataset | POST passes, PRE does not | PRE passes, POST does not |", "|---|---|---|---|"]
    for m, mlabel in MEASURES:
        for ds, label, _ in DATASETS:
            post = {n: v for n, _, v in T[ds]["post"][m]}
            pre = {n: v for n, _, v in T[ds]["pre"][m]}
            both = set(post) & set(pre)
            lines.append(f"| {mlabel} | {label} | {sum(post[n] and not pre[n] for n in both)} | {sum(pre[n] and not post[n] for n in both)} |")
    lines += ["", "## Direction labels against the map (one template on all cells, where it matches a map)", "",
              "| dataset | epoch | sessions with a map match | labels correct (pooled) |", "|---|---|---|---|"]
    for ds, label, _ in DATASETS:
        for ep in ("post", "pre"):
            labs = [r["epochs"][ep]["direction"]["labels"] for r in R[ds] if r["epochs"].get(ep) and r["epochs"][ep].get("direction")
                    and r["epochs"][ep]["direction"]["labels"]]
            k = sum(x["correct_expressing"] for x in labs)
            n = sum(x["n_significant_expressing"] for x in labs)
            lines.append(f"| {label} | {ep.upper()} | {len(labs)} | {k}/{n}" + (f" ({k / n:.0%})" if n else "") + " |")
    (RES / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    R = {ds: load(ds) for ds, *_ in DATASETS}
    FIG.mkdir(exist_ok=True)
    figure(R)
    summary(R)
    print(f"{len(R['hc11'])} hc-11, {len(R['huszar'])} 000552 sessions; figures/huszar_vs_hc11.png; results/huszar/summary.md")


if __name__ == "__main__":
    main()
