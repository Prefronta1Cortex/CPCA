"""Figure and tables for Step 0 (oracle bound) and Step 1 (how much replay each method needs) on hc-11.

Reads results/oracle/ (scripts/run_oracle_hc11.py), results/spikein/ and results/power/ (scripts/run_power_hc11.py).
Output: figures/power_hc11.png/.pdf, results/power/summary.md
"""

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_power_hc11 import OUT, PPSEQ_FRACTIONS, PPSEQ_SESSIONS  # noqa: E402
from scripts.run_spikein_hc11 import FRACTIONS, HC11_ORDER, OUT as SPIKE  # noqa: E402

FIG = ROOT / "figures"
ORACLE = ROOT / "results" / "oracle"
METHODS = [("cpca", "Complex PCA (1500 events)", "#1f4e79", "-"), ("cpca_all", "Complex PCA (all events)", "#5d8fc4", "--"),
           ("sparse", "Event-sparse templates (raw match)", "#c0392b", "-"), ("sparse_norm", "Event-sparse templates (cosine match)", "#e3998f", "--"),
           ("rastermap", "Rastermap", "#2a8c82", "-"),
           ("ppseq", "PP-Seq (fixed setting)", "#b8901a", "-")]
READOUTS = [("oracle", "replays chosen by the decoder (clear)", "#1d2430", "D"), ("loose", "replays chosen by the decoder (loose)", "#6b7380", "d"),
            ("cpca", "complex PCA, no map", "#1f4e79", "o"), ("random", "random events", "#b3b9c2", "x")]
BIG = set(PPSEQ_SESSIONS)


def passed(p):
    return bool(np.isfinite(p) and p < 0.05)


def load_power():
    P = {}
    for n in HC11_ORDER:
        rows = pickle.load(open(SPIKE / f"{n}.pkl", "rb"))["rows"]
        for f in FRACTIONS:
            P[(n, "cpca", f)] = [r for r in rows if r["kind"] == "ideal" and r["f"] == f]
    for path in OUT.glob("*_*_*.pkl"):
        d = pickle.load(open(path, "rb"))
        P[(d["name"], d["method"], d["f"])] = d["rows"]
    return P


def rate(P, method, f, sessions):
    rs = [r for n in sessions for r in P.get((n, method, f), [])]
    return (np.mean([passed(r["p"]) for r in rs]), len(rs)) if rs else (np.nan, 0)


def figure(O, P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    INK, SOFT = "#1d2430", "#6b7380"
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10, "axes.titlesize": 11.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9aa1ab", "axes.linewidth": 0.8,
        "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": INK, "text.color": INK, "axes.titlecolor": INK,
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    })
    fig = plt.figure(figsize=(16.5, 7.9))
    fig.text(0.03, 0.955, "On hc-11 sleep, perfect event choice does no better than complex PCA, and no method needs less replay", fontsize=16, fontweight="bold")
    fig.text(0.03, 0.918, "A: perfect choice of events (the map-based decoder picks the replays with half of the place cells; the other half is scored) "
             "against complex PCA without the map, scored on the same half.", fontsize=10.3, color=SOFT)
    fig.text(0.03, 0.892, "B–C: real sleep events from before the maze, a known share turned into ideal replays of the map; each method run without the map.",
             fontsize=10.3, color=SOFT)
    gs = fig.add_gridspec(1, 3, left=0.105, right=0.985, top=0.765, bottom=0.235, wspace=0.34, width_ratios=[1.05, 1.0, 1.0])

    # A: oracle bound, POST
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A   The bound from perfect event choice (POST)", pad=34)
    ax.text(0, 1.025, "share of 10 splits in which the order recovered from the\nchosen events matches the map on the held-out cells (p < 0.05)",
            transform=ax.transAxes, fontsize=8.6, color=SOFT, va="bottom", linespacing=1.25)
    y = np.arange(len(HC11_ORDER))
    for k, (lab, text, colour, marker) in enumerate(READOUTS):
        v = [np.mean([passed(s[lab]["p"]) for s in O[n]["epochs"]["post"]["splits"]]) for n in HC11_ORDER]
        ax.scatter(v, y + (k - 1.5) * 0.15, marker=marker, s=34 if marker != "x" else 30, color=colour,
                   edgecolor="white" if marker not in ("x",) else None, linewidth=0.5, zorder=3, label=text)
    for i in y[:-1]:
        ax.axhline(i + 0.5, color="#edf0f3", lw=0.8, zorder=0)
    ax.axvline(0.5, color="#9aa1ab", lw=0.8, ls=(0, (3, 2)))
    chosen = [np.median([s["n_chosen"] for s in O[n]["epochs"]["post"]["splits"]]) for n in HC11_ORDER]
    for i, n in enumerate(HC11_ORDER):
        a, d = n.split("_")
        ax.text(-0.02, i, f"{a} {d[:2]}/{d[2:4]}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9.3, color=INK)
        ax.text(1.01, i, f"{chosen[i]:.0f}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.6, color=SOFT)
    ax.text(1.01, -0.75, "clear\nreplays", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=8, color=SOFT, linespacing=1.15)
    ax.set_yticks([])
    ax.set_ylim(len(y) - 0.5, -0.5)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "5/10", "10/10"])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("splits passing")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.15), fontsize=8.3, frameon=False, handletextpad=0.4, borderaxespad=0, ncol=2, columnspacing=1.0)

    # B, C: power curves
    xs = np.arange(len(FRACTIONS))
    def xpos(f):
        return np.interp(f, FRACTIONS, xs)
    for c, (title, sessions) in enumerate((("B   The two sessions with the most place cells", [n for n in HC11_ORDER if n in BIG]),
                                           ("C   The other six sessions", [n for n in HC11_ORDER if n not in BIG]))):
        ax = fig.add_subplot(gs[0, 1 + c])
        ax.set_title(title, pad=34)
        ax.text(0, 1.025, "share of draws in which the best output matches the map\n(p < 0.05), against the share of events made into replays",
                transform=ax.transAxes, fontsize=8.6, color=SOFT, va="bottom", linespacing=1.25)
        ax.axvspan(xpos(0), xpos(0.034), color="#e6edf5", zorder=0)
        if c == 0:
            ax.text(xpos(0) + 0.08, 0.98, "hc-11's real\nexcess of clear\nreplays: 0–3.4%", fontsize=8.2, color="#1f4e79", va="top", ha="left", linespacing=1.2)
        for m, label, colour, ls in METHODS:
            fr = PPSEQ_FRACTIONS if m == "ppseq" else FRACTIONS
            pts = [(f, *rate(P, m, f, sessions)) for f in fr]
            pts = [(f, v, n) for f, v, n in pts if n]
            if not pts:
                continue
            ax.plot([xpos(f) for f, _, _ in pts], [v for _, v, _ in pts], color=colour, ls=ls, lw=1.8 if m != "ppseq" else 0,
                    marker="o" if m != "ppseq" else "s", ms=4.5 if m != "ppseq" else 6, zorder=3, label=label)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{100 * f:.0f}%" for f in FRACTIONS])
        ax.set_xlabel("share of events turned into replays")
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "50%", "100%"])
        if c == 0:
            handles = [Line2D([], [], color=col, ls=ls if m != "ppseq" else "none", lw=1.8, marker="o" if m != "ppseq" else "s", ms=5, label=lab)
                       for m, lab, col, ls in METHODS]
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.15), fontsize=8.3, frameon=False, ncol=2, columnspacing=1.2,
                      handletextpad=0.5, borderaxespad=0)
    fig.text(0.03, 0.045, "Draws per level: 5 per session (10 in B, 30 in C). PP-Seq: one draw per level in the two sessions of B, hyperparameters fixed to those "
             "its own sweep chose on the same events before replays were added (a new sweep might suit it better).", fontsize=8.3, color=SOFT)
    fig.text(0.03, 0.025, "Event-sparse templates (cpca/sparse.py) lower the threshold 2–4 times on planted data, but on real events they keep the largest "
             "bursts (raw match) or repeated co-firing (cosine), not the replays (docs/power.md).", fontsize=8.3, color=SOFT)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"power_hc11.{ext}", dpi=200)
    plt.close(fig)


def summary(O, P):
    lines = ["# hc-11: the bound from perfect event choice (Step 0) and how much replay each method needs (Step 1)", "",
             "Produced by `scripts/run_oracle_hc11.py`, `scripts/run_power_hc11.py`, `scripts/summarize_power.py`. Design: `docs/power.md`.", "",
             "## Step 0: splits (of 10) whose order matches the map on the held-out half of the place cells", "",
             "| session | epoch | clear replays chosen (median) | decoder, clear replays | decoder, loose | random events | complex PCA (no map) |",
             "|---|---|---|---|---|---|---|"]
    for n in HC11_ORDER:
        for ep in ("post", "pre"):
            sp = O[n]["epochs"][ep]["splits"]
            lines.append(f"| {n} | {ep.upper()} | {np.median([s['n_chosen'] for s in sp]):.0f} of {np.median([s['n_testable'] for s in sp]):.0f} testable | " +
                         " | ".join(f"{sum(passed(s[lab]['p']) for s in sp)}/10" for lab in ("oracle", "loose", "random", "cpca")) + " |")
    lines += ["", "## Step 1: share of draws whose best output matches the map", "",
              "| method | sessions | " + " | ".join(f"{100 * f:.0f}%" for f in FRACTIONS) + " |", "|---" * (len(FRACTIONS) + 2) + "|"]
    for m, label, *_ in METHODS:
        for group, sessions in (("two largest", [n for n in HC11_ORDER if n in BIG]), ("other six", [n for n in HC11_ORDER if n not in BIG])):
            cells = []
            for f in FRACTIONS:
                v, k = rate(P, m, f, sessions)
                cells.append(f"{sum(passed(r['p']) for n in sessions for r in P.get((n, m, f), []))}/{k}" if k else "–")
            if any(c != "–" for c in cells):
                lines.append(f"| {label} | {group} | " + " | ".join(cells) + " |")
    lines += ["", "## Runtime per fit (median, s)", "", "| method | seconds |", "|---|---|"]
    for m, label, *_ in METHODS:
        rt = [r["runtime_s"] for key, rows in P.items() if key[1] == m for r in rows if "runtime_s" in r]
        if rt:
            lines.append(f"| {label} | {np.median(rt):.1f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    O = {n: pickle.load(open(ORACLE / f"{n}.pkl", "rb")) for n in HC11_ORDER}
    P = load_power()
    FIG.mkdir(exist_ok=True)
    figure(O, P)
    summary(O, P)
    print("figures/power_hc11.png; results/power/summary.md")


if __name__ == "__main__":
    main()
