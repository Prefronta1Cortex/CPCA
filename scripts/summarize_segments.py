"""Tables and figure for the G1 grid with the direction oracle (docs/segments.md, scripts/run_segments.py).

Usage: python scripts/summarize_segments.py
Output: results/segments/summary.md, figures/segments_g1.png/.pdf
"""

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_segments import CELLS, CONTROLS, N_DRAWS, REVERSE, SHARES  # noqa: E402

OUT = ROOT / "results" / "segments"
FIG = ROOT / "figures"
CANDIDATES = [("cpca_centred", "complex PCA, centred"), ("cpca_uncentred", "complex PCA, uncentred"), ("leadlag", "lead-lag component"),
              ("em", "direction EM"), ("oracle_direction", "direction oracle"), ("oracle_events", "event + direction oracle")]
ALPHA = 0.05


def load():
    grid, controls = {}, {}
    for p in sorted(OUT.glob("*.pkl")):
        r = pickle.load(open(p, "rb"))
        if "control" in r:
            controls.setdefault(r["control"], []).append(r)
        else:
            grid.setdefault((r["cells"], r["share"], r["reverse"]), []).append(r)
    return grid, controls


def passes(rows, cand, kind="whole"):
    return sum(np.isfinite(r[kind][cand]["p"]) and r[kind][cand]["p"] < ALPHA for r in rows if cand in r[kind])


def both_bars(rows, cand):
    """Fragment bar 1 and the timing bar (whole-track statistic against shuffled refits) for the candidates that have it."""
    n = 0
    for r in rows:
        f = r["fragment"].get(cand, {}).get("p", np.nan)
        t = r.get("timing", {}).get(cand, {}).get("p", np.nan)
        n += bool(np.isfinite(f) and f < ALPHA and np.isfinite(t) and t < ALPHA)
    return n


def whole_and_timing(rows, cand):
    n = 0
    for r in rows:
        w = r["whole"].get(cand, {}).get("p", np.nan)
        t = r.get("timing", {}).get(cand, {}).get("p", np.nan)
        n += bool(np.isfinite(w) and w < ALPHA and np.isfinite(t) and t < ALPHA)
    return n


def cell_rows(grid, cells, share, reverse):
    key = (cells, share, 0.5 if share == 0 else reverse)
    return grid.get(key, [])


def summary(grid, controls):
    lines = ["# G1 with the direction oracle: constant-speed partial replays", "",
             "Produced by `scripts/summarize_segments.py` from `scripts/run_segments.py`; design in `docs/segments.md`. "
             "Each cell: draws (of 5) whose best output passes the whole-track order test (p < 0.05) against the planted positions. "
             "Share 0 is shared between the two direction balances.", ""]
    for reverse in REVERSE:
        lines += [f"## Reverse fraction {reverse:.1f}: whole-track order test, draws passing", "",
                  "| ordered cells per replay | share | n | " + " | ".join(lab for _, lab in CANDIDATES) + " |", "|---" * (len(CANDIDATES) + 3) + "|"]
        for cells in CELLS:
            for share in SHARES:
                rows = cell_rows(grid, cells, share, reverse)
                if not rows:
                    continue
                opr = np.mean([r["ordered_per_replay"] for r in rows])
                lines.append(f"| {cells} (measured {opr:.1f}) | {share:.0%} | {len(rows)} | " + " | ".join(f"{passes(rows, c)}/{len(rows)}" for c, _ in CANDIDATES) + " |")
        lines.append("")
    lines += ["## Fragment criterion, bar 1 (a 40% stretch, at least 8 cells), draws passing", "",
              "| reverse | cells | share | " + " | ".join(lab for _, lab in CANDIDATES) + " |", "|---" * (len(CANDIDATES) + 3) + "|"]
    for reverse in REVERSE:
        for cells in CELLS:
            for share in SHARES:
                rows = cell_rows(grid, cells, share, reverse)
                if rows and (share > 0 or reverse == REVERSE[0]):
                    lines.append(f"| {reverse:.1f} | {cells} | {share:.0%} | " + " | ".join(f"{passes(rows, c, 'fragment')}/{len(rows)}" for c, _ in CANDIDATES) + " |")
    lines += ["", "## Timing control (30 shuffled refits): draws passing whole-track + timing, and fragment bar 1 + timing", "",
              "| reverse | cells | share | cPCA whole+timing | cPCA fragment+timing | lead-lag whole+timing | lead-lag fragment+timing | median timing p cPCA / lead-lag |",
              "|---|---|---|---|---|---|---|---|"]
    for reverse in REVERSE:
        for cells in CELLS:
            for share in SHARES:
                rows = cell_rows(grid, cells, share, reverse)
                if rows and (share > 0 or reverse == REVERSE[0]):
                    tp = lambda c: np.nanmedian([r.get("timing", {}).get(c, {}).get("p", np.nan) for r in rows])
                    lines.append(f"| {reverse:.1f} | {cells} | {share:.0%} | {whole_and_timing(rows, 'cpca_centred')}/{len(rows)} | {both_bars(rows, 'cpca_centred')}/{len(rows)} | "
                                 f"{whole_and_timing(rows, 'leadlag')}/{len(rows)} | {both_bars(rows, 'leadlag')}/{len(rows)} | {tp('cpca_centred'):.2f} / {tp('leadlag'):.2f} |")
    lines += ["", "## Per-event direction labels on true replays (accuracy, median over draws)", "",
              "| reverse | cells | share | direction EM (label-free) | labels from the oracle template | EM forward share |", "|---|---|---|---|---|---|"]
    for reverse in REVERSE:
        for cells in CELLS:
            for share in SHARES[1:]:
                rows = cell_rows(grid, cells, share, reverse)
                if rows:
                    lines.append(f"| {reverse:.1f} | {cells} | {share:.0%} | {np.nanmedian([r['em_label_accuracy'] for r in rows]):.2f} | "
                                 f"{np.nanmedian([r['oracle_template_label_accuracy'] for r in rows]):.2f} | {np.nanmedian([r.get('em_forward_share', np.nan) for r in rows]):.2f} |")
    lines += ["", "## Controls (9 ordered cells per replay)", "",
              "| control | n | " + " | ".join(lab for _, lab in CANDIDATES) + " | cPCA fragment+timing | lead-lag fragment+timing |", "|---" * (len(CANDIDATES) + 4) + "|"]
    for name in CONTROLS:
        rows = controls.get(name, [])
        if rows:
            lines.append(f"| {name} | {len(rows)} | " + " | ".join(f"{passes(rows, c)}/{len(rows)}" for c, _ in CANDIDATES) +
                         f" | {both_bars(rows, 'cpca_centred')}/{len(rows)} | {both_bars(rows, 'leadlag')}/{len(rows)} |")
    # criteria
    lines += ["", "## Pass criteria of docs/segments.md", ""]
    zero = [r for (c, s, v), rs in grid.items() if s == 0 for r in rs]
    for cand, lab in CANDIDATES:
        lines.append(f"- False positives at share 0, {lab}: {passes(zero, cand)}/{len(zero)} whole-track passes (criterion: at most 2 of 15).")
    for cand, lab in CANDIDATES[1:4]:
        better, worse = [], []
        for reverse in REVERSE:
            for cells in CELLS:
                for share in SHARES[1:]:
                    rows = cell_rows(grid, cells, share, reverse)
                    if not rows:
                        continue
                    d = passes(rows, cand) - passes(rows, "cpca_centred")
                    if d >= 3 and (share <= 0.05 or cells <= 9):
                        better.append((reverse, cells, share, d))
                    if d <= -3:
                        worse.append((reverse, cells, share, d))
        verdict = "beats complex PCA" if better and not worse else "does not beat complex PCA"
        lines.append(f"- {lab}: {verdict}. Cells with +3 or more draws in the low-share/few-cell region: {better or 'none'}; cells with -3 or fewer: {worse or 'none'}.")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def figure(grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, SOFT = "#1d2430", "#6b7380"
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.facecolor": "white"})
    colours = {"cpca_centred": "#1f4e79", "cpca_uncentred": "#7fa7cc", "leadlag": "#d4652f", "em": "#b8901a",
               "oracle_direction": "#2e8b57", "oracle_events": "#9fd3b2"}
    fig, axes = plt.subplots(len(REVERSE), len(CELLS), figsize=(4.2 * len(CELLS), 3.4 * len(REVERSE)), sharex=True, sharey=True)
    for i, reverse in enumerate(REVERSE):
        for j, cells in enumerate(CELLS):
            ax = axes[i, j]
            for cand, lab in CANDIDATES:
                ys = []
                for share in SHARES:
                    rows = cell_rows(grid, cells, share, reverse)
                    ys.append(passes(rows, cand) / len(rows) if rows else np.nan)
                ax.plot([100 * s for s in SHARES], ys, "-o", color=colours[cand], label=lab, lw=2 if "oracle" in cand or cand == "cpca_centred" else 1.3,
                        ms=5, alpha=0.95)
            ax.axhline(0.05, color="#c9ced6", lw=0.8, ls="--")
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks([100 * s for s in SHARES])
            if i == 0:
                ax.set_title(f"{cells} ordered cells per replay")
            if j == 0:
                ax.set_ylabel(f"reverse fraction {reverse:.1f}\nshare of draws recovering the map", color=INK)
            if i == len(REVERSE) - 1:
                ax.set_xlabel("replay share (%)", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Constant-speed partial replays: the map is recovered when each event's play direction is known, and not otherwise",
                 x=0.01, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.945, "120 cells, 2,500 events, assemblies, per-cell participation and latency; whole-track order test against the planted positions, "
             "p < 0.05; 5 draws per point.", fontsize=8.8, color=SOFT)
    fig.text(0.01, 0.920, "Oracles use the planted direction of every event (and, for the second, which events are replays); nothing else sees the answer. "
             "Label-free candidates: complex PCA (both conventions), the lead-lag component, the direction EM.", fontsize=8.8, color=SOFT)
    fig.tight_layout(rect=(0, 0.05, 1, 0.895))
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"segments_g1.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    grid, controls = load()
    summary(grid, controls)
    if grid:
        figure(grid)
        print("figures/segments_g1.png")
