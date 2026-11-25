"""Tables and figure for checks A and B (docs/segments.md, addendum): the conjugate-pair read-out and the centred
direction oracle on the G1 grid, and the HMM on a slice.

Usage: python scripts/summarize_segments_pairs.py
Output: results/segments/pairs_summary.md, figures/segments_pairs.png/.pdf
"""

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_segments import CELLS, CONTROLS, REVERSE, SHARES  # noqa: E402
from scripts.summarize_segments import load as load_main  # noqa: E402

OUT = ROOT / "results" / "segments"
PAIRS = OUT / "pairs"
FIG = ROOT / "figures"
ALPHA = 0.05
READOUTS = [("standard", "standard read-out"), ("pairs_adjacent", "adjacent pairs"), ("pairs_all", "all pairs"), ("oracle_centred", "centred direction oracle")]


def load():
    grid, controls = {}, {}
    for p in sorted(PAIRS.glob("*.pkl")):
        r = pickle.load(open(p, "rb"))
        if "control" in r:
            controls.setdefault(r["control"], []).append(r)
        else:
            grid.setdefault((r["cells"], r["share"], r["reverse"]), []).append(r)
    return grid, controls


def rows_of(grid, cells, share, reverse):
    return grid.get((cells, share, 0.5 if share == 0 else reverse), [])


def passes(rows, key, kind="whole"):
    return sum(np.isfinite(r[kind].get(key, {}).get("p", np.nan)) and r[kind][key]["p"] < ALPHA for r in rows)


def with_timing(rows, key, kind="whole"):
    n = 0
    for r in rows:
        a = r[kind].get(key, {}).get("p", np.nan)
        t = r.get("timing", {}).get(key, {}).get("p", np.nan)
        n += bool(np.isfinite(a) and a < ALPHA and np.isfinite(t) and t < ALPHA)
    return n


def summary(grid, controls, main_grid, main_controls):
    lines = ["# Checks A and B: the conjugate-pair read-out, the centred direction oracle, and the HMM on the G1 grid", "",
             "Produced by `scripts/summarize_segments_pairs.py` from `scripts/run_segments_pairs.py`; design in `docs/segments.md` (addendum). "
             "Same planted events as `results/segments/summary.md`; one centred complex PCA fit per draw read out three ways. The uncentred "
             "direction oracle is copied from the main grid for comparison.", ""]
    for reverse in REVERSE:
        lines += [f"## Reverse fraction {reverse:.1f}: whole-track order test, draws (of 5) passing", "",
                  "| ordered cells | share | " + " | ".join(lab for _, lab in READOUTS) + " | uncentred direction oracle (main grid) |", "|---" * (len(READOUTS) + 3) + "|"]
        for cells in CELLS:
            for share in SHARES:
                rows = rows_of(grid, cells, share, reverse)
                if not rows:
                    continue
                main_rows = rows_of(main_grid, cells, share, reverse)
                lines.append(f"| {cells} | {share:.0%} | " + " | ".join(f"{passes(rows, k)}/{len(rows)}" for k, _ in READOUTS) +
                             f" | {passes(main_rows, 'oracle_direction')}/{len(main_rows)} |")
        lines.append("")
    lines += ["## Whole-track test with the timing bar (30 shuffled refits read out the same way)", "",
              "| reverse | cells | share | " + " | ".join(lab for _, lab in READOUTS[:3]) + " |", "|---|---|---|---|---|---|"]
    for reverse in REVERSE:
        for cells in CELLS:
            for share in SHARES:
                rows = rows_of(grid, cells, share, reverse)
                if rows and (share > 0 or reverse == REVERSE[0]):
                    lines.append(f"| {reverse:.1f} | {cells} | {share:.0%} | " + " | ".join(f"{with_timing(rows, k)}/{len(rows)}" for k, _ in READOUTS[:3]) + " |")
    lines += ["", "## Fragment criterion: bar 1, and bar 1 with the timing bar", "",
              "| reverse | cells | share | " + " | ".join(f"{lab} bar 1" for _, lab in READOUTS) + " | " + " | ".join(f"{lab} both bars" for _, lab in READOUTS[:3]) + " |",
              "|---" * (len(READOUTS) + 6) + "|"]
    for reverse in REVERSE:
        for cells in CELLS:
            for share in SHARES:
                rows = rows_of(grid, cells, share, reverse)
                if rows and (share > 0 or reverse == REVERSE[0]):
                    lines.append(f"| {reverse:.1f} | {cells} | {share:.0%} | " + " | ".join(f"{passes(rows, k, 'fragment')}/{len(rows)}" for k, _ in READOUTS) +
                                 " | " + " | ".join(f"{with_timing(rows, k, 'fragment')}/{len(rows)}" for k, _ in READOUTS[:3]) + " |")
    lines += ["", "## Controls (9 ordered cells per replay): whole-track passes, and fragment + timing", "",
              "| control | n | " + " | ".join(lab for _, lab in READOUTS) + " | " + " | ".join(f"{lab} fragment+timing" for _, lab in READOUTS[:3]) + " |",
              "|---" * (len(READOUTS) + 5) + "|"]
    for name in CONTROLS:
        rows = controls.get(name, [])
        if rows:
            lines.append(f"| {name} | {len(rows)} | " + " | ".join(f"{passes(rows, k)}/{len(rows)}" for k, _ in READOUTS) + " | " +
                         " | ".join(f"{with_timing(rows, k, 'fragment')}/{len(rows)}" for k, _ in READOUTS[:3]) + " |")
    # criteria: pairs against the standard read-out on the same data
    lines += ["", "## Pass criteria (against the standard read-out on the same draws)", ""]
    zero = [r for (c, s, v), rs in grid.items() if s == 0 for r in rs]
    for key, lab in READOUTS:
        lines.append(f"- False positives at share 0, {lab}: {passes(zero, key)}/{len(zero)} whole-track passes; with the timing bar "
                     f"{with_timing(zero, key) if key != 'oracle_centred' else '–'}/{len(zero)}.")
    for key, lab in READOUTS[1:3]:
        better, worse = [], []
        for reverse in REVERSE:
            for cells in CELLS:
                for share in SHARES[1:]:
                    rows = rows_of(grid, cells, share, reverse)
                    if not rows:
                        continue
                    d = passes(rows, key) - passes(rows, "standard")
                    if d >= 3 and (share <= 0.05 or cells <= 9):
                        better.append((reverse, cells, share, d))
                    if d <= -3:
                        worse.append((reverse, cells, share, d))
        total_pairs = sum(passes(rows_of(grid, c, s, v), key) for v in REVERSE for c in CELLS for s in SHARES[1:])
        total_std = sum(passes(rows_of(grid, c, s, v), "standard") for v in REVERSE for c in CELLS for s in SHARES[1:])
        verdict = "beats the standard read-out" if better and not worse else "does not beat the standard read-out"
        lines.append(f"- {lab}: {verdict}. Cells gained by 3 or more draws (low-share or few-cell region): {better or 'none'}; lost by 3 or more: {worse or 'none'}. "
                     f"Total passes over the 18 cells with replays: {total_pairs} against {total_std}.")
    ass = controls.get("assemblies_no_order", [])
    for key, lab in READOUTS[1:3]:
        lines.append(f"- Assemblies without order, {lab}: {with_timing(ass, key, 'fragment')}/{len(ass)} under fragment + timing (criterion: at most 1 of 5).")
    # HMM
    lines += ["", "## Check B: the HMM (20 ms bins, 30 states) on 9 ordered cells per replay", "",
              "| reverse | share | n | congruent at 95% (share of events) | at 99% | false alarms at 95% | recall of replays at 95% | AUC of congruence |", "|---|---|---|---|---|---|---|---|"]
    for reverse in REVERSE:
        for share in SHARES:
            rows = [r for r in rows_of(grid, 9, share, reverse) if "hmm" in r]
            if rows and (share > 0 or reverse == REVERSE[0]):
                h = lambda k: np.nanmedian([r["hmm"][k] for r in rows])
                lines.append(f"| {reverse:.1f} | {share:.0%} | {len(rows)} | {h('congruent_95'):.2f} | {h('congruent_99'):.2f} | {h('false_alarm_95'):.2f} | "
                             f"{h('recall_95'):.2f} | {h('auc'):.2f} |")
    for name in CONTROLS:
        rows = [r for r in controls.get(name, []) if "hmm" in r]
        if rows:
            h = lambda k: np.nanmedian([r["hmm"][k] for r in rows])
            lines.append(f"| {name} | – | {len(rows)} | {h('congruent_95'):.2f} | {h('congruent_99'):.2f} | {h('false_alarm_95'):.2f} | {h('recall_95'):.2f} | {h('auc'):.2f} |")
    (OUT / "pairs_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def figure(grid, main_grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, SOFT = "#1d2430", "#6b7380"
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.facecolor": "white"})
    series = [("standard", "complex PCA, standard read-out", "#1f4e79", grid), ("pairs_adjacent", "complex PCA, adjacent-pair read-out", "#d4652f", grid),
              ("pairs_all", "complex PCA, all-pairs read-out", "#f0a070", grid), ("oracle_centred", "direction oracle, centred", "#2e8b57", grid),
              ("oracle_direction", "direction oracle, uncentred", "#9fd3b2", main_grid)]
    fig, axes = plt.subplots(len(REVERSE), len(CELLS), figsize=(4.2 * len(CELLS), 3.4 * len(REVERSE)), sharex=True, sharey=True)
    for i, reverse in enumerate(REVERSE):
        for j, cells in enumerate(CELLS):
            ax = axes[i, j]
            for key, lab, colour, src in series:
                ys = []
                for share in SHARES:
                    rows = rows_of(src, cells, share, reverse)
                    ys.append(passes(rows, key) / len(rows) if rows else np.nan)
                ax.plot([100 * s for s in SHARES], ys, "-o", color=colour, label=lab, lw=2 if key in ("standard", "pairs_adjacent") else 1.4, ms=5)
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
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Pair read-out and centred direction oracle against the standard read-out: direction labels inside the centred fit buy little; "
                 "the uncentred oracle's margin is its read-out", x=0.01, y=0.99, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.945, "Same planted events as figures/segments_g1.png; one centred complex PCA fit per draw. Whole-track order test, p < 0.05, "
             "best-of-outputs null per read-out; 5 draws per point.", fontsize=8.8, color=SOFT)
    fig.text(0.01, 0.920, "Pair read-out: the combination of two components whose phases are most spread (least like a real vector), label-free. "
             "Oracles use the planted direction of every event.", fontsize=8.8, color=SOFT)
    fig.tight_layout(rect=(0, 0.05, 1, 0.895))
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"segments_pairs.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    grid, controls = load()
    main_grid, main_controls = load_main()
    summary(grid, controls, main_grid, main_controls)
    if grid:
        figure(grid, main_grid)
        print("figures/segments_pairs.png")
