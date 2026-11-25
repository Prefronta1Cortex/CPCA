"""Table: complex PCA and count PCA on awake rest on the maze, with the timing control (results/maze_rest/).

Usage: python scripts/summarize_maze_rest.py
Output: results/maze_rest/summary.md
"""

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_maze_rest import HC11_ORDER, OUT  # noqa: E402


def p_cell(s, key="p"):
    return "–" if not s or not np.isfinite(s.get(key, np.nan)) else f"{s[key]:.3f}"


def main():
    R = {n: pickle.load(open(OUT / f"{n}.pkl", "rb")) for n in HC11_ORDER}
    Tm = {n: pickle.load(open(OUT / f"{n}_timing_maze.pkl", "rb")) for n in HC11_ORDER}
    Tu = {n: pickle.load(open(OUT / f"{n}_timing_maze_uncentred.pkl", "rb")) for n in HC11_ORDER}
    Tp = {n: pickle.load(open(OUT / f"{n}_timing_post.pkl", "rb")) for n in HC11_ORDER}
    lines = ["# hc-11: awake rest on the maze", "", "Produced by `scripts/run_maze_rest.py` and `scripts/summarize_maze_rest.py`. Design: `docs/maze_rest.md`.", "",
             "## Events and replay content", "", "| session | rest (min) | events | place cells active per event (median) | clear replays (tested) | "
             "PRE control: clear replays |", "|---|---|---|---|---|---|"]
    for n in HC11_ORDER:
        r = R[n]
        lines.append(f"| {n} | {r['rest_s'] / 60:.0f} | {r['n_events']} | {r['place_active_median']:.0f} | {r['replay']['share']:.0%} ({r['replay']['n_tested']}) | "
                     f"{r['pre_replay']['share']:.0%} |")
    lines += ["", "## Order and topology (p; planned analyses)", "",
              "| session | epoch | complex PCA centred, best component | uncentred | seriation, complex PCA | seriation, count PCA | topology, complex PCA | topology, count PCA |",
              "|---|---|---|---|---|---|---|---|"]
    for n in HC11_ORDER:
        for ep, lab in (("maze", "MAZE rest"), ("pre", "PRE control")):
            a = R[n][ep]
            lines.append(f"| {n} | {lab} | " + " | ".join(p_cell(a.get(k)) for k in ("cpca_centred", "cpca_uncentred", "seriation_cpca", "seriation_count",
                                                                                     "topology_cpca", "topology_count")) + " |")
    for ep, lab in (("maze", "MAZE rest"), ("pre", "PRE control")):
        lines.append(f"| **sessions with p < 0.05** | {lab} | " + " | ".join(
            f"{sum(bool(R[n][ep].get(k)) and np.isfinite(R[n][ep][k]['p']) and R[n][ep][k]['p'] < 0.05 for n in HC11_ORDER)}/8"
            for k in ("cpca_centred", "cpca_uncentred", "seriation_cpca", "seriation_count", "topology_cpca", "topology_count")) + " |")
    lines += ["", "## Timing control (added after the first results): the same events with spike trains shuffled among each event's active cells", "",
              "Share of 100 timing-shuffled fits whose best output matches the map (p < 0.05), and the share whose |ρ| is at least the real fit's "
              "(the timing p: small = the real order is more than co-firing explains). \\|ρ\\|: best output's absolute Spearman correlation.", "",
              "| session | MAZE centred: real \\|ρ\\| (p) | shuffled pass | timing p | MAZE uncentred: real \\|ρ\\| (p) | shuffled pass | timing p | "
              "POST sleep centred: real \\|ρ\\| (p) | shuffled pass | timing p |", "|---" * 10 + "|"]
    for n in HC11_ORDER:
        cells = []
        for T in (Tm, Tu, Tp):
            t = T[n]
            ok = sum(np.isfinite(r["p"]) and r["p"] < 0.05 for r in t["rows"])
            cells += [f"{t['real']['abs']:.2f} ({t['real']['p']:.3f})", f"{ok}/100", f"{t['p_timing']:.3f}"]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("results/maze_rest/summary.md")


if __name__ == "__main__":
    main()
