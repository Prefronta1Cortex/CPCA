"""Table: the recurring-fragment criterion on planted data and hc-11 (results/fragment/).

Usage: python scripts/summarize_fragment.py
Output: results/fragment/summary.md
"""

import glob
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_fragment import HC11_ORDER, OUT, PLANTED  # noqa: E402


def ok(p):
    return bool(np.isfinite(p) and p < 0.05)


def main():
    V = [pickle.load(open(f, "rb")) for f in glob.glob(str(OUT / "validation" / "*.pkl"))]
    lines = ["# The recurring-fragment criterion", "", "Produced by `scripts/run_fragment.py` and `scripts/summarize_fragment.py`. Design: `docs/fragment.md`.", "",
             "## Planted data (80 cells, 60 on the route, 2000 events, 5 draws)", "",
             "| condition | complex PCA bar 1 | complex PCA bar 2 (timing) | complex PCA both | count PCA bar 1 | median \\|ρ\\| complex / count |",
             "|---|---|---|---|---|---|"]
    for c in PLANTED:
        rs = [r for r in V if r["condition"] == c]
        b1, b2 = [ok(r["cpca"]["p"]) for r in rs], [ok(r["p_timing"]) for r in rs]
        med = lambda k: np.nanmedian([r[k]["abs"] for r in rs]) if any(np.isfinite(r[k]["abs"]) for r in rs) else np.nan
        lines.append(f"| {c} | {sum(b1)}/{len(rs)} | {sum(b2)}/{len(rs)} | {sum(a and b for a, b in zip(b1, b2))}/{len(rs)} | "
                     f"{sum(ok(r['count']['p']) for r in rs)}/{len(rs)} | {med('cpca'):.2f} / {med('count'):.2f} |")
    lines += ["", "## hc-11", "", "| session | epoch | events | complex PCA \\|ρ\\| (bar 1 p) | timing p | both | count PCA \\|ρ\\| (bar 1 p) |",
              "|---|---|---|---|---|---|---|"]
    tally = {}
    for e in ("post", "pre", "maze"):
        for n in HC11_ORDER:
            r = pickle.load(open(OUT / "hc11" / f"{n}_{e}.pkl", "rb"))
            both = ok(r["cpca"]["p"]) and ok(r["p_timing"])
            tally.setdefault(e, []).append((ok(r["cpca"]["p"]), ok(r["p_timing"]), both, ok(r["count"]["p"])))
            lines.append(f"| {n} | {e.upper()} | {r['n_events']} | {r['cpca']['abs']:.2f} ({r['cpca']['p']:.3f}) | {r['p_timing']:.3f} | "
                         f"{'pass' if both else '–'} | {r['count']['abs']:.2f} ({r['count']['p']:.3f}) |")
    lines += ["", "| epoch | complex PCA bar 1 | bar 2 | both | count PCA bar 1 |", "|---|---|---|---|---|"]
    for e, v in tally.items():
        v = np.array(v)
        lines.append(f"| {e.upper()} | {v[:, 0].sum()}/8 | {v[:, 1].sum()}/8 | {v[:, 2].sum()}/8 | {v[:, 3].sum()}/8 |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("results/fragment/summary.md")


if __name__ == "__main__":
    main()
