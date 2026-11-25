"""Table: are complex PCA's waking components found within its sleep (POST) components? (results/containment/)

Usage: python scripts/summarize_containment.py
Output: results/containment/summary.md
"""

import glob
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_containment import HC11_ORDER, OUT, SYNTHETIC  # noqa: E402


def ok(p):
    return bool(np.isfinite(p) and p < 0.05)


def main():
    lines = ["# Are the waking components found within the sleep components?", "",
             "Produced by `scripts/run_containment.py` and `scripts/summarize_containment.py`. Design: `docs/containment.md`. "
             "Waking inside sleep: mean share of the kept waking components (uncentred, laps) inside the span of the kept sleep components "
             "(centred, bursts) and their conjugates, against the same with the sleep fit's cell identities permuted (its dimension kept).", "",
             "## Planted data (5 draws per condition)", "",
             "| condition | draws passing (p < 0.05) | share inside sleep (median) vs null | waking components contained individually |",
             "|---|---|---|---|"]
    V = [pickle.load(open(f, "rb")) for f in glob.glob(str(OUT / "synthetic" / "*.pkl"))]
    for c in SYNTHETIC:
        rs = [r for r in V if r["condition"] == c]
        wk = [r for r in rs if r["k_sleep"]]
        if not wk:
            lines.append(f"| {c} | 0/{len(rs)} (no sleep components) | – | – |")
            continue
        lines.append(f"| {c} | {sum(ok(r['p']) for r in rs)}/{len(rs)} | {np.median([r['real'] for r in wk]):.2f} vs {np.median([r['null_mean'] for r in wk]):.2f} | "
                     f"{sum(int((r['per_component_p'] < 0.05).sum()) for r in wk)}/{sum(r['k_wake'] for r in wk)} |")
    lines += ["", "## hc-11, sleep after the maze (all NREM bursts)", "",
              "| session | laps | sleep events | cells | components, waking / sleep | share inside sleep vs null (p) | waking components contained individually |",
              "|---|---|---|---|---|---|---|"]
    R = {n: pickle.load(open(OUT / "hc11" / f"{n}_post.pkl", "rb")) for n in HC11_ORDER}
    for n in HC11_ORDER:
        r = R[n]
        lines.append(f"| {n} | {r['n_laps']} | {r['n_events']} | {r['n_cells']} | {r['k_wake']} / {r['k_sleep']} | "
                     f"{r['real']:.2f} vs {r['null_mean']:.2f} ({r['p']:.3f}) | {int((r['per_component_p'] < 0.05).sum())}/{r['k_wake']} |")
    lines += ["", f"Sessions passing: {sum(ok(R[n]['p']) for n in HC11_ORDER)}/8. Waking components contained individually: "
              f"{sum(int((R[n]['per_component_p'] < 0.05).sum()) for n in HC11_ORDER)}/{sum(R[n]['k_wake'] for n in HC11_ORDER)}."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[6:]))


if __name__ == "__main__":
    main()
