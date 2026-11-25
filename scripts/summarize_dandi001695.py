"""Tables: complex PCA on DANDI 001695 (results/dandi001695/, scripts/run_dandi001695.py).

Usage: python scripts/summarize_dandi001695.py
Output: results/dandi001695/summary.md
"""

import glob
import pickle
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_dandi001695 import OUT  # noqa: E402

EPOCHS = [("post", "POST"), ("pre", "PRE"), ("maze", "MAZE rest")]


def ok(p):
    return bool(np.isfinite(p) and p < 0.05)


def binom_sf(k, n, p=0.05):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def section(tag):
    R = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(str(OUT / tag / "*.pkl")))]
    if not R:
        return []
    lines = [f"## Cells: {tag.replace('+', ' + ')} pyramidal", "",
             "| session | epoch | events | place cells active per event | clear replays | whole-track \\|ρ\\| (p; timing p) | fragment \\|ρ\\| (p; timing p) | "
             "ripple-gated: events, whole-track p; timing p |", "|---|---|---|---|---|---|---|---|"]
    tally = {}
    for ep, lab in EPOCHS:
        for r in sorted([r for r in R if r["epoch"] == ep], key=lambda r: r["name"]):
            if "all" not in r:
                lines.append(f"| {r['name']} | {lab} | {r['n_events']} | – | – | skipped | – | – |")
                continue
            a = r["all"]
            w, f = a["whole"], a["fragment"]
            t = tally.setdefault(ep, dict(n=0, w=0, wt=0, f=0, ft=0, rn=0, rwt=0, mice=set()))
            t["n"] += 1
            t["w"] += ok(w["p"])
            t["f"] += ok(f["p"])
            if ok(w["p"]) and ok(a["timing_whole"]):
                t["wt"] += 1
                t["mice"].add(r["mouse"])
            t["ft"] += ok(f["p"]) and ok(a["timing_fragment"])
            rip = "–"
            if "ripple" in r:
                rr = r["ripple"]
                t["rn"] += 1
                t["rwt"] += ok(rr["whole"]["p"]) and ok(rr["timing_whole"])
                rip = f"{rr['n_events']}, {rr['whole']['p']:.3f}; {rr['timing_whole']:.3f}"
            fr = f"{f['abs']:.2f} ({f['p']:.3f}; {a['timing_fragment']:.3f})" if np.isfinite(f["abs"]) else "no output with 8 cells in a stretch"
            lines.append(f"| {r['name']} | {lab} | {r['n_events']} | {r['place_active_median']:.0f} | {r['replay']['clear_share']:.1%} | "
                         f"{w['abs']:.2f} ({w['p']:.3f}; {a['timing_whole']:.3f}) | {fr} | {rip} |")
    lines += ["", "| epoch | whole-track order | whole-track order + timing (mice) | fragment | fragment + timing | ripple-gated order + timing |",
              "|---|---|---|---|---|---|"]
    for ep, lab in EPOCHS:
        t = tally.get(ep)
        if not t:
            continue
        lines.append(f"| {lab} | {t['w']}/{t['n']} | {t['wt']}/{t['n']} ({len(t['mice'])} mice; binomial p {binom_sf(t['wt'], t['n']):.3f}) | {t['f']}/{t['n']} | "
                     f"{t['ft']}/{t['n']} | {t['rwt']}/{t['rn']} |")
    return lines + [""]


def main():
    lines = ["# Complex PCA on DANDI 001695", "", "Produced by `scripts/run_dandi001695.py` and `scripts/summarize_dandi001695.py`. Design: `docs/dandi001695.md`. "
             "p values: best output against permuted field centres; timing p: share of 100 refits on spike trains shuffled among each event's "
             "active cells that score at least as high (50 for the ripple-gated events). Clear replays: the paper's Bayesian analysis, both shuffles.", ""]
    for tag in ("CA1", "CA1+CA3"):
        lines += section(tag)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("results/dandi001695/summary.md")


if __name__ == "__main__":
    main()
