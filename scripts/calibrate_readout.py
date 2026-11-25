"""Calibrate the read-out check (cpca/readout_check.py; docs/readout_check.md) on planted single-route data.

Conditions (sim/segments.py: constant-speed partial replays, latency, assemblies; 2500 events, 120 cells, about 13
ordered cells per replay, 6 draws):
  replays both directions 10%, 20%; forward only 20%;  no replays;  map-ordered assemblies without order.
Second design (docs/readout_check.md): fit on one half of the events, timing check on the other half (both folds);
per kept component (centred and uncentred, PCA and ICA rotation): its timing p, its circularity (first design, kept
for the record), and whether it matches the planted route (members on the route >= 6, |Spearman| with positions,
1000-permutation p < 0.05).

Usage: python scripts/calibrate_readout.py
Output: results/readout_check/calibration.pkl, printed table
"""

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "readout_check"
CONDITIONS = {"replays both ways 10%": dict(replay_frac=0.10), "replays both ways 20%": dict(replay_frac=0.20),
              "replays forward 20%": dict(replay_frac=0.20, reverse_frac=0.0), "no replays": dict(replay_frac=0.0),
              "map-ordered assemblies, no order": dict(replay_frac=0.0, spatial=True)}


def route_match(order, pos, rng, n_perm=1000, min_n=6):
    from scipy.stats import rankdata
    m = np.isfinite(order) & np.isfinite(pos)
    if m.sum() < min_n:
        return np.nan, np.nan
    a, b = rankdata(order[m]), rankdata(pos[m])
    a, b = a - a.mean(), b - b.mean()
    r = abs(a @ b) / np.sqrt((a @ a) * (b @ b))
    null = np.array([abs(a @ rng.permutation(b)) for _ in range(n_perm)]) / np.sqrt((a @ a) * (b @ b))
    return float(r), float((1 + np.sum(null >= r)) / (1 + n_perm))


def job(cond, draw):
    from cpca import fit
    from cpca.readout_check import outputs
    from sim.segments import generate, n_route_for, spatial_assemblies
    kw = dict(CONDITIONS[cond])
    spatial = kw.pop("spatial", False)
    rng = np.random.default_rng(1000 * draw + len(cond))
    n_route = n_route_for(13)
    extra = {}
    if spatial:
        tmp = generate(np.random.default_rng(draw), n_route=n_route, n_events=10, replay_frac=0.0)
        extra["assembly_membership"] = spatial_assemblies(rng, 120, tmp["route"], tmp["centres"])
        d = generate(np.random.default_rng(draw), n_route=n_route, n_events=2500, **kw, **extra)
    else:
        d = generate(rng, n_route=n_route, n_events=2500, **kw)
    from cpca.readout_check import timing_check
    rows = []
    half = np.random.default_rng(draw + 99).permutation(d["times"].shape[0])
    halves = (half[: half.size // 2], half[half.size // 2:])
    for fold, (tr, te) in enumerate((halves, halves[::-1])):
        for conv, c in (("centred", True), ("uncentred", False)):
            for rot in ("PCA", "ICA"):
                _, comps = timing_check(d["times"][tr], d["T"][tr], d["times"][te], d["T"][te], np.random.default_rng(draw + 7 * fold),
                                        center=c, scale=c, ica=(rot == "ICA"))
                for o in comps:
                    rho, p = route_match(o["order"], d["centres"], np.random.default_rng(draw + o["index"]))
                    rows.append({"condition": cond, "draw": draw, "fold": fold, "convention": conv, "rotation": rot,
                                 "circularity": o["circularity"], "p_timing": o["p_timing"], "rho": rho, "p": p,
                                 "n_members": int(o["members"].sum())})
    return rows


def main():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    with ProcessPoolExecutor(9) as pool:
        rows = [r for rs in pool.map(job, *zip(*[(c, d) for c in CONDITIONS for d in range(6)])) for r in rs]
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(rows, open(OUT / "calibration.pkl", "wb"))
    ok = lambda r: np.isfinite(r["p"]) and r["p"] < 0.05
    seqlike = lambda r: r["p_timing"] < 0.05
    thresholds = {}
    for conv in ("centred", "uncentred"):
        for rot in ("PCA", "ICA"):
            print(f"\n{conv}, {rot}:")
            for cond in CONDITIONS:
                rs = [r for r in rows if r["convention"] == conv and r["rotation"] == rot and r["condition"] == cond]
                match = [r for r in rs if ok(r)]
                non = [r for r in rs if not ok(r)]
                print(f"  {cond:34s} components {len(rs):3d} | route-matching {len(match):3d}, kept by the check {np.mean([seqlike(r) for r in match]) if match else np.nan:.0%} "
                      f"| not matching {len(non):3d}, kept {np.mean([seqlike(r) for r in non]) if non else np.nan:.0%} | all components kept {np.mean([seqlike(r) for r in rs]) if rs else np.nan:.0%}")
    pickle.dump(thresholds, open(OUT / "thresholds.pkl", "wb"))


if __name__ == "__main__":
    main()
