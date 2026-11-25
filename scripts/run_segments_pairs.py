"""Checks A and B of docs/segments.md (addendum) on the G1 grid: the conjugate-pair read-out, the centred direction
oracle, and the Maboudi HMM on a slice.

The data of every grid cell and control are regenerated from the seeds of scripts/run_segments.py (identical events).
Per draw: complex PCA centred (one fit), read out as single components (standard), as adjacent pairs and as all pairs
(cpca/pairs.py); the centred direction oracle (the planted backward replays time-reversed, non-replays reversed at
random, then the standard fit and read-out); whole-track and fragment bar 1 scores with the best-of-outputs null per
output set; the timing bar (30 shuffled refits, each read out all three ways). On 9 ordered cells per replay at shares
0, 5 and 10%: the HMM's congruence (share called congruent; AUC against the planted labels).

Usage: python scripts/run_segments_pairs.py [--workers=9]
Output: results/segments/pairs/<cells>_<share>_<reverse>_<draw>.pkl, results/segments/pairs/control_<name>_<draw>.pkl
"""

import os
import pickle
import sys
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_segments import CELLS, CONTROLS, N_DRAWS, N_NULL, N_TIMING, REVERSE, SHARES, best_abs, control_data, grid_data  # noqa: E402

OUT = ROOT / "results" / "segments" / "pairs"
HMM_CELLS = (9,)
HMM_SHARES = (0.0, 0.05, 0.10)
READOUTS = ("standard", "pairs_adjacent", "pairs_all")


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def readouts(patterns):
    from cpca.direction import order_of
    from cpca.pairs import pair_orders
    return {"standard": [order_of(patterns[:, j], member_frac=0.2) for j in range(patterns.shape[1])],
            "pairs_adjacent": pair_orders(patterns, how="adjacent"), "pairs_all": pair_orders(patterns, how="all")}


def reverse_events(times, T, flip):
    out = times.copy()
    out[flip] = T[flip, None, None] - times[flip]
    return out


def analyse(data, rng, hmm=False):
    from cpca import fit
    from cpca.fragment import best_fragment
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_maze_rest import shuffle_timing
    times, T, pos, label, direction = data["times"], data["T"], data["centres"], data["label"], data["direction"]
    key = {1.0: pos * 100}
    rep = label > 0
    out = {"n_replays": int(rep.sum()), "ordered_per_replay": float(np.mean(data["n_ordered"][rep])) if rep.any() else 0.0}
    res = fit(times, T, rng=rng, ica=False)
    out["n_kept"] = res["n_kept"]
    outputs = readouts(res["patterns"]) if res["n_kept"] else {k: [] for k in READOUTS}
    # centred direction oracle: play every event in one direction (backward replays reversed; others at random)
    flip = np.where(direction != 0, direction < 0, rng.random(direction.size) < 0.5)
    res_o = fit(reverse_events(times, T, flip), T, rng=rng, ica=False)
    outputs["oracle_centred"] = readouts(res_o["patterns"])["standard"] if res_o["n_kept"] else []
    out["n_kept_oracle"] = res_o["n_kept"]
    out["n_outputs"] = {k: len(v) for k, v in outputs.items()}
    out["whole"] = {k: best_map_match(v, key, False, 100.0, rng, n_null=N_NULL) if v else {"abs": np.nan, "p": np.nan} for k, v in outputs.items()}
    out["fragment"] = {k: best_fragment(v, key, 100.0, rng, n_null=N_NULL) if v else {"abs": np.nan, "p": np.nan} for k, v in outputs.items()}
    shuffled = {k: [] for k in READOUTS}
    for s in range(N_TIMING):
        r2 = np.random.default_rng(seed("segments-pairs-timing", rng.integers(1 << 30), s))
        rs = fit(shuffle_timing(times, r2), T, rng=r2, ica=False)
        o = readouts(rs["patterns"]) if rs["n_kept"] else {k: [] for k in READOUTS}
        for k in READOUTS:
            shuffled[k].append(best_abs(o[k], pos) if o[k] else 0.0)
    out["timing"] = {}
    for k, arr in shuffled.items():
        real = out["whole"][k]["abs"]
        arr = np.asarray(arr)
        out["timing"][k] = {"shuffled_abs": arr, "p": float((1 + np.sum(arr >= real)) / (1 + arr.size)) if np.isfinite(real) else np.nan}
    if hmm:
        from baselines import hmm_baseline
        from scripts.run_baselines import auc, blocks_from_times
        t0 = time.perf_counter()
        blocks = blocks_from_times(times, T, 0.02)
        model, events = hmm_baseline.fit(blocks, rng, n_states=30)
        p = hmm_baseline.congruence(model, events, rng, n_shuffles=100)
        kept = hmm_baseline.kept_events(blocks)
        out["hmm"] = {"congruent_99": float(np.mean(p < 0.01)), "congruent_95": float(np.mean(p < 0.05)), "n_events": int(kept.sum()),
                      "auc": auc(-p, label[kept]) if rep.any() else np.nan,
                      "recall_95": float(np.mean(p[label[kept] > 0] < 0.05)) if rep.any() else np.nan,
                      "false_alarm_95": float(np.mean(p[label[kept] == 0] < 0.05)), "runtime_s": time.perf_counter() - t0}
    return out


def grid_job(cells, share, reverse, draw):
    data, _ = grid_data(cells, share, reverse, draw)
    rng = np.random.default_rng(seed("segments-pairs", cells, share, reverse, draw))
    out = analyse(data, rng, hmm=cells in HMM_CELLS and share in HMM_SHARES)
    out.update(cells=cells, share=share, reverse=reverse, draw=draw)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT / f"{cells}_{share:.2f}_{reverse:.1f}_{draw}.pkl", "wb"))
    return out


def control_job(name, draw):
    data, _ = control_data(name, draw)
    rng = np.random.default_rng(seed("segments-pairs-control", name, draw))
    out = analyse(data, rng, hmm=True)
    out.update(control=name, draw=draw)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT / f"control_{name}_{draw}.pkl", "wb"))
    return out


def fmt(out):
    p = lambda k: (f"{out['whole'][k]['p']:.3f}" if np.isfinite(out['whole'][k]['p']) else "  –  ")
    t = lambda k: (f"{out['timing'][k]['p']:.2f}" if k in out["timing"] and np.isfinite(out["timing"][k]["p"]) else " – ")
    h = out.get("hmm")
    return (f"k={out['n_kept']} | whole-track p: standard {p('standard')} pairs {p('pairs_adjacent')} all {p('pairs_all')} oracle-centred {p('oracle_centred')} | "
            f"timing p {t('standard')} / {t('pairs_adjacent')} / {t('pairs_all')}" +
            (f" | HMM congruent95 {h['congruent_95']:.2f} AUC {h['auc']:.2f} ({h['runtime_s']:.0f}s)" if h else ""))


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    jobs = []
    for cells in CELLS:
        for share in SHARES:
            for reverse in (REVERSE if share > 0 else (0.5,)):
                for draw in range(N_DRAWS):
                    if not (OUT / f"{cells}_{share:.2f}_{reverse:.1f}_{draw}.pkl").exists():
                        jobs.append((grid_job, (cells, share, reverse, draw)))
    for name in CONTROLS:
        for draw in range(N_DRAWS):
            if not (OUT / f"control_{name}_{draw}.pkl").exists():
                jobs.append((control_job, (name, draw)))
    jobs.sort(key=lambda j: not (j[0] is control_job or (j[1][0] in HMM_CELLS and j[1][1] in HMM_SHARES)))   # HMM jobs first
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *a): a for fn, a in jobs}
        for f in as_completed(futures):
            try:
                print(f"{futures[f]}: {fmt(f.result())}", flush=True)
            except Exception:
                print(f"FAILED {futures[f]}:\n{traceback.format_exc()}", flush=True)
    print(f"done in {(time.perf_counter() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
