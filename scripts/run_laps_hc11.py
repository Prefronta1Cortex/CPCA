"""hc-11 waking laps: complex PCA against count PCA, cross-validated as on hc-5 (docs/laps_hc11.md).

Per session: laps (cpca.hc11.laps) split into halves stratified by running direction, 3 random splits x 2 folds.
Fitted on one half: complex PCA uncentred (primary: the sequence is in every lap) and centred (the rerun's
read-out, T = each lap's duration); count PCA (z-scored spike counts per lap, components above the shuffle null,
members ordered by weight, plus the seriation of its reduced space); and an averaging reference (each cell's mean
relative time within the lap over training laps of each running direction; it uses the laps' running direction,
so it is not label-free). Scored against place-field centres from the held-out laps only (cpca.hc11.place_centres
on the held-out laps, at least 20 spikes, peak at least 3 Hz), best output, best-of-outputs null
(scripts/plot_hc11_sequences.best_map_match; rotations on the circular tracks, where laps start anywhere).

Usage: python scripts/run_laps_hc11.py [--workers=9]
Output: results/laps_hc11/<session>_<split>_<fold>.pkl, results/laps_hc11/summary.md (--summary)
"""

import os
import pickle
import sys
import traceback
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "laps_hc11"
HC11_ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
              "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]
METHODS = [("cpca_uncentred", "complex PCA, uncentred"), ("cpca_centred", "complex PCA, centred"), ("count", "count PCA"),
           ("average", "averaging reference (uses lap direction)")]


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def heldout_run(run, intervals):
    """The running record with direction set to 0 outside the given lap intervals."""
    from cpca.events import in_intervals
    r = dict(run)
    r["direction"] = np.where(in_intervals(run["t"], intervals), run["direction"], 0)
    return r


def averaged(times, T, dirs):
    warnings.filterwarnings("ignore", "Mean of empty slice")
    rel = np.nanmean(times, axis=-1) / T[:, None]
    return [np.nanmean(rel[dirs == d], axis=0) for d in np.unique(dirs)]


def job(name, split, fold):
    from cpca import event_spike_times
    from cpca.hc11 import laps, load_session, place_centres, pyramidal_spikes, running, session_paths
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_fragment import count_orders
    from scripts.run_rerun import run_cpca
    s = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(s)
    L = laps(run)
    starts, ends, dirs = (np.array([x[i] for x in L]) for i in range(3))
    rng = np.random.default_rng(seed("laps", name, split))
    half = np.zeros(len(L), bool)
    for d in np.unique(dirs):
        idx = rng.permutation(np.flatnonzero(dirs == d))
        half[idx[: idx.size // 2]] = True
    train = half if fold == 0 else ~half
    by_cell, cell_ids, _ = pyramidal_spikes(s, np.column_stack([starts, ends]))
    times, _ = event_spike_times(by_cell, cell_ids, starts[train], ends[train], max_spikes=400)
    T = (ends[train] - starts[train]) * 1000
    centres = place_centres(s, cell_ids, heldout_run(run, np.column_stack([starts[~train], ends[~train]])), min_spikes=20)
    frng = np.random.default_rng(seed("laps-fit", name, split, fold))
    cp = run_cpca(times, T, frng)
    outputs = {"cpca_uncentred": cp["uncentred"]["orders"], "cpca_centred": cp["centred"]["orders"],
               "count": count_orders(times, frng), "average": averaged(times, T, dirs[train])}
    srng = np.random.default_rng(seed("laps-score", name, split, fold))
    scores = {m: (best_map_match(o, centres, run["circular"], run["length"], srng) if o else {"abs": np.nan, "p": np.nan, "n": 0, "index": -1})
              for m, o in outputs.items()}
    out = {"name": name, "split": split, "fold": fold, "n_train": int(train.sum()), "n_test": int((~train).sum()), "circular": run["circular"],
           "n_key": {k: int(np.isfinite(v).sum()) for k, v in centres.items()}, "k": {m: len(o) for m, o in outputs.items()}, "scores": scores}
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT / f"{name}_{split}_{fold}.pkl", "wb"))
    return " | ".join(f"{m} {s_['abs']:.2f} p {s_['p']:.3f}" for m, s_ in scores.items())


def summary():
    R = [pickle.load(open(f, "rb")) for f in sorted(OUT.glob("*_*_*.pkl"))]
    ok = lambda p: bool(np.isfinite(p) and p < 0.05)
    lines = ["# hc-11 waking laps: complex PCA against count PCA", "", "Produced by `scripts/run_laps_hc11.py`. Design: `docs/laps_hc11.md`. "
             "Per session: fits (of 6, 3 splits x 2 folds) whose best output matches the held-out place fields (p < 0.05), and the median |ρ|.", "",
             "| session | track | laps (train/test) | place cells, held-out key | " + " | ".join(lab for _, lab in METHODS) + " |", "|---" * (4 + len(METHODS)) + "|"]
    tot = {m: [0, 0] for m, _ in METHODS}
    for n in HC11_ORDER:
        rs = [r for r in R if r["name"] == n]
        if not rs:
            continue
        cells = []
        for m, _ in METHODS:
            k = sum(ok(r["scores"][m]["p"]) for r in rs)
            tot[m][0] += k
            tot[m][1] += len(rs)
            cells.append(f"{k}/{len(rs)} ({np.nanmedian([r['scores'][m]['abs'] for r in rs]):.2f})")
        key = "/".join(str(int(np.median([r["n_key"][d] for r in rs]))) for d in (1.0, -1.0))
        lines.append(f"| {n} | {'circular' if rs[0]['circular'] else 'linear'} | {rs[0]['n_train']}/{rs[0]['n_test']} | {key} | " + " | ".join(cells) + " |")
    lines.append("| **all fits** | | | | " + " | ".join(f"{tot[m][0]}/{tot[m][1]}" for m, _ in METHODS) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv):
    if "--summary" in argv:
        return summary()
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    todo = [(n, s_, f) for n in HC11_ORDER for s_ in range(3) for f in (0, 1) if not (OUT / f"{n}_{s_}_{f}.pkl").exists()]
    print(f"{len(todo)} jobs on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, *a): a for a in todo}
        for fut in as_completed(futures):
            try:
                print(f"{futures[fut]}: {fut.result()}", flush=True)
            except Exception:
                print(f"FAILED {futures[fut]}:\n{traceback.format_exc()}", flush=True)
    summary()


if __name__ == "__main__":
    main(sys.argv[1:])
