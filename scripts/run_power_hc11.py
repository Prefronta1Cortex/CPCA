"""Step 1: how much replay does each method need? The spike-in power test (scripts/run_spikein_hc11.py) for more methods.

The same benchmark: real PRE sleep events of each hc-11 session, a share f of them turned into ideal replays of
the waking map (each place cell that fires keeps its spike count; its spikes move to its field's position along
the track, forward or reverse). For the matched 1500-event subset, the spiked data sets are regenerated with the
seeds of run_spikein_hc11, so every method sees exactly the data complex PCA saw there (results/spikein/).

  cpca_all   complex PCA on all PRE events instead of the 1500-event subset (more events, lower noise bound);
  cpca_uncentred  complex PCA uncentred on the 1500 events (the same spiked data sets as results/spikein/, centred);
  sparse     event-sparse complex templates (cpca/sparse.py), first version (raw inner products), on the 1500 events;
  sparse_norm  the same with cosine matching (cpca/sparse.py, normalize=True), on the 1500 events;
  rastermap  Rastermap (baselines/rastermap_baseline.py) on the 1500 events;
  ppseq      PP-Seq (the authors' package) on the 1500 events, the two sessions with the most place cells,
             f = 0, 5, 10, 20%, one draw: to keep it short, its hyperparameters are not swept again but fixed to
             the setting its own sweep chose on the same PRE events in the matched rerun (results/rerun/); one
             refit with the rerun's sampler schedule per data set.
Every output is scored as for every method: best output against the waking map, best-of-outputs null.

Usage: python scripts/run_power_hc11.py [--methods=cpca_all,sparse,rastermap] [--workers=8]
       python scripts/run_power_hc11.py --methods=ppseq --workers=2
Output: results/power/<session>_<method>_<f>.pkl
"""

import os
import pickle
import sys
import tempfile
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_spikein_hc11 import FRACTIONS, HC11_ORDER, N_DRAWS, ideal  # noqa: E402

OUT = ROOT / "results" / "power"
RERUN = ROOT / "results" / "rerun"
PPSEQ_SESSIONS = ["Achilles_10252013", "Achilles_11012013"]
PPSEQ_FRACTIONS = [0.0, 0.05, 0.10, 0.20]


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def base_events(name, all_events):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    from scripts.run_rerun import ORDER_INDEX, fixed_subset
    session = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(session)
    intervals = nrem_intervals(session, "pre")
    by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
    bursts, peaks = population_bursts(pooled, intervals)
    duration = np.diff(bursts, axis=1).ravel()
    sub = np.arange(len(peaks)) if all_events else fixed_subset(len(peaks), 555 + 2 * ORDER_INDEX[name] + 1)
    times, _ = event_spike_times(by_cell, cell_ids, (peaks - duration / 2)[sub], (peaks + duration / 2)[sub], max_spikes=60)
    return times, duration[sub] * 1000, place_centres(session, cell_ids, run), run


def outputs(method, times, T, rng, name):
    if method == "cpca_all":
        from scripts.run_rerun import run_cpca
        return run_cpca(times, T, rng)["centred"]["orders"]
    if method == "cpca_uncentred":
        from scripts.run_rerun import run_cpca
        return run_cpca(times, T, rng)["uncentred"]["orders"]
    if method in ("sparse", "sparse_norm"):
        from cpca import sparse
        return sparse.fit(times, T, rng, normalize=method == "sparse_norm")["orders"]
    if method == "rastermap":
        from baselines import rastermap_baseline
        return rastermap_baseline.fit(times, T, rng)["orders"]
    from baselines import ppseq_baseline as pp
    setting = pickle.load(open(RERUN / f"{name}.ppseq.pkl", "rb"))["epochs"]["pre"]["ppseq"]["best_setting"]
    plan = pp.plan_fit(times, T, rng)
    with tempfile.TemporaryDirectory() as wd:
        parsed, _, _ = pp.refit_best(plan, [{"setting": setting, "heldout_log_like": 0.0}], Path(wd), threads=pp.THREADS)
    return [np.asarray(t["lag_ms"], float) for t in parsed["types"]]


def job(name, method, f):
    from scripts.plot_hc11_sequences import best_map_match
    base, T, centres, run = base_events(name, method == "cpca_all")
    draws = 1 if method == "ppseq" else N_DRAWS
    rows = []
    for draw in range(draws):
        if method == "cpca_all":
            rng = np.random.default_rng(seed("power-all", name, f, draw))
        else:
            rng = np.random.default_rng(seed("spikein", name, "ideal", f, draw))
        k = int(round(f * len(T)))
        chosen = rng.choice(len(T), k, replace=False)
        times = ideal(base, T, chosen, centres, rng) if k else base
        t0 = time.perf_counter()
        orders = outputs(method, times, T, np.random.default_rng(seed("power-fit", name, method, f, draw)), name)
        runtime = time.perf_counter() - t0
        s = best_map_match(orders, centres, run["circular"], run["length"], np.random.default_rng(seed("power-score", name, method, f, draw))) \
            if orders else {"abs": np.nan, "p": np.nan, "n": 0}
        rows.append({"draw": draw, "k": k, "n_events": int(len(T)), "n_outputs": len(orders), "abs": s["abs"], "p": s["p"], "n": s["n"], "runtime_s": runtime})
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump({"name": name, "method": method, "f": f, "rows": rows}, open(OUT / f"{name}_{method}_{f:.2f}.pkl", "wb"))
    return f"{sum(np.isfinite(r['p']) and r['p'] < 0.05 for r in rows)}/{len(rows)} (|ρ| {np.nanmedian([r['abs'] for r in rows]):.2f}, {np.median([r['runtime_s'] for r in rows]):.0f}s per fit)"


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 8)
    methods = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--methods=")), ["cpca_all", "sparse", "rastermap"])
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = "1"
    todo = []
    for m in methods:
        sessions, fracs = (PPSEQ_SESSIONS, PPSEQ_FRACTIONS) if m == "ppseq" else (HC11_ORDER, FRACTIONS)
        todo += [(n, m, f) for n in sessions for f in fracs if not (OUT / f"{n}_{m}_{f:.2f}.pkl").exists()]
    todo.sort(key=lambda a: (a[1] != "rastermap", a[0] not in PPSEQ_SESSIONS))          # longest jobs first
    print(f"{len(todo)} jobs on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, *a): a for a in todo}
        for fut in as_completed(futures):
            try:
                print(f"{futures[fut]}: {fut.result()}", flush=True)
            except Exception:
                print(f"FAILED {futures[fut]}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
