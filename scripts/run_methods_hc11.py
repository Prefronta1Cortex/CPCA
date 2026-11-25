"""hc-11: six methods on the same sleep events, scored against the waking map the same way.

Complex PCA, seqNMF and PP-Seq are the outputs of the matched rerun as scored for figures/hc11_sequences.png
(results/hc11_sequences/<session>_<epoch>.pkl; their scores are reused unchanged). Count PCA/ICA assemblies,
SPOTDisClust and Rastermap are run here on the identical events: the rerun's 1500-event subset of the session's
sleep bursts, spike times read exactly as the rerun read them (scripts/plot_hc11_sequences.py checks that the
reproduced complex PCA fit equals the stored one). No method sees the waking session.

Scores (same functions for every method):
  order test: scripts/plot_hc11_sequences.best_map_match, the best output over outputs, running directions and
    senses against the place-field centres, p against the same maximum under permuted centres (1000 draws).
    Count assemblies give no order and are not scored on it.
  assembly locality (descriptive only): cpca.evaluate.best_compactness, do some output's cells have fields close
    together, p against the same best under permuted centres. A route-wide sequence scores no better than chance
    on it by construction (its cells cover the whole track), so it is not a ranking of sequence methods.
  runtime of one fit on the events (complex PCA timed again here; seqNMF its K = 15, L = 120 ms fit including its
    lambda search; PP-Seq its full run including the hyperparameter sweep that chooses its setting).

Usage: python scripts/run_methods_hc11.py [--workers=8]
Cache: results/methods_hc11/<session>_<epoch>.pkl (existing files are reused).
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

OUT = ROOT / "results" / "methods_hc11"
SEQ = ROOT / "results" / "hc11_sequences"
RERUN = ROOT / "results" / "rerun"
HC11_ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
              "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]
NEW = ["count_ica", "spotdis", "rastermap"]


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def new_method(method, times, T, rng):
    """(orders, member sets, details) of one of the new methods."""
    if method == "count_ica":
        from baselines import count_assemblies
        r = count_assemblies.fit(times, T, rng)
        return [], list(r["members"]), {"k": r["k"]}
    if method == "spotdis":
        from baselines import spotdis_baseline
        r = spotdis_baseline.fit(times, T, rng)
        return r["orders"], r["members"], {"n_clusters": r["n_clusters"], "noise_fraction": r["noise_fraction"],
                                           "cluster_sizes": np.bincount(r["labels"][r["labels"] >= 0]).tolist() if r["n_clusters"] else []}
    from baselines import rastermap_baseline
    r = rastermap_baseline.fit(times, T, rng)
    return r["orders"], [], {"n_active": r["n_active"], "n_pcs": r["n_pcs"]}


def job(name, epoch):
    from cpca import event_spike_times, fit, population_bursts
    from cpca.evaluate import best_compactness
    from cpca.hc11 import load_session, nrem_intervals, pyramidal_spikes, session_paths
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_rerun import ORDER_INDEX, fixed_subset
    cached = pickle.load(open(SEQ / f"{name}_{epoch}.pkl", "rb"))
    session = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    ei = ("post", "pre").index(epoch)
    intervals = nrem_intervals(session, epoch)
    by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
    bursts, peaks = population_bursts(pooled, intervals)
    duration = np.diff(bursts, axis=1).ravel()
    sub = fixed_subset(len(peaks), 555 + 2 * ORDER_INDEX[name] + ei)
    assert len(sub) == cached["n_events"]
    times, _ = event_spike_times(by_cell, cell_ids, (peaks - duration / 2)[sub], (peaks + duration / 2)[sub], max_spikes=60)
    T = duration[sub] * 1000
    centres, circular, length = cached["centres"], cached["circular"], cached["length"]

    orders = {m: cached["outputs"][m] for m in ("cpca", "seqnmf", "ppseq", "chance")}
    sets = {m: [np.isfinite(o) for o in orders[m]] for m in ("cpca", "seqnmf", "ppseq")}
    order_scores = {m: cached["scores"][m] for m in orders}
    details, runtime = {}, {}
    t0 = time.perf_counter()
    fit(times, T, rng=np.random.default_rng(31 + ei), ica=False, center=True, scale=True)
    runtime["cpca"] = time.perf_counter() - t0
    seq = pickle.load(open(RERUN / f"{name}.seqnmf.pkl", "rb"))["epochs"][epoch]["seqnmf"]
    runtime["seqnmf"] = next(g["runtime_s"] for g in seq["grid"] if g["K"] == 15 and abs(g["L_ms"] - 120) < 1e-6)
    runtime["ppseq"] = pickle.load(open(RERUN / f"{name}.ppseq.pkl", "rb"))["epochs"][epoch]["ppseq"]["runtime_s"]
    for m in NEW:
        t0 = time.perf_counter()
        orders[m], sets[m], details[m] = new_method(m, times, T, np.random.default_rng(seed("methods-hc11", m, name, epoch)))
        runtime[m] = time.perf_counter() - t0
        if orders[m]:
            order_scores[m] = best_map_match(orders[m], centres, circular, length, np.random.default_rng(seed("methods-hc11-order", m, name, epoch)))
    compact = {m: best_compactness(s, centres, circular, length, np.random.default_rng(seed("methods-hc11-compact", m, name, epoch)))
               for m, s in sets.items() if s}
    result = {"name": name, "epoch": epoch, "circular": circular, "length": length, "n_events": int(len(sub)),
              "n_cells": len(cell_ids), "maps": cached["maps"], "centres": centres, "place": cached["place"],
              "orders": orders, "sets": sets, "order": order_scores, "compact": compact, "runtime_s": runtime, "details": details}
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{name}_{epoch}.tmp"
    pickle.dump(result, open(tmp, "wb"))
    tmp.replace(OUT / f"{name}_{epoch}.pkl")
    return " | ".join(f"{m} {s['abs']:.2f} p={s['p']:.3f}" for m, s in order_scores.items() if s["index"] >= 0) + \
        f" || count_ica k={details['count_ica']['k']}, spotdis {details['spotdis']['n_clusters']} clusters"


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 8)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = "1"
    todo = [(n, e) for n in HC11_ORDER for e in ("post", "pre") if not (OUT / f"{n}_{e}.pkl").exists()]
    print(f"{len(todo)} jobs on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, *a): a for a in todo}
        for f in as_completed(futures):
            try:
                print(f"{futures[f]}: {f.result()}", flush=True)
            except Exception:
                print(f"FAILED {futures[f]}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
