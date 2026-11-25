"""Topology test: does closeness of cells in the reduced space match closeness on the map?

For each data set, closeness of every pair of cells in a method's representation is compared with
closeness of their maps (cpca.topology: Spearman over pairs, cell-permutation Mantel null).

Representations (all from sleep or event spikes only; the map is used only to score):
- "cPCA reduced":   centred complex PCA, components above the shuffle null, Re(C_K) normalised
                    (co-membership and timing);
- "cPCA magnitude": the same components, |C_K| normalised (co-membership only, timing ignored);
- "cPCA full":      all components, i.e. the complex correlation of the arrows (no reduction);
- "cPCA uncentred": uncentred complex PCA, components above the shuffle null;
- "count PCA":      z-scored spike counts per event, components above the shuffle null
                    (who fires together, no timing at all; Lopes-dos-Santos et al., 2013).

Data:
- hc-11: 8 sessions, POST and PRE sleep, all population bursts; map = direction-resolved rate
  maps on the maze, place cells only (a field in either direction); PRE is the control;
- synthetic (sim.routes): null conditions (volleys only; assemblies unrelated to the map) for
  calibration, a control in which assemblies follow the map but fire without order, and planted
  routes for power; map = Gaussian bumps at the planted positions, one map per planted map;
- hc-5: wheel runs, fit on half of the runs, map = time rate maps of the other half (the splits of
  scripts/run_hc5.py), aligned and unaligned windows.

Usage: python scripts/run_topology.py [--workers=10] [--only=hc11,synthetic,hc5]
Results: results/topology/<family>/<job>.pkl (existing files are skipped).
"""

import os
import pickle
import sys
import time
import traceback
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpca import fit  # noqa: E402
from cpca.decode import readout  # noqa: E402
from cpca.decompose import decompose, n_above_null, prepare, shuffle_null  # noqa: E402
from cpca.theory import eigh_desc, lag_matrix  # noqa: E402
from cpca.topology import mantel, map_similarity, reduced_similarity  # noqa: E402

OUT = ROOT / "results" / "topology"
N_PERM = 2000
N_SHUFFLES = 100


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


# ----------------------------------------------------------------------------- representations
def cpca_similarities(times, T, rng):
    """Pairwise closeness in the cPCA representations (all cells). Returns dict name -> (N, N), and k per convention."""
    out, ks = {}, {}
    for conv, centred in (("centred", True), ("uncentred", False)):
        res = fit(times, T, rng=rng, ica=False, center=centred, scale=centred, n_shuffles=N_SHUFFLES)
        values, vectors, _, _ = decompose(res["encoded"], center=centred, scale=centred)
        k = res["n_kept"]
        ks[conv] = k
        if centred:
            out["cPCA reduced"] = reduced_similarity(values, vectors, k=k, kind="phase")
            out["cPCA magnitude"] = reduced_similarity(values, vectors, k=k, kind="magnitude")
            out["cPCA full"] = reduced_similarity(values, vectors, k=None, kind="phase")
            out["_readout_orders"] = [np.where(r["included"], r["times_ms"], np.nan) for r in res["readouts"]]
        else:
            out["cPCA uncentred"] = reduced_similarity(values, vectors, k=k, kind="phase")
    return out, ks


def count_similarity(times, rng):
    """Count PCA: z-scored spike counts per event, components above the shuffle null."""
    counts = np.isfinite(times).sum(axis=-1).astype(float)
    sd = counts.std(axis=0)
    Z = np.where(sd > 0, (counts - counts.mean(axis=0)) / np.where(sd > 0, sd, 1), 0.0)
    values, vectors = eigh_desc(lag_matrix(Z))
    k, _ = n_above_null(values, shuffle_null(Z, rng, n_shuffles=N_SHUFFLES), centred=True)
    return reduced_similarity(values, np.real(vectors), k=k, kind="phase"), k


def score_all(sims, M, cells, rng):
    """Mantel test of every representation against the map, on the given cells."""
    out = {}
    for name, S in sims.items():
        if name.startswith("_"):
            continue
        out[name] = mantel(S[np.ix_(cells, cells)], M, rng, n_perm=N_PERM)
    return out


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pickle.dump(result, open(tmp, "wb"))
    tmp.replace(path)


# ----------------------------------------------------------------------------- hc-11
HC11_ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
              "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]


def hc11_job(name, epoch):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, place_maps, pyramidal_spikes, running, session_paths
    t0 = time.perf_counter()
    path = next(p for p in session_paths() if p.name.startswith(name))
    session = load_session(path)
    run = running(session)
    intervals = nrem_intervals(session, epoch)
    by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
    bursts, peaks = population_bursts(pooled, intervals)
    duration = np.diff(bursts, axis=1).ravel()
    maps, n_spikes, mid = place_maps(session, cell_ids, run)
    centres = place_centres(session, cell_ids, run)
    cells = np.flatnonzero(np.isfinite(centres[1.0]) | np.isfinite(centres[-1.0]))
    M = map_similarity(maps[cells].reshape(len(cells), -1))
    rng = np.random.default_rng(seed("topology", name, epoch))
    # all events, as in the main hc-11 analysis (scripts/run_hc11.py)
    times, _ = event_spike_times(by_cell, cell_ids, peaks - duration / 2, peaks + duration / 2)
    sims, ks = cpca_similarities(times, duration * 1000, rng)
    sims["count PCA"], ks["count"] = count_similarity(times, rng)
    scores = score_all(sims, M, cells, rng)
    keep = {k: v[np.ix_(cells, cells)].astype(np.float32) for k, v in sims.items() if not k.startswith("_")}
    result = {"family": "hc11", "name": name, "epoch": epoch, "circular": run["circular"], "length": run["length"],
              "n_cells": len(cell_ids), "cells": cells, "n_events": len(peaks), "k": ks, "scores": scores,
              "map_similarity": M.astype(np.float32), "similarities": keep, "centres": {s: c[cells] for s, c in centres.items()},
              "peak_direction": np.argmax(maps[cells].max(axis=2), axis=1), "runtime_s": time.perf_counter() - t0}
    save(OUT / "hc11" / f"{name}_{epoch}.pkl", result)
    return f"k={ks}, cPCA reduced rho {scores['cPCA reduced']['rho']:.3f} p {scores['cPCA reduced']['p']:.4f}"


# ----------------------------------------------------------------------------- synthetic
def bump_maps(routes, n_cells, n_bins=40, width=0.08):
    """One Gaussian bump per cell per planted map (zeros off the map), maps concatenated."""
    x = (np.arange(n_bins) + 0.5) / n_bins
    maps = np.zeros((n_cells, len(routes), n_bins))
    for r, route in enumerate(routes):
        maps[route["cells"], r] = np.exp(-0.5 * ((x[None, :] - route["pos"][:, None]) / width) ** 2)
    return maps.reshape(n_cells, -1)


def synthetic_conditions():
    from sim.routes import CONDITIONS, LADDER
    base = dict(participation=0.15, jitter_ms=8.0)
    return {
        "null: volleys only": ("null", dict(base, replay_frac=0.0), dict(n_routes=1, cells_per_route=60), None),
        "null: assemblies unrelated to the map": ("null", dict(base, replay_frac=0.0, assemblies=6), dict(n_routes=1, cells_per_route=60), None),
        "control: assemblies along the map, no order": ("control", dict(base, replay_frac=0.0), dict(n_routes=1, cells_per_route=60), "spatial"),
        "D two routes on different cells, 40% replay": ("planted",) + CONDITIONS["D two routes on different cells, 40% replay"] + (None,),
        "F two routes, both directions, 15% replay, 15% cells": ("planted",) + CONDITIONS["F two routes, both directions, 15% replay, 15% cells"] + (None,),
        "L0 one map on 60 cells, both directions": ("planted",) + LADDER["L0 one map on 60 cells, both directions"] + (None,),
        "L2 two maps on the same cells, both directions": ("planted",) + LADDER["L2 two maps on the same cells, both directions"] + (None,),
        "L3 L2 + partial replays (40% segments)": ("planted",) + LADDER["L3 L2 + partial replays (40% segments)"] + (None,),
        "L6 L2 + partial replays + assemblies (= H)": ("planted",) + LADDER["L6 L2 + partial replays + assemblies (= H)"] + (None,),
    }


N_DRAWS = {"null": 30, "control": 30, "planted": 10}


def synthetic_job(cond, draw):
    from sim.routes import generate, make_routes
    t0 = time.perf_counter()
    kind, gen_kwargs, route_kwargs, special = synthetic_conditions()[cond]
    rng = np.random.default_rng(seed("topology-synthetic", cond, draw))
    routes = make_routes(rng, n_cells=80, **route_kwargs)
    extra = {}
    if special == "spatial":
        membership = rng.integers(0, 6, size=80)
        route = routes[0]
        membership[route["cells"]] = np.minimum((route["pos"] * 6).astype(int), 5)       # six contiguous stretches of the map
        extra["assembly_membership"] = membership
    data = generate(rng, n_cells=80, n_events=3000, routes=routes, **gen_kwargs, **extra)
    maps = bump_maps(routes, 80)
    cells = np.flatnonzero(maps.max(axis=1) > 0)
    M = map_similarity(maps[cells])
    sims, ks = cpca_similarities(data["times"], data["event_ms"], rng)
    sims["count PCA"], ks["count"] = count_similarity(data["times"], rng)
    result = {"family": "synthetic", "condition": cond, "kind": kind, "draw": draw, "k": ks, "scores": score_all(sims, M, cells, rng),
              "runtime_s": time.perf_counter() - t0}
    save(OUT / "synthetic" / f"{cond.split(':')[0].split(' ')[0]}_{zlib.crc32(cond.encode()) % 10000:04d}_d{draw:02d}.pkl", result)
    return f"k={ks}, reduced rho {result['scores']['cPCA reduced']['rho']:.3f}"


# ----------------------------------------------------------------------------- hc-5
def hc5_job(name, condition, repeat, fold):
    from cpca import hc5
    from scripts.run_hc5 import training_data
    t0 = time.perf_counter()
    session = hc5.load_session(name)
    times, T, train, test, _ = training_data(session, condition, repeat, fold)
    key = hc5.time_fields(session, test)
    maps, _ = hc5.rate_maps(session, test)
    cells = np.flatnonzero(np.isfinite(key))
    M = map_similarity(maps[cells])
    rng = np.random.default_rng(seed("topology-hc5", name, condition, repeat, fold))
    sims, ks = cpca_similarities(times, T, rng)
    sims["count PCA"], ks["count"] = count_similarity(times, rng)
    result = {"family": "hc5", "name": name, "condition": condition, "repeat": repeat, "fold": fold, "k": ks,
              "scores": score_all(sims, M, cells, rng), "runtime_s": time.perf_counter() - t0}
    save(OUT / "hc5" / f"{name}_{condition}_r{repeat}_f{fold}.pkl", result)
    return f"k={ks}, uncentred rho {result['scores']['cPCA uncentred']['rho']:.3f}"


# ----------------------------------------------------------------------------- scheduler
def all_jobs(families):
    jobs = []
    if "hc11" in families:
        for name in HC11_ORDER:
            for epoch in ("post", "pre"):
                if not (OUT / "hc11" / f"{name}_{epoch}.pkl").exists():
                    jobs.append((hc11_job, (name, epoch)))
    if "hc5" in families:
        from cpca import hc5
        from scripts.run_hc5 import CONDITIONS as HC5_CONDITIONS
        for name in hc5.session_names():
            for condition in ("aligned", "unaligned"):
                for repeat in range(HC5_CONDITIONS[condition]):
                    for fold in (0, 1):
                        if not (OUT / "hc5" / f"{name}_{condition}_r{repeat}_f{fold}.pkl").exists():
                            jobs.append((hc5_job, (name, condition, repeat, fold)))
    if "synthetic" in families:
        for cond, (kind, *_rest) in synthetic_conditions().items():
            for draw in range(N_DRAWS[kind]):
                p = OUT / "synthetic" / f"{cond.split(':')[0].split(' ')[0]}_{zlib.crc32(cond.encode()) % 10000:04d}_d{draw:02d}.pkl"
                if not p.exists():
                    jobs.append((synthetic_job, (cond, draw)))
    return jobs


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), os.cpu_count() or 2)
    families = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--only=")), ["hc11", "synthetic", "hc5"])
    for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(n, "1")
    jobs = all_jobs(families)
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        running = {pool.submit(fn, *args): (fn.__name__, args) for fn, args in jobs}
        done = 0
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                label = running.pop(fut)
                done += 1
                try:
                    msg = fut.result()
                    print(f"[{done}/{len(jobs)} {time.perf_counter() - t0:6.0f}s] {label[0]} {label[1]}: {msg}", flush=True)
                except Exception:
                    print(f"[{done}/{len(jobs)}] FAILED {label}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
