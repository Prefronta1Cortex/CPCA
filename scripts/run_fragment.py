"""The recurring-fragment criterion (cpca/fragment.py; docs/fragment.md): validation on planted data, then hc-11.

A session (or planted draw) passes if complex PCA's best fragment beats the fragment null (bar 1) and the real
statistic beats the same refit after spike trains are shuffled among each event's active cells (bar 2). Count PCA
is scored on bar 1 only (it uses spike counts, so the timing shuffle leaves it unchanged); its outputs are each
component's members ordered by weight, and the seriation of its reduced space.

  validate   planted conditions, 5 draws, 50 timing shuffles, 500 null permutations;
  hc11       POST (all NREM bursts), PRE (the same number of NREM bursts, drawn at random; if PRE has fewer, POST
             is subsampled to PRE's number instead), MAZE rest (all immobility bursts); 200 timing shuffles, 1000
             null permutations. Complex PCA centred (the primary convention; the rerun's read-out).

Usage: python scripts/run_fragment.py validate|hc11 [--workers=9]
Output: results/fragment/validation/*.pkl, results/fragment/hc11/*.pkl
"""

import os
import pickle
import sys
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "fragment"
HC11_ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
              "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]
_BASE = dict(participation=0.15, jitter_ms=8.0)
PLANTED = {
    "whole-route replays, 20%": (dict(_BASE, replay_frac=0.2, reverse_frac=0.5), None),
    "40% segment replays, 20%": (dict(_BASE, replay_frac=0.2, reverse_frac=0.5, segment=0.4), None),
    "40% segment replays, 10%, busy events": (dict(replay_frac=0.1, reverse_frac=0.5, segment=0.4, participation=0.4, jitter_ms=8.0), None),
    "assemblies along the map, no order": (dict(_BASE, replay_frac=0.0), "spatial"),
    "no replays": (dict(_BASE, replay_frac=0.0), None),
}


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def cpca_orders(times, T, rng):
    from scripts.run_rerun import run_cpca
    return run_cpca(times, T, rng)["centred"]["orders"]


def count_orders(times, rng, member_frac=0.2):
    from cpca.decompose import n_above_null, shuffle_null
    from cpca.theory import eigh_desc, lag_matrix
    from cpca.subspace import seriate
    from cpca.topology import reduced_similarity
    counts = np.isfinite(times).sum(axis=-1).astype(float)
    sd = counts.std(axis=0)
    Z = np.where(sd > 0, (counts - counts.mean(axis=0)) / np.where(sd > 0, sd, 1), 0.0)
    values, vectors = eigh_desc(lag_matrix(Z))
    k, _ = n_above_null(values, shuffle_null(Z, rng, n_shuffles=50), centred=True)
    V = np.real(vectors)
    orders = []
    for j in range(k):
        w = V[:, j]
        m = np.abs(w) >= member_frac * np.abs(w).max()
        orders.append(np.where(m, w, np.nan))
    if k:
        orders.append(seriate(reduced_similarity(values, V, k=k, kind="phase")))
    return orders


def criterion(times, T, centres, length, rng, n_shuffles, n_null):
    from cpca.fragment import best_fragment, timing_p
    from scripts.run_maze_rest import shuffle_timing
    real = best_fragment(cpca_orders(times, T, rng), centres, length, rng, n_null=n_null)
    shuffled = []
    for _ in range(n_shuffles):
        shuffled.append(best_fragment(cpca_orders(shuffle_timing(times, rng), T, rng), centres, length, n_null=0)["abs"])
    count = best_fragment(count_orders(times, rng), centres, length, rng, n_null=n_null)
    return {"cpca": real, "shuffled_abs": shuffled, "p_timing": timing_p(real["abs"], shuffled), "count": count}


def validate_job(cond, draw):
    from sim.routes import generate, make_routes
    gen, special = PLANTED[cond]
    rng = np.random.default_rng(seed("fragment-validate", cond, draw))
    routes = make_routes(rng, n_cells=80, n_routes=1, cells_per_route=60)
    extra = {}
    if special == "spatial":
        membership = rng.integers(0, 6, size=80)
        membership[routes[0]["cells"]] = np.minimum((routes[0]["pos"] * 6).astype(int), 5)
        extra["assembly_membership"] = membership
    data = generate(rng, n_cells=80, n_events=2000, routes=routes, **gen, **extra)
    T = np.full(data["times"].shape[0], data["event_ms"])
    res = criterion(data["times"], T, {1.0: data["centres"][0] * 100}, 100.0, rng, n_shuffles=50, n_null=500)
    path = OUT / "validation" / f"{zlib.crc32(cond.encode()) % 10000:04d}_d{draw}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"condition": cond, "draw": draw, **res}, open(path, "wb"))
    return f"cPCA |ρ| {res['cpca']['abs']:.2f} p {res['cpca']['p']:.3f}, timing p {res['p_timing']:.3f} | count PCA p {res['count']['p']:.3f}"


def hc11_job(name, epoch):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    from cpca.replay import immobile_intervals
    from scripts.run_replay_hc11 import intersect, two_d
    path = next(p for p in session_paths() if p.name.startswith(name))
    session = load_session(path)
    session["pos_xy"] = two_d(path)
    run = running(session)
    rng = np.random.default_rng(seed("fragment-hc11", name, epoch))

    def bursts_of(iv):
        by_cell, cell_ids, pooled = pyramidal_spikes(session, iv)
        b, pk = population_bursts(pooled, iv)
        return by_cell, cell_ids, pk, np.diff(b, axis=1).ravel()

    if epoch == "maze":
        by_cell, cell_ids, pk, d = bursts_of(intersect(immobile_intervals(session), [tuple(session["maze_epoch"])]))
        sub = np.arange(len(pk))
    else:
        n_post, n_pre = (len(bursts_of(nrem_intervals(session, e))[2]) for e in ("post", "pre"))
        by_cell, cell_ids, pk, d = bursts_of(nrem_intervals(session, epoch))
        n = min(n_post, n_pre)
        sub = np.sort(np.random.default_rng(seed("fragment-subset", name, epoch)).choice(len(pk), n, replace=False)) if len(pk) > n else np.arange(len(pk))
    times, _ = event_spike_times(by_cell, cell_ids, (pk - d / 2)[sub], (pk + d / 2)[sub], max_spikes=60)
    centres = place_centres(session, cell_ids, run)
    res = criterion(times, d[sub] * 1000, centres, run["length"], rng, n_shuffles=200, n_null=1000)
    out = OUT / "hc11" / f"{name}_{epoch}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"name": name, "epoch": epoch, "n_events": int(len(sub)), **res}, open(out, "wb"))
    return (f"{len(sub)} events | cPCA |ρ| {res['cpca']['abs']:.2f} p {res['cpca']['p']:.3f}, timing p {res['p_timing']:.3f} | "
            f"count PCA |ρ| {res['count']['abs']:.2f} p {res['count']['p']:.3f}")


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    if "validate" in argv:
        jobs = [(validate_job, (c, d)) for c in PLANTED for d in range(5)
                if not (OUT / "validation" / f"{zlib.crc32(c.encode()) % 10000:04d}_d{d}.pkl").exists()]
    else:
        jobs = [(hc11_job, (n, e)) for e in ("post", "pre", "maze") for n in HC11_ORDER if not (OUT / "hc11" / f"{n}_{e}.pkl").exists()]
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *a): a for fn, a in jobs}
        for f in as_completed(futures):
            try:
                print(f"{futures[f]}: {f.result()}", flush=True)
            except Exception:
                print(f"FAILED {futures[f]}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
