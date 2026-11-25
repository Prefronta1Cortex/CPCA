"""Sleep fit contained in the waking fit (cpca/containment.py; docs/containment.md): planted check, then hc-11.

Per data set: the waking fit (uncentred, laps) and the sleep fit (centred, bursts) on the same cells (active in
at least 5 waking events and at least 1% of sleep events). Statistic: waking inside sleep, the mean share of the
kept waking components inside the span of the kept sleep components and their conjugates, against the same with
the sleep fit's cell identities permuted (1000 permutations; 500 planted); per waking component, its own share
against the same null. Earlier designs (sleep inside waking, with and without a timing bar) are kept in
results/containment/sleep_in_waking/ and described in docs/containment.md.

  synthetic  80 cells, 60 waking laps playing one of two maps on the same 60 cells (forward), 3000 sleep events:
             both maps replayed; one map only (the subset case); 40% segments of the maps; two other maps on the
             same cells; assemblies along the map firing without order; no replays. 5 draws.
  hc11       POST, all NREM bursts.

Usage: python scripts/run_containment.py synthetic|hc11 [--workers=9]
Output: results/containment/<family>/*.pkl
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

from scripts.run_wake_sleep import HC11_ORDER, MIN_SLEEP_FRACTION, MIN_WAKE_EVENTS, SLEEP_KW, WAKE_KW  # noqa: E402

OUT = ROOT / "results" / "containment"
SYNTHETIC = ["both maps replayed", "one map only (subset)", "40% segments of the maps", "other maps on the same cells",
             "assemblies along the map, no order", "no replays"]


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def components(times, T, rng, centred):
    from cpca import fit
    res = fit(times, T, rng=rng, ica=False, center=centred, scale=centred, n_shuffles=50)
    return res["pca_components"][:, :res["n_kept"]]


def evaluate(wake_times, wake_T, sleep_times, sleep_T, rng, n_perm, n_shuffles=0):
    """Waking inside sleep: each kept waking component's share inside the span of the kept sleep components and
    their conjugates, against the same with the sleep fit's cell identities permuted (its dimension kept)."""
    from cpca.containment import basis, containment, null_containment
    W = components(wake_times, wake_T, rng, centred=False)
    S = components(sleep_times, sleep_T, rng, centred=True)
    out = {"k_wake": W.shape[1], "k_sleep": S.shape[1], "n_cells": wake_times.shape[1]}
    if not W.shape[1] or not S.shape[1]:
        return {**out, "real": np.nan, "p": np.nan, "per_component": np.zeros(0), "per_component_p": np.zeros(0)}
    B = basis(S)
    c = containment(W, B)
    null, _ = null_containment(W, B, rng, n_perm)
    real = float(c.mean())
    return {**out, "dim_sleep": B.shape[1], "per_component": c, "per_component_p": (1 + (null >= c[None, :]).sum(axis=0)) / (1 + n_perm),
            "real": real, "null_mean": float(null.mean()), "p": float((1 + np.sum(null.mean(axis=1) >= real)) / (1 + n_perm))}


def restrict(wake_times, sleep_times):
    wa = np.isfinite(wake_times).any(axis=-1).sum(axis=0)
    sf = np.isfinite(sleep_times).any(axis=-1).mean(axis=0)
    return np.flatnonzero((wa >= MIN_WAKE_EVENTS) & (sf >= MIN_SLEEP_FRACTION))


def synthetic_job(cond, draw, n_shuffles=0):
    from sim.routes import generate, make_routes
    rng = np.random.default_rng(seed("containment-synthetic", cond, draw))
    routes = make_routes(rng, n_cells=80, n_routes=1, cells_per_route=60, direction_maps=True)
    wake = generate(rng, n_cells=80, n_events=60, routes=routes, **WAKE_KW)
    sleep_routes, kw, extra = routes, dict(SLEEP_KW), {}
    if cond == "one map only (subset)":
        sleep_routes = routes[:1]
    elif cond == "40% segments of the maps":
        kw["segment"] = 0.4
    elif cond == "other maps on the same cells":
        sleep_routes = make_routes(rng, n_cells=80, n_routes=1, cells_per_route=60, direction_maps=True)
        for r_new, r_old in zip(sleep_routes, routes):
            r_new["cells"] = r_old["cells"]
    elif cond == "assemblies along the map, no order":
        kw["replay_frac"] = 0.0
        membership = rng.integers(0, 6, size=80)
        membership[routes[0]["cells"]] = np.minimum((routes[0]["pos"] * 6).astype(int), 5)
        extra["assembly_membership"] = membership
    elif cond == "no replays":
        kw["replay_frac"] = 0.0
    sleep = generate(rng, n_cells=80, n_events=3000, routes=sleep_routes, **kw, **extra)
    cells = restrict(wake["times"], sleep["times"])
    res = evaluate(wake["times"][:, cells], wake["event_ms"], sleep["times"][:, cells], sleep["event_ms"], rng, n_perm=500, n_shuffles=n_shuffles)
    path = OUT / "synthetic" / f"{SYNTHETIC.index(cond)}_d{draw}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"condition": cond, "draw": draw, **res}, open(path, "wb"))
    return f"waking in sleep {res['real']:.2f} (null {res.get('null_mean', np.nan):.2f}, p {res['p']:.3f}), k wake/sleep {res['k_wake']}/{res['k_sleep']}"


def hc11_job(name, epoch="post", n_shuffles=0):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import laps, load_session, nrem_intervals, pyramidal_spikes, running, session_paths
    session = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(session)
    L = laps(run)
    starts, ends = np.array([a for a, _, _ in L]), np.array([b for _, b, _ in L])
    by_cell, cell_ids, _ = pyramidal_spikes(session, np.column_stack([starts, ends]))
    wake_times, _ = event_spike_times(by_cell, cell_ids, starts, ends, max_spikes=400)
    iv = nrem_intervals(session, epoch)
    bc, _, pooled = pyramidal_spikes(session, iv)
    b, pk = population_bursts(pooled, iv)
    d = np.diff(b, axis=1).ravel()
    sub = np.arange(len(pk))
    sleep_times, _ = event_spike_times(bc, cell_ids, (pk - d / 2)[sub], (pk + d / 2)[sub], max_spikes=60)
    cells = restrict(wake_times, sleep_times)
    rng = np.random.default_rng(seed("containment-hc11", name, epoch))
    res = evaluate(wake_times[:, cells], (ends - starts) * 1000, sleep_times[:, cells], d[sub] * 1000, rng, n_perm=1000, n_shuffles=n_shuffles)
    path = OUT / "hc11" / f"{name}_{epoch}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"name": name, "epoch": epoch, "n_events": int(len(sub)), "n_laps": len(L), **res}, open(path, "wb"))
    return (f"{len(sub)} events, {len(cells)} cells | waking in sleep {res['real']:.2f} (null {res.get('null_mean', np.nan):.2f}, p {res['p']:.3f}), "
            f"waking components contained {int((res['per_component_p'] < 0.05).sum())}/{res['k_wake']}, k wake/sleep {res['k_wake']}/{res['k_sleep']}")


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    n_shuffles = 0
    if "synthetic" in argv:
        jobs = [(synthetic_job, (c, d, n_shuffles)) for c in SYNTHETIC for d in range(5) if not (OUT / "synthetic" / f"{SYNTHETIC.index(c)}_d{d}.pkl").exists()]
    else:
        jobs = [(hc11_job, (n, "post", 2 * n_shuffles)) for n in HC11_ORDER if not (OUT / "hc11" / f"{n}_post.pkl").exists()]
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
