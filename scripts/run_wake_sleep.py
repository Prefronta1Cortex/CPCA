"""Waking fit against sleep fit (docs/wake_sleep.md): the same model on waking laps and on sleep reactivation events.

- hc-11: 8 sessions. Waking events: continuous runs covering at least 60% of the track
  (cpca.hc11.laps); sleep events: all POST and all PRE population bursts. Per session and sleep
  epoch, closeness of every pair of cells in the waking fit's reduced space against closeness in
  the sleep fit's (Spearman over pairs, Mantel null), for waking uncentred / centred x sleep
  centred / uncentred (headline: waking uncentred, sleep centred); and the waking fit against the
  waking place map.
- synthetic: 60 waking laps (each plays one of two maps on the same cells, forward) and 3000
  sleep events from the same maps (as L2; as H), from two other maps, or without replays.

Usage: python scripts/run_wake_sleep.py [--workers=10] [--only=hc11,synthetic]
Results: results/wake_sleep/<family>/<job>.pkl (existing files are skipped).
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
from cpca.topology import mantel, map_similarity, reduced_similarity  # noqa: E402

OUT = ROOT / "results" / "wake_sleep"
N_PERM = 2000
N_SHUFFLES = 100
N_DRAWS = 10
MIN_WAKE_EVENTS = 5          # a cell must fire in at least this many waking events
MIN_SLEEP_FRACTION = 0.01    # and in at least this fraction of the epoch's sleep events
HEADLINE = ("uncentred", "centred")          # (waking fit, sleep fit)
HC11_ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
              "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pickle.dump(result, open(tmp, "wb"))
    tmp.replace(path)


def fits(times, T, rng):
    """Both conventions: reduced-space closeness over all cells, components kept, and the fit itself."""
    out = {}
    for conv, centred in (("uncentred", False), ("centred", True)):
        res = fit(times, T, rng=rng, ica=False, center=centred, scale=centred, n_shuffles=N_SHUFFLES)
        k = res["n_kept"]
        out[conv] = {"k": k, "similarity": reduced_similarity(res["eigenvalues"][:k], res["pca_components"][:, :k]) if k else None,
                     "first": res["pca_components"][:, 0] if k else None}
    return out


def active(times):
    """Events in which each cell fires at least once: (events, cells) boolean."""
    return np.isfinite(times).any(axis=-1)


def compare(wake, sleep, cells, rng):
    """Mantel test of every (waking fit, sleep fit) convention pair, on the given cells."""
    out = {}
    for wc in ("uncentred", "centred"):
        for sc in ("centred", "uncentred"):
            W, S = wake[wc]["similarity"], sleep[sc]["similarity"]
            if W is None or S is None or len(cells) < 4:
                out[(wc, sc)] = {"rho": np.nan, "p": np.nan, "null95": np.nan, "null_mean": np.nan, "n_cells": len(cells), "n_pairs": 0}
                continue
            out[(wc, sc)] = mantel(S[np.ix_(cells, cells)], W[np.ix_(cells, cells)], rng, n_perm=N_PERM)
    return out


# ----------------------------------------------------------------------------- hc-11
def hc11_job(name):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import laps, load_session, nrem_intervals, place_centres, place_maps, pyramidal_spikes, running, session_paths
    t0 = time.perf_counter()
    session = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(session)
    L = laps(run)
    starts, ends = np.array([a for a, _, _ in L]), np.array([b for _, b, _ in L])
    by_cell, cell_ids, _ = pyramidal_spikes(session, np.column_stack([starts, ends]))
    wake_times, truncated = event_spike_times(by_cell, cell_ids, starts, ends, max_spikes=400)
    rng = np.random.default_rng(seed("wake-sleep", name))
    wake = fits(wake_times, (ends - starts) * 1000, rng)
    wake_active = active(wake_times).sum(axis=0)
    # the waking fit against the waking place map, place cells only
    maps, _, _ = place_maps(session, cell_ids, run)
    centres = place_centres(session, cell_ids, run)
    place = np.flatnonzero((np.isfinite(centres[1.0]) | np.isfinite(centres[-1.0])) & (wake_active >= MIN_WAKE_EVENTS))
    M = map_similarity(maps[place].reshape(len(place), -1))
    wake_vs_map = {wc: (mantel(wake[wc]["similarity"][np.ix_(place, place)], M, rng, n_perm=N_PERM) if wake[wc]["similarity"] is not None
                        else {"rho": np.nan, "p": np.nan}) for wc in wake}
    result = {"family": "hc11", "name": name, "circular": run["circular"], "n_wake_events": len(L), "lap_directions": [d for *_, d in L],
              "lap_duration_s": ends - starts, "truncated": truncated, "cell_ids": cell_ids, "wake_k": {c: wake[c]["k"] for c in wake},
              "wake_active": wake_active, "wake_vs_map": wake_vs_map, "place": place, "map_similarity": M.astype(np.float32),
              "wake_similarity": {c: (v["similarity"].astype(np.float32) if v["similarity"] is not None else None) for c, v in wake.items()},
              "wake_first": wake["uncentred"]["first"], "epochs": {}}
    for epoch in ("post", "pre"):
        intervals = nrem_intervals(session, epoch)
        by_cell_s, _, pooled = pyramidal_spikes(session, intervals)
        bursts, peaks = population_bursts(pooled, intervals)
        duration = np.diff(bursts, axis=1).ravel()
        times, _ = event_spike_times(by_cell_s, cell_ids, peaks - duration / 2, peaks + duration / 2)
        sleep = fits(times, duration * 1000, rng)
        frac = active(times).mean(axis=0)
        cells = np.flatnonzero((wake_active >= MIN_WAKE_EVENTS) & (frac >= MIN_SLEEP_FRACTION))
        result["epochs"][epoch] = {"n_events": len(peaks), "k": {c: sleep[c]["k"] for c in sleep}, "cells": cells,
                                   "scores": compare(wake, sleep, cells, rng),
                                   "sleep_similarity": {c: (v["similarity"].astype(np.float32) if v["similarity"] is not None else None)
                                                        for c, v in sleep.items()}}
    result["runtime_s"] = time.perf_counter() - t0
    save(OUT / "hc11" / f"{name}.pkl", result)
    h = HEADLINE
    return f"{len(L)} waking events, k wake {result['wake_k']}, wake vs map rho {wake_vs_map['uncentred']['rho']:.2f} | " + " | ".join(
        f"{e}: rho {result['epochs'][e]['scores'][h]['rho']:+.3f} p {result['epochs'][e]['scores'][h]['p']:.4f} ({len(result['epochs'][e]['cells'])} cells)"
        for e in ("post", "pre"))


# ----------------------------------------------------------------------------- synthetic
SYNTHETIC = {
    "same maps, sleep as L2": "same",
    "same maps, sleep as H (partial replays, assemblies)": "same-H",
    "other maps": "other",
    "no replays in sleep": "none",
}
WAKE_KW = dict(replay_frac=1.0, reverse_frac=0.0, participation=0.05, seq_participation=0.9, span_ms=90.0, jitter_ms=10.0)
SLEEP_KW = dict(replay_frac=0.4, reverse_frac=0.5, participation=0.15, jitter_ms=8.0)


def synthetic_target(cond, draw):
    return OUT / "synthetic" / f"{SYNTHETIC[cond]}_d{draw:02d}.pkl"


def synthetic_job(cond, draw):
    from sim.routes import generate, make_routes
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed("wake-sleep-synthetic", cond, draw))
    routes = make_routes(rng, n_cells=80, n_routes=1, cells_per_route=60, direction_maps=True)
    wake_data = generate(rng, n_cells=80, n_events=60, routes=routes, **WAKE_KW)
    kind = SYNTHETIC[cond]
    if kind == "other":
        other = make_routes(rng, n_cells=80, n_routes=1, cells_per_route=60, direction_maps=True)
        for r_new, r_old in zip(other, routes):
            r_new["cells"] = r_old["cells"]                       # same cells, new positions
        sleep_routes, sleep_kw = other, SLEEP_KW
    elif kind == "same-H":
        sleep_routes, sleep_kw = routes, dict(SLEEP_KW, segment=0.4, assemblies=6)
    elif kind == "none":
        sleep_routes, sleep_kw = routes, dict(SLEEP_KW, replay_frac=0.0)
    else:
        sleep_routes, sleep_kw = routes, SLEEP_KW
    sleep_data = generate(rng, n_cells=80, n_events=3000, routes=sleep_routes, **sleep_kw)
    wake = fits(wake_data["times"], wake_data["event_ms"], rng)
    sleep = fits(sleep_data["times"], sleep_data["event_ms"], rng)
    wa, sf = active(wake_data["times"]).sum(axis=0), active(sleep_data["times"]).mean(axis=0)
    cells = np.flatnonzero((wa >= MIN_WAKE_EVENTS) & (sf >= MIN_SLEEP_FRACTION))
    out = {"family": "synthetic", "condition": cond, "draw": draw, "wake_k": {c: wake[c]["k"] for c in wake},
           "sleep_k": {c: sleep[c]["k"] for c in sleep}, "n_cells": len(cells), "scores": compare(wake, sleep, cells, rng),
           "runtime_s": time.perf_counter() - t0}
    save(synthetic_target(cond, draw), out)
    s = out["scores"][HEADLINE]
    return f"rho {s['rho']:+.3f} p {s['p']:.4f}, k wake {out['wake_k']}, k sleep {out['sleep_k']}"


# ----------------------------------------------------------------------------- scheduler
def all_jobs(families):
    jobs = []
    if "hc11" in families:
        jobs += [(hc11_job, (n,)) for n in HC11_ORDER if not (OUT / "hc11" / f"{n}.pkl").exists()]
    if "synthetic" in families:
        jobs += [(synthetic_job, (c, d)) for c in SYNTHETIC for d in range(N_DRAWS) if not synthetic_target(c, d).exists()]
    return jobs


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), os.cpu_count() or 2)
    families = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--only=")), ["hc11", "synthetic"])
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
                    print(f"[{done}/{len(jobs)} {time.perf_counter() - t0:6.0f}s] {label[0]} {label[1]}: {fut.result()}", flush=True)
                except Exception:
                    print(f"[{done}/{len(jobs)}] FAILED {label}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
