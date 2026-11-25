"""Supervised ceiling (docs/ceiling.md): replays chosen with half of the place cells, order recovered on the other half.

Usage: python scripts/run_ceiling.py [--workers=10] [--only=hc11,huszar]
Results: results/ceiling/<dataset>/<session>_<epoch>.pkl
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

OUT = ROOT / "results" / "ceiling"
N_SPLITS = 5
MIN_A = 5
MIN_SELECTED = 30
N_EVENT_NULL = 200
N_SCORE_NULL = 500
HC11_ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
              "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]
READOUTS = ["centred", "uncentred", "aligned", "averaging"]


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


# ----------------------------------------------------------------------------- data
def hc11_data(name, epoch):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    session = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(session)
    iv = nrem_intervals(session, epoch)
    by_cell, cell_ids, pooled = pyramidal_spikes(session, iv)
    bursts, peaks = population_bursts(pooled, iv)
    dur = np.diff(bursts, axis=1).ravel()
    times, _ = event_spike_times(by_cell, cell_ids, peaks - dur / 2, peaks + dur / 2)
    return times, dur * 1000, place_centres(session, cell_ids, run), run["circular"], run["length"]


SYNTHETIC = {"replays 10%": (0.10, 0.6), "replays 25%": (0.25, 0.6), "weak replays 10%": (0.10, 0.3), "no replays": (0.0, 0.6),
             "segment replays 10%, busy events": (0.10, 0.6)}  # (share of replays, share of route cells per replay)
SEGMENT = {"segment replays 10%, busy events": dict(segment=0.4, participation=0.4)}


def synthetic_data(cond, draw):
    """Planted check of the diagnostic: 80 cells, one map on 60 of them, both directions; maps = the planted positions (x 100 cm)."""
    from sim.routes import generate, make_routes
    rng = np.random.default_rng(seed("ceiling-synthetic", cond, draw))
    routes = make_routes(rng, n_cells=80, n_routes=1, cells_per_route=60)
    frac, seq = SYNTHETIC[cond]
    kw = dict(dict(participation=0.15), **SEGMENT.get(cond, {}))
    data = generate(rng, n_cells=80, n_events=3000, routes=routes, replay_frac=frac, seq_participation=seq, reverse_frac=0.5, jitter_ms=8.0, **kw)
    cen = data["centres"][0] * 100
    return data["times"], np.full(data["times"].shape[0], data["event_ms"]), {1.0: cen}, False, 100.0


def huszar_data(path, epoch):
    from cpca import event_spike_times, population_bursts
    from cpca import huszar as hz
    s = hz.load_session(path)
    pre_w, task_w, post_w = hz.task_window(s)
    pyr = hz.putative_pyramidal(s)
    maps, n_spikes, mid = hz.arm_maps(s, pyr, task_w)
    centres = hz.place_centres(maps, n_spikes, mid)
    w = pre_w if epoch == "pre" else post_w
    iv = hz.nrem_in(s, w)
    if np.sum(np.diff(iv, axis=1)) < 300:
        return None
    pooled = np.sort(np.concatenate([x[(x >= w[0]) & (x < w[1])] for x in (s["spikes"][c] for c in pyr)]))
    bursts, peaks = population_bursts(pooled, iv)
    dur = np.diff(bursts, axis=1).ravel()
    times, _ = event_spike_times({c: s["spikes"][c] for c in pyr}, pyr, peaks - dur / 2, peaks + dur / 2)
    return times, dur * 1000, centres, False, hz.LENGTH_CM


# ----------------------------------------------------------------------------- pieces
def mean_times(times):
    """(events, cells) mean spike time in ms, NaN where silent."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(times, axis=-1)


def select_replays(mt, centres_A, circular, length, rng):
    """Per event: (A correlation, its null 95th percentile) using only the A cells (NaN if fewer than MIN_A fire)."""
    from cpca.evaluate import order_vs_place
    rho = np.full(mt.shape[0], np.nan)
    n95 = np.full(mt.shape[0], np.nan)
    has = np.isfinite(centres_A)
    for i in range(mt.shape[0]):
        if np.sum(np.isfinite(mt[i]) & has) < MIN_A:
            continue
        rho[i], n95[i], _ = order_vs_place(mt[i], centres_A, circular, length, rng, min_members=MIN_A, n_null=N_EVENT_NULL)
    return rho, n95


def leading_order(X):
    """Order of the first uncentred component of arrows X (events x cells)."""
    from cpca.direction import order_of
    from cpca.theory import eigh_desc, lag_matrix
    return order_of(eigh_desc(lag_matrix(X))[1][:, 0])


def readouts(times, T, keep_cells, sign, rng):
    """All read-outs on these events, cells restricted to keep_cells; orders are over all cells (NaN outside)."""
    from cpca import encode, fit
    from cpca.direction import order_of
    n_cells = times.shape[1]
    sub = times[:, keep_cells]
    out = {}
    for conv, centred in (("centred", True), ("uncentred", False)):
        res = fit(sub, T, rng=rng, ica=False, center=centred, scale=centred, n_shuffles=50)
        orders = []
        for j in range(res["n_kept"]):
            o = np.full(n_cells, np.nan)
            o[keep_cells] = order_of(res["pca_components"][:, j])
            orders.append(o)
        out[conv] = orders
    X = encode(sub, T)
    X = np.where((sign < 0)[:, None], np.conj(X), X)
    o = np.full(n_cells, np.nan)
    o[keep_cells] = leading_order(X)
    out["aligned"] = [o]
    mt = mean_times(sub) / np.asarray(T, float)[:, None]
    mt = np.where((sign < 0)[:, None], 1 - mt, mt)
    o = np.full(n_cells, np.nan)
    with np.errstate(invalid="ignore"):
        avg = np.nanmean(mt, axis=0)
    o[keep_cells] = np.where(np.isfinite(mt).sum(axis=0) >= 3, avg, np.nan)
    out["averaging"] = [o]
    return out


def score(orders, key_B, circular, length, rng):
    from scripts.plot_hc11_sequences import best_map_match
    if not orders:
        return {"abs": np.nan, "p": np.nan}
    s = best_map_match(orders, {1.0: key_B}, circular, length, rng, n_null=N_SCORE_NULL)
    return {"abs": s["abs"], "p": s["p"], "n": s["n"]}


# ----------------------------------------------------------------------------- one session and epoch
def job(dataset, ident, epoch):
    from cpca import fit
    from cpca.direction import order_of
    t0 = time.perf_counter()
    if dataset == "synthetic":
        data, name = synthetic_data(ident, int(epoch)), f"{ident.replace(' ', '_').replace('%', '').replace(',', '')}_d{int(epoch):02d}"
    else:
        data = hc11_data(ident, epoch) if dataset == "hc11" else huszar_data(Path(ident), epoch)
        name = ident if dataset == "hc11" else Path(ident).name.replace("_behavior+ecephys.nwb", "")
    if data is None:
        return dataset, name, epoch, "too little NREM"
    times, T, centres, circular, length = data
    T = np.asarray(T, float)
    rng = np.random.default_rng(seed("ceiling", dataset, name, epoch))
    mt = mean_times(times)
    # the standard pipeline once (all cells, all events); scored per split on B cells
    base = fit(times, T, rng=rng, ica=False, n_shuffles=50)
    base_orders = [order_of(base["pca_components"][:, j]) for j in range(base["n_kept"])]
    result = {"dataset": dataset, "name": name, "epoch": epoch, "n_events": int(times.shape[0]), "maps": {}}
    for d, cen in centres.items():
        place = np.flatnonzero(np.isfinite(cen))
        rows = []
        if place.size < 2 * MIN_A + 6:
            result["maps"][d] = {"n_place": int(place.size), "splits": rows}
            continue
        for s in range(N_SPLITS):
            perm = rng.permutation(place)
            A, B = np.sort(perm[: place.size // 2]), np.sort(perm[place.size // 2:])
            cen_A = np.full(cen.size, np.nan)
            cen_A[A] = cen[A]
            key_B = np.full(cen.size, np.nan)
            key_B[B] = cen[B]
            rho, n95 = select_replays(mt, cen_A, circular, length, rng)
            testable = np.isfinite(rho)
            selected = np.flatnonzero(testable & (np.abs(rho) > n95))
            # this session's own false-positive rate: the same selection with A's centres shuffled among the A cells
            cen_S = np.full(cen.size, np.nan)
            cen_S[A] = cen[rng.permutation(A)]
            rho_s, n95_s = select_replays(mt, cen_S, circular, length, rng)
            row = {"n_testable": int(testable.sum()), "n_selected": int(selected.size),
                   "n_selected_shuffled": int(np.sum(np.isfinite(rho_s) & (np.abs(rho_s) > n95_s))),
                   "baseline": score(base_orders, key_B, circular, length, rng)}
            if selected.size >= MIN_SELECTED:
                pool = np.flatnonzero(testable)
                random = np.sort(rng.choice(pool, selected.size, replace=False))
                keep = np.setdiff1d(np.arange(cen.size), A)
                for label, ev in (("selected", selected), ("random", random)):
                    outs = readouts(times[ev], T[ev], keep, np.sign(np.nan_to_num(rho[ev])), rng)
                    row[label] = {r: score(outs[r], key_B, circular, length, rng) for r in READOUTS}
            rows.append(row)
        result["maps"][d] = {"n_place": int(place.size), "splits": rows}
    result["runtime_s"] = time.perf_counter() - t0
    path = OUT / dataset / (f"{name}.pkl" if dataset == "synthetic" else f"{name}_{epoch}.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(result, open(path, "wb"))

    def med(key, r):
        v = [x[key][r]["abs"] for m in result["maps"].values() for x in m["splits"] if key in x]
        return f"{np.nanmedian(v):.2f}" if v else "–"
    sel = [x["n_selected"] / max(x["n_testable"], 1) for m in result["maps"].values() for x in m["splits"]]
    return dataset, name, epoch, (f"selected {np.mean(sel):.1%} | selected: " + " ".join(f"{r} {med('selected', r)}" for r in READOUTS) +
                                  " | random: " + " ".join(f"{r} {med('random', r)}" for r in READOUTS) + f" | {result['runtime_s']:.0f}s")


def all_jobs(families):
    from cpca import huszar as hz
    jobs = []
    if "synthetic" in families:
        jobs += [("synthetic", c, str(d)) for c in SYNTHETIC for d in range(5)
                 if not (OUT / "synthetic" / f"{c.replace(' ', '_').replace('%', '').replace(',', '')}_d{d:02d}.pkl").exists()]
    if "hc11" in families:
        jobs += [("hc11", n, e) for n in HC11_ORDER for e in ("post", "pre") if not (OUT / "hc11" / f"{n}_{e}.pkl").exists()]
    if "huszar" in families:
        for p in hz.session_paths():
            name = p.name.replace("_behavior+ecephys.nwb", "")
            jobs += [("huszar", str(p), e) for e in ("post", "pre") if not (OUT / "huszar" / f"{name}_{e}.pkl").exists()]
    return jobs


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), os.cpu_count() or 2)
    families = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--only=")), ["synthetic", "hc11", "huszar"])
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    jobs = all_jobs(families)
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        running = {pool.submit(job, *a): a for a in jobs}
        done = 0
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                a = running.pop(fut)
                done += 1
                try:
                    ds, name, ep, msg = fut.result()
                    print(f"[{done}/{len(jobs)} {time.perf_counter() - t0:6.0f}s] {ds} {name[:36]} {ep}: {msg}", flush=True)
                except Exception:
                    print(f"[{done}/{len(jobs)}] FAILED {a}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
