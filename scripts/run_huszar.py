"""Complex PCA on DANDI 000552 (Huszar et al., 2022) and on hc-11 with identical measures (docs/huszar.md).

Per session and sleep epoch (PRE, POST), centred complex PCA on all NREM population bursts, then:
components kept; component tests against each map; the best output against its best-of-outputs null;
topology (reduced space against map similarity); waking against sleep (waking fit uncentred on the
trials / laps); one direction template on all cells (held-out test, labels against the map); templates
of Step 3 (held-out check, map matches). Maps: left-arm and right-arm trials (000552), running
directions (hc-11), keyed +1 / -1 in both.

Usage: python scripts/run_huszar.py [--workers=10] [--only=huszar,hc11]
Results: results/huszar/<dataset>/<session>.pkl (existing files are skipped).
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

OUT = ROOT / "results" / "huszar"
N_SHUFFLES = 100
N_PERM = 2000
HC11_ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
              "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pickle.dump(result, open(tmp, "wb"))
    tmp.replace(path)


def hit(m):
    return bool(np.isfinite(m["rho"]) and abs(m["rho"]) > m["null95"])


def analyse_epoch(times, T, centres, map_sim, place, wake_sim, wake_active, run, rng):
    """All measures for one sleep epoch. Cell indices refer to the columns of `times`."""
    from cpca import fit
    from cpca.direction import order_of, split_members
    from cpca.mixture import sequences
    from cpca.topology import mantel, reduced_similarity
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_direction import label_check, map_test
    t0 = time.perf_counter()
    res = fit(times, T, rng=rng, ica=False, n_shuffles=N_SHUFFLES)
    k = res["n_kept"]
    out = {"n_events": int(times.shape[0]), "k": k}
    orders = [order_of(res["pca_components"][:, j]) for j in range(k)]
    tests = [{s: map_test(o, centres[s], run, rng) for s in centres} for o in orders]
    out["component_tests"] = tests
    out["component_hits"] = sum(hit(m) for t in tests for m in t.values())
    out["component_testable"] = sum(np.isfinite(m["rho"]) for t in tests for m in t.values())
    out["best"] = best_map_match(orders, centres, run["circular"], run["length"], rng) if orders else {"abs": np.nan, "p": np.nan, "index": -1}
    S = reduced_similarity(res["eigenvalues"][:k], res["pca_components"][:, :k]) if k else None
    out["topology"] = mantel(S[np.ix_(place, place)], map_sim, rng, n_perm=N_PERM) if S is not None else {"rho": np.nan, "p": np.nan}
    frac = np.isfinite(times).any(axis=-1).mean(axis=0)
    cells = np.flatnonzero((wake_active >= 5) & (frac >= 0.01))
    out["wake_sleep"] = (mantel(S[np.ix_(cells, cells)], wake_sim[np.ix_(cells, cells)], rng, n_perm=N_PERM)
                         if S is not None and wake_sim is not None else {"rho": np.nan, "p": np.nan})
    out["wake_sleep"]["n_cells"] = len(cells)
    # Step 2: one direction template on all cells
    sp = split_members(res["encoded"], np.ones(res["encoded"].shape[1], bool), rng)
    if sp is not None:
        tmaps = {s: map_test(order_of(sp["template"]), centres[s], run, rng) for s in centres}
        lab = label_check(res["encoded"], sp, tmaps, centres, run, rng)
        out["direction"] = {"heldout_share": sp["heldout_share"], "heldout_p": sp["heldout_p"], "template_maps": tmaps,
                            "minority_fraction": sp["minority_fraction"],
                            "labels": None if lab is None else {k2: lab[k2] for k2 in ("map", "template_rho", "n_significant_expressing",
                                                                                     "correct_expressing", "n_expressing")}}
    # Step 3: templates with event assignment
    sq = sequences(res, rng)
    if sq is not None and sq["templates"].shape[1]:
        keep = sq["check"]["keep"]
        tm = [{s: map_test(order_of(sq["templates"][:, j]), centres[s], run, rng) for s in centres} for j in range(sq["templates"].shape[1])]
        out["templates"] = {"n": int(len(keep)), "valid": int(keep.sum()),
                            "valid_on_map": int(sum(keep[j] and any(hit(m) for m in tm[j].values()) for j in range(len(keep))))}
    else:
        out["templates"] = {"n": 0, "valid": 0, "valid_on_map": 0}
    out["runtime_s"] = time.perf_counter() - t0
    return out


def waking_fit(times, T, rng):
    from cpca import fit
    from cpca.topology import reduced_similarity
    res = fit(times, T, rng=rng, ica=False, center=False, scale=False, n_shuffles=N_SHUFFLES)
    k = res["n_kept"]
    return (reduced_similarity(res["eigenvalues"][:k], res["pca_components"][:, :k]) if k else None), k


# ----------------------------------------------------------------------------- DANDI 000552
def huszar_job(path):
    from cpca import event_spike_times, population_bursts
    from cpca import huszar as hz
    from cpca.topology import map_similarity
    t0 = time.perf_counter()
    s = hz.load_session(path)
    win = hz.task_window(s)
    if win is None:
        return None, "epoch layout does not fit (no sleep before and after the task)"
    pre_w, task_w, post_w = win
    pyr = hz.putative_pyramidal(s)
    maps, n_spikes, mid = hz.arm_maps(s, pyr, task_w)
    centres = hz.place_centres(maps, n_spikes, mid)
    place = np.flatnonzero(np.isfinite(centres[1.0]) | np.isfinite(centres[-1.0]))
    M = map_similarity(maps[place].reshape(len(place), -1))
    run = {"circular": False, "length": hz.LENGTH_CM}
    rng = np.random.default_rng(seed("huszar", s["name"]))
    by_cell = {c: s["spikes"][c] for c in pyr}
    tr = s["trials"][hz.task_trials(s, task_w)]
    wt, _ = event_spike_times(by_cell, pyr, tr[:, 0], tr[:, 1], max_spikes=1000)
    W, k_wake = waking_fit(wt, (tr[:, 1] - tr[:, 0]) * 1000, rng)
    wake_active = np.isfinite(wt).any(axis=-1).sum(axis=0)
    result = {"dataset": "huszar", "name": s["name"], "subject": s["subject"], "n_units": len(s["spikes"]), "n_pyramidal": len(pyr),
              "n_place": {k: int(np.isfinite(v).sum()) for k, v in centres.items()}, "n_place_any": len(place), "n_trials": len(tr),
              "k_wake": k_wake, "epochs": {}}
    for label, w in (("pre", pre_w), ("post", post_w)):
        iv = hz.nrem_in(s, w)
        if np.sum(np.diff(iv, axis=1)) < 300:
            result["epochs"][label] = None
            continue
        pooled = np.sort(np.concatenate([x[(x >= w[0]) & (x < w[1])] for x in (s["spikes"][c] for c in pyr)]))
        bursts, peaks = population_bursts(pooled, iv)
        dur = np.diff(bursts, axis=1).ravel()
        times, _ = event_spike_times(by_cell, pyr, peaks - dur / 2, peaks + dur / 2)
        result["epochs"][label] = analyse_epoch(times, dur * 1000, centres, M, place, W, wake_active, run, rng)
        result["epochs"][label]["nrem_min"] = float(np.sum(np.diff(iv, axis=1)) / 60)
    result["runtime_s"] = time.perf_counter() - t0
    save(OUT / "huszar" / f"{s['name']}.pkl", result)
    e = result["epochs"]
    return result, " | ".join(f"{k}: k={v['k']} hits {v['component_hits']}/{v['component_testable']} best p={v['best']['p']:.3f} topo p={v['topology']['p']:.3f}"
                              for k, v in e.items() if v) + f" | place {len(place)}"


# ----------------------------------------------------------------------------- hc-11
def hc11_job(name):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import laps, load_session, nrem_intervals, place_centres, place_maps, pyramidal_spikes, running, session_paths
    from cpca.topology import map_similarity
    t0 = time.perf_counter()
    session = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(session)
    L = laps(run)
    starts, ends = np.array([a for a, _, _ in L]), np.array([b for _, b, _ in L])
    by_cell_w, cell_ids, _ = pyramidal_spikes(session, np.column_stack([starts, ends]))
    wt, _ = event_spike_times(by_cell_w, cell_ids, starts, ends, max_spikes=400)
    rng = np.random.default_rng(seed("huszar-hc11", name))
    W, k_wake = waking_fit(wt, (ends - starts) * 1000, rng)
    wake_active = np.isfinite(wt).any(axis=-1).sum(axis=0)
    maps, _, _ = place_maps(session, cell_ids, run)
    centres = place_centres(session, cell_ids, run)
    place = np.flatnonzero(np.isfinite(centres[1.0]) | np.isfinite(centres[-1.0]))
    M = map_similarity(maps[place].reshape(len(place), -1))
    result = {"dataset": "hc11", "name": name, "subject": name.split("_")[0], "n_pyramidal": len(cell_ids),
              "n_place": {k: int(np.isfinite(v).sum()) for k, v in centres.items()}, "n_place_any": len(place), "n_trials": len(L),
              "k_wake": k_wake, "epochs": {}}
    for epoch in ("pre", "post"):
        iv = nrem_intervals(session, epoch)
        by_cell, _, pooled = pyramidal_spikes(session, iv)
        bursts, peaks = population_bursts(pooled, iv)
        dur = np.diff(bursts, axis=1).ravel()
        times, _ = event_spike_times(by_cell, cell_ids, peaks - dur / 2, peaks + dur / 2)
        result["epochs"][epoch] = analyse_epoch(times, dur * 1000, centres, M, place, W, wake_active,
                                                {"circular": run["circular"], "length": run["length"]}, rng)
        result["epochs"][epoch]["nrem_min"] = float(np.sum(np.diff(iv, axis=1)) / 60)
    result["runtime_s"] = time.perf_counter() - t0
    save(OUT / "hc11" / f"{name}.pkl", result)
    e = result["epochs"]
    return result, " | ".join(f"{k}: k={v['k']} hits {v['component_hits']}/{v['component_testable']} best p={v['best']['p']:.3f} topo p={v['topology']['p']:.3f}"
                              for k, v in e.items()) + f" | place {len(place)}"


# ----------------------------------------------------------------------------- scheduler
def all_jobs(families):
    from cpca import huszar as hz
    jobs = []
    if "hc11" in families:
        jobs += [(hc11_job, (n,)) for n in HC11_ORDER if not (OUT / "hc11" / f"{n}.pkl").exists()]
    if "huszar" in families:
        jobs += [(huszar_job, (p,)) for p in hz.session_paths()
                 if not (OUT / "huszar" / f"{p.name.replace('_behavior+ecephys.nwb', '')}.pkl").exists()]
    return jobs


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), os.cpu_count() or 2)
    families = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--only=")), ["hc11", "huszar"])
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
                    _, msg = fut.result()
                    print(f"[{done}/{len(jobs)} {time.perf_counter() - t0:6.0f}s] {label[0]} {Path(str(label[1][0])).name[:40]}: {msg}", flush=True)
                except Exception:
                    print(f"[{done}/{len(jobs)}] FAILED {label}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
