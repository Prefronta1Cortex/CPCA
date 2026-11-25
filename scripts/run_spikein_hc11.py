"""hc-11: how much replay does complex PCA need? Replays spiked into real sleep events at known rates.

Base: the matched 1500-event subset of PRE sleep that every method was given (scripts/run_rerun.py), with its
real cells, spike counts, durations and background. A fraction f of the events is turned into replays of the
waking map in one of two ways:
  ideal   in each chosen event, every place cell (with a field in the replayed direction) that fires keeps its
          spike count, and its spikes are moved to the time of its field centre along the track (forward or
          reverse at random, the event's duration spanning the track, 5% of the duration jitter);
  awake   the chosen events are replaced by real replay events from rest on the maze (significant under both
          of the paper's shuffles, results/replay/), each used at most once.
Complex PCA (centred, the rerun's read-out) is fitted to the 1500 events and scored as in figures/methods_hc11.png:
best component against the waking map, best-of-outputs null. f = 0 is PRE as it is.

For comparison: the real excess of doubly significant replays in POST over PRE (results/replay/) is 0.8 points
(median over sessions) and at most 3.4 points.

Usage: python scripts/run_spikein_hc11.py [--workers=10]
Output: results/spikein/<session>.pkl
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

OUT = ROOT / "results" / "spikein"
REPLAY = ROOT / "results" / "replay"
HC11_ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
              "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]
FRACTIONS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
N_DRAWS = 5


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def ideal(times, T, chosen, centres, rng, jitter=0.05):
    out = times.copy()
    keys = [k for k, c in centres.items() if np.isfinite(c).sum() >= 6]
    for e in chosen:
        key = keys[rng.integers(len(keys))]
        cen = centres[key]
        has = np.isfinite(cen)
        lo, hi = np.nanmin(cen), np.nanmax(cen)
        sense = 1 if rng.random() < 0.5 else -1
        for c in np.flatnonzero(has & np.isfinite(times[e]).any(axis=-1)):
            u = (cen[c] - lo) / max(hi - lo, 1e-9)
            u = u if sense > 0 else 1 - u
            n = int(np.isfinite(times[e, c]).sum())
            t = np.clip((0.1 + 0.8 * u + jitter * rng.standard_normal(n)) * T[e], 0, T[e])
            out[e, c, :] = np.nan
            out[e, c, :n] = np.sort(t)
    return out


def job(name):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_rerun import ORDER_INDEX, fixed_subset, run_cpca
    session = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(session)
    intervals = nrem_intervals(session, "pre")
    by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
    bursts, peaks = population_bursts(pooled, intervals)
    duration = np.diff(bursts, axis=1).ravel()
    sub = fixed_subset(len(peaks), 555 + 2 * ORDER_INDEX[name] + 1)
    base, _ = event_spike_times(by_cell, cell_ids, (peaks - duration / 2)[sub], (peaks + duration / 2)[sub], max_spikes=60)
    T = duration[sub] * 1000
    centres = place_centres(session, cell_ids, run)

    # real awake replays from rest on the maze
    rep = pickle.load(open(REPLAY / f"{name}.pkl", "rb"))["paper"]["maze"]
    sig = (rep["p"] < 0.05) & (rep["p_time"] < 0.05)
    maze_cells = {c: session["times"][(session["ids"] == c)] for c in cell_ids}
    maze_cells = {c: np.sort(v) for c, v in maze_cells.items()}
    awake, _ = event_spike_times(maze_cells, cell_ids, rep["start"][sig], rep["end"][sig], max_spikes=60)
    awake_T = (rep["end"][sig] - rep["start"][sig]) * 1000

    rows = []
    for kind in ("ideal", "awake"):
        for f in FRACTIONS:
            k = int(round(f * len(T)))
            if kind == "awake" and k > len(awake_T):
                continue
            for draw in range(N_DRAWS if f > 0 or kind == "ideal" else 0):
                rng = np.random.default_rng(seed("spikein", name, kind, f, draw))
                chosen = rng.choice(len(T), k, replace=False)
                times, Te = base, T.copy()
                if k and kind == "ideal":
                    times = ideal(base, T, chosen, centres, rng)
                elif k:
                    times = base.copy()
                    pick = rng.choice(len(awake_T), k, replace=False)
                    times[chosen] = awake[pick]
                    Te[chosen] = awake_T[pick]
                orders = run_cpca(times, Te, np.random.default_rng(seed("spikein-fit", name, kind, f, draw)))["centred"]["orders"]
                s = best_map_match(orders, centres, run["circular"], run["length"], np.random.default_rng(seed("spikein-score", name, kind, f, draw)))
                rows.append({"kind": kind, "f": f, "draw": draw, "k": k, "n_components": len(orders), "abs": s["abs"], "p": s["p"], "n": s["n"]})
    out = {"name": name, "n_awake": int(len(awake_T)), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT / f"{name}.pkl", "wb"))
    summary = []
    for kind in ("ideal", "awake"):
        for f in FRACTIONS:
            rs = [r for r in rows if r["kind"] == kind and r["f"] == f]
            if rs:
                summary.append(f"{kind} {f:.0%}: {sum(np.isfinite(r['p']) and r['p'] < 0.05 for r in rs)}/{len(rs)}")
    return f"{len(awake_T)} awake replays | " + ", ".join(summary)


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 8)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    todo = [n for n in HC11_ORDER if not (OUT / f"{n}.pkl").exists()]
    print(f"{len(todo)} sessions on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, n): n for n in todo}
        for f in as_completed(futures):
            try:
                print(f"{futures[f]}: {f.result()}", flush=True)
            except Exception:
                print(f"FAILED {futures[f]}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
