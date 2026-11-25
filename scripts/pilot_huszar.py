"""Pilot: the hc-11 pipeline on two sessions of DANDI 000552 (Huszar et al., 2022; figure-eight alternation, mice).

Sessions fixed in advance: the one with the most sorted units from each of the two mice with the most units.
Epochs: 0 = sleep before the task (PRE), 1 = figure-eight alternation, 2 = sleep after (POST).

- putative pyramidal cells: mean rate below 8 Hz over the session (units are not typed in the files);
- place cells: rate map on the linearized figure-eight (40 bins over 0-222 cm, running > 3 cm/s, smoothed
  one bin), at least 40 spikes while running and a peak of at least 2 Hz; field centre = peak bin;
- sleep events: population bursts of the pyramidal cells in NREM of each sleep epoch (cpca.events, as on hc-11);
- complex PCA centred, components above the shuffle null; each component's order against the field centres
  (cpca.evaluate.order_vs_place, n_null 500), and the best output with its best-of-outputs null
  (scripts/plot_hc11_sequences.best_map_match);
- waking against sleep (docs/wake_sleep.md): waking events = the alternation trials (start-stop), uncentred
  fit; pair closeness against the centred sleep fit, Mantel null.

Usage: python scripts/pilot_huszar.py
Output: results/datasets/huszar_pilot.md
"""

import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "dataset_checks" / "huszar"
OUT = ROOT / "results" / "datasets"
SESSIONS = ["sub-e15-13f1_ses-e15-13f1-220117", "sub-e13-26m1_ses-e13-26m1-211019"]
LENGTH_CM = 222.0
N_BINS = 40


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def load(name):
    import h5py
    f = h5py.File(DATA / f"{name}_behavior+ecephys.nwb", "r")
    idx = f["units/spike_times_index"][:]
    st = f["units/spike_times"][:]
    spikes = np.split(st, idx[:-1])
    ep = f["intervals/epochs"]
    epochs = np.column_stack([ep["start_time"][:], ep["stop_time"][:]])
    lp = f["processing/behavior/LinearizedPosition/LinearizedSpatialSeries"]
    ss = f["processing/behavior/SleepStates"]
    labels = np.array([x.decode() if isinstance(x, bytes) else x for x in ss["label"][:]])
    nrem = np.column_stack([ss["start_time"][:], ss["stop_time"][:]])[labels == "Non-REM"]
    tr = f["intervals/trials"]
    trials = np.column_stack([tr["start_time"][:], tr["stop_time"][:]])
    return {"spikes": spikes, "epochs": epochs, "x": lp["data"][:], "t": lp["timestamps"][:], "nrem": nrem,
            "trials": trials[np.all(np.isfinite(trials), axis=1)]}


def clip(intervals, window):
    a, b = window
    out = np.column_stack([np.maximum(intervals[:, 0], a), np.minimum(intervals[:, 1], b)])
    return out[out[:, 1] > out[:, 0]]


def place_fields(s, pyr, maze):
    from scipy.ndimage import gaussian_filter1d
    t, x = s["t"], s["x"]
    keep = (t >= maze[0]) & (t < maze[1]) & np.isfinite(x)
    t, x = t[keep], x[keep]
    speed = np.r_[0, np.abs(np.diff(x)) / np.maximum(np.diff(t), 1e-6)]
    run = speed > 3.0
    dt = float(np.median(np.diff(t)))
    edges = np.linspace(0, LENGTH_CM, N_BINS + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    occ = np.histogram(x[run], bins=edges)[0] * dt
    valid = occ > 0.2
    centres = np.full(len(s["spikes"]), np.nan)
    maps = np.zeros((len(s["spikes"]), N_BINS))
    for c in pyr:
        sp = s["spikes"][c]
        sp = sp[(sp >= maze[0]) & (sp < maze[1])]
        j = np.clip(np.searchsorted(t, sp), 1, len(t) - 1)
        on = run[j]
        rate = np.zeros(N_BINS)
        rate[valid] = np.histogram(x[j[on]], bins=edges)[0][valid] / occ[valid]
        rate = gaussian_filter1d(rate, 1.0, mode="nearest")
        maps[c] = rate
        if on.sum() >= 40 and rate.max() >= 2.0:
            centres[c] = mid[np.argmax(rate)]
    return centres, maps


def session_job(name):
    from cpca import event_spike_times, fit, population_bursts
    from cpca.direction import order_of
    from cpca.evaluate import order_vs_place
    from cpca.topology import mantel, reduced_similarity
    from scripts.plot_hc11_sequences import best_map_match
    s = load(name)
    rng = np.random.default_rng(seed("huszar-pilot", name))
    pre_ep, maze, post_ep = s["epochs"][0], s["epochs"][1], s["epochs"][2]
    total = s["epochs"][-1, 1] - s["epochs"][0, 0]
    rates = np.array([len(x) / total for x in s["spikes"]])
    pyr = np.flatnonzero(rates < 8.0)
    centres, _ = place_fields(s, pyr, maze)
    place = np.flatnonzero(np.isfinite(centres))
    by_cell = {c: s["spikes"][c] for c in pyr}
    out = {"name": name, "units": len(s["spikes"]), "pyramidal": len(pyr), "place": len(place), "trials": len(s["trials"]), "epochs": {}}
    # waking fit on the trials
    wt, _ = event_spike_times(by_cell, pyr, s["trials"][:, 0], s["trials"][:, 1], max_spikes=400)
    wake = fit(wt, (s["trials"][:, 1] - s["trials"][:, 0]) * 1000, rng=rng, ica=False, center=False, scale=False)
    W = reduced_similarity(wake["eigenvalues"][:wake["n_kept"]], wake["pca_components"][:, :wake["n_kept"]])
    wake_active = np.isfinite(wt).any(axis=-1).sum(axis=0)
    for label, window in (("PRE", pre_ep), ("POST", post_ep)):
        intervals = clip(s["nrem"], window)
        pooled = np.sort(np.concatenate([x[(x >= window[0]) & (x < window[1])] for x in (s["spikes"][c] for c in pyr)]))
        bursts, peaks = population_bursts(pooled, intervals)
        dur = np.diff(bursts, axis=1).ravel()
        times, _ = event_spike_times(by_cell, pyr, peaks - dur / 2, peaks + dur / 2)
        res = fit(times, dur * 1000, rng=rng, ica=False)
        k = res["n_kept"]
        key = centres[pyr]
        orders = [order_of(res["pca_components"][:, j]) for j in range(k)]
        tests = [order_vs_place(o, key, False, LENGTH_CM, rng, n_null=500) for o in orders]
        hits = sum(np.isfinite(r) and abs(r) > n95 for r, n95, _ in tests)
        testable = sum(np.isfinite(r) for r, _, _ in tests)
        best = best_map_match(orders, {1.0: key}, False, LENGTH_CM, rng)
        S = reduced_similarity(res["eigenvalues"][:k], res["pca_components"][:, :k]) if k else None
        frac = np.isfinite(times).any(axis=-1).mean(axis=0)
        cells = np.flatnonzero((wake_active >= 5) & (frac >= 0.01))
        ws = mantel(S[np.ix_(cells, cells)], W[np.ix_(cells, cells)], rng, n_perm=2000) if S is not None else {"rho": np.nan, "p": np.nan}
        out["epochs"][label] = {"nrem_min": float(np.sum(np.diff(intervals, axis=1)) / 60), "events": len(peaks), "k": k,
                                "hits": hits, "testable": testable, "best": best, "wake_sleep": ws,
                                "rhos": [float(r) for r, _, _ in tests if np.isfinite(r)]}
    return out


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=len(SESSIONS)) as pool:
        results = list(pool.map(session_job, SESSIONS))
    lines = ["# Pilot: the hc-11 pipeline on DANDI 000552 (Huszar et al., 2022)", "",
             "Produced by `scripts/pilot_huszar.py`. Two sessions fixed in advance (most units in each of the two mice with most units).", "",
             "| session | units | putative pyramidal | place cells | trials | epoch | NREM (min) | events | components | components on the map | best output |ρ| (p) | waking vs sleep ρ (p) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for e in ("PRE", "POST"):
            x = r["epochs"][e]
            b = x["best"]
            lines.append(f"| {r['name']} | {r['units']} | {r['pyramidal']} | {r['place']} | {r['trials']} | {e} | {x['nrem_min']:.0f} | {x['events']} | {x['k']} | "
                         f"{x['hits']}/{x['testable']} | {b['abs']:.2f} ({b['p']:.3f}) | {x['wake_sleep']['rho']:+.3f} ({x['wake_sleep']['p']:.4f}) |")
            print(lines[-1])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "huszar_pilot.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
