"""Follow-up to the ceiling (docs/ceiling.md): does the event definition hide the replays?

Only the selection step: the share of events whose A-cell order matches the map, against the same share with
A's centres shuffled, for four event definitions:
- bursts, mean spike time per cell (as in the ceiling run);
- bursts, first spike per cell;
- a short window around each burst's peak (peak +/- 75 ms), first spike;
- DANDI 000552 only: the dataset's own ripples in NREM (ripple peak +/- 75 ms), first spike.
hc-11: all 8 sessions; 000552: the 16 sessions with the most units; POST and PRE; 3 cell splits per map.

Usage: python scripts/run_ceiling_events.py [--workers=10]
Results: results/ceiling/events/<dataset>_<session>_<epoch>.pkl, results/ceiling/events_summary.md
"""

import csv
import os
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "ceiling" / "events"
N_SPLITS = 3
HALF_WINDOW_S = 0.075
VARIANTS = ["bursts, mean spike", "bursts, first spike", "burst peak +/- 75 ms", "ripples +/- 75 ms"]


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def first_times(times):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(times, axis=-1)


def mean_times(times):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(times, axis=-1)


def events(dataset, ident, epoch):
    """by_cell, cells, burst windows (start, end, peak), ripple peaks or None, centres, circular, length."""
    from cpca import population_bursts
    if dataset == "hc11":
        from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
        session = load_session(next(p for p in session_paths() if p.name.startswith(ident)))
        run = running(session)
        iv = nrem_intervals(session, epoch)
        by_cell, cells, pooled = pyramidal_spikes(session, iv)
        bursts, peaks = population_bursts(pooled, iv)
        return by_cell, cells, bursts, peaks, None, place_centres(session, cells, run), run["circular"], run["length"]
    import h5py
    from cpca import huszar as hz
    from cpca.events import in_intervals
    s = hz.load_session(ident)
    pre_w, task_w, post_w = hz.task_window(s)
    pyr = hz.putative_pyramidal(s)
    maps, n_spikes, mid = hz.arm_maps(s, pyr, task_w)
    centres = hz.place_centres(maps, n_spikes, mid)
    w = pre_w if epoch == "pre" else post_w
    iv = hz.nrem_in(s, w)
    pooled = np.sort(np.concatenate([x[(x >= w[0]) & (x < w[1])] for x in (s["spikes"][c] for c in pyr)]))
    bursts, peaks = population_bursts(pooled, iv)
    with h5py.File(ident, "r") as f:
        rp = f["processing/ecephys/Ripples/peaks"][:]
    rp = rp[in_intervals(rp, iv)]
    return {c: s["spikes"][c] for c in pyr}, pyr, bursts, peaks, rp, centres, False, hz.LENGTH_CM


def job(dataset, ident, epoch):
    from cpca import event_spike_times
    from scripts.run_ceiling import select_replays
    by_cell, cells, bursts, peaks, ripples, centres, circular, length = events(dataset, ident, epoch)
    name = ident if dataset == "hc11" else Path(ident).name.replace("_behavior+ecephys.nwb", "")
    dur = np.diff(bursts, axis=1).ravel()
    t_burst, _ = event_spike_times(by_cell, cells, peaks - dur / 2, peaks + dur / 2)
    t_peak, _ = event_spike_times(by_cell, cells, peaks - HALF_WINDOW_S, peaks + HALF_WINDOW_S)
    per_cell = {VARIANTS[0]: mean_times(t_burst), VARIANTS[1]: first_times(t_burst), VARIANTS[2]: first_times(t_peak)}
    if ripples is not None and ripples.size:
        t_rip, _ = event_spike_times(by_cell, cells, ripples - HALF_WINDOW_S, ripples + HALF_WINDOW_S)
        per_cell[VARIANTS[3]] = first_times(t_rip)
    rng = np.random.default_rng(seed("ceiling-events", dataset, name, epoch))
    out = {"dataset": dataset, "name": name, "epoch": epoch, "n_events": {k: int(v.shape[0]) for k, v in per_cell.items()}, "rows": []}
    for d, cen in centres.items():
        place = np.flatnonzero(np.isfinite(cen))
        if place.size < 16:
            continue
        for s in range(N_SPLITS):
            A = np.sort(rng.permutation(place)[: place.size // 2])
            cen_A = np.full(cen.size, np.nan)
            cen_A[A] = cen[A]
            cen_S = np.full(cen.size, np.nan)
            cen_S[A] = cen[rng.permutation(A)]
            for variant, mt in per_cell.items():
                rho, n95 = select_replays(mt, cen_A, circular, length, rng)
                rho_s, n95_s = select_replays(mt, cen_S, circular, length, rng)
                n = max(int(np.isfinite(rho).sum()), 1)
                out["rows"].append({"map": d, "split": s, "variant": variant, "n_testable": n,
                                    "selected": float(np.sum(np.isfinite(rho) & (np.abs(rho) > n95)) / n),
                                    "shuffled": float(np.sum(np.isfinite(rho_s) & (np.abs(rho_s) > n95_s)) / max(int(np.isfinite(rho_s).sum()), 1))})
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT / f"{dataset}_{name}_{epoch}.pkl", "wb"))
    return name, epoch


def summary():
    R = [pickle.load(open(p, "rb")) for p in sorted(OUT.glob("*.pkl"))]
    lines = ["# Does the event definition hide the replays?", "", "Generated by `scripts/run_ceiling_events.py`. Share of events whose A-cell order "
             "matches the map (selected) against the same share with A's centres shuffled (chance), median over sessions, maps and splits; "
             "excess = selected − chance.", "",
             "| dataset | epoch | event definition | events per session (median) | selected | chance | excess | splits with excess > 0 |", "|---|---|---|---|---|---|---|---|"]
    for ds in ("hc11", "huszar"):
        for ep in ("post", "pre"):
            rows = [r for r in R if r["dataset"] == ds and r["epoch"] == ep]
            for v in VARIANTS:
                x = [row for r in rows for row in r["rows"] if row["variant"] == v]
                if not x:
                    continue
                sel = np.array([row["selected"] for row in x])
                shf = np.array([row["shuffled"] for row in x])
                nev = np.median([r["n_events"][v] for r in rows if v in r["n_events"]])
                lines.append(f"| {ds} | {ep.upper()} | {v} | {nev:.0f} | {np.median(sel):.3f} | {np.median(shf):.3f} | {np.median(sel - shf):+.3f} | "
                             f"{np.mean(sel - shf > 0):.2f} |")
    (ROOT / "results" / "ceiling" / "events_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), os.cpu_count() or 2)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    from scripts.run_ceiling import HC11_ORDER
    rows = [r for r in csv.DictReader(open(ROOT / "results" / "datasets" / "huszar_sessions.csv")) if r["trials"] and int(r["trials"]) > 0]
    top = [str(ROOT / "data" / "huszar" / Path(r["path"]).name) for r in sorted(rows, key=lambda r: -int(r["units"]))[:16]]
    jobs = [("hc11", n, e) for n in HC11_ORDER for e in ("post", "pre")] + [("huszar", p, e) for p in top for e in ("post", "pre")]
    jobs = [j for j in jobs if not (OUT / f"{j[0]}_{j[1] if j[0] == 'hc11' else Path(j[1]).name.replace('_behavior+ecephys.nwb', '')}_{j[2]}.pkl").exists()]
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, epoch in pool.map(job, *zip(*jobs)) if jobs else []:
            print("done", name[:40], epoch, flush=True)
    summary()


if __name__ == "__main__":
    main(sys.argv[1:])
