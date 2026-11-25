"""hc-11: the original paper's Bayesian replay analysis, to check whether sleep events carry the maze's order.

Grosmark & Buzsaki (2016) report more sequence content (rZ) in POST than in PRE sleep and the most in awake
immobility on the maze. This script reproduces their procedure (cpca/replay.py) on the same eight sessions:
  paper events   population-synchrony events as in the paper (no LFP ripple condition: no LFP in the files)
                 in NREM of PRE and POST, yoked to the shorter epoch (PRE: its last T, POST: its first T), and
                 in immobility on the maze (MAZE, 2-D head speed under 5 cm/s);
  our events     the population bursts every method was given (cpca.events.population_bursts on NREM of each
                 epoch, windows centred on the burst peak), scored the same way, to see whether our event
                 definition loses sequence content.
Sanity check of the templates: position decoded from running in 500 ms bins (median error, against one
shuffled template set), as in the paper's fig. S3.

Usage: python scripts/run_replay_hc11.py [--workers=8]
Output: results/replay/<session>.pkl
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

OUT = ROOT / "results" / "replay"
HC11_ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
              "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def intersect(a, b):
    out = []
    for s, e in a:
        for s2, e2 in b:
            lo, hi = max(s, s2), min(e, e2)
            if hi > lo:
                out.append((lo, hi))
    return np.array(out) if out else np.zeros((0, 2))


def two_d(path):
    import h5py
    with h5py.File(path, "r") as f:
        return np.array(f["sessInfo/Position/TwoDLocation"])


def running_check(session, by_cell, cells, fields, run, direction, rng, bin_s=0.5):
    """Median error (m) of position decoded from running in 500 ms bins, observed template and one shuffled set."""
    from cpca.replay import EPS_HZ, null_templates
    t, x, d = run["t"], run["x"], run["direction"]
    use = d == direction
    starts = np.arange(t[use].min(), t[use].max() - bin_s, bin_s)
    mid = starts + bin_s / 2
    j = np.clip(np.searchsorted(t, mid), 0, t.size - 1)
    ok = use[j] & (np.abs(t[j] - mid) < 0.05)
    starts, true_x = starts[ok], x[j[ok]]
    counts = np.zeros((starts.size, len(cells)))
    for k, c in enumerate(cells):
        sp = by_cell[c]
        counts[:, k] = np.searchsorted(sp, starts + bin_s) - np.searchsorted(sp, starts)
    errors = []
    for F in (fields["rate"][fields["place"]], null_templates(fields["raw"][fields["place"]], rng, 1)[0]):
        L = counts[:, fields["place"]] @ np.log(F + EPS_HZ) - bin_s * F.sum(axis=0)
        errors.append(float(np.median(np.abs(fields["x"][np.argmax(L, axis=1)] - true_x))))
    return {"median_error_m": errors[0], "median_error_shuffled_m": errors[1], "n_bins": int(starts.size)}


def job(name):
    from cpca import population_bursts
    from cpca.hc11 import load_session, nrem_intervals, running, session_paths
    from cpca.replay import (TAU_S, binned, decoder, direction_of, immobile_intervals, null_templates, place_fields,
                             score_event, synchrony_events)
    t_start = time.perf_counter()
    path = next(p for p in session_paths() if p.name.startswith(name))
    session = load_session(path)
    session["pos_xy"] = two_d(path)
    run = running(session)
    rng = np.random.default_rng(seed("replay", name))
    cells = np.array(sorted(session["pyr"]))
    keep = np.isin(session["ids"], cells)
    st, sid = session["times"][keep], session["ids"][keep]
    order = np.argsort(st, kind="stable")
    st, sid = st[order], sid[order]
    by_cell = {c: st[sid == c] for c in cells}

    # epochs: NREM of PRE and POST yoked to the shorter epoch; immobility on the maze
    nrem_pre, nrem_post = nrem_intervals(session, "pre"), nrem_intervals(session, "post")
    T = min(np.diff(session["pre"])[0], np.diff(session["post"])[0])
    allowed = {"pre": intersect(nrem_pre, [(session["pre"][1] - T, session["pre"][1])]),
               "post": intersect(nrem_post, [(session["post"][0], session["post"][0] + T)]),
               "maze": intersect(immobile_intervals(session), [tuple(session["maze_epoch"])])}
    baseline = np.vstack([nrem_pre, nrem_post])

    # templates per running direction
    directions = [1.0, -1.0] if not run["circular"] else [max((1.0, -1.0), key=lambda s: np.sum(run["direction"] == s))]
    fields, decs, checks = {}, {}, {}
    for dirn in directions:
        f = place_fields(session, cells, run, dirn, rng)
        fields[dirn] = f
        decs[dirn] = decoder(f["rate"][f["place"]], null_templates(f["raw"][f["place"]], rng))
        checks[dirn] = running_check(session, by_cell, cells, f, run, dirn, rng)
    mean_rates = [fields[d]["mean_rate"] for d in directions]

    def score(events):
        rows = []
        for s, e in events:
            M = int(np.floor((e - s) / TAU_S + 1e-9))
            if M < 5:
                continue
            counts = binned(by_cell, cells, s, M)
            k = direction_of(counts, mean_rates, M * TAU_S) if len(directions) > 1 else 0
            dirn = directions[k]
            pc = counts[:, fields[dirn]["place"]]
            n_place = int(fields[dirn]["place"].sum())
            n_active = int((pc.sum(axis=0) > 0).sum())
            if n_active < max(5, int(np.ceil(0.1 * n_place))):
                continue
            res = score_event(pc, decs[dirn], fields[dirn]["x"], rng)
            rows.append({"start": s, "end": e, "direction": dirn, "n_active": n_active, **res})
        return {key: np.array([r[key] for r in rows]) for key in ("start", "end", "direction", "n_active", "r", "rZ", "p", "p_time")} if rows else {}

    out = {"name": name, "circular": run["circular"], "length": run["length"], "directions": directions,
           "n_place": {d: int(fields[d]["place"].sum()) for d in directions}, "running_check": checks, "yoke_s": float(T),
           "paper": {}, "ours": {}, "n_candidates": {}}
    for ep in ("pre", "maze", "post"):
        ev = synchrony_events(st, baseline, allowed[ep], by_cell, cells)
        out["n_candidates"][ep] = int(len(ev))
        out["paper"][ep] = score(ev)
    for ep in ("pre", "post"):
        iv = nrem_pre if ep == "pre" else nrem_post
        pooled = st[_inside(st, iv)]
        bursts, peaks = population_bursts(pooled, iv)
        d = np.diff(bursts, axis=1).ravel()
        out["ours"][ep] = score(np.c_[peaks - d / 2, peaks + d / 2])
        out["n_candidates"][f"ours_{ep}"] = int(len(bursts))
    out["runtime_s"] = time.perf_counter() - t_start
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{name}.tmp"
    pickle.dump(out, open(tmp, "wb"))
    tmp.replace(OUT / f"{name}.pkl")

    def line(res):
        if not res:
            return "none"
        return f"n {res['rZ'].size}, mean rZ {np.nanmean(res['rZ']):.2f}, p<.05 {np.mean(res['p'] < 0.05):.1%}"
    return (f"place cells {out['n_place']} | run error {[round(c['median_error_m'], 2) for c in checks.values()]} m "
            f"(shuffled {[round(c['median_error_shuffled_m'], 2) for c in checks.values()]}) | "
            + " | ".join(f"{ep.upper()} {line(out['paper'][ep])}" for ep in ("pre", "maze", "post"))
            + " || ours " + " | ".join(f"{ep.upper()} {line(out['ours'][ep])}" for ep in ("pre", "post"))
            + f" | {out['runtime_s']:.0f}s")


def _inside(times, intervals):
    from cpca.events import in_intervals
    return in_intervals(times, intervals)


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 8)
    only = [a for a in argv if not a.startswith("--")]
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    todo = [n for n in HC11_ORDER if (not only or any(n.startswith(o) for o in only)) and not (OUT / f"{n}.pkl").exists()]
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
