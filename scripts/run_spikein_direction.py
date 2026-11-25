"""Check C of docs/segments.md (addendum): what direction balance costs on real hc-11 events.

The spike-in power test of docs/replay.md (scripts/run_spikein_hc11.py: the matched 1500 PRE events of each session, a
share f of them turned into ideal replays of the waking map, 5 draws) is repeated with the same seeds, so the balanced
case reproduces the published data event for event, and once more with every planted replay forward. Read-outs of the
centred complex PCA fit: standard (single components), adjacent conjugate pairs, all pairs (cpca/pairs.py). Oracles on
the same real events: the planted backward replays played forward (arrows conjugated; other events conjugated at
random) with the uncentred leading eigenvector, and the same alignment as time reversal of the spike times followed by
the standard centred fit and read-out. Scoring: best_map_match, as in the power test.

Usage: python scripts/run_spikein_direction.py [--workers=8]
Output: results/spikein_direction/<session>.pkl
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

from scripts.run_spikein_hc11 import FRACTIONS, HC11_ORDER, N_DRAWS  # noqa: E402

OUT = ROOT / "results" / "spikein_direction"
N_NULL = 500
READOUTS = ("standard", "pairs_adjacent", "pairs_all", "oracle_uncentred", "oracle_centred")


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def ideal_directed(times, T, chosen, centres, rng, forward_only, jitter=0.05):
    """scripts.run_spikein_hc11.ideal with the same random stream, returning each chosen event's sense (+1 forward,
    -1 backward); with forward_only the sense draw is made but every replay is played forward."""
    out = times.copy()
    sense_of = np.zeros(times.shape[0], int)
    keys = [k for k, c in centres.items() if np.isfinite(c).sum() >= 6]
    for e in chosen:
        key = keys[rng.integers(len(keys))]
        cen = centres[key]
        has = np.isfinite(cen)
        lo, hi = np.nanmin(cen), np.nanmax(cen)
        sense = 1 if rng.random() < 0.5 else -1
        if forward_only:
            sense = 1
        sense_of[e] = sense
        for c in np.flatnonzero(has & np.isfinite(times[e]).any(axis=-1)):
            u = (cen[c] - lo) / max(hi - lo, 1e-9)
            u = u if sense > 0 else 1 - u
            n = int(np.isfinite(times[e, c]).sum())
            t = np.clip((0.1 + 0.8 * u + jitter * rng.standard_normal(n)) * T[e], 0, T[e])
            out[e, c, :] = np.nan
            out[e, c, :n] = np.sort(t)
    return out, sense_of


def readouts(times, T, sense, rng):
    from cpca import encode, fit
    from cpca.direction import order_of
    from cpca.pairs import pair_orders
    from cpca.theory import eigh_desc, lag_matrix
    res = fit(times, T, rng=rng, ica=False)
    P = res["patterns"]
    out = {"standard": [order_of(P[:, j], member_frac=0.2) for j in range(res["n_kept"])],
           "pairs_adjacent": pair_orders(P, how="adjacent") if res["n_kept"] else [],
           "pairs_all": pair_orders(P, how="all") if res["n_kept"] else []}
    signs = np.where(sense != 0, sense, rng.choice([-1, 1], size=sense.size))
    S = encode(times, T)
    q = eigh_desc(lag_matrix(np.where((signs > 0)[:, None], S, np.conj(S))))[1][:, 0]
    out["oracle_uncentred"] = [order_of(q, member_frac=0.2)]
    flip = signs < 0
    reversed_times = times.copy()
    reversed_times[flip] = T[flip, None, None] - times[flip]
    res_o = fit(reversed_times, T, rng=rng, ica=False)
    out["oracle_centred"] = [order_of(res_o["patterns"][:, j], member_frac=0.2) for j in range(res_o["n_kept"])]
    return out, res["n_kept"], res_o["n_kept"]


def job(name):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_rerun import ORDER_INDEX, fixed_subset
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
    rows = []
    for kind in ("balanced", "forward"):
        for f in FRACTIONS:
            if f == 0 and kind == "forward":
                continue
            for draw in range(N_DRAWS):
                rng = np.random.default_rng(seed("spikein", name, "ideal", f, draw))          # the published spike-in stream
                k = int(round(f * len(T)))
                chosen = rng.choice(len(T), k, replace=False)
                times, sense = (ideal_directed(base, T, chosen, centres, rng, kind == "forward") if k else (base, np.zeros(len(T), int)))
                orders, n_kept, n_kept_oracle = readouts(times, T, sense, np.random.default_rng(seed("spikein-direction-fit", name, kind, f, draw)))
                row = {"kind": kind, "f": f, "draw": draw, "k": k, "n_kept": n_kept, "n_kept_oracle": n_kept_oracle}
                for key, o in orders.items():
                    s = best_map_match(o, centres, run["circular"], run["length"], np.random.default_rng(seed("spikein-direction-score", name, kind, f, draw, key)),
                                       n_null=N_NULL) if o else {"abs": np.nan, "p": np.nan}
                    row[key] = {"abs": s["abs"], "p": s["p"], "n_outputs": len(o)}
                rows.append(row)
    out = {"name": name, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT / f"{name}.pkl", "wb"))
    parts = []
    for kind in ("balanced", "forward"):
        for f in FRACTIONS:
            rs = [r for r in rows if r["kind"] == kind and r["f"] == f]
            if rs:
                parts.append(f"{kind[0]}{f:.0%}: " + "/".join(str(sum(np.isfinite(r[k]["p"]) and r[k]["p"] < 0.05 for r in rs)) for k in READOUTS))
    return "passes of 5 (standard/pairs/all/oracle-unc/oracle-cen) " + ", ".join(parts)


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
