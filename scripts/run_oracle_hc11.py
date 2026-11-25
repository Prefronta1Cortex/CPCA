"""Step 0: the most any label-free method could recover from hc-11 sleep, given perfect event selection.

Per session and sleep epoch, on the matched 1500-event subset every method was given (scripts/run_rerun.py):
the place cells (a field in our answer key or under the paper's criterion, either direction) are split at
random into halves A and B, 10 times. Half A selects: the paper's Bayesian decoder (cpca/replay.py) built from
A's place cells only decodes each event; clear replays are those significant under both shuffles; the sign of
the weighted correlation says forward or reverse, the decoded direction says which map. Half B is scored: each
B cell's mean relative spike time in the selected events (flipped for reverse replays) is averaged per decoded
map, and these orders are scored by the test used for every method (scripts/plot_hc11_sequences.best_map_match,
B's field centres only). Also: a looser selection (significant against shuffled place fields only, more events);
the control, the same number of events drawn at random among those the decoder could test, averaged the same way
with random senses; and complex PCA (centred, the rerun's read-out) fitted once to the events without the map
and scored on the same B cells, so that it and the oracle are compared on equal terms.

No cell both selects and scores. Averaging the chosen events is the read-out a perfect label-free method would
reduce to once it knew which events are replays and which way they run (it recovered planted order in 92-100%
of splits in docs/ceiling.md), so the share of splits that pass bounds every method on these events.

Usage: python scripts/run_oracle_hc11.py [--workers=8]
Output: results/oracle/<session>.pkl
"""

import os
import pickle
import sys
import time
import traceback
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "oracle"
HC11_ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
              "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]
N_SPLITS = 10
N_NULL_TEMPLATES = 500


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def averaged_orders(times, T, events, senses, groups, b_cells):
    """One order per group: each B cell's mean relative time over the group's events (1 - t for reverse)."""
    warnings.filterwarnings("ignore", "Mean of empty slice")
    with np.errstate(invalid="ignore"):
        rel = np.nanmean(times[events][:, b_cells], axis=-1) / T[events, None]          # (k, B)
    rel = np.where(senses[:, None] > 0, rel, 1 - rel)
    orders = []
    for g in np.unique(groups):
        sel = groups == g
        if sel.sum() < 3:
            continue
        o = np.full(times.shape[1], np.nan)
        with np.errstate(invalid="ignore"):
            o[b_cells] = np.nanmean(rel[sel], axis=0)
        orders.append(o)
    return orders


def job(name):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    from cpca.replay import TAU_S, binned, decoder, direction_of, null_templates, place_fields, score_event
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_rerun import ORDER_INDEX, fixed_subset, run_cpca
    t0 = time.perf_counter()
    session = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(session)
    rng = np.random.default_rng(seed("oracle-fields", name))
    directions = [1.0, -1.0] if not run["circular"] else [max((1.0, -1.0), key=lambda s: np.sum(run["direction"] == s))]
    out = {"name": name, "epochs": {}}
    for ei, epoch in enumerate(("post", "pre")):
        intervals = nrem_intervals(session, epoch)
        by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
        if ei == 0:
            fields = {d: place_fields(session, cell_ids, run, d, rng) for d in directions}
            key = place_centres(session, cell_ids, run)
            candidates = np.flatnonzero(np.isfinite(key[1.0]) | np.isfinite(key[-1.0]) | np.any([fields[d]["place"] for d in directions], axis=0))
        bursts, peaks = population_bursts(pooled, intervals)
        duration = np.diff(bursts, axis=1).ravel()
        sub = fixed_subset(len(peaks), 555 + 2 * ORDER_INDEX[name] + ei)
        starts, T = (peaks - duration / 2)[sub], duration[sub] * 1000
        times, _ = event_spike_times(by_cell, cell_ids, starts, starts + duration[sub], max_spikes=60)
        n_bins = np.floor(duration[sub] / TAU_S + 1e-9).astype(int)
        counts = [binned(by_cell, cell_ids, s, m) if m >= 5 else None for s, m in zip(starts, n_bins)]
        cpca_orders = run_cpca(times, T, np.random.default_rng(31 + ei))["centred"]["orders"]
        splits = []
        for split in range(N_SPLITS):
            srng = np.random.default_rng(seed("oracle-split", name, split))       # the same split in both epochs
            perm = srng.permutation(candidates)
            A, B = np.sort(perm[: perm.size // 2]), np.sort(perm[perm.size // 2:])
            inA = np.zeros(len(cell_ids), bool)
            inA[A] = True
            decs, use = {}, {}
            for d in directions:
                use[d] = fields[d]["place"] & inA
                decs[d] = decoder(fields[d]["rate"][use[d]], null_templates(fields[d]["raw"][use[d]], srng, N_NULL_TEMPLATES))
            mean_rates = [np.where(inA, fields[d]["mean_rate"], 0.0) for d in directions]
            erng = np.random.default_rng(seed("oracle-events", name, epoch, split))
            testable, chosen, senses, groups, loose = [], [], [], [], []
            for e, c in enumerate(counts):
                if c is None:
                    continue
                k = direction_of(c[:, inA], [m[inA] for m in mean_rates], c.shape[0] * TAU_S) if len(directions) > 1 else 0
                d = directions[k]
                pc = c[:, use[d]]
                if (pc.sum(axis=0) > 0).sum() < max(5, int(np.ceil(0.1 * use[d].sum()))):
                    continue
                testable.append((e, k))
                res = score_event(pc, decs[d], fields[d]["x"], erng, n_time=500)
                if res["p"] < 0.05:
                    loose.append((e, np.sign(res["r"]), k))
                if res["p"] < 0.05 and res["p_time"] < 0.05:
                    chosen.append(e)
                    senses.append(np.sign(res["r"]))
                    groups.append(k)
            centres_B = {s: np.where(np.isin(np.arange(len(cell_ids)), B), c, np.nan) for s, c in key.items()}
            res = {"split": split, "n_testable": len(testable), "n_chosen": len(chosen), "n_A": int(A.size), "n_B": int(B.size)}
            res["n_loose"] = len(loose)
            for label, ev, sn, gr in (("oracle", np.array(chosen, int), np.array(senses), np.array(groups)),
                                      ("loose", np.array([x[0] for x in loose], int), np.array([x[1] for x in loose]), np.array([x[2] for x in loose], int)),
                                      ("random", *_random_like(testable, len(chosen), erng)),
                                      ("cpca", None, None, None)):
                if label == "cpca":
                    orders = cpca_orders
                else:
                    orders = averaged_orders(times, T, ev, sn, gr, B) if ev.size else []
                s = best_map_match(orders, centres_B, run["circular"], run["length"], np.random.default_rng(seed("oracle-score", name, epoch, split, label))) \
                    if orders else {"abs": np.nan, "p": np.nan, "n": 0}
                res[label] = {"abs": s["abs"], "p": s["p"], "n": s["n"]}
            splits.append(res)
        out["epochs"][epoch] = {"n_events": int(len(sub)), "splits": splits}
    out["runtime_s"] = time.perf_counter() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT / f"{name}.pkl", "wb"))

    def line(ep):
        sp = out["epochs"][ep]["splits"]
        return (f"{ep.upper()} chosen {np.median([s['n_chosen'] for s in sp]):.0f}/{np.median([s['n_testable'] for s in sp]):.0f} testable, "
                f"oracle {sum(np.isfinite(s['oracle']['p']) and s['oracle']['p'] < 0.05 for s in sp)}/{len(sp)} "
                f"(|ρ| {np.nanmedian([s['oracle']['abs'] for s in sp]):.2f}), " +
                ", ".join(f"{lab} {sum(np.isfinite(s[lab]['p']) and s[lab]['p'] < 0.05 for s in sp)}/{len(sp)}" for lab in ("loose", "random", "cpca")))
    return " | ".join(line(ep) for ep in ("post", "pre")) + f" | {out['runtime_s']:.0f}s"


def _random_like(testable, k, rng):
    if not testable or k == 0:
        return np.zeros(0, int), np.zeros(0), np.zeros(0, int)
    pick = rng.choice(len(testable), min(k, len(testable)), replace=False)
    ev = np.array([testable[i][0] for i in pick], int)
    gr = np.array([testable[i][1] for i in pick], int)
    return ev, rng.choice([-1.0, 1.0], ev.size), gr


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
