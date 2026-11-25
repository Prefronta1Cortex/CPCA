"""Complex PCA on DANDI 001695 (docs/dandi001695.md): POST, PRE (matched counts) and MAZE rest, CA1 pyramidal cells.

Per session and epoch: centred complex PCA; whole-track order test and fragment criterion, each with the timing
control (100 refits on spike trains shuffled among each event's active cells); the replay content of the same
events (cpca.replay, clear replays); the same on the ripple-gated events where there are at least 200.

Usage: python scripts/run_dandi001695.py [--areas=CA1|CA1,CA3] [--workers=9]
Output: results/dandi001695/<areas>/<session>_<epoch>.pkl
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

OUT = ROOT / "results" / "dandi001695"
N_SHUFFLES = 100


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def overlaps(events, intervals):
    """Boolean per event: its window overlaps at least one interval."""
    if not len(intervals):
        return np.zeros(len(events), bool)
    iv = intervals[np.argsort(intervals[:, 0])]
    j = np.searchsorted(iv[:, 0], events[:, 1])                  # intervals starting before the event ends
    stop_max = np.maximum.accumulate(iv[:, 1])
    return (j > 0) & (stop_max[np.maximum(j - 1, 0)] > events[:, 0])


def order_scores(times, T, centres, length, rng, n_shuffles):
    from cpca.fragment import best_fragment, timing_p
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_maze_rest import shuffle_timing
    from scripts.run_rerun import run_cpca
    orders = run_cpca(times, T, rng)["centred"]["orders"]
    none = {"abs": np.nan, "p": np.nan, "n": 0, "index": -1}
    whole = best_map_match(orders, centres, False, length, rng) if orders else dict(none)
    frag = best_fragment(orders, centres, length, rng, n_null=1000) if orders else dict(none)
    sw, sf = [], []
    for _ in range(n_shuffles):
        o = run_cpca(shuffle_timing(times, rng), T, rng)["centred"]["orders"]
        sw.append(best_map_match(o, centres, False, length, rng, n_null=1)["abs"] if o else np.nan)
        sf.append(best_fragment(o, centres, length, n_null=0)["abs"] if o else np.nan)
    return {"k": len(orders), "whole": whole, "fragment": frag, "timing_whole": timing_p(whole["abs"], sw),
            "timing_fragment": timing_p(frag["abs"], sf), "shuffled_whole": np.array(sw, float), "shuffled_fragment": np.array(sf, float)}


def replay_content(session, by_cell, cell_ids, starts, ends, run, rng, max_events=2000):
    from cpca.replay import TAU_S, binned, decoder, direction_of, null_templates, place_fields, score_event
    fields = {d: place_fields(session, cell_ids, run, d, rng) for d in (1.0, -1.0)}
    decs = {d: decoder(fields[d]["rate"][fields[d]["place"]], null_templates(fields[d]["raw"][fields[d]["place"]], rng, 500)) for d in fields}
    rates = [fields[d]["mean_rate"] for d in (1.0, -1.0)]
    pick = np.sort(rng.choice(len(starts), min(len(starts), max_events), replace=False))
    tested, clear, rz = 0, 0, []
    for s, e in zip(starts[pick], ends[pick]):
        M = int(np.floor((e - s) / TAU_S + 1e-9))
        if M < 5:
            continue
        c = binned(by_cell, cell_ids, s, M)
        d = (1.0, -1.0)[direction_of(c, rates, M * TAU_S)]
        pc = c[:, fields[d]["place"]]
        if (pc.sum(axis=0) > 0).sum() < max(5, int(np.ceil(0.1 * fields[d]["place"].sum()))):
            continue
        r = score_event(pc, decs[d], fields[d]["x"], rng, n_time=500)
        tested += 1
        clear += r["p"] < 0.05 and r["p_time"] < 0.05
        rz.append(r["rZ"])
    return {"n_tested": tested, "clear_share": clear / tested if tested else np.nan, "mean_rZ": float(np.nanmean(rz)) if rz else np.nan,
            "n_place": {d: int(fields[d]["place"].sum()) for d in fields}}


def job(path, epoch, areas):
    from cpca import event_spike_times, population_bursts
    from cpca.dandi001695 import load_session
    from cpca.hc11 import nrem_intervals, place_centres, pyramidal_spikes, running
    from cpca.replay import immobile_intervals
    from scripts.run_replay_hc11 import intersect
    t0 = time.perf_counter()
    s = load_session(path, areas=areas)
    run = running(s)
    rng = np.random.default_rng(seed("dandi001695", s["name"], epoch, ",".join(areas)))

    def bursts(iv):
        by_cell, cell_ids, pooled = pyramidal_spikes(s, iv)
        b, pk = population_bursts(pooled, iv)
        return by_cell, cell_ids, pk, np.diff(b, axis=1).ravel()

    if epoch == "maze":
        by_cell, cell_ids, pk, d = bursts(intersect(immobile_intervals(s), [tuple(s["maze_epoch"])]))
        sub = np.arange(len(pk))
    else:
        n = min(len(bursts(nrem_intervals(s, e))[2]) for e in ("pre", "post"))
        by_cell, cell_ids, pk, d = bursts(nrem_intervals(s, epoch))
        sub = np.sort(np.random.default_rng(seed("dandi001695-subset", s["name"], epoch)).choice(len(pk), n, replace=False)) if len(pk) > n else np.arange(len(pk))
    starts, ends = (pk - d / 2)[sub], (pk + d / 2)[sub]
    times, _ = event_spike_times(by_cell, cell_ids, starts, ends, max_spikes=60)
    T = d[sub] * 1000
    centres = place_centres(s, cell_ids, run)
    place = np.isfinite(centres[1.0]) | np.isfinite(centres[-1.0])
    out = {"name": s["name"], "mouse": s["name"].split("_")[0], "epoch": epoch, "areas": areas, "n_cells": len(cell_ids),
           "n_place": {k: int(np.isfinite(v).sum()) for k, v in centres.items()}, "n_events": int(len(sub)),
           "place_active_median": float(np.median(np.isfinite(times[:, place]).any(axis=-1).sum(axis=1))) if len(sub) else np.nan}
    if len(sub) < 100:
        out["skipped"] = "fewer than 100 events"
    else:
        out["all"] = order_scores(times, T, centres, run["length"], rng, N_SHUFFLES)
        rip = overlaps(np.column_stack([starts, ends]), s["ripples"])
        out["ripple_share"] = float(rip.mean())
        if rip.sum() >= 200:
            out["ripple"] = order_scores(times[rip], T[rip], centres, run["length"], rng, N_SHUFFLES // 2)
            out["ripple"]["n_events"] = int(rip.sum())
        out["replay"] = replay_content(s, by_cell, cell_ids, starts, ends, run, rng)
    out["runtime_s"] = time.perf_counter() - t0
    tag = "+".join(areas)
    target = OUT / tag / f"{s['name']}_{epoch}.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(target, "wb"))
    if "all" not in out:
        return f"{out['n_events']} events: skipped"
    a = out["all"]
    return (f"{out['n_events']} events, {out['place_active_median']:.0f} place cells active/event, clear replays {out['replay']['clear_share']:.1%} | "
            f"whole |ρ| {a['whole']['abs']:.2f} p {a['whole']['p']:.3f} timing {a['timing_whole']:.3f} | fragment p {a['fragment']['p']:.3f} timing {a['timing_fragment']:.3f}"
            f" | {out['runtime_s']:.0f}s")


def main(argv):
    from cpca.dandi001695 import session_paths
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    areas = tuple(next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--areas=")), ["CA1"]))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    tag = "+".join(areas)
    todo = [(p, e) for e in ("post", "pre", "maze") for p in session_paths()
            if not (OUT / tag / f"{p.name.split('_behavior')[0]}_{e}.pkl").exists()]
    print(f"{len(todo)} jobs on {workers} workers ({tag})", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, p, e, areas): (p.name.split("_behavior")[0], e) for p, e in todo}
        for f in as_completed(futures):
            try:
                print(f"{futures[f]}: {f.result()}", flush=True)
            except Exception:
                print(f"FAILED {futures[f]}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
