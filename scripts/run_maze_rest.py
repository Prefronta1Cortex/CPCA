"""hc-11: complex PCA and count PCA on awake rest on the maze, where replay is common.

Design (docs/maze_rest.md, written before running):
  events    population bursts (cpca.events.population_bursts, as for sleep) during immobility on the maze (2-D head
            speed under 5 cm/s, cpca.replay.immobile_intervals); spike windows centred on the burst peak as for
            sleep. The answer key uses only spikes while running faster than 5 cm/s, so no spike is in both.
  methods   complex PCA, centred (primary) and uncentred, best component (the rerun's read-out); seriation of the
            reduced space (one order of all cells, cpca.subspace.seriate) for complex PCA and for count PCA
            (z-scored spike counts per event, components above the shuffle null), the same read-out on both.
  scores    order: scripts/plot_hc11_sequences.best_map_match (best output, best-of-outputs null); topology: Mantel
            test of each reduced space's cell-pair similarity against the place maps' similarity (cpca.topology).
  replay    the share of these events that are clear replays under the paper's Bayesian analysis (cpca.replay,
            significant under both shuffles) and the place cells active per event, to place each session against
            complex PCA's measured threshold (docs/power.md).
  control   the same number of NREM bursts from PRE (drawn at random), analysed identically.
  timing    (added after the first results, because count PCA seriation recovered the order as often as complex
            PCA seriation) --timing: complex PCA on the same rest events with each event's spike trains shuffled
            among the cells active in it (who fires is kept, who fires when is destroyed), 100 draws; the same
            control on the matched 1500 POST sleep events of the method comparison (--timing --epoch=post).

Usage: python scripts/run_maze_rest.py [--workers=8] [--timing [--epoch=post] [--uncentred]]
Output: results/maze_rest/<session>.pkl
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

OUT = ROOT / "results" / "maze_rest"
HC11_ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
              "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def clear_replays(by_cell, cell_ids, starts, ends, fields, decs, directions, rng):
    """Share of events that are clear replays (paper's analysis) among those it can test, and how many it can test."""
    from cpca.replay import TAU_S, binned, direction_of, score_event
    mean_rates = [fields[d]["mean_rate"] for d in directions]
    tested, clear = 0, 0
    for s, e in zip(starts, ends):
        M = int(np.floor((e - s) / TAU_S + 1e-9))
        if M < 5:
            continue
        c = binned(by_cell, cell_ids, s, M)
        d = directions[direction_of(c, mean_rates, M * TAU_S) if len(directions) > 1 else 0]
        pc = c[:, fields[d]["place"]]
        if (pc.sum(axis=0) > 0).sum() < max(5, int(np.ceil(0.1 * fields[d]["place"].sum()))):
            continue
        tested += 1
        r = score_event(pc, decs[d], fields[d]["x"], rng, n_time=500)
        clear += r["p"] < 0.05 and r["p_time"] < 0.05
    return {"n_tested": tested, "clear": clear, "share": clear / tested if tested else np.nan}


def analyse(times, T, centres, maps, run, rng):
    from cpca.subspace import reduced_covariance, seriate
    from cpca.topology import map_similarity, mantel, reduced_similarity
    from cpca import fit
    from cpca.decompose import decompose
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_rerun import run_cpca
    from scripts.run_topology import count_similarity
    out = {}
    cp = run_cpca(times, T, rng)
    for conv in ("centred", "uncentred"):
        o = cp[conv]["orders"]
        out[f"cpca_{conv}"] = best_map_match(o, centres, run["circular"], run["length"], rng) if o else {"abs": np.nan, "p": np.nan, "n": 0, "index": -1}
        out[f"k_{conv}"] = cp[conv]["n_kept"]
    res = fit(times, T, rng=rng, ica=False)
    k = res["n_kept"]
    values, vectors, _, _ = decompose(res["encoded"], center=True, scale=True)
    S_count, k_count = count_similarity(times, rng)
    out["k_count"] = int(k_count)
    ring = run["circular"]
    if k:
        C = reduced_covariance(res["eigenvalues"], res["pca_components"], k)
        d = np.sqrt(np.clip(np.real(np.diag(C)), 1e-300, None))
        out["seriation_cpca"] = best_map_match([seriate(np.real(C) / np.outer(d, d), ring=ring)], centres, run["circular"], run["length"], rng)
    else:
        out["seriation_cpca"] = {"abs": np.nan, "p": np.nan, "n": 0, "index": -1}
    out["seriation_count"] = best_map_match([seriate(S_count, ring=ring)], centres, run["circular"], run["length"], rng) if k_count \
        else {"abs": np.nan, "p": np.nan, "n": 0, "index": -1}
    cells = np.flatnonzero(np.isfinite(centres[1.0]) | np.isfinite(centres[-1.0]))
    Mref = map_similarity(maps[cells].reshape(len(cells), -1))
    S_cpca = reduced_similarity(values, vectors, k=k, kind="phase")
    out["topology_cpca"] = mantel(S_cpca[np.ix_(cells, cells)], Mref, rng) if k else {"rho": np.nan, "p": np.nan}
    out["topology_count"] = mantel(S_count[np.ix_(cells, cells)], Mref, rng) if k_count else {"rho": np.nan, "p": np.nan}
    return out


def job(name):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, place_maps, pyramidal_spikes, running, session_paths
    from cpca.replay import decoder, immobile_intervals, null_templates, place_fields
    from scripts.run_replay_hc11 import intersect, two_d
    t0 = time.perf_counter()
    path = next(p for p in session_paths() if p.name.startswith(name))
    session = load_session(path)
    session["pos_xy"] = two_d(path)
    run = running(session)
    rng = np.random.default_rng(seed("maze-rest", name))
    rest = intersect(immobile_intervals(session), [tuple(session["maze_epoch"])])
    by_cell, cell_ids, pooled = pyramidal_spikes(session, rest)
    bursts, peaks = population_bursts(pooled, rest)
    dur = np.diff(bursts, axis=1).ravel()
    times, _ = event_spike_times(by_cell, cell_ids, peaks - dur / 2, peaks + dur / 2, max_spikes=60)
    centres = place_centres(session, cell_ids, run)
    maps, _, _ = place_maps(session, cell_ids, run)
    place = np.isfinite(centres[1.0]) | np.isfinite(centres[-1.0])
    out = {"name": name, "rest_s": float(np.diff(rest, axis=1).sum()), "n_events": int(len(peaks)), "circular": run["circular"],
           "place_active_median": float(np.median(np.isfinite(times[:, place]).any(axis=-1).sum(axis=1))) if len(peaks) else np.nan}
    directions = [1.0, -1.0] if not run["circular"] else [max((1.0, -1.0), key=lambda s: np.sum(run["direction"] == s))]
    fields = {d: place_fields(session, cell_ids, run, d, rng) for d in directions}
    decs = {d: decoder(fields[d]["rate"][fields[d]["place"]], null_templates(fields[d]["raw"][fields[d]["place"]], rng)) for d in directions}
    out["replay"] = clear_replays(by_cell, cell_ids, peaks - dur / 2, peaks + dur / 2, fields, decs, directions, rng)
    out["maze"] = analyse(times, dur * 1000, centres, maps, run, rng) if len(peaks) >= 50 else {}

    # control: the same number of PRE sleep bursts
    iv = nrem_intervals(session, "pre")
    bc, ids, pl = pyramidal_spikes(session, iv)
    b, pk = population_bursts(pl, iv)
    d = np.diff(b, axis=1).ravel()
    pick = np.sort(rng.choice(len(pk), min(len(pk), len(peaks)), replace=False))
    tp, _ = event_spike_times(bc, ids, (pk - d / 2)[pick], (pk + d / 2)[pick], max_spikes=60)
    out["pre"] = analyse(tp, d[pick] * 1000, centres, maps, run, rng) if len(pick) >= 50 else {}
    out["pre_replay"] = clear_replays(bc, ids, (pk - d / 2)[pick], (pk + d / 2)[pick], fields, decs, directions, rng)
    out["runtime_s"] = time.perf_counter() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT / f"{name}.pkl", "wb"))

    def fmt(r):
        if not r:
            return "too few events"
        return (f"cPCA {r['cpca_centred']['abs']:.2f} p={r['cpca_centred']['p']:.3f} (k {r['k_centred']}) | uncentred p={r['cpca_uncentred']['p']:.3f} | "
                f"seriation cPCA p={r['seriation_cpca']['p']:.3f}, count p={r['seriation_count']['p']:.3f} | "
                f"topology cPCA p={r['topology_cpca']['p']:.3f}, count p={r['topology_count']['p']:.3f}")
    return (f"{out['n_events']} rest events ({out['rest_s'] / 60:.0f} min rest), {out['place_active_median']:.0f} place cells active/event, "
            f"clear replays {out['replay']['share']:.0%} of {out['replay']['n_tested']} | MAZE {fmt(out['maze'])} || PRE {fmt(out['pre'])}, "
            f"clear {out['pre_replay']['share']:.0%} | {out['runtime_s']:.0f}s")


def shuffle_timing(times, rng):
    """Within each event, give the active cells each other's spike trains (which cells fire is kept)."""
    out = times.copy()
    for e in range(times.shape[0]):
        act = np.flatnonzero(np.isfinite(times[e]).any(axis=-1))
        out[e, act] = times[e, act[rng.permutation(act.size)]]
    return out


def timing_job(name, epoch="maze", n_draws=100, conv="centred"):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    from cpca.replay import immobile_intervals
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_replay_hc11 import intersect, two_d
    from scripts.run_rerun import ORDER_INDEX, fixed_subset, run_cpca
    path = next(p for p in session_paths() if p.name.startswith(name))
    session = load_session(path)
    session["pos_xy"] = two_d(path)
    run = running(session)
    iv = intersect(immobile_intervals(session), [tuple(session["maze_epoch"])]) if epoch == "maze" else nrem_intervals(session, "post")
    by_cell, cell_ids, pooled = pyramidal_spikes(session, iv)
    bursts, peaks = population_bursts(pooled, iv)
    dur = np.diff(bursts, axis=1).ravel()
    sub = np.arange(len(peaks)) if epoch == "maze" else fixed_subset(len(peaks), 555 + 2 * ORDER_INDEX[name])
    times, _ = event_spike_times(by_cell, cell_ids, (peaks - dur / 2)[sub], (peaks + dur / 2)[sub], max_spikes=60)
    T = dur[sub] * 1000
    centres = place_centres(session, cell_ids, run)
    real = run_cpca(times, T, np.random.default_rng(seed("timing-real", name, epoch)))[conv]["orders"]
    real = best_map_match(real, centres, run["circular"], run["length"], np.random.default_rng(seed("timing-real-score", name, epoch))) if real \
        else {"abs": np.nan, "p": np.nan}
    rows = []
    for draw in range(n_draws):
        rng = np.random.default_rng(seed("maze-rest-timing", name, epoch, draw))
        o = run_cpca(shuffle_timing(times, rng), T, rng)[conv]["orders"]
        s = best_map_match(o, centres, run["circular"], run["length"], rng, n_null=300) if o else {"abs": np.nan, "p": np.nan}
        rows.append({"draw": draw, "abs": s["abs"], "p": s["p"], "k": len(o)})
    a = np.array([r["abs"] for r in rows], float)
    p_timing = float((1 + np.sum(np.nan_to_num(a, nan=-1) >= real["abs"])) / (1 + n_draws)) if np.isfinite(real["abs"]) else np.nan
    tag = epoch if conv == "centred" else f"{epoch}_{conv}"
    pickle.dump({"name": name, "epoch": epoch, "conv": conv, "real": real, "rows": rows, "p_timing": p_timing}, open(OUT / f"{name}_timing_{tag}.pkl", "wb"))
    return (f"real |ρ| {real['abs']:.2f} (p {real['p']:.3f}); timing shuffled: {sum(np.isfinite(r['p']) and r['p'] < 0.05 for r in rows)}/{n_draws} pass, "
            f"median |ρ| {np.nanmedian(a):.2f}; share of shuffles at least as high as real: {p_timing:.3f}")


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 8)
    only = [a for a in argv if not a.startswith("--")]
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    timing = "--timing" in argv
    epoch = next((a.split("=", 1)[1] for a in argv if a.startswith("--epoch=")), "maze")
    conv = "uncentred" if "--uncentred" in argv else "centred"
    suffix = (f"_timing_{epoch}" + ("" if conv == "centred" else f"_{conv}")) if timing else ""
    todo = [n for n in HC11_ORDER if (not only or any(n.startswith(o) for o in only)) and not (OUT / f"{n}{suffix}.pkl").exists()]
    print(f"{len(todo)} sessions on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {(pool.submit(timing_job, n, epoch, 100, conv) if timing else pool.submit(job, n)): n for n in todo}
        for f in as_completed(futures):
            try:
                print(f"{futures[f]}: {f.result()}", flush=True)
            except Exception:
                print(f"FAILED {futures[f]}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
