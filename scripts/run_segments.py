"""Gate G1 with the direction oracle (docs/segments.md): constant-speed partial replays in both directions, swept over
replay share, cells per replay and direction balance; complex PCA against label-free candidates and two oracles.

Per draw the outputs scored are: complex PCA centred and uncentred (all kept components), the lead-lag component
(leading eigenvector of j*Im of the centred, scaled lag matrix), the direction EM seeded from the kept centred components
(docs/segments.md, development check 5), the direction oracle (aligned uncentred leading eigenvector over all events,
planted directions given, random signs for non-replays) and the event-and-direction oracle (replay events only).
Scores: whole-track order test and fragment bar 1 against the planted positions; timing control (30 shuffled refits) for
complex PCA centred and the lead-lag component.

Usage: python scripts/run_segments.py [--workers=9]
Output: results/segments/<cells>_<share>_<reverse>_<draw>.pkl, results/segments/control_<name>_<draw>.pkl
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

OUT = ROOT / "results" / "segments"
CELLS = (5, 9, 13)
SHARES = (0.0, 0.02, 0.05, 0.10)
REVERSE = (0.5, 0.3)
N_DRAWS = 5
N_TIMING = 30
N_NULL = 500
CONTROLS = {"assemblies_no_order": dict(replay_frac=0.0, spatial=True), "forward_only_5pct": dict(replay_frac=0.05, reverse_frac=0.0)}


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


# ----------------------------------------------------------------------------- candidates
def leadlag_orders(prepared, n_out=2):
    """Orders from the eigenvectors of j*Im(C) with the largest positive eigenvalues (C: lag matrix of the prepared arrows)."""
    from cpca.direction import order_of
    from cpca.theory import eigh_desc, lag_matrix
    J = 1j * np.imag(lag_matrix(prepared))
    w, v = eigh_desc(J)
    return [order_of(v[:, j], member_frac=0.2) for j in range(min(n_out, (w > 0).sum()))], w[:n_out]


def direction_em(S, q0, n_iter=50):
    """Alternate per-event labels (forward if |<s,q>| >= |<s,conj q>|) and the aligned leading eigenvector; all events."""
    from cpca.theory import eigh_desc, lag_matrix
    q = q0 / max(np.linalg.norm(q0), 1e-12)
    labels = None
    for it in range(n_iter):
        new = np.abs(S @ np.conj(q)) >= np.abs(S @ q)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        w, v = eigh_desc(lag_matrix(np.where(labels[:, None], S, np.conj(S))))
        q = v[:, 0]
    w, v = eigh_desc(lag_matrix(np.where(labels[:, None], S, np.conj(S))))
    return q, labels, float(w[0])


def aligned_leading(S, forward):
    from cpca.theory import eigh_desc, lag_matrix
    return eigh_desc(lag_matrix(np.where(forward[:, None], S, np.conj(S))))[1][:, 0]


def best_abs(orders, key):
    """Largest |Spearman| over outputs between member times and the key (no null), for the timing control."""
    from scipy.stats import spearmanr
    best = 0.0
    for o in orders:
        o = np.asarray(o, float)
        use = np.isfinite(o) & np.isfinite(key)
        if use.sum() >= 6:
            r = abs(float(spearmanr(o[use], key[use]).statistic))
            best = max(best, r if np.isfinite(r) else 0.0)
    return best


def analyse(data, rng, timing=True):
    from cpca import encode, fit, prepare
    from cpca.direction import order_of
    from cpca.fragment import best_fragment
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_maze_rest import shuffle_timing
    from scripts.run_rerun import run_cpca
    times, T, pos, label, direction = data["times"], data["T"], data["centres"], data["label"], data["direction"]
    key = {1.0: pos * 100}
    rep = label > 0
    S = encode(times, T)
    out = {"n_active_median": float(np.median(np.isfinite(times).any(-1).sum(1))),
           "cells_per_replay": float(np.mean(np.isfinite(times[rep][:, np.isfinite(pos)]).any(-1).sum(1))) if rep.any() else 0.0,
           "ordered_per_replay": float(np.mean(data["n_ordered"][rep])) if rep.any() else 0.0, "n_replays": int(rep.sum())}
    outputs = {}
    cp = run_cpca(times, T, rng)
    outputs["cpca_centred"] = cp["centred"]["orders"]
    outputs["cpca_uncentred"] = cp["uncentred"]["orders"]
    res = fit(times, T, rng=rng, ica=False)
    outputs["leadlag"], out["leadlag_eigenvalues"] = leadlag_orders(res["prepared"])
    # direction EM from the kept centred components, best objective
    em = []
    for j in range(res["n_kept"]):
        q, labels, obj = direction_em(S, res["patterns"][:, j])
        em.append((obj, q, labels))
    if em:
        obj, q, labels = max(em, key=lambda e: e[0])
        outputs["em"] = [order_of(q, member_frac=0.2)]
        acc = float(np.mean(np.where(labels[rep], 1, -1) == direction[rep])) if rep.any() else np.nan
        out["em_label_accuracy"] = max(acc, 1 - acc) if np.isfinite(acc) else np.nan
        out["em_forward_share"] = float(labels.mean())
    else:
        outputs["em"] = []
        out["em_label_accuracy"] = np.nan
    # oracles
    signs = np.where(direction != 0, direction, rng.choice([-1, 1], size=direction.size))
    q_dir = aligned_leading(S, signs > 0)
    outputs["oracle_direction"] = [order_of(q_dir, member_frac=0.2)]
    if rep.sum() >= 10:
        outputs["oracle_events"] = [order_of(aligned_leading(S[rep], direction[rep] > 0), member_frac=0.2)]
        F, B = np.abs(S[rep] @ np.conj(q_dir)), np.abs(S[rep] @ q_dir)
        acc = float(np.mean(np.where(F >= B, 1, -1) == direction[rep]))
        out["oracle_template_label_accuracy"] = max(acc, 1 - acc)
    else:
        outputs["oracle_events"] = []
        out["oracle_template_label_accuracy"] = np.nan
    out["n_outputs"] = {k: len(v) for k, v in outputs.items()}
    out["whole"] = {k: best_map_match(v, key, False, 100.0, rng, n_null=N_NULL) if v else {"abs": np.nan, "p": np.nan} for k, v in outputs.items()}
    out["fragment"] = {k: best_fragment(v, key, 100.0, rng, n_null=N_NULL) if v else {"abs": np.nan, "p": np.nan} for k, v in outputs.items()}
    if timing:
        shuffled = {"cpca_centred": [], "leadlag": []}
        for s in range(N_TIMING):
            r2 = np.random.default_rng(seed("segments-timing", rng.integers(1 << 30), s))
            ts = shuffle_timing(times, r2)
            rs = fit(ts, T, rng=r2, ica=False, center=True, scale=True)            # the centred refit only (run_cpca's read-out)
            cps = [order_of(rs["patterns"][:, j], member_frac=0.2) for j in range(rs["n_kept"])]
            shuffled["cpca_centred"].append(best_abs(cps, pos) if cps else 0.0)
            lls, _ = leadlag_orders(prepare(encode(ts, T), center=True, scale=True))
            shuffled["leadlag"].append(best_abs(lls, pos) if lls else 0.0)
        out["timing"] = {}
        for k, arr in shuffled.items():
            real = out["whole"][k]["abs"]
            arr = np.asarray(arr)
            out["timing"][k] = {"shuffled_abs": arr, "p": float((1 + np.sum(arr >= real)) / (1 + arr.size)) if np.isfinite(real) else np.nan}
    return out


# ----------------------------------------------------------------------------- data (same random stream as the stored results)
def grid_data(cells, share, reverse, draw):
    """The planted data of one grid cell and the generator continued past it (as the stored runs used it)."""
    from sim.segments import generate, n_route_for
    rng = np.random.default_rng(seed("segments", cells, share, reverse, draw))
    return generate(rng, n_route=n_route_for(cells), replay_frac=share, reverse_frac=reverse), rng


def control_data(name, draw):
    from sim.segments import generate, n_route_for, spatial_assemblies
    cfg = dict(CONTROLS[name])
    spatial = cfg.pop("spatial", False)
    n_route = n_route_for(9)
    s = seed("segments-control", name, draw)
    if spatial:
        # the route and its positions are the first draws of generate(), so a second call with the same seed keeps them
        probe = generate(np.random.default_rng(s), n_route=n_route, n_events=1, replay_frac=0.0)
        membership = spatial_assemblies(np.random.default_rng(s + 1), probe["n_cells"], probe["route"], probe["centres"])
        data = generate(np.random.default_rng(s), n_route=n_route, assembly_membership=membership, **cfg)
        assert np.array_equal(data["route"], probe["route"])
    else:
        data = generate(np.random.default_rng(s), n_route=n_route, **cfg)
    return data, np.random.default_rng(s + 2)


# ----------------------------------------------------------------------------- jobs
def grid_job(cells, share, reverse, draw):
    from sim.segments import n_route_for
    data, rng = grid_data(cells, share, reverse, draw)
    out = analyse(data, rng)
    out.update(cells=cells, share=share, reverse=reverse, draw=draw, n_route=n_route_for(cells))
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT / f"{cells}_{share:.2f}_{reverse:.1f}_{draw}.pkl", "wb"))
    return out


def control_job(name, draw):
    from sim.segments import n_route_for
    n_route = n_route_for(9)
    data, rng = control_data(name, draw)
    out = analyse(data, rng)
    out.update(control=name, draw=draw, n_route=n_route)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT / f"control_{name}_{draw}.pkl", "wb"))
    return out


def fmt(out):
    def p(k):
        v = out["whole"].get(k, {}).get("p", np.nan)
        return f"{v:.3f}" if np.isfinite(v) else "  –  "
    return (f"active {out['n_active_median']:.0f}, ordered/replay {out['ordered_per_replay']:.1f} | whole-track p: cPCA {p('cpca_centred')} unc {p('cpca_uncentred')} "
            f"leadlag {p('leadlag')} em {p('em')} | oracle dir {p('oracle_direction')} events {p('oracle_events')} | "
            f"timing p cPCA {out.get('timing', {}).get('cpca_centred', {}).get('p', np.nan):.3f} leadlag {out.get('timing', {}).get('leadlag', {}).get('p', np.nan):.3f} | "
            f"labels em {out['em_label_accuracy']:.2f} oracle {out['oracle_template_label_accuracy']:.2f}")


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    jobs = []
    for cells in CELLS:
        for share in SHARES:
            for reverse in (REVERSE if share > 0 else (0.5,)):
                for draw in range(N_DRAWS):
                    if not (OUT / f"{cells}_{share:.2f}_{reverse:.1f}_{draw}.pkl").exists():
                        jobs.append((grid_job, (cells, share, reverse, draw)))
    for name in CONTROLS:
        for draw in range(N_DRAWS):
            if not (OUT / f"control_{name}_{draw}.pkl").exists():
                jobs.append((control_job, (name, draw)))
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *a): a for fn, a in jobs}
        for f in as_completed(futures):
            try:
                print(f"{futures[f]}: {fmt(f.result())}", flush=True)
            except Exception:
                print(f"FAILED {futures[f]}:\n{traceback.format_exc()}", flush=True)
    print(f"done in {(time.perf_counter() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
