"""Experiment 5: Clean waking evaluation with exact circular time-shift null + Riemannian Complex ICA.

Tests ONE unified waking fit (uncentred cPCA + circular time-shift shuffle null + Riemannian
kurtosis Complex ICA) across ALL waking datasets and controls:
- Planted laps W1-W5 (dev seeds 20000, confirm seeds 21000)
- hc-11 linear & circular laps + circularly-shifted control
- hc-5 wheel (aligned & unaligned!) + shuffled control
- DANDI 001695 linear laps
- DANDI 000552 figure-eight (all 65 sessions: unsupervised Left/Right trial separation accuracy
  AND comparison vs supervised Oracle!)
"""

import sys
import time
import zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exploration.core import encode_arc, prepare_matrix, lag_matrix, eigh_desc, readout_gap, count_above_null
from exploration.exp4_unified_screen import complex_ica_fast
from sim.laps import generate as gen_laps
from scripts.run_separation import judge
from scripts.run_pure_capabilities import judge_share, halves_by, LAPS, HC11
from scripts.plot_hc11_sequences import best_map_match


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def shuffle_null_circ_time(times, T, rng, alpha=1.5 * np.pi, scale="none", n_shuffles=50):
    """Exact circular time-shift null: shift each cell's spikes in each event by U(0, T_i) mod T_i.

    Preserves each cell's spike count and burstiness in every event; destroys sequence order.
    """
    M, N = times.shape[:2]
    T_arr = np.asarray(T, dtype=float)
    if T_arr.ndim == 0:
        T_arr = np.full(M, float(T_arr))
    T3 = T_arr[:, None, None]
    null = np.empty((n_shuffles, N), dtype=float)
    for s in range(n_shuffles):
        shift = rng.uniform(0.0, 1.0, size=(M, N, 1)) * T3
        sh = np.mod(times + shift, T3)
        enc = encode_arc(sh, T_arr, alpha=alpha)
        X = prepare_matrix(enc, mode="uncentred", scale=scale)
        null[s] = eigh_desc(lag_matrix(X))[0]
    return null


def fit_waking_unified(times, T, rng, alpha=1.5 * np.pi, scale="none", power=2.0,
                       n_shuffles=50, member_rule="frac", member_frac=0.2, max_components=10):
    """Unified waking cPCA + exact circular time-shift null + Riemannian Complex ICA."""
    T_arr = np.asarray(T, dtype=float)
    if T_arr.ndim == 0:
        T_arr = np.full(times.shape[0], float(T_arr))
    enc = encode_arc(times, T_arr, alpha=alpha)
    X = prepare_matrix(enc, mode="uncentred", scale=scale)
    vals, vecs = eigh_desc(lag_matrix(X))
    null = shuffle_null_circ_time(times, T_arr, rng, alpha=alpha, scale=scale, n_shuffles=n_shuffles)
    k, bound = count_above_null(vals, null, structural=0, quantile=95.0)
    k = int(min(max(k, 0), max_components))
    if k == 0:
        return {"k": 0, "patterns": np.zeros((X.shape[1], 0), complex), "scores": np.zeros((X.shape[0], 0), complex), "orders": []}
    if k == 1:
        P = vecs[:, :1].copy()
        P /= np.maximum(np.linalg.norm(P, axis=0, keepdims=True), 1e-12)
        Y = X @ np.conj(P)
    else:
        Z = (X @ np.conj(vecs[:, :k])) / np.sqrt(np.maximum(vals[:k], 1e-12))[None, :]
        W = complex_ica_fast(Z, rng, power=power)
        P = vecs[:, :k] @ (np.sqrt(vals[:k])[:, None] * np.conj(W))
        norms = np.linalg.norm(P, axis=0)
        P = P / np.maximum(norms, 1e-12)
        order = np.argsort(-norms)
        P, Y = P[:, order], (Z @ W)[:, order]
    T_read = float(np.median(T_arr))
    orders = [readout_gap(P[:, j], T_read=T_read, alpha=alpha, member_rule=member_rule, member_frac=member_frac)[0]
              for j in range(P.shape[1])]
    return {"k": k, "patterns": P, "scores": Y, "orders": orders}


def run_all(alpha=1.5 * np.pi, scale="none", member_rule="frac", member_frac=0.2, power=2.0):
    from cpca import event_spike_times
    from cpca.hc11 import laps, load_session as load_hc11, place_centres, pyramidal_spikes, running, session_paths as hc11_paths
    import cpca.hc5 as hc5
    from scripts.run_hc5 import training_data
    from scripts.run_laps_hc11 import heldout_run
    import cpca.dandi001695 as dd
    import cpca.huszar as hz
    from experimental.scripts.run_initial_real import figure8_events
    from experimental.scripts.dev_unmixing import judge_partial

    cfg = dict(alpha=alpha, scale=scale, power=power, member_rule=member_rule, member_frac=member_frac)
    print(f"\n==================== WAKING UNIFIED: alpha={alpha/np.pi:.2f}*pi scale={scale} mem={member_rule}({member_frac}) ====================", flush=True)

    # 1. Planted laps (confirm seeds 21000)
    print("  [Planted Laps, confirm seeds 21000]", flush=True)
    for ci, (cname, kw) in enumerate(LAPS.items()):
        ks, clean_o, clean_s, clean_p, pass_s, tot = [], 0, 0, 0, 0, 0
        for draw in range(5):
            d = gen_laps(np.random.default_rng(21000 + 97 * ci + draw), **kw)
            half = halves_by(d["direction"], np.random.default_rng(seed("half", "confirm", cname, draw)))
            n_dir = kw.get("laps_per_direction", (30, 30))
            maps = [] if kw.get("shuffle") else [m for m, n in zip(d["maps"], n_dir) if n > 0]
            for fold, train in enumerate((half, ~half)):
                tr = np.flatnonzero(train)
                res = fit_waking_unified(d["times"][tr], d["T"][tr], np.random.default_rng(seed("w5", cname, draw, fold)), **cfg)
                ks.append(res["k"])
                jrng = np.random.default_rng(seed("w5-j", cname, draw, fold))
                if not maps:
                    tot += len(res["orders"])
                    if len(res["orders"]) == 0:
                        clean_o += 1; clean_s += 1; clean_p += 1
                    continue
                vo, vs, vp = [], [], []
                for o in res["orders"]:
                    vo.append(judge(o, maps, True, jrng)[:2])
                    vs.append(judge_share(o, maps, jrng))
                    vp.append(judge_partial(o, maps, jrng))
                pass_s += sum(ok for ok, _ in vs); tot += len(vs)
                if len({k for ok, k in vo if ok}) == len(maps) and all(ok for ok, _ in vo): clean_o += 1
                if len({k for ok, k in vs if ok}) == len(maps) and all(ok for ok, _ in vs): clean_s += 1
                if len({k for ok, k in vp if ok}) == len(maps) and all(ok for ok, _ in vp): clean_p += 1
        print(f"    {cname:<28} | k={np.median(ks):.1f} | share={pass_s:2d}/{tot:<3d} | clean orig={clean_o:2d}/10 share={clean_s:2d}/10 partial={clean_p:2d}/10", flush=True)

    # 2. hc-11 linear, circular, and circularly-shifted control
    print("  [hc-11 Laps + control]", flush=True)
    for kind in ("linear", "circular", "control"):
        clean_s, clean_p, pass_s, tot, n_fits = 0, 0, 0, 0, 0
        det = []
        for sname in HC11:
            s = load_hc11(next(p for p in hc11_paths() if p.name.startswith(sname)))
            run = running(s)
            if kind == "linear" and run["circular"]: continue
            if kind == "circular" and not run["circular"]: continue
            L = laps(run)
            starts, ends, dirs = (np.array([x[i] for x in L]) for i in range(3))
            half = halves_by(dirs, np.random.default_rng(seed("waking-sep", sname)))
            by_cell, ids, _ = pyramidal_spikes(s, np.column_stack([starts, ends]))
            for fold in (0, 1):
                train = half if fold == 0 else ~half
                tr, te = np.flatnonzero(train), np.flatnonzero(~train)
                X, _ = event_spike_times(by_cell, ids, starts[tr], ends[tr], max_spikes=400)
                TX = (ends[tr] - starts[tr]) * 1000
                if kind == "control":
                    crng = np.random.default_rng(seed("ctrl", sname, fold))
                    shift = crng.uniform(0, 1, X.shape[:2])[..., None] * TX[:, None, None]
                    X = np.sort(np.mod(X + shift, TX[:, None, None]), axis=-1)
                c = place_centres(s, ids, heldout_run(run, np.column_stack([starts[te], ends[te]])), min_spikes=20)
                maps = [] if kind == "control" else [c[sg] for sg in (1.0, -1.0) if np.isfinite(c[sg]).sum() >= 6 and np.sum(dirs[te] == sg) >= 3]
                res = fit_waking_unified(X, TX, np.random.default_rng(seed("w5-hc11", sname, fold)), **cfg)
                jrng = np.random.default_rng(seed("w5-hc11-j", sname, fold))
                n_fits += 1
                if not maps:
                    tot += len(res["orders"])
                    if len(res["orders"]) == 0: clean_s += 1; clean_p += 1
                    continue
                if run["circular"]:
                    vs = [(bool(np.sum(np.isfinite(o) & np.isfinite(maps[0])) >= 6 and
                                best_map_match([o], {1.0: maps[0]}, True, run["length"], jrng, n_null=500)["p"] < 0.05), 0)
                          for o in res["orders"]]
                    vp = vs
                else:
                    vs = [judge_share(o, maps, jrng) for o in res["orders"]]
                    vp = [judge_partial(o, maps, jrng) for o in res["orders"]]
                pass_s += sum(ok for ok, _ in vs); tot += len(vs)
                if len({k for ok, k in vs if ok}) == len(maps) and all(ok for ok, _ in vs): clean_s += 1
                if len({k for ok, k in vp if ok}) == len(maps) and all(ok for ok, _ in vp): clean_p += 1
                if kind == "linear":
                    det.append(f"{sname[:7]}f{fold}:k={res['k']}:{[k if ok else 'X' for ok, k in vp]}")
        print(f"    hc-11 {kind:<10} | pass={pass_s:2d}/{tot:<3d} | clean share={clean_s:2d}/{n_fits} partial={clean_p:2d}/{n_fits} | {' '.join(det)}", flush=True)

    # 3. hc-5 wheel (aligned, unaligned, shuffled control)
    print("  [hc-5 Wheel + shuffled control]", flush=True)
    for cond in ("aligned", "unaligned", "shuffled"):
        clean_s, pass_s, tot, n_fits = 0, 0, 0, 0
        for sname in hc5.session_names():
            sess = hc5.load_session(sname)
            for fold in (0, 1):
                X, T, train, test, _ = training_data(sess, cond, 0, fold)
                TX = np.full(X.shape[0], T) if np.ndim(T) == 0 else np.asarray(T)
                f = hc5.time_fields(sess, test)
                maps = [] if cond == "shuffled" else ([f] if np.isfinite(f).sum() >= 6 else [])
                res = fit_waking_unified(X, TX, np.random.default_rng(seed("w5-hc5", cond, sname, fold)), **cfg)
                jrng = np.random.default_rng(seed("w5-hc5-j", cond, sname, fold))
                n_fits += 1
                if not maps:
                    tot += len(res["orders"])
                    if len(res["orders"]) == 0: clean_s += 1
                    continue
                vs = [judge_share(o, maps, jrng) for o in res["orders"]]
                pass_s += sum(ok for ok, _ in vs); tot += len(vs)
                if len({k for ok, k in vs if ok}) == len(maps) and all(ok for ok, _ in vs): clean_s += 1
        print(f"    hc-5 {cond:<10}  | pass={pass_s:2d}/{tot:<3d} | clean={clean_s:2d}/{n_fits}", flush=True)

    # 4. DANDI 001695 linear laps
    clean_s, clean_p, pass_s, pass_p, tot, n_fits = 0, 0, 0, 0, 0, 0
    for path in dd.session_paths():
        s = dd.load_session(path); run = running(s); L = laps(run)
        starts, ends, dirs = (np.array([x[i] for x in L]) for i in range(3))
        half = halves_by(dirs, np.random.default_rng(seed("dandi-laps-half", path.name)))
        by_cell, ids, _ = pyramidal_spikes(s, np.column_stack([starts, ends]))
        for fold in (0, 1):
            train = half if fold == 0 else ~half
            tr, te = np.flatnonzero(train), np.flatnonzero(~train)
            X, _ = event_spike_times(by_cell, ids, starts[tr], ends[tr], max_spikes=400)
            TX = (ends[tr] - starts[tr]) * 1000
            c = place_centres(s, ids, heldout_run(run, np.column_stack([starts[te], ends[te]])), min_spikes=20)
            maps = [c[sg] for sg in (1.0, -1.0) if np.isfinite(c[sg]).sum() >= 6 and np.sum(dirs[te] == sg) >= 3]
            res = fit_waking_unified(X, TX, np.random.default_rng(seed("w5-dd", path.name, fold)), **cfg)
            jrng = np.random.default_rng(seed("w5-dd-j", path.name, fold))
            vs = [judge_share(o, maps, jrng) for o in res["orders"]]
            vp = [judge_partial(o, maps, jrng) for o in res["orders"]]
            pass_s += sum(ok for ok, _ in vs); pass_p += sum(ok for ok, _ in vp); tot += len(vs); n_fits += 1
            if maps and len({k for ok, k in vs if ok}) == len(maps) and all(ok for ok, _ in vs): clean_s += 1
            if maps and len({k for ok, k in vp if ok}) == len(maps) and all(ok for ok, _ in vp): clean_p += 1
    print(f"    001695 laps      | pass share={pass_s:2d}/{tot} partial={pass_p:2d}/{tot} | clean share={clean_s:2d}/{n_fits} partial={clean_p:2d}/{n_fits}", flush=True)

    # 5. DANDI 000552 figure-eight: unsupervised trial classification accuracy (Left vs Right) on all 65 sessions!
    accs, k2_count, n_f8 = [], 0, 0
    for path in hz.session_paths():
        s = hz.load_session(path)
        task_w = hz.task_window(s)[1]
        trials = hz.task_trials(s, task_w)
        arms = s["arm"][trials]
        cells = hz.putative_pyramidal(s)
        times, T = figure8_events(s, trials, cells)
        if (arms == "left").sum() < 2 or (arms == "right").sum() < 2:
            continue
        res = fit_waking_unified(times, T, np.random.default_rng(seed("w5-f8", s["name"])), **cfg)
        n_f8 += 1
        if res["k"] >= 2:
            k2_count += 1
            # Unsupervised classification of left vs right trials by top-2 ICs
            Y = np.abs(res["scores"][:, :2]) ** 2
            pred = (Y[:, 0] >= Y[:, 1])
            y_true = (arms == "left")
            acc = max(np.mean(pred == y_true), np.mean(pred != y_true))
            accs.append(acc)
    print(f"    000552 fig8 (all {n_f8} sess) | k>=2 in {k2_count}/{n_f8} | unsupervised L/R trial accuracy: median={np.median(accs):.1%}, mean={np.mean(accs):.1%}, >=90% in {sum(a>=0.9 for a in accs)}/{len(accs)}", flush=True)


if __name__ == "__main__":
    t0 = time.perf_counter()
    run_all(alpha=1.5 * np.pi, scale="none", member_rule="frac", member_frac=0.2, power=2.0)
    run_all(alpha=np.pi, scale="none", member_rule="frac", member_frac=0.2, power=2.0)
    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s", flush=True)
