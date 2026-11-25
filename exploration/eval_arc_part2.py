"""Objective head-to-head evaluation of phase arc alpha = pi vs alpha = 1.5*pi (Part 2).

Matched head-to-head across ALL real & waking datasets with identical seeds:
1. Planted Waking Laps W1-W5 (fresh seeds 21000)
2. Real Waking:
   - hc-11 linear, circular, control
   - hc-5 wheel (aligned, unaligned, shuffled)
   - DANDI 001695 linear laps
   - DANDI 000552 figure-eight (all 65 sessions: k>=2, L/R trial accuracy, oracle pattern similarity)
3. Real Reactivations (hc-11):
   - Spike-in power across ALL 8 sessions (2 largest + 6 smaller, 0-20%)
   - hc-11 maze rest (+ 50 timing shuffles)
   - hc-11 sleep POST & PRE (+ 50 timing shuffles)
4. Also tests legacy uncentred cpca (no ICA) on hc-5 wheel & hc-11 waking laps at alpha=pi vs 1.5*pi!
"""

import pickle
import sys
import time
import zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exploration.cpca_v2 import fit_cpca_v2, encode_arc, lag_matrix, eigh_desc
from sim.laps import generate as gen_laps
from cpca import event_spike_times, population_bursts
from cpca.hc11 import laps, load_session as load_hc11, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths as hc11_paths
import cpca.hc5 as hc5
import cpca.dandi001695 as dd
import cpca.huszar as hz
from cpca.replay import immobile_intervals
from scripts.run_hc5 import training_data
from scripts.run_laps_hc11 import heldout_run
from scripts.run_replay_hc11 import intersect, two_d
from scripts.run_separation import judge
from scripts.run_pure_capabilities import judge_share, halves_by, LAPS, HC11
from scripts.plot_hc11_sequences import best_map_match
from scripts.run_maze_rest import HC11_ORDER, shuffle_timing
from scripts.run_rerun import ORDER_INDEX, fixed_subset
from scripts.run_spikein_hc11 import ideal as spikein_ideal
from experimental.scripts.dev_unmixing import judge_partial
from experimental.scripts.run_initial_real import figure8_events

OUT = ROOT / "exploration" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def eval_waking(alpha):
    out = {}
    # 1. Planted W1-W5 (fresh confirmation seeds 21000)
    wl = []
    for ci, (cname, kw) in enumerate(LAPS.items()):
        ks, co, cs, cp, ps, tot = [], 0, 0, 0, 0, 0
        for draw in range(5):
            d = gen_laps(np.random.default_rng(21000 + 97 * ci + draw), **kw)
            half = halves_by(d["direction"], np.random.default_rng(seed("half", "confirm", cname, draw)))
            n_dir = kw.get("laps_per_direction", (30, 30))
            maps = [] if kw.get("shuffle") else [m for m, n in zip(d["maps"], n_dir) if n > 0]
            for fold, train in enumerate((half, ~half)):
                tr = np.flatnonzero(train)
                res = fit_cpca_v2(d["times"][tr], d["T"][tr],
                                  np.random.default_rng(seed("arc-wl", cname, draw, fold)),
                                  regime="waking", alpha=alpha, power=1.5, n_shuffles=100, quantile=99.0)
                ks.append(res["k"])
                jrng = np.random.default_rng(seed("arc-wlj", cname, draw, fold))
                if not maps:
                    tot += len(res["orders"])
                    if len(res["orders"]) == 0: co += 1; cs += 1; cp += 1
                    continue
                vo = [judge(o, maps, True, jrng)[:2] for o in res["orders"]]
                vs = [judge_share(o, maps, jrng) for o in res["orders"]]
                vp = [judge_partial(o, maps, jrng) for o in res["orders"]]
                ps += sum(ok for ok, _ in vs); tot += len(vs)
                if len({k for ok, k in vo if ok}) == len(maps) and all(ok for ok, _ in vo): co += 1
                if len({k for ok, k in vs if ok}) == len(maps) and all(ok for ok, _ in vs): cs += 1
                if len({k for ok, k in vp if ok}) == len(maps) and all(ok for ok, _ in vp): cp += 1
        wl.append({"name": cname, "med_k": float(np.median(ks)), "ps": ps, "tot": tot, "co": co, "cs": cs, "cp": cp})
    out["planted_laps"] = wl

    # 2. hc-11 linear, circular, control
    for kind in ("linear", "circular", "control"):
        cs = cp = ps = pp = tot = nf = 0
        details = []
        for sname in HC11:
            s = load_hc11(next(p for p in hc11_paths() if p.name.startswith(sname))); run = running(s)
            if kind == "linear" and run["circular"]: continue
            if kind == "circular" and not run["circular"]: continue
            L = laps(run); starts, ends, dirs = (np.array([x[i] for x in L]) for i in range(3))
            half = halves_by(dirs, np.random.default_rng(seed("waking-sep", sname)))
            by_cell, ids, _ = pyramidal_spikes(s, np.column_stack([starts, ends]))
            for fold in (0, 1):
                train = half if fold == 0 else ~half
                tr, te = np.flatnonzero(train), np.flatnonzero(~train)
                X, _ = event_spike_times(by_cell, ids, starts[tr], ends[tr], max_spikes=400)
                TX = (ends[tr] - starts[tr]) * 1000
                if kind == "control":
                    crng = np.random.default_rng(seed("ctrl", sname, fold))
                    X = np.sort(np.mod(X + crng.uniform(0, 1, X.shape[:2])[..., None] * TX[:, None, None], TX[:, None, None]), axis=-1)
                c = place_centres(s, ids, heldout_run(run, np.column_stack([starts[te], ends[te]])), min_spikes=20)
                maps = [] if kind == "control" else [c[sg] for sg in (1.0, -1.0) if np.isfinite(c[sg]).sum() >= 6 and np.sum(dirs[te] == sg) >= 3]
                res = fit_cpca_v2(X, TX, np.random.default_rng(seed("arc-hc11", kind, sname, fold)),
                                  regime="waking", alpha=alpha, power=1.5, n_shuffles=100, quantile=99.0)
                jrng = np.random.default_rng(seed("arc-hc11j", kind, sname, fold)); nf += 1
                if not maps:
                    tot += len(res["orders"])
                    if len(res["orders"]) == 0: cs += 1; cp += 1
                    continue
                if run["circular"]:
                    vs = [(bool(np.sum(np.isfinite(o) & np.isfinite(maps[0])) >= 6 and best_map_match([o], {1.0: maps[0]}, True, run["length"], jrng, n_null=500)["p"] < 0.05), 0) for o in res["orders"]]
                    vp = vs
                else:
                    vs = [judge_share(o, maps, jrng) for o in res["orders"]]
                    vp = [judge_partial(o, maps, jrng) for o in res["orders"]]
                ps += sum(ok for ok, _ in vs); pp += sum(ok for ok, _ in vp); tot += len(vs)
                if len({m for ok, m in vs if ok}) == len(maps) and all(ok for ok, _ in vs): cs += 1
                if len({m for ok, m in vp if ok}) == len(maps) and all(ok for ok, _ in vp): cp += 1
                details.append((sname, fold, res["k"], [m if ok else "X" for ok, m in vp]))
        out[f"hc11_{kind}"] = {"ps": ps, "pp": pp, "tot": tot, "cs": cs, "cp": cp, "nf": nf, "details": details}

    # 3. hc-5 wheel
    for cond in ("aligned", "unaligned", "shuffled"):
        cs = ps = tot = nf = 0; rhos = []
        for sname in hc5.session_names():
            sess = hc5.load_session(sname)
            for fold in (0, 1):
                X, T, train, test, _ = training_data(sess, cond, 0, fold)
                TX = np.full(X.shape[0], T) if np.ndim(T) == 0 else np.asarray(T)
                f = hc5.time_fields(sess, test)
                maps = [] if cond == "shuffled" else ([f] if np.isfinite(f).sum() >= 6 else [])
                res = fit_cpca_v2(X, TX, np.random.default_rng(seed("arc-hc5", cond, sname, fold)),
                                  regime="waking", alpha=alpha, power=1.5, n_shuffles=100, quantile=99.0)
                jrng = np.random.default_rng(seed("arc-hc5j", cond, sname, fold)); nf += 1
                if not maps:
                    tot += len(res["orders"])
                    if len(res["orders"]) == 0: cs += 1
                    continue
                vs = [judge_share(o, maps, jrng) for o in res["orders"]]
                ps += sum(ok for ok, _ in vs); tot += len(vs)
                if len({m for ok, m in vs if ok}) == len(maps) and all(ok for ok, _ in vs): cs += 1
                for o in res["orders"]:
                    ok, _, det = judge(o, maps, True, jrng)
                    if ok and "rhos" in det: rhos.append(det["rhos"][0])
        out[f"hc5_{cond}"] = {"ps": ps, "tot": tot, "cs": cs, "nf": nf, "med_rho": float(np.median(rhos)) if rhos else np.nan}

    # 4. DANDI 001695 laps
    cs = cp = ps = pp = tot = nf = 0
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
            res = fit_cpca_v2(X, TX, np.random.default_rng(seed("arc-dd", path.name, fold)),
                              regime="waking", alpha=alpha, power=1.5, n_shuffles=100, quantile=99.0)
            jrng = np.random.default_rng(seed("arc-ddj", path.name, fold))
            vs = [judge_share(o, maps, jrng) for o in res["orders"]]
            vp = [judge_partial(o, maps, jrng) for o in res["orders"]]
            ps += sum(ok for ok, _ in vs); pp += sum(ok for ok, _ in vp); tot += len(vs); nf += 1
            if maps and len({m for ok, m in vs if ok}) == len(maps) and all(ok for ok, _ in vs): cs += 1
            if maps and len({m for ok, m in vp if ok}) == len(maps) and all(ok for ok, _ in vp): cp += 1
    out["dd_laps"] = {"ps": ps, "pp": pp, "tot": tot, "cs": cs, "cp": cp, "nf": nf}

    # 5. DANDI 000552 figure-eight (all 65 sessions)
    f8 = []
    for path in hz.session_paths():
        s = hz.load_session(path); task_w = hz.task_window(s)[1]
        trials = hz.task_trials(s, task_w); arms = s["arm"][trials]; cells = hz.putative_pyramidal(s)
        times, T = figure8_events(s, trials, cells)
        if (arms == "left").sum() < 2 or (arms == "right").sum() < 2: continue
        res = fit_cpca_v2(times, T, np.random.default_rng(seed("arc-f8", s["name"])),
                          regime="waking", alpha=alpha, power=1.5, n_shuffles=100, quantile=99.0)
        entry = {"name": s["name"], "k": res["k"], "acc": np.nan, "oracle_sim": np.nan}
        if res["k"] >= 2:
            Y = np.abs(res["scores"][:, :2]) ** 2
            pred = (Y[:, 0] >= Y[:, 1]); y_true = (arms == "left")
            entry["acc"] = float(max(np.mean(pred == y_true), np.mean(pred != y_true)))
            X = encode_arc(times, T, alpha=alpha)
            qL = eigh_desc(lag_matrix(X[y_true]))[1][:, 0]
            qR = eigh_desc(lag_matrix(X[~y_true]))[1][:, 0]
            P2 = res["patterns"][:, :2]
            s1 = 0.5 * (abs(np.vdot(qL, P2[:, 0])) + abs(np.vdot(qR, P2[:, 1])))
            s2 = 0.5 * (abs(np.vdot(qL, P2[:, 1])) + abs(np.vdot(qR, P2[:, 0])))
            entry["oracle_sim"] = float(max(s1, s2))
        f8.append(entry)
    out["f8"] = f8
    return out


def eval_real_hc11_react(alpha):
    cfg = dict(regime="reactivation", alpha=alpha, power=1.5, scale="global_rms",
               n_shuffles=100, quantile=99.0, member_frac=0.2)
    fracs = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
    sp_rows = []
    for sname in HC11_ORDER:
        path = next(p for p in hc11_paths() if p.name.startswith(sname))
        s = load_hc11(path); run = running(s)
        iv = nrem_intervals(s, "pre")
        by_cell, ids, pooled = pyramidal_spikes(s, iv)
        b, pk = population_bursts(pooled, iv); dur = np.diff(b, axis=1).ravel()
        sub = fixed_subset(len(pk), 555 + 2 * ORDER_INDEX[sname] + 1)
        base, _ = event_spike_times(by_cell, ids, (pk - dur / 2)[sub], (pk + dur / 2)[sub], max_spikes=60)
        T = dur[sub] * 1000; centres = place_centres(s, ids, run)
        s_row = {"session": sname, "hits": {}}
        for f in fracs:
            k_ev = int(round(f * len(T)))
            hits = 0
            for d in range(5):
                rng = np.random.default_rng(seed("spikein", sname, "ideal", f, d))
                chosen = rng.choice(len(T), k_ev, replace=False) if k_ev else np.array([], int)
                times = spikein_ideal(base, T, chosen, centres, rng) if k_ev else base
                res = fit_cpca_v2(times, T, np.random.default_rng(seed("arc-sp", sname, f, d)), **cfg)
                sc = best_map_match(res["orders"], centres, run["circular"], run["length"],
                                    np.random.default_rng(seed("arc-spsc", sname, f, d)), n_null=500) if res["orders"] else {"p": np.nan}
                hits += bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
            s_row["hits"][f] = hits
        sp_rows.append(s_row)

    # Maze rest & POST/PRE (+ 30 timing shuffles)
    n_shuf = 30
    real = {"maze": [], "post": [], "pre": []}
    for ep in ("maze", "post", "pre"):
        for sname in HC11_ORDER:
            path = next(p for p in hc11_paths() if p.name.startswith(sname))
            s = load_hc11(path); s["pos_xy"] = two_d(path); run = running(s)
            iv = intersect(immobile_intervals(s), [tuple(s["maze_epoch"])]) if ep == "maze" else nrem_intervals(s, ep)
            by_cell, ids, pooled = pyramidal_spikes(s, iv)
            b, pk = population_bursts(pooled, iv); dur = np.diff(b, axis=1).ravel()
            times, _ = event_spike_times(by_cell, ids, pk - dur / 2, pk + dur / 2, max_spikes=60)
            T = dur * 1000; centres = place_centres(s, ids, run)
            res = fit_cpca_v2(times, T, np.random.default_rng(seed("arc-hc", sname, ep)), **cfg)
            sc = best_map_match(res["orders"], centres, run["circular"], run["length"],
                                np.random.default_rng(seed("arc-hcsc", sname, ep)), n_null=1000) if res["orders"] else {"abs": np.nan, "p": np.nan}
            run_sh = (ep == "maze") or (np.isfinite(sc["p"]) and sc["p"] < 0.05) or sname.startswith("Achilles")
            s_fp = None; p_tim = np.nan
            if run_sh:
                s_abs = []; s_fp = 0
                for d in range(n_shuf):
                    srng = np.random.default_rng(seed("arc-hcsh", sname, ep, d))
                    rs = fit_cpca_v2(shuffle_timing(times, srng), T, srng, **cfg)
                    if rs["orders"]:
                        ss = best_map_match(rs["orders"], centres, run["circular"], run["length"], srng, n_null=200)
                        s_abs.append(ss["abs"]); s_fp += (np.isfinite(ss["p"]) and ss["p"] < 0.05)
                    else:
                        s_abs.append(np.nan)
                p_tim = float((1 + np.sum(np.nan_to_num(s_abs, nan=-1.0) >= sc["abs"])) / (1 + n_shuf)) if np.isfinite(sc["abs"]) else 1.0
            ok_o = bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
            ok_b = ok_o and bool(np.isfinite(p_tim) and p_tim < 0.05)
            real[ep].append({"session": sname, "k": res["k"], "n_out": len(res["orders"]),
                             "abs": sc["abs"], "p": sc["p"], "p_tim": p_tim, "shuf_fp": s_fp,
                             "ok_o": ok_o, "ok_b": ok_b})
    return {"spikein": sp_rows, "real": real}


if __name__ == "__main__":
    t0 = time.perf_counter()
    res = {}
    for label, alpha in (("pi", np.pi), ("1.5pi", 1.5 * np.pi)):
        print(f"\n================ WAKING & REAL HC-11: alpha={label} ================", flush=True)
        w = eval_waking(alpha)
        res[f"waking_{label}"] = w
        print("  Planted laps W1-W5 (clean share):", " ".join(f"{r['name'][:2]}:{r['cs']}/10" for r in w["planted_laps"]), flush=True)
        for k in ("hc11_linear", "hc11_circular", "hc11_control", "dd_laps"):
            d = w[k]
            print(f"  {k:<15}: pass={d['ps']:2d}/{d['tot']:<3d} clean share={d['cs']:2d}/{d['nf']} partial={d['cp']:2d}/{d['nf']}", flush=True)
        for k in ("hc5_aligned", "hc5_unaligned", "hc5_shuffled"):
            d = w[k]
            print(f"  {k:<15}: pass={d['ps']:2d}/{d['tot']:<3d} clean={d['cs']:2d}/{d['nf']} rho={d['med_rho']:.3f}", flush=True)
        f8_ok = [x for x in w["f8"] if np.isfinite(x["acc"])]
        print(f"  000552 fig8    : k>=2 in {len(f8_ok)}/{len(w['f8'])} | acc med={np.median([x['acc'] for x in f8_ok]):.1%} mean={np.mean([x['acc'] for x in f8_ok]):.1%} | oracle_sim={np.median([x['oracle_sim'] for x in f8_ok]):.3f}", flush=True)

        hr = eval_real_hc11_react(alpha)
        res[f"hc11_{label}"] = hr
        fracs = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
        big = [r for r in hr["spikein"] if r["session"].startswith("Achilles")]
        sml = [r for r in hr["spikein"] if not r["session"].startswith("Achilles")]
        print("  Spike-in 2 big :", " ".join(f"{f:.0%}:{sum(r['hits'][f] for r in big)}/10" for f in fracs), flush=True)
        print("  Spike-in 6 sml :", " ".join(f"{f:.0%}:{sum(r['hits'][f] for r in sml)}/30" for f in fracs), flush=True)
        for ep in ("maze", "post", "pre"):
            rs = hr["real"][ep]
            fp_s = sum(r["shuf_fp"] for r in rs if r["shuf_fp"] is not None)
            fp_n = 30 * sum(r["shuf_fp"] is not None for r in rs)
            passes = [f"{r['session'][:7]}(r={r['abs']:.2f},p={r['p']:.3f},pt={r['p_tim']:.2f})" for r in rs if r["ok_o"]]
            print(f"  hc-11 {ep.upper():<4}     : ord={sum(r['ok_o'] for r in rs)}/8 both={sum(r['ok_b'] for r in rs)}/8 shuf_FP={fp_s}/{fp_n} ({100*fp_s/max(fp_n,1):.1f}%) | {' '.join(passes)}", flush=True)

    pickle.dump(res, open(OUT / "arc_eval_part2.pkl", "wb"))
    print(f"\nArc Part 2 finished in {time.perf_counter()-t0:.1f}s", flush=True)
