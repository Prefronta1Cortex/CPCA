"""Confirmation Part 1: Fresh-seed planted reactivations (A-F, Stress G-M, spatial assemblies),
fresh-seed planted waking laps (W1-W5), ALL real waking datasets (hc-11, hc-5, 001695, 000552),
and pillar ablation study.

Locked parameters (exploration/cpca_v2.py):
  alpha = 1.5*pi, power = 1.5, n_shuffles = 100, quantile = 99.0, member_frac = 0.2.
"""

import os
import pickle
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exploration.cpca_v2 import fit_cpca_v2
from sim.two_routes import generate as gen_routes
from sim.segments import spatial_assemblies
from sim.laps import generate as gen_laps
from scripts.run_separation import CONDITIONS, judge
from scripts.run_pure_capabilities import judge_share, halves_by, LAPS, HC11
from scripts.plot_hc11_sequences import best_map_match
from experimental.scripts.dev_unmixing import judge_partial

OUT = ROOT / "exploration" / "results"
OUT.mkdir(parents=True, exist_ok=True)

STRESS = {
    "G 3 routes diff 7%": dict(rates=(0.07, 0.07, 0.07), shared=False),
    "H diff 5% + 5%": dict(rates=(0.05, 0.05), shared=False),
    "I diff 3% + 3%": dict(rates=(0.03, 0.03), shared=False),
    "J diff 25 cells 10%+10%": dict(rates=(0.10, 0.10), shared=False, n_route=25),
    "K same 5% + 5% both": dict(rates=(0.05, 0.05), shared=True),
    "L 3 maps same 7% each": dict(rates=(0.07, 0.07, 0.07), shared=True),
    "M 1 map 5%": dict(rates=(0.05,), shared=False),
}


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def eval_planted_react(conds, seed_base, phase_tag, fit_kw=None):
    fit_kw = dict(regime="reactivation", alpha=1.5 * np.pi, power=1.5, scale="global_rms",
                  n_shuffles=100, quantile=99.0, member_frac=0.2, ica=True, reconcile=True,
                  fold_harmonics=True) | (fit_kw or {})

    def one(args):
        ci, cname, kw = args
        ks, nouts, covs, times_s = [], [], [], []
        pass_o = pass_s = pass_p = tot = 0
        clean_o = clean_s = clean_p = 0
        for draw in range(5):
            d = gen_routes(np.random.default_rng(seed_base + 97 * ci + draw), **kw)
            perm = np.random.default_rng(seed("split", phase_tag, cname, draw)).permutation(d["times"].shape[0])
            halves = (perm[: perm.size // 2], perm[perm.size // 2:])
            maps = d["maps"] if sum(kw["rates"]) > 0 else []
            for fold in (0, 1):
                tr = halves[fold]
                t0 = time.perf_counter()
                res = fit_cpca_v2(d["times"][tr], d["T"][tr], np.random.default_rng(seed("c1", phase_tag, cname, draw, fold)), **fit_kw)
                times_s.append(time.perf_counter() - t0)
                ks.append(res["k"]); nouts.append(len(res["orders"]))
                jrng = np.random.default_rng(seed("c1-j", phase_tag, cname, draw, fold))
                if not maps:
                    tot += len(res["orders"])
                    if len(res["orders"]) == 0:
                        clean_o += 1; clean_s += 1; clean_p += 1
                    continue
                vo, vs, vp = [], [], []
                for o in res["orders"]:
                    oko, ko, _ = judge(o, maps, kw["shared"], jrng)
                    vo.append((oko, ko))
                    if kw["shared"]:
                        vs.append(judge_share(o, maps, jrng))
                        vp.append(judge_partial(o, maps, jrng))
                    else:
                        vs.append((oko, ko))
                        vp.append((oko, ko))
                    if oko:
                        covs.append(float(np.sum(np.isfinite(o) & np.isfinite(maps[ko])) / np.sum(np.isfinite(maps[ko]))))
                pass_o += sum(ok for ok, _ in vo)
                pass_s += sum(ok for ok, _ in vs)
                pass_p += sum(ok for ok, _ in vp)
                tot += len(vo)
                if len({k for ok, k in vo if ok}) == len(maps) and all(ok for ok, _ in vo): clean_o += 1
                if len({k for ok, k in vs if ok}) == len(maps) and all(ok for ok, _ in vs): clean_s += 1
                if len({k for ok, k in vp if ok}) == len(maps) and all(ok for ok, _ in vp): clean_p += 1
        return {
            "name": cname, "med_k": float(np.median(ks)), "med_out": float(np.median(nouts)),
            "pass_o": pass_o, "pass_s": pass_s, "pass_p": pass_p, "tot": tot,
            "clean_o": clean_o, "clean_s": clean_s, "clean_p": clean_p,
            "cov": float(np.median(covs)) if covs else np.nan,
            "med_time_s": float(np.median(times_s)),
        }

    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(one, [(ci, cn, kw) for ci, (cn, kw) in enumerate(conds.items())]))
    return rows


def eval_spatial_control():
    ks, nouts = [], []
    for draw in range(10):
        rng_d = np.random.default_rng(19000 + draw)
        route = rng_d.permutation(120)[:40]
        pos = np.full(120, np.nan); pos[route] = rng_d.uniform(0, 1, 40)
        mem = spatial_assemblies(rng_d, 120, route, pos, n_groups=6)
        d = gen_routes(rng_d, rates=(0.0,), shared=False, assembly_membership=mem)
        tr = rng_d.permutation(d["times"].shape[0])[:1250]
        res = fit_cpca_v2(d["times"][tr], d["T"][tr], np.random.default_rng(19500 + draw),
                          regime="reactivation", n_shuffles=100, quantile=99.0)
        ks.append(res["k"]); nouts.append(len(res["orders"]))
    return {"n_draws": 10, "clean": sum(n == 0 for n in nouts), "tot_out": sum(nouts)}


def eval_planted_laps_confirm():
    rows = []
    for ci, (cname, kw) in enumerate(LAPS.items()):
        ks, clean_o, clean_s, clean_p, pass_s, pass_p, tot, times_s = [], 0, 0, 0, 0, 0, 0, []
        for draw in range(5):
            d = gen_laps(np.random.default_rng(21000 + 97 * ci + draw), **kw)
            half = halves_by(d["direction"], np.random.default_rng(seed("half", "confirm", cname, draw)))
            n_dir = kw.get("laps_per_direction", (30, 30))
            maps = [] if kw.get("shuffle") else [m for m, n in zip(d["maps"], n_dir) if n > 0]
            for fold, train in enumerate((half, ~half)):
                tr = np.flatnonzero(train)
                t0 = time.perf_counter()
                res = fit_cpca_v2(d["times"][tr], d["T"][tr], np.random.default_rng(seed("c1-wl", cname, draw, fold)),
                                  regime="waking", power=1.5, n_shuffles=100, quantile=99.0)
                times_s.append(time.perf_counter() - t0)
                ks.append(res["k"])
                jrng = np.random.default_rng(seed("c1-wlj", cname, draw, fold))
                if not maps:
                    tot += len(res["orders"])
                    if len(res["orders"]) == 0:
                        clean_o += 1; clean_s += 1; clean_p += 1
                    continue
                vo = [judge(o, maps, True, jrng)[:2] for o in res["orders"]]
                vs = [judge_share(o, maps, jrng) for o in res["orders"]]
                vp = [judge_partial(o, maps, jrng) for o in res["orders"]]
                pass_s += sum(ok for ok, _ in vs); pass_p += sum(ok for ok, _ in vp); tot += len(vs)
                if len({k for ok, k in vo if ok}) == len(maps) and all(ok for ok, _ in vo): clean_o += 1
                if len({k for ok, k in vs if ok}) == len(maps) and all(ok for ok, _ in vs): clean_s += 1
                if len({k for ok, k in vp if ok}) == len(maps) and all(ok for ok, _ in vp): clean_p += 1
        rows.append({
            "name": cname, "med_k": float(np.median(ks)), "pass_s": pass_s, "pass_p": pass_p, "tot": tot,
            "clean_o": clean_o, "clean_s": clean_s, "clean_p": clean_p, "med_time_s": float(np.median(times_s)),
        })
    return rows


def eval_real_waking():
    from cpca import event_spike_times
    from cpca.hc11 import laps, load_session as load_hc11, place_centres, pyramidal_spikes, running, session_paths as hc11_paths
    import cpca.hc5 as hc5
    from scripts.run_hc5 import training_data
    from scripts.run_laps_hc11 import heldout_run
    import cpca.dandi001695 as dd
    import cpca.huszar as hz
    from experimental.scripts.run_initial_real import figure8_events

    out = {}
    # 1. hc-11 linear, circular, control
    for kind in ("linear", "circular", "control"):
        cs = cp = ps = pp = tot = nf = 0
        per_fit = []
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
                res = fit_cpca_v2(X, TX, np.random.default_rng(seed("c1-hc11", kind, sname, fold)),
                                  regime="waking", power=1.5, n_shuffles=100, quantile=99.0)
                jrng = np.random.default_rng(seed("c1-hc11j", kind, sname, fold)); nf += 1
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
                ok_s = len({m for ok, m in vs if ok}) == len(maps) and all(ok for ok, _ in vs)
                ok_p = len({m for ok, m in vp if ok}) == len(maps) and all(ok for ok, _ in vp)
                cs += ok_s; cp += ok_p
                per_fit.append({"session": sname, "fold": fold, "k": res["k"], "share": vs, "partial": vp})
        out[f"hc11_{kind}"] = {"pass_s": ps, "pass_p": pp, "tot": tot, "clean_s": cs, "clean_p": cp, "n_fits": nf, "per_fit": per_fit}

    # 2. hc-5 wheel (aligned, unaligned, shuffled)
    for cond in ("aligned", "unaligned", "shuffled"):
        cs = ps = tot = nf = 0
        for sname in hc5.session_names():
            sess = hc5.load_session(sname)
            for fold in (0, 1):
                X, T, train, test, _ = training_data(sess, cond, 0, fold)
                TX = np.full(X.shape[0], T) if np.ndim(T) == 0 else np.asarray(T)
                f = hc5.time_fields(sess, test)
                maps = [] if cond == "shuffled" else ([f] if np.isfinite(f).sum() >= 6 else [])
                res = fit_cpca_v2(X, TX, np.random.default_rng(seed("c1-hc5", cond, sname, fold)),
                                  regime="waking", power=1.5, n_shuffles=100, quantile=99.0)
                jrng = np.random.default_rng(seed("c1-hc5j", cond, sname, fold)); nf += 1
                if not maps:
                    tot += len(res["orders"])
                    if len(res["orders"]) == 0: cs += 1
                    continue
                vs = [judge_share(o, maps, jrng) for o in res["orders"]]
                ps += sum(ok for ok, _ in vs); tot += len(vs)
                if len({m for ok, m in vs if ok}) == len(maps) and all(ok for ok, _ in vs): cs += 1
        out[f"hc5_{cond}"] = {"pass": ps, "tot": tot, "clean": cs, "n_fits": nf}

    # 3. DANDI 001695 laps
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
            res = fit_cpca_v2(X, TX, np.random.default_rng(seed("c1-dd", path.name, fold)),
                              regime="waking", power=1.5, n_shuffles=100, quantile=99.0)
            jrng = np.random.default_rng(seed("c1-ddj", path.name, fold))
            vs = [judge_share(o, maps, jrng) for o in res["orders"]]
            vp = [judge_partial(o, maps, jrng) for o in res["orders"]]
            ps += sum(ok for ok, _ in vs); pp += sum(ok for ok, _ in vp); tot += len(vs); nf += 1
            if maps and len({m for ok, m in vs if ok}) == len(maps) and all(ok for ok, _ in vs): cs += 1
            if maps and len({m for ok, m in vp if ok}) == len(maps) and all(ok for ok, _ in vp): cp += 1
    out["dandi001695_laps"] = {"pass_s": ps, "pass_p": pp, "tot": tot, "clean_s": cs, "clean_p": cp, "n_fits": nf}

    # 4. DANDI 000552 figure-eight (all 65 sessions: unsupervised L/R trial separation accuracy + pattern alignment to oracle)
    f8_rows = []
    from exploration.cpca_v2 import encode_arc, lag_matrix, eigh_desc
    for path in hz.session_paths():
        s = hz.load_session(path); task_w = hz.task_window(s)[1]
        trials = hz.task_trials(s, task_w); arms = s["arm"][trials]; cells = hz.putative_pyramidal(s)
        times, T = figure8_events(s, trials, cells)
        if (arms == "left").sum() < 2 or (arms == "right").sum() < 2: continue
        res = fit_cpca_v2(times, T, np.random.default_rng(seed("c1-f8", s["name"])),
                          regime="waking", power=1.5, n_shuffles=100, quantile=99.0)
        entry = {"name": s["name"], "n_trials": len(trials), "k": res["k"], "acc": np.nan, "oracle_sim": np.nan}
        if res["k"] >= 2:
            Y = np.abs(res["scores"][:, :2]) ** 2
            pred = (Y[:, 0] >= Y[:, 1]); y_true = (arms == "left")
            entry["acc"] = float(max(np.mean(pred == y_true), np.mean(pred != y_true)))
            # Compare top-2 ICA patterns against supervised Oracle (leading evec of true L & true R trials)
            X = encode_arc(times, T, alpha=1.5 * np.pi)
            qL = eigh_desc(lag_matrix(X[y_true]))[1][:, 0]
            qR = eigh_desc(lag_matrix(X[~y_true]))[1][:, 0]
            P2 = res["patterns"][:, :2]
            s1 = 0.5 * (abs(np.vdot(qL, P2[:, 0])) + abs(np.vdot(qR, P2[:, 1])))
            s2 = 0.5 * (abs(np.vdot(qL, P2[:, 1])) + abs(np.vdot(qR, P2[:, 0])))
            entry["oracle_sim"] = float(max(s1, s2))
        f8_rows.append(entry)
    out["dandi000552_fig8"] = f8_rows
    return out


def eval_ablation():
    """Ablation on planted A-F (confirm 9000) showing each pillar's necessity."""
    from cpca import fit as fit_old
    abl = {}
    # Full cpca_v2 already computed; test removing one pillar at a time
    abl["no_ica"] = eval_planted_react(CONDITIONS, 9000, "confirm", fit_kw=dict(ica=False, reconcile=False))
    abl["no_symmetry"] = eval_planted_react(CONDITIONS, 9000, "confirm", fit_kw=dict(reconcile=False))
    abl["alpha_pi"] = eval_planted_react(CONDITIONS, 9000, "confirm", fit_kw=dict(alpha=np.pi))
    abl["no_harmonic_fold"] = eval_planted_react(CONDITIONS, 9000, "confirm", fit_kw=dict(fold_harmonics=False))
    return abl


if __name__ == "__main__":
    t0 = time.perf_counter()
    print(">>> Running Confirmation Part 1 (fresh seeds + real waking + ablation)...", flush=True)
    res = {}
    res["planted_AF_confirm"] = eval_planted_react(CONDITIONS, 9000, "confirm")
    print("\n--- PLANTED REACTIVATIONS A-F (FRESH CONFIRMATION SEEDS 9000) ---", flush=True)
    for r in res["planted_AF_confirm"]:
        pct = f"{r['pass_o']}/{r['tot']} ({100*r['pass_o']/max(r['tot'],1):.0f}%)" if r["tot"] else "0/0"
        print(f"  {r['name'][:36]:<36} | k={r['med_k']:3.1f} out={r['med_out']:3.1f} | pass={pct:>13} | clean orig={r['clean_o']:2d}/10 share={r['clean_s']:2d}/10 partial={r['clean_p']:2d}/10 | cov={r['cov']:.2f} | {1000*r['med_time_s']:.0f}ms", flush=True)

    res["planted_stress_confirm"] = eval_planted_react(STRESS, 15000, "stress_confirm")
    print("\n--- PLANTED REACTIVATIONS STRESS G-M (FRESH CONFIRMATION SEEDS 15000) ---", flush=True)
    for r in res["planted_stress_confirm"]:
        pct = f"{r['pass_o']}/{r['tot']} ({100*r['pass_o']/max(r['tot'],1):.0f}%)" if r["tot"] else "0/0"
        print(f"  {r['name'][:36]:<36} | k={r['med_k']:3.1f} out={r['med_out']:3.1f} | pass={pct:>13} | clean orig={r['clean_o']:2d}/10 share={r['clean_s']:2d}/10 partial={r['clean_p']:2d}/10 | cov={r['cov']:.2f}", flush=True)

    res["spatial_control"] = eval_spatial_control()
    print(f"\n--- SPATIAL ASSEMBLIES CONTROL (no replays, 10 draws): clean={res['spatial_control']['clean']}/10 (tot outputs={res['spatial_control']['tot_out']}) ---", flush=True)

    res["planted_laps_confirm"] = eval_planted_laps_confirm()
    print("\n--- PLANTED WAKING LAPS W1-W5 (FRESH CONFIRMATION SEEDS 21000) ---", flush=True)
    for r in res["planted_laps_confirm"]:
        print(f"  {r['name']:<28} | k={r['med_k']:3.1f} | share={r['pass_s']:2d}/{r['tot']:<3d} | clean orig={r['clean_o']:2d}/10 share={r['clean_s']:2d}/10 partial={r['clean_p']:2d}/10 | {1000*r['med_time_s']:.0f}ms", flush=True)

    res["real_waking"] = eval_real_waking()
    rw = res["real_waking"]
    print("\n--- REAL WAKING DATA ---", flush=True)
    for k in ("hc11_linear", "hc11_circular", "hc11_control", "dandi001695_laps"):
        d = rw[k]
        print(f"  {k:<18} | pass share={d['pass_s']:2d}/{d['tot']:<3d} partial={d['pass_p']:2d}/{d['tot']:<3d} | clean share={d['clean_s']:2d}/{d['n_fits']} partial={d['clean_p']:2d}/{d['n_fits']}", flush=True)
    for k in ("hc5_aligned", "hc5_unaligned", "hc5_shuffled"):
        d = rw[k]
        print(f"  {k:<18} | pass={d['pass']:2d}/{d['tot']:<3d} | clean={d['clean']:2d}/{d['n_fits']}", flush=True)
    f8 = rw["dandi000552_fig8"]
    f8_ok = [x for x in f8 if np.isfinite(x["acc"])]
    print(f"  dandi000552_fig8   | k>=2 in {len(f8_ok)}/{len(f8)} | L/R trial acc: med={np.median([x['acc'] for x in f8_ok]):.1%}, mean={np.mean([x['acc'] for x in f8_ok]):.1%}, >=90% in {sum(x['acc']>=0.9 for x in f8_ok)}/{len(f8_ok)} | oracle pattern sim med={np.median([x['oracle_sim'] for x in f8_ok]):.3f}", flush=True)

    res["ablation"] = eval_ablation()
    print("\n--- ABLATION ON FRESH CONFIRMATION A-F (clean fits, share criterion) ---", flush=True)
    for k, rows in res["ablation"].items():
        print(f"  {k:<18}: " + " ".join(f"{r['name'][0]}:{r['clean_s']}/10" for r in rows) + f" | TOT={sum(r['clean_s'] for r in rows)}/60", flush=True)

    pickle.dump(res, open(OUT / "confirm_part1.pkl", "wb"))
    print(f"\nPart 1 complete in {time.perf_counter() - t0:.1f}s -> exploration/results/confirm_part1.pkl", flush=True)
