"""Objective head-to-head evaluation of phase arc alpha = pi vs alpha = 1.5*pi (Part 1).

Tests:
1. Legacy cpca (global centring, no ICA/reconcile) with alpha=pi vs alpha=1.5*pi:
   - hc-11 spike-in power (2 largest sessions)
   - hc-11 maze rest (+ timing control)
   - hc-11 waking laps (best-of-outputs & per-component) & hc-5 wheel
2. Unified cpca_v2 with alpha=pi vs alpha=1.5*pi (identical seeds, identical code):
   - Planted Reactivations A-F (fresh seeds 9000)
   - Planted Reactivations Stress G-M (fresh seeds 15000)
   - Short partial-replay G1 grid (sim/segments.py, L in [0.3, 0.5], 90 draws + controls)
   - High-jitter / broad-timing sweep (jitter_ms = 8, 16, 24, 32, 40 ms on Condition A & E)
   - Partial-segment length sweep (segment = [0.8,1.0], [0.5,1.0], [0.3,0.5], [0.2,0.35] on Condition E)
"""

import pickle
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exploration.cpca_v2 import fit_cpca_v2, encode_arc, lag_matrix, eigh_desc, readout_gap
from cpca.decompose import prepare, shuffle_null, n_above_null
from sim.two_routes import generate as gen_routes
from sim.segments import generate as gen_seg, n_route_for, spatial_assemblies
from scripts.run_separation import CONDITIONS, judge
from scripts.run_pure_capabilities import judge_share
from scripts.plot_hc11_sequences import best_map_match
from exploration.run_confirm_part1 import STRESS

OUT = ROOT / "exploration" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def legacy_fit(times, T, rng, alpha=np.pi, center=True, scale=True, n_shuffles=100, member_frac=0.2):
    """Exact legacy cpca.fit logic, parameterized only by phase arc alpha."""
    enc = encode_arc(times, T, alpha=alpha)
    X = prepare(enc, center=center, scale=scale)
    vals, vecs = eigh_desc(lag_matrix(X))
    null = shuffle_null(X, rng, n_shuffles=n_shuffles)
    k, bound = n_above_null(vals, null, centred=center)
    k = int(min(max(k, 0), 20))
    T_read = float(np.median(T)) if np.ndim(T) else float(T)
    orders = [readout_gap(vecs[:, j], T_read=T_read, alpha=alpha, member_frac=member_frac)[0] for j in range(k)]
    return {"k": k, "orders": orders, "vecs": vecs[:, :k]}


def eval_planted_block(conds, seed_base, tag, alpha):
    def one(args):
        ci, cname, kw = args
        ks, nouts, covs, rhos = [], [], [], []
        po = ps = tot = co = cs = 0
        for draw in range(5):
            d = gen_routes(np.random.default_rng(seed_base + 97 * ci + draw), **kw)
            perm = np.random.default_rng(seed("split", tag, cname, draw)).permutation(d["times"].shape[0])
            halves = (perm[: perm.size // 2], perm[perm.size // 2:])
            maps = d["maps"] if sum(kw["rates"]) > 0 else []
            for fold in (0, 1):
                tr = halves[fold]
                res = fit_cpca_v2(d["times"][tr], d["T"][tr],
                                  np.random.default_rng(seed("arc-fit", tag, cname, draw, fold)),
                                  regime="reactivation", alpha=alpha, power=1.5, scale="global_rms",
                                  n_shuffles=100, quantile=99.0, member_frac=0.2)
                ks.append(res["k"]); nouts.append(len(res["orders"]))
                jrng = np.random.default_rng(seed("arc-j", tag, cname, draw, fold))
                if not maps:
                    tot += len(res["orders"])
                    if len(res["orders"]) == 0: co += 1; cs += 1
                    continue
                vo, vs = [], []
                for o in res["orders"]:
                    oko, ko, det = judge(o, maps, kw["shared"], jrng)
                    vo.append((oko, ko))
                    if kw["shared"]:
                        vs.append(judge_share(o, maps, jrng))
                    else:
                        vs.append((oko, ko))
                    if oko:
                        covs.append(float(np.sum(np.isfinite(o) & np.isfinite(maps[ko])) / np.sum(np.isfinite(maps[ko]))))
                        if "rho" in det: rhos.append(det["rho"])
                po += sum(ok for ok, _ in vo); ps += sum(ok for ok, _ in vs); tot += len(vo)
                if len({k for ok, k in vo if ok}) == len(maps) and all(ok for ok, _ in vo): co += 1
                if len({k for ok, k in vs if ok}) == len(maps) and all(ok for ok, _ in vs): cs += 1
        return {
            "name": cname, "med_k": float(np.median(ks)), "med_out": float(np.median(nouts)),
            "po": po, "ps": ps, "tot": tot, "co": co, "cs": cs,
            "cov": float(np.median(covs)) if covs else np.nan,
            "rho": float(np.median(rhos)) if rhos else np.nan,
        }
    with ThreadPoolExecutor(max_workers=6) as ex:
        return list(ex.map(one, [(ci, cn, kw) for ci, (cn, kw) in enumerate(conds.items())]))


def eval_g1(alpha):
    grid = {}
    tot_pass = 0
    for rev in (0.5, 0.3):
        for cpr in (5, 9, 13):
            nr = n_route_for(cpr)
            counts = []
            for sh in (0.02, 0.05, 0.10):
                hits = 0
                for draw in range(5):
                    d = gen_seg(np.random.default_rng(seed("g1-arc", rev, cpr, sh, draw)),
                                n_route=nr, replay_frac=sh, reverse_frac=rev)
                    res = fit_cpca_v2(d["times"], d["T"],
                                      np.random.default_rng(seed("g1-arc-f", rev, cpr, sh, draw)),
                                      regime="reactivation", alpha=alpha, power=1.5, scale="global_rms",
                                      n_shuffles=100, quantile=99.0, member_frac=0.2)
                    sc = best_map_match(res["orders"], {1.0: d["centres"]}, False, 1.0,
                                        np.random.default_rng(seed("g1-arc-s", rev, cpr, sh, draw)), n_null=500) if res["orders"] else {"p": np.nan}
                    hits += bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
                counts.append(hits); tot_pass += hits
            grid[(rev, cpr)] = tuple(counts)
    fp0 = 0
    for draw in range(15):
        d = gen_seg(np.random.default_rng(seed("g1-arc-z", draw)), n_route=n_route_for(9), replay_frac=0.0)
        res = fit_cpca_v2(d["times"], d["T"], np.random.default_rng(seed("g1-arc-zf", draw)),
                          regime="reactivation", alpha=alpha, power=1.5, scale="global_rms",
                          n_shuffles=100, quantile=99.0, member_frac=0.2)
        sc = best_map_match(res["orders"], {1.0: d["centres"]}, False, 1.0, np.random.default_rng(draw), n_null=500) if res["orders"] else {"p": np.nan}
        fp0 += bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
    fp_sa = 0
    for draw in range(5):
        rng_d = np.random.default_rng(seed("g1-arc-sa", draw))
        nr = n_route_for(9); route = rng_d.permutation(120)[:nr]; pos = np.full(120, np.nan); pos[route] = rng_d.uniform(0, 1, nr)
        mem = spatial_assemblies(rng_d, 120, route, pos)
        d = gen_seg(rng_d, n_route=nr, replay_frac=0.0, assembly_membership=mem)
        res = fit_cpca_v2(d["times"], d["T"], np.random.default_rng(seed("g1-arc-saf", draw)),
                          regime="reactivation", alpha=alpha, power=1.5, scale="global_rms",
                          n_shuffles=100, quantile=99.0, member_frac=0.2)
        sc = best_map_match(res["orders"], {1.0: pos}, False, 1.0, np.random.default_rng(draw), n_null=500) if res["orders"] else {"p": np.nan}
        fp_sa += bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
    return {"grid": grid, "tot": tot_pass, "fp0": fp0, "fp_sa": fp_sa}


def eval_mechanistic_sweeps():
    """Explicitly test the two regimes where theory predicts pi vs 1.5*pi trade off:
    (a) High temporal jitter (jitter_ms = 8, 16, 24, 32, 40 ms) on Condition A (2 routes) & E (1 route)
    (b) Short partial segments (segment = (0.8,1.0), (0.5,1.0), (0.3,0.5), (0.2,0.35)) on Condition E (1 route, 20%)
    (c) Fine alpha sweep (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)*pi on Condition D (same cells both) & A (diff cells both)
    """
    out = {"jitter": [], "segment": [], "alpha_sweep": []}

    # (a) Jitter sweep on Condition A (rates 10%+10%, diff cells, both ways)
    for jit in (8.0, 16.0, 24.0, 32.0, 40.0):
        row = {"jitter_ms": jit}
        for label, alpha in (("pi", np.pi), ("1.5pi", 1.5 * np.pi)):
            clean, rhos, ks = 0, [], []
            for draw in range(5):
                d = gen_routes(np.random.default_rng(seed("jit-sw", jit, draw)),
                               rates=(0.10, 0.10), shared=False, jitter_ms=jit)
                perm = np.random.default_rng(seed("jit-sp", jit, draw)).permutation(d["times"].shape[0])
                for fold, tr in enumerate((perm[:1250], perm[1250:])):
                    res = fit_cpca_v2(d["times"][tr], d["T"][tr],
                                      np.random.default_rng(seed("jit-f", jit, draw, fold)),
                                      regime="reactivation", alpha=alpha, n_shuffles=100, quantile=99.0)
                    ks.append(res["k"])
                    jrng = np.random.default_rng(seed("jit-j", jit, draw, fold))
                    vo = [judge(o, d["maps"], False, jrng) for o in res["orders"]]
                    if len({k for ok, k, _ in vo if ok}) == 2 and all(ok for ok, _, _ in vo):
                        clean += 1
                    for ok, _, det in vo:
                        if ok and "rho" in det: rhos.append(det["rho"])
            row[label] = {"clean": clean, "med_rho": float(np.median(rhos)) if rhos else np.nan, "med_k": float(np.median(ks))}
        out["jitter"].append(row)

    # (b) Partial segment length sweep on Condition E (1 map, 20%, both ways)
    for seg in ((0.8, 1.0), (0.5, 1.0), (0.3, 0.5), (0.2, 0.35)):
        row = {"segment": seg}
        for label, alpha in (("pi", np.pi), ("1.5pi", 1.5 * np.pi)):
            clean, det_any, rhos, spans = 0, 0, [], []
            for draw in range(5):
                d = gen_routes(np.random.default_rng(seed("seg-sw", seg, draw)),
                               rates=(0.20,), shared=False, segment=seg)
                perm = np.random.default_rng(seed("seg-sp", seg, draw)).permutation(d["times"].shape[0])
                for fold, tr in enumerate((perm[:1250], perm[1250:])):
                    res = fit_cpca_v2(d["times"][tr], d["T"][tr],
                                      np.random.default_rng(seed("seg-f", seg, draw, fold)),
                                      regime="reactivation", alpha=alpha, n_shuffles=100, quantile=99.0)
                    jrng = np.random.default_rng(seed("seg-j", seg, draw, fold))
                    vo = [judge(o, d["maps"], False, jrng) for o in res["orders"]]
                    if any(ok for ok, _, _ in vo): det_any += 1
                    if len({k for ok, k, _ in vo if ok}) == 1 and all(ok for ok, _, _ in vo): clean += 1
                    for ok, _, det in vo:
                        if ok and "rho" in det: rhos.append(det["rho"])
                    spans.extend([sp / np.pi for sp in res["spans"]])
            row[label] = {"clean": clean, "det_any": det_any,
                          "med_rho": float(np.median(rhos)) if rhos else np.nan,
                          "med_span_pi": float(np.median(spans)) if spans else np.nan}
        out["segment"].append(row)

    # (c) Fine alpha sweep on Condition D (same cells, 10%+10%, both) and Condition K (same cells, 5%+5%, both)
    for mult in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
        alpha = mult * np.pi
        row = {"mult": mult}
        for cname, kw, sbase in (("D_same_10", dict(rates=(0.10, 0.10), shared=True), 9200),
                                 ("K_same_5", dict(rates=(0.05, 0.05), shared=True), 15400),
                                 ("I_diff_3", dict(rates=(0.03, 0.03), shared=False), 15200)):
            cs, rhos, ks = 0, [], []
            for draw in range(5):
                d = gen_routes(np.random.default_rng(sbase + draw), **kw)
                perm = np.random.default_rng(seed("asw-sp", cname, draw)).permutation(d["times"].shape[0])
                for fold, tr in enumerate((perm[:1250], perm[1250:])):
                    res = fit_cpca_v2(d["times"][tr], d["T"][tr],
                                      np.random.default_rng(seed("asw-f", cname, draw, fold)),
                                      regime="reactivation", alpha=alpha, n_shuffles=100, quantile=99.0)
                    ks.append(res["k"])
                    jrng = np.random.default_rng(seed("asw-j", cname, draw, fold))
                    vs = [judge_share(o, d["maps"], jrng) if kw["shared"] else judge(o, d["maps"], False, jrng)[:2] for o in res["orders"]]
                    if len({k for ok, k in vs if ok}) == 2 and all(ok for ok, _ in vs): cs += 1
                    for o in res["orders"]:
                        ok, k, det = judge(o, d["maps"], kw["shared"], jrng)
                        if ok and "rho" in det: rhos.append(det["rho"])
            row[cname] = {"clean_s": cs, "med_rho": float(np.median(rhos)) if rhos else np.nan, "med_k": float(np.median(ks))}
        out["alpha_sweep"].append(row)
    return out


def eval_legacy_cpca():
    """Compare alpha=pi vs alpha=1.5*pi inside the EXISTING legacy cpca.fit (global centring, no ICA)."""
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session as load_hc11, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths as hc11_paths
    from scripts.run_rerun import ORDER_INDEX, fixed_subset
    from scripts.run_spikein_hc11 import ideal as spikein_ideal

    out = {"spikein_legacy": {}}
    fracs = [0.0, 0.03, 0.05, 0.10, 0.20]
    for label, alpha in (("pi", np.pi), ("1.5pi", 1.5 * np.pi)):
        totals = {f: 0 for f in fracs}
        for sname in ("Achilles_10252013", "Achilles_11012013"):
            path = next(p for p in hc11_paths() if p.name.startswith(sname))
            s = load_hc11(path); run = running(s)
            iv = nrem_intervals(s, "pre")
            by_cell, ids, pooled = pyramidal_spikes(s, iv)
            b, pk = population_bursts(pooled, iv); dur = np.diff(b, axis=1).ravel()
            sub = fixed_subset(len(pk), 555 + 2 * ORDER_INDEX[sname] + 1)
            base, _ = event_spike_times(by_cell, ids, (pk - dur / 2)[sub], (pk + dur / 2)[sub], max_spikes=60)
            T = dur[sub] * 1000; centres = place_centres(s, ids, run)
            for f in fracs:
                k_ev = int(round(f * len(T)))
                for d in range(5):
                    rng = np.random.default_rng(seed("spikein", sname, "ideal", f, d))
                    chosen = rng.choice(len(T), k_ev, replace=False) if k_ev else np.array([], int)
                    times = spikein_ideal(base, T, chosen, centres, rng) if k_ev else base
                    res = legacy_fit(times, T, np.random.default_rng(seed("leg-sp", sname, f, d)),
                                     alpha=alpha, center=True, scale=True, n_shuffles=100)
                    sc = best_map_match(res["orders"], centres, run["circular"], run["length"],
                                        np.random.default_rng(seed("leg-sc", sname, f, d)), n_null=500) if res["orders"] else {"p": np.nan}
                    totals[f] += bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
        out["spikein_legacy"][label] = totals
    return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    res = {}
    print("=== PART 1: Head-to-head alpha=pi vs alpha=1.5*pi ===", flush=True)

    for label, alpha in (("pi", np.pi), ("1.5pi", 1.5 * np.pi)):
        res[f"AF_{label}"] = eval_planted_block(CONDITIONS, 9000, "confirm", alpha)
        res[f"STRESS_{label}"] = eval_planted_block(STRESS, 15000, "stress_confirm", alpha)
        print(f"\n[cpca_v2 alpha={label}] A-F total clean (share): {sum(r['cs'] for r in res[f'AF_{label}'])}/60 | STRESS total clean (share): {sum(r['cs'] for r in res[f'STRESS_{label}'])}/70", flush=True)
        for r in res[f"AF_{label}"] + res[f"STRESS_{label}"]:
            print(f"  {r['name'][:34]:<34} | k={r['med_k']:3.1f} out={r['med_out']:3.1f} | orig={r['co']:2d}/10 share={r['cs']:2d}/10 | rho={r['rho']:.3f} cov={r['cov']:.2f}", flush=True)

    print("\n--- G1 Short Partial-Replay Grid (L in [0.3, 0.5]) ---", flush=True)
    for label, alpha in (("pi", np.pi), ("1.5pi", 1.5 * np.pi)):
        res[f"G1_{label}"] = eval_g1(alpha)
        g = res[f"G1_{label}"]
        print(f"  G1 alpha={label:<5}: TOTAL={g['tot']}/90 | share=0 FP={g['fp0']}/15 | spatial_asm FP={g['fp_sa']}/5", flush=True)
        for k_g, v_g in g["grid"].items():
            print(f"    rev={k_g[0]} cells={k_g[1]:2d}: {v_g[0]}/{v_g[1]}/{v_g[2]}", flush=True)

    print("\n--- Mechanistic Sweeps (Jitter, Segment Length, Fine Alpha Sweep) ---", flush=True)
    res["sweeps"] = eval_mechanistic_sweeps()
    print("  Jitter sweep (Condition A, clean/10, med_rho):", flush=True)
    for r in res["sweeps"]["jitter"]:
        print(f"    jit={r['jitter_ms']:4.0f}ms | pi: clean={r['pi']['clean']:2d}/10 rho={r['pi']['med_rho']:.3f} k={r['pi']['med_k']:.1f} | 1.5pi: clean={r['1.5pi']['clean']:2d}/10 rho={r['1.5pi']['med_rho']:.3f} k={r['1.5pi']['med_k']:.1f}", flush=True)
    print("  Segment length sweep (Condition E, clean/10, det_any/10, med_rho, med_span/pi):", flush=True)
    for r in res["sweeps"]["segment"]:
        print(f"    seg={str(r['segment']):<12} | pi: clean={r['pi']['clean']:2d}/10 any={r['pi']['det_any']:2d}/10 rho={r['pi']['med_rho']:.3f} span={r['pi']['med_span_pi']:.2f}pi | 1.5pi: clean={r['1.5pi']['clean']:2d}/10 any={r['1.5pi']['det_any']:2d}/10 rho={r['1.5pi']['med_rho']:.3f} span={r['1.5pi']['med_span_pi']:.2f}pi", flush=True)
    print("  Fine alpha sweep (clean_s/10):", flush=True)
    for r in res["sweeps"]["alpha_sweep"]:
        print(f"    alpha={r['mult']:.2f}*pi | D(same 10%): {r['D_same_10']['clean_s']:2d}/10 (rho={r['D_same_10']['med_rho']:.3f}) | K(same 5%): {r['K_same_5']['clean_s']:2d}/10 (rho={r['K_same_5']['med_rho']:.3f}) | I(diff 3%): {r['I_diff_3']['clean_s']:2d}/10 (rho={r['I_diff_3']['med_rho']:.3f})", flush=True)

    print("\n--- Legacy cpca.fit (global centring, no ICA) spike-in power (2 largest hc-11 sessions) ---", flush=True)
    res["legacy"] = eval_legacy_cpca()
    for label, tot in res["legacy"]["spikein_legacy"].items():
        print(f"  Legacy cpca alpha={label:<5}: " + "  ".join(f"{f:.0%}:{v}/10" for f, v in tot.items()), flush=True)

    pickle.dump(res, open(OUT / "arc_eval_part1.pkl", "wb"))
    print(f"\nArc Part 1 finished in {time.perf_counter()-t0:.1f}s", flush=True)
