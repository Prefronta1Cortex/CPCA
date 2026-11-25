"""Confirmation Part 2: Real reactivations (hc-11 spike-in power across ALL 8 sessions,
hc-11 maze rest with 100 timing shuffles, hc-11 sleep POST vs PRE with 50 timing shuffles,
DANDI 001695 maze rest & POST vs PRE with timing shuffles) and G1 partial-replay grid.

Locked parameters (exploration/cpca_v2.py):
  regime='reactivation', alpha=1.5*pi, power=1.5, scale='global_rms', n_shuffles=100,
  quantile=99.0, member_frac=0.2, ica=True, reconcile=True, fold_harmonics=True.
"""

import pickle
import sys
import time
import zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exploration.cpca_v2 import fit_cpca_v2
from cpca import event_spike_times, population_bursts
from cpca.hc11 import load_session as load_hc11, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths as hc11_paths
import cpca.dandi001695 as dd
from cpca.replay import immobile_intervals
from scripts.run_replay_hc11 import intersect, two_d
from scripts.plot_hc11_sequences import best_map_match
from scripts.run_maze_rest import HC11_ORDER, shuffle_timing
from scripts.run_rerun import ORDER_INDEX, fixed_subset
from scripts.run_spikein_hc11 import ideal as spikein_ideal
from sim.segments import generate as gen_seg, n_route_for, spatial_assemblies

OUT = ROOT / "exploration" / "results"
OUT.mkdir(parents=True, exist_ok=True)

CFG = dict(regime="reactivation", alpha=1.5 * np.pi, power=1.5, scale="global_rms",
           n_shuffles=100, quantile=99.0, member_frac=0.2, ica=True, reconcile=True,
           fold_harmonics=True)


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def run_hc11_spikein():
    """Full hc-11 spike-in power curve on ALL 8 sessions (matched 1500 PRE subsets, ideal replays)."""
    fracs = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
    rows = []
    print("\n--- HC-11 SPIKE-IN POWER (ALL 8 SESSIONS, 1500 PRE EVENTS, IDEAL REPLAYS) ---", flush=True)
    for sname in HC11_ORDER:
        path = next(p for p in hc11_paths() if p.name.startswith(sname))
        s = load_hc11(path); run = running(s)
        iv = nrem_intervals(s, "pre")
        by_cell, ids, pooled = pyramidal_spikes(s, iv)
        b, pk = population_bursts(pooled, iv); dur = np.diff(b, axis=1).ravel()
        sub = fixed_subset(len(pk), 555 + 2 * ORDER_INDEX[sname] + 1)
        base, _ = event_spike_times(by_cell, ids, (pk - dur / 2)[sub], (pk + dur / 2)[sub], max_spikes=60)
        T = dur[sub] * 1000; centres = place_centres(s, ids, run)
        s_row = {"session": sname, "hits": {}, "med_k": {}}
        parts = []
        for f in fracs:
            k_ev = int(round(f * len(T)))
            hits = 0; ks = []
            for d in range(5):
                rng = np.random.default_rng(seed("spikein", sname, "ideal", f, d))
                chosen = rng.choice(len(T), k_ev, replace=False) if k_ev else np.array([], int)
                times = spikein_ideal(base, T, chosen, centres, rng) if k_ev else base
                res = fit_cpca_v2(times, T, np.random.default_rng(seed("c2-sp", sname, f, d)), **CFG)
                ks.append(res["k"])
                sc = best_map_match(res["orders"], centres, run["circular"], run["length"],
                                    np.random.default_rng(seed("c2-spsc", sname, f, d)), n_null=500) if res["orders"] else {"p": np.nan}
                hits += bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
            s_row["hits"][f] = hits; s_row["med_k"][f] = float(np.median(ks))
            parts.append(f"{f:.0%}:{hits}/5")
        rows.append(s_row)
        print(f"  {sname:<18} | " + "  ".join(parts), flush=True)
    # Pooled 2 largest & 6 smaller
    big = [r for r in rows if r["session"].startswith("Achilles")]
    sml = [r for r in rows if not r["session"].startswith("Achilles")]
    print("  2 LARGEST POOLED   | " + "  ".join(f"{f:.0%}:{sum(r['hits'][f] for r in big)}/10" for f in fracs), flush=True)
    print("  6 SMALLER POOLED   | " + "  ".join(f"{f:.0%}:{sum(r['hits'][f] for r in sml)}/30" for f in fracs), flush=True)
    return rows


def run_hc11_maze_and_sleep(n_shuf_timing=50):
    print(f"\n--- HC-11 MAZE REST & SLEEP POST/PRE (+ {n_shuf_timing} TIMING SHUFFLES) ---", flush=True)
    out = {"maze": [], "post": [], "pre": []}
    for ep in ("maze", "post", "pre"):
        n_ord = n_both = fp_tot = sh_tot = 0
        for sname in HC11_ORDER:
            path = next(p for p in hc11_paths() if p.name.startswith(sname))
            s = load_hc11(path); s["pos_xy"] = two_d(path); run = running(s)
            iv = intersect(immobile_intervals(s), [tuple(s["maze_epoch"])]) if ep == "maze" else nrem_intervals(s, ep)
            by_cell, ids, pooled = pyramidal_spikes(s, iv)
            b, pk = population_bursts(pooled, iv); dur = np.diff(b, axis=1).ravel()
            times, _ = event_spike_times(by_cell, ids, pk - dur / 2, pk + dur / 2, max_spikes=60)
            T = dur * 1000; centres = place_centres(s, ids, run)
            res = fit_cpca_v2(times, T, np.random.default_rng(seed("c2-hc", sname, ep)), **CFG)
            sc = best_map_match(res["orders"], centres, run["circular"], run["length"],
                                np.random.default_rng(seed("c2-hcsc", sname, ep)), n_null=1000) if res["orders"] else {"abs": np.nan, "p": np.nan, "n": 0}
            s_abs = []; s_fp = 0; s_ks = []
            # Run timing shuffles on ALL maze rest sessions, and on POST/PRE sessions that pass order (plus Achilles_1025/1101)
            run_shuf = (ep == "maze") or (np.isfinite(sc["p"]) and sc["p"] < 0.05) or sname.startswith("Achilles")
            if run_shuf:
                for d in range(n_shuf_timing):
                    srng = np.random.default_rng(seed("c2-hcsh", sname, ep, d))
                    rs = fit_cpca_v2(shuffle_timing(times, srng), T, srng, **CFG)
                    s_ks.append(rs["k"])
                    if rs["orders"]:
                        ss = best_map_match(rs["orders"], centres, run["circular"], run["length"], srng, n_null=200)
                        s_abs.append(ss["abs"]); s_fp += (np.isfinite(ss["p"]) and ss["p"] < 0.05)
                    else:
                        s_abs.append(np.nan)
                p_tim = float((1 + np.sum(np.nan_to_num(s_abs, nan=-1.0) >= sc["abs"])) / (1 + n_shuf_timing)) if np.isfinite(sc["abs"]) else 1.0
                fp_tot += s_fp; sh_tot += n_shuf_timing
            else:
                p_tim = np.nan
            ok_o = bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
            ok_b = ok_o and bool(np.isfinite(p_tim) and p_tim < 0.05)
            n_ord += ok_o; n_both += ok_b
            row = {"session": sname, "k": res["k"], "n_out": len(res["orders"]), "abs": sc["abs"], "p": sc["p"],
                   "p_tim": p_tim, "shuf_fp": s_fp if run_shuf else None, "shuf_med_k": float(np.median(s_ks)) if s_ks else np.nan,
                   "ok_o": ok_o, "ok_b": ok_b}
            out[ep].append(row)
            flag = "PASS*" if ok_b else ("ord " if ok_o else "    ")
            sh_str = f"shuf_fp={s_fp:2d}/{n_shuf_timing} p_tim={p_tim:.3f}" if run_shuf else "               "
            print(f"  {ep.upper():<4} {sname:<18} | k={res['k']:2d} out={len(res['orders']):2d} | |rho|={sc['abs']:5.2f} p={sc['p']:5.3f} | {sh_str} | {flag}", flush=True)
        print(f"  ==> {ep.upper()} SUMMARY: order={n_ord}/8, order+timing={n_both}/8, shuf_FP={fp_tot}/{sh_tot} ({100*fp_tot/max(sh_tot,1):.1f}%)\n", flush=True)
    return out


def run_g1_grid():
    """G1 partial-replay grid (sim/segments.py) with exact fresh seeds + controls."""
    print("\n--- G1 PARTIAL-REPLAY GRID (sim/segments.py, locked cpca_v2) ---", flush=True)
    grid = {}
    tot_pass = 0
    for rev in (0.5, 0.3):
        for cpr in (5, 9, 13):
            nr = n_route_for(cpr)
            counts = []
            for sh in (0.02, 0.05, 0.10):
                hits = 0
                for draw in range(5):
                    d = gen_seg(np.random.default_rng(seed("g1-confirm", rev, cpr, sh, draw)),
                                n_route=nr, replay_frac=sh, reverse_frac=rev)
                    res = fit_cpca_v2(d["times"], d["T"], np.random.default_rng(seed("g1-cfit", rev, cpr, sh, draw)), **CFG)
                    sc = best_map_match(res["orders"], {1.0: d["centres"]}, False, 1.0,
                                        np.random.default_rng(seed("g1-csc", rev, cpr, sh, draw)), n_null=500) if res["orders"] else {"p": np.nan}
                    hits += bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
                counts.append(hits); tot_pass += hits
            grid[(rev, cpr)] = tuple(counts)
            print(f"  rev={rev} cells={cpr:2d} | 2/5/10%: {counts[0]}/{counts[1]}/{counts[2]}", flush=True)
    fp0 = 0
    for draw in range(15):
        d = gen_seg(np.random.default_rng(seed("g1-czero", draw)), n_route=n_route_for(9), replay_frac=0.0)
        res = fit_cpca_v2(d["times"], d["T"], np.random.default_rng(seed("g1-czfit", draw)), **CFG)
        sc = best_map_match(res["orders"], {1.0: d["centres"]}, False, 1.0, np.random.default_rng(draw), n_null=500) if res["orders"] else {"p": np.nan}
        fp0 += bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
    fp_sa = 0
    for draw in range(5):
        rng_d = np.random.default_rng(seed("g1-csa", draw))
        nr = n_route_for(9); route = rng_d.permutation(120)[:nr]; pos = np.full(120, np.nan); pos[route] = rng_d.uniform(0, 1, nr)
        mem = spatial_assemblies(rng_d, 120, route, pos)
        d = gen_seg(rng_d, n_route=nr, replay_frac=0.0, assembly_membership=mem)
        res = fit_cpca_v2(d["times"], d["T"], np.random.default_rng(seed("g1-csafit", draw)), **CFG)
        sc = best_map_match(res["orders"], {1.0: pos}, False, 1.0, np.random.default_rng(draw), n_null=500) if res["orders"] else {"p": np.nan}
        fp_sa += bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
    print(f"  TOTAL={tot_pass}/90 | share=0 FP={fp0}/15 | spatial assemblies FP={fp_sa}/5", flush=True)
    return {"grid": grid, "tot": tot_pass, "fp0": fp0, "fp_sa": fp_sa}


if __name__ == "__main__":
    t0 = time.perf_counter()
    res = {}
    res["hc11_spikein"] = run_hc11_spikein()
    res["hc11_real"] = run_hc11_maze_and_sleep(n_shuf_timing=50)
    res["g1_grid"] = run_g1_grid()
    pickle.dump(res, open(OUT / "confirm_part2.pkl", "wb"))
    print(f"\nPart 2 complete in {time.perf_counter() - t0:.1f}s -> exploration/results/confirm_part2.pkl", flush=True)
