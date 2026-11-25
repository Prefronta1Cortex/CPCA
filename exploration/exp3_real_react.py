"""Experiment 3: Real reactivations (hc-11 maze rest with timing control, hc-11 spike-in power test, hc-11 POST vs PRE).

Tests whether active-centring + active-phase shuffle null + wider arc alpha (+ fixed T vs event T)
1) eliminates co-firing false positives on hc-11 maze rest (timing-shuffled shuffles should be ~5%, not 98%!)
   while recovering genuine timing-driven order in MORE sessions;
2) lowers the spike-in replay detection threshold on real hc-11 PRE events below 10%;
3) improves POST vs PRE timing-specific recovery on hc-11 sleep.
"""

import sys
import time
import zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exploration.core import fit_clean
from cpca import event_spike_times, population_bursts
from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
from cpca.replay import immobile_intervals
from scripts.run_replay_hc11 import intersect, two_d
from scripts.plot_hc11_sequences import best_map_match
from scripts.run_maze_rest import HC11_ORDER, shuffle_timing
from scripts.run_rerun import ORDER_INDEX, fixed_subset
from scripts.run_spikein_hc11 import ideal as spikein_ideal


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def test_maze_rest(name, cfg, n_timing_shuffles=40):
    print(f"\n=== hc-11 MAZE REST (+ timing control, {n_timing_shuffles} shuffles): {name} ===", flush=True)
    n_ord, n_both, n_shuf_pass_tot, n_shuf_tot = 0, 0, 0, 0
    for sname in HC11_ORDER:
        path = next(p for p in session_paths() if p.name.startswith(sname))
        s = load_session(path)
        s["pos_xy"] = two_d(path)
        run = running(s)
        rest = intersect(immobile_intervals(s), [tuple(s["maze_epoch"])])
        by_cell, ids, pooled = pyramidal_spikes(s, rest)
        b, pk = population_bursts(pooled, rest)
        dur = np.diff(b, axis=1).ravel()
        times, _ = event_spike_times(by_cell, ids, pk - dur / 2, pk + dur / 2, max_spikes=60)
        T = dur * 1000
        centres = place_centres(s, ids, run)

        res = fit_clean(times, T, np.random.default_rng(seed("mrest", sname)), **cfg)
        sc = best_map_match(res["orders"], centres, run["circular"], run["length"],
                            np.random.default_rng(seed("mrest-sc", sname)), n_null=500) if res["orders"] else {"abs": np.nan, "p": np.nan}

        shuf_abs, shuf_p, shuf_k = [], [], []
        for d in range(n_timing_shuffles):
            srng = np.random.default_rng(seed("mrest-sh", sname, d))
            st = shuffle_timing(times, srng)
            r_s = fit_clean(st, T, srng, **cfg)
            shuf_k.append(r_s["n_kept"])
            if r_s["orders"]:
                ss = best_map_match(r_s["orders"], centres, run["circular"], run["length"], srng, n_null=200)
                shuf_abs.append(ss["abs"])
                shuf_p.append(ss["p"])
            else:
                shuf_abs.append(np.nan)
                shuf_p.append(np.nan)
        a = np.array(shuf_abs, float)
        p_arr = np.array(shuf_p, float)
        n_sp = int(np.sum(np.isfinite(p_arr) & (p_arr < 0.05)))
        n_shuf_pass_tot += n_sp
        n_shuf_tot += n_timing_shuffles
        p_tim = float((1 + np.sum(np.nan_to_num(a, nan=-1.0) >= sc["abs"])) / (1 + n_timing_shuffles)) if np.isfinite(sc["abs"]) else 1.0
        ord_ok = bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
        both_ok = ord_ok and (p_tim < 0.05)
        n_ord += ord_ok
        n_both += both_ok
        flag = "PASS*" if both_ok else ("ord " if ord_ok else "    ")
        print(f"  {sname:<18} | k={res['n_kept']:2d} (shuf med k={np.median(shuf_k):.1f}) | |rho|={sc['abs']:5.2f} p={sc['p']:5.3f} | shuf_pass={n_sp:2d}/{n_timing_shuffles} p_tim={p_tim:5.3f} | {flag}", flush=True)
    print(f"  SUMMARY: order={n_ord}/8, order+timing={n_both}/8, shuffled FP={n_shuf_pass_tot}/{n_shuf_tot} ({100*n_shuf_pass_tot/max(n_shuf_tot,1):.1f}%)", flush=True)


def test_spikein(name, cfg, sessions=("Achilles_10252013", "Achilles_11012013"), fracs=(0.0, 0.03, 0.05, 0.07, 0.10), n_draws=5):
    print(f"\n=== hc-11 SPIKE-IN POWER (ideal replays on PRE 1500): {name} ===", flush=True)
    totals = {f: [0, 0] for f in fracs}
    for sname in sessions:
        path = next(p for p in session_paths() if p.name.startswith(sname))
        s = load_session(path)
        run = running(s)
        iv = nrem_intervals(s, "pre")
        by_cell, ids, pooled = pyramidal_spikes(s, iv)
        b, pk = population_bursts(pooled, iv)
        dur = np.diff(b, axis=1).ravel()
        sub = fixed_subset(len(pk), 555 + 2 * ORDER_INDEX[sname] + 1)
        base, _ = event_spike_times(by_cell, ids, (pk - dur / 2)[sub], (pk + dur / 2)[sub], max_spikes=60)
        T = dur[sub] * 1000
        centres = place_centres(s, ids, run)
        row = []
        for f in fracs:
            k_ev = int(round(f * len(T)))
            hits = 0
            ks = []
            for d in range(n_draws):
                rng = np.random.default_rng(seed("spikein", sname, "ideal", f, d))
                chosen = rng.choice(len(T), k_ev, replace=False) if k_ev else np.array([], int)
                times = spikein_ideal(base, T, chosen, centres, rng) if k_ev else base
                res = fit_clean(times, T, np.random.default_rng(seed("exp3-sp", sname, f, d)), **cfg)
                ks.append(res["n_kept"])
                sc = best_map_match(res["orders"], centres, run["circular"], run["length"],
                                    np.random.default_rng(seed("exp3-sc", sname, f, d)), n_null=500) if res["orders"] else {"p": np.nan}
                if np.isfinite(sc["p"]) and sc["p"] < 0.05:
                    hits += 1
            totals[f][0] += hits
            totals[f][1] += n_draws
            row.append(f"{f:.0%}:{hits}/{n_draws}(k={np.median(ks):.0f})")
        print(f"  {sname:<18} | " + "  ".join(row), flush=True)
    print("  POOLED (2 largest) | " + "  ".join(f"{f:.0%}:{totals[f][0]}/{totals[f][1]}" for f in fracs), flush=True)


def test_post_pre(name, cfg):
    print(f"\n=== hc-11 SLEEP POST vs PRE (all NREM events): {name} ===", flush=True)
    for ep in ("post", "pre"):
        n_ord, n_ks = 0, []
        details = []
        for sname in HC11_ORDER:
            path = next(p for p in session_paths() if p.name.startswith(sname))
            s = load_session(path)
            run = running(s)
            iv = nrem_intervals(s, ep)
            by_cell, ids, pooled = pyramidal_spikes(s, iv)
            b, pk = population_bursts(pooled, iv)
            dur = np.diff(b, axis=1).ravel()
            times, _ = event_spike_times(by_cell, ids, pk - dur / 2, pk + dur / 2, max_spikes=60)
            T = dur * 1000
            centres = place_centres(s, ids, run)
            res = fit_clean(times, T, np.random.default_rng(seed("sleep", sname, ep)), **cfg)
            n_ks.append(res["n_kept"])
            sc = best_map_match(res["orders"], centres, run["circular"], run["length"],
                                np.random.default_rng(seed("sleep-sc", sname, ep)), n_null=500) if res["orders"] else {"abs": np.nan, "p": np.nan}
            ok = bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
            n_ord += ok
            details.append(f"{sname[:7]}:k={res['n_kept']}({sc['abs']:.2f},p={sc['p']:.3f})" if np.isfinite(sc["p"]) else f"{sname[:7]}:k=0")
        print(f"  {ep.upper():<4} | pass={n_ord}/8 | med k={np.median(n_ks):.1f} | " + " ".join(details), flush=True)


if __name__ == "__main__":
    t0 = time.perf_counter()
    configs = [
        ("1. Active-centred, alpha=pi, event-T, active-rms, ICA+merge",
         dict(alpha=np.pi, mode="active_centred", scale="active_rms", null_kind="active",
              ica=True, contrast="log", a=0.1, merge_conj=True, merge_thresh=0.5, member_rule="frac", member_frac=0.2)),
        ("2. Active-centred, alpha=1.6*pi, event-T, active-rms, ICA+merge",
         dict(alpha=1.6*np.pi, mode="active_centred", scale="active_rms", null_kind="active",
              ica=True, contrast="log", a=0.1, merge_conj=True, merge_thresh=0.5, member_rule="frac", member_frac=0.2)),
        ("3. Active-centred, alpha=1.6*pi, event-T, global-rms, ICA+merge",
         dict(alpha=1.6*np.pi, mode="active_centred", scale="global_rms", null_kind="active",
              ica=True, contrast="log", a=0.1, merge_conj=True, merge_thresh=0.5, member_rule="frac", member_frac=0.2)),
        ("4. Active-centred, alpha=1.6*pi, fixed-T=200, active-rms, ICA+merge",
         dict(alpha=1.6*np.pi, mode="active_centred", scale="active_rms", null_kind="active",
              fixed_T=200.0, ica=True, contrast="log", a=0.1, merge_conj=True, merge_thresh=0.5, member_rule="frac", member_frac=0.2)),
        ("5. Active-centred, alpha=1.6*pi, event-T, global-rms, PCA only (no ICA)",
         dict(alpha=1.6*np.pi, mode="active_centred", scale="global_rms", null_kind="active",
              ica=False, merge_conj=False, member_rule="frac", member_frac=0.2)),
    ]
    for name, cfg in configs:
        test_maze_rest(name, cfg, n_timing_shuffles=30)
        test_spikein(name, cfg)
        test_post_pre(name, cfg)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s", flush=True)
