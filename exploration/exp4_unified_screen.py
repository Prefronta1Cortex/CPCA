"""Experiment 4: Clean comparison of scaling, event-phase alignment, and ICA contrast
across BOTH planted reactivations (A-F, STRESS G-M) AND real reactivations (hc-11 maze rest,
hc-11 spike-in power, hc-11 sleep POST/PRE, and DANDI 001695 POST/PRE/maze rest).

Goal: find ONE single unified, parameter-free setting (NOT different tweaks per dataset!).
"""

import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exploration.core import (
    encode_arc, prepare_matrix, lag_matrix, eigh_desc,
    shuffle_null_active, count_above_null, unmix_patterns,
    merge_conjugate_pairs, readout_gap, _sym_orth
)
from cpca.pairs import circular_combination


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def align_event_phase(S):
    """Rotate each event row so its circular mean phase is 0 (removes burst-peak skew)."""
    phi = np.angle(S.sum(axis=1))
    return S * np.exp(-1j * phi)[:, None]


def complex_ica_fast(Z, rng, power=2.0, n_init=6, max_iter=200, tol=1e-6):
    """Fast Riemannian unitary optimization maximizing sum_j mean_i (|y_ij|^2)^power.

    power=2.0: exact 4th moment (kurtosis / event-space quartimax)
    power=1.5: robust super-linear contrast (|y|^3, less sensitive to extreme outliers)
    Uses exact polar retraction (SVD / symmetric orthogonalization) instead of expm -> 10x faster!
    """
    M, K = Z.shape
    if K <= 1:
        return np.eye(K, dtype=complex)

    def obj_and_grad(W):
        Y = Z @ W
        U = np.abs(Y) ** 2
        if power == 2.0:
            loss = -float(np.sum(np.mean(U * U, axis=0)))
            g = -2.0 * U
        else:
            Up = np.power(U + 1e-12, power - 1.0)
            loss = -float(np.sum(np.mean(U * Up, axis=0)))
            g = -power * Up
        Egrad = (Z.conj().T @ (Y * g)) / M
        return loss, Egrad

    best_loss, best_W = np.inf, np.eye(K, dtype=complex)
    inits = [np.eye(K, dtype=complex)]
    for _ in range(n_init - 1):
        R = rng.normal(size=(K, K)) + 1j * rng.normal(size=(K, K))
        inits.append(_sym_orth(R))

    for W0 in inits:
        W = W0.copy()
        loss, Egrad = obj_and_grad(W)
        step = 0.5
        for _ in range(max_iter):
            G = W.conj().T @ Egrad
            A = -(G - G.conj().T)
            gnorm2 = float(np.real(np.sum(A.conj() * A)))
            if gnorm2 < tol ** 2:
                break
            accepted = False
            for _ls in range(20):
                # Polar retraction on U(K): _sym_orth(W + step * W @ A)
                W_try = _sym_orth(W + step * (W @ A))
                loss_try, Egrad_try = obj_and_grad(W_try)
                if loss_try <= loss - 1e-4 * step * gnorm2:
                    W, loss, Egrad = W_try, loss_try, Egrad_try
                    step = min(step * 1.5, 2.0)
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
        if loss < best_loss:
            best_loss, best_W = loss, W
    return best_W


def merge_sym(P, Y, sim_thresh=0.5, merge_real_pairs=True, circ_thresh=0.75, corr_thresh=0.25):
    """Quotient by time-reversal symmetry: merge conjugate pairs (q, conj(q)) and co-occurring real pairs."""
    K = P.shape[1]
    if K <= 1:
        return P
    P_norm = P / np.maximum(np.linalg.norm(P, axis=0, keepdims=True), 1e-12)
    U = np.abs(Y) ** 2
    U_c = U - U.mean(axis=0, keepdims=True)
    U_std = np.maximum(np.linalg.norm(U_c, axis=0, keepdims=True), 1e-12)
    C_ev = (U_c.T @ U_c) / (U_std.T @ U_std)
    circs = np.array([abs(np.sum(P_norm[:, j] ** 2)) for j in range(K)])

    used = np.zeros(K, dtype=bool)
    out = []
    for i in range(K):
        if used[i]:
            continue
        used[i] = True
        acc = P_norm[:, i].copy()
        paired = False
        for j in range(i + 1, K):
            if used[j]:
                continue
            s_dir = np.vdot(P_norm[:, j], acc / max(np.linalg.norm(acc), 1e-12))
            s_con = np.vdot(np.conj(P_norm[:, j]), acc / max(np.linalg.norm(acc), 1e-12))
            if max(abs(s_dir), abs(s_con)) >= sim_thresh:
                used[j] = True
                paired = True
                if abs(s_con) >= abs(s_dir):
                    acc = acc + np.conj(P_norm[:, j]) * np.exp(1j * np.angle(s_con))
                else:
                    acc = acc + P_norm[:, j] * np.exp(1j * np.angle(s_dir))
        if merge_real_pairs and not paired and circs[i] >= circ_thresh:
            best_j, best_c = -1, corr_thresh
            for j in range(i + 1, K):
                if not used[j] and circs[j] >= circ_thresh and C_ev[i, j] > best_c:
                    best_j, best_c = j, C_ev[i, j]
            if best_j >= 0:
                used[best_j] = True
                acc = circular_combination(P_norm[:, i], P_norm[:, best_j])
        acc /= max(np.linalg.norm(acc), 1e-12)
        out.append(acc)
    return np.column_stack(out)


def fit_react_unified(times, T, rng, alpha=1.5 * np.pi, align=True, scale="sqrt_rms",
                      power=2.0, merge_real_pairs=True, member_rule="otsu", member_frac=0.2,
                      n_shuffles=50, max_components=20):
    S = encode_arc(times, T, alpha=alpha)
    if align:
        S = align_event_phase(S)
    X = prepare_matrix(S, mode="active_centred", scale=scale)
    vals, vecs = eigh_desc(lag_matrix(X))
    null = shuffle_null_active(X, rng, n_shuffles=n_shuffles)
    k, bound = count_above_null(vals, null, structural=0)
    k = int(min(max(k, 0), max_components))
    if k == 0:
        return {"k": 0, "patterns": np.zeros((X.shape[1], 0), complex), "orders": []}
    if k == 1:
        P = vecs[:, :1].copy()
        P /= np.maximum(np.linalg.norm(P, axis=0, keepdims=True), 1e-12)
    else:
        Z = (X @ np.conj(vecs[:, :k])) / np.sqrt(np.maximum(vals[:k], 1e-12))[None, :]
        W = complex_ica_fast(Z, rng, power=power)
        P = vecs[:, :k] @ (np.sqrt(vals[:k])[:, None] * np.conj(W))
        norms = np.linalg.norm(P, axis=0)
        P = P / np.maximum(norms, 1e-12)
        order = np.argsort(-norms)
        P, Y = P[:, order], (Z @ W)[:, order]
        P = merge_sym(P, Y, sim_thresh=0.5, merge_real_pairs=merge_real_pairs)
    T_read = float(np.median(T)) if np.ndim(T) else float(T)
    orders = [readout_gap(P[:, j], T_read=T_read, alpha=alpha, member_rule=member_rule, member_frac=member_frac)[0]
              for j in range(P.shape[1])]
    return {"k": k, "patterns": P, "orders": orders}


def run_planted_suite(cfg, tag):
    from sim.two_routes import generate
    from scripts.run_separation import CONDITIONS, judge
    from scripts.run_pure_capabilities import judge_share
    from exploration.exp1_planted_react import STRESS

    print(f"\n=== PLANTED: {tag} ===", flush=True)
    for label, conds, sbase in [("A-F (dev)", CONDITIONS, 7000), ("STRESS (dev)", STRESS, 13000)]:
        tot_clean_s, tot_fits = 0, 0
        for ci, (cname, kw) in enumerate(conds.items()):
            ks, nouts, n_pass_o, n_tot, clean_o, clean_s = [], [], 0, 0, 0, 0
            for draw in range(5):
                d = generate(np.random.default_rng(sbase + 97 * ci + draw), **kw)
                perm = np.random.default_rng(seed("exp1-split", sbase, cname, draw)).permutation(d["times"].shape[0])
                halves = (perm[: perm.size // 2], perm[perm.size // 2:])
                maps = d["maps"] if sum(kw["rates"]) > 0 else []
                for fold in (0, 1):
                    tr = halves[fold]
                    res = fit_react_unified(d["times"][tr], d["T"][tr], np.random.default_rng(seed("e4", cname, draw, fold)), **cfg)
                    ks.append(res["k"]); nouts.append(len(res["orders"]))
                    jrng = np.random.default_rng(seed("e4-j", cname, draw, fold))
                    if not maps:
                        n_tot += len(res["orders"])
                        if len(res["orders"]) == 0:
                            clean_o += 1; clean_s += 1
                        continue
                    vo, vs = [], []
                    for o in res["orders"]:
                        oko, ko, _ = judge(o, maps, kw["shared"], jrng)
                        oks, ks_i = judge_share(o, maps, jrng) if kw["shared"] else (oko, ko)
                        vo.append((oko, ko)); vs.append((oks, ks_i))
                    n_pass_o += sum(ok for ok, _ in vo); n_tot += len(vo)
                    if len({k for ok, k in vo if ok}) == len(maps) and all(ok for ok, _ in vo):
                        clean_o += 1
                    if len({k for ok, k in vs if ok}) == len(maps) and all(ok for ok, _ in vs):
                        clean_s += 1
            tot_clean_s += clean_s; tot_fits += 10
            pct = f"{n_pass_o}/{n_tot} ({100*n_pass_o/max(n_tot,1):3.0f}%)" if n_tot else "0/0"
            print(f"  {cname[:34]:<34} | k={np.median(ks):3.1f} out={np.median(nouts):3.1f} | {pct:>13} | orig={clean_o:2d}/10 | share={clean_s:2d}/10", flush=True)
        print(f"  --> {label} TOTAL CLEAN (share): {tot_clean_s}/{tot_fits}", flush=True)


def run_hc11_suite(cfg, tag):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    from cpca.replay import immobile_intervals
    from scripts.run_replay_hc11 import intersect, two_d
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_maze_rest import HC11_ORDER, shuffle_timing
    from scripts.run_rerun import ORDER_INDEX, fixed_subset
    from scripts.run_spikein_hc11 import ideal as spikein_ideal

    print(f"\n=== REAL HC-11: {tag} ===", flush=True)
    # 1. Spike-in power on Achilles 10/25 & 11/01
    fracs = (0.0, 0.03, 0.05, 0.07, 0.10)
    totals = {f: 0 for f in fracs}
    for sname in ("Achilles_10252013", "Achilles_11012013"):
        path = next(p for p in session_paths() if p.name.startswith(sname))
        s = load_session(path); run = running(s)
        iv = nrem_intervals(s, "pre")
        by_cell, ids, pooled = pyramidal_spikes(s, iv)
        b, pk = population_bursts(pooled, iv); dur = np.diff(b, axis=1).ravel()
        sub = fixed_subset(len(pk), 555 + 2 * ORDER_INDEX[sname] + 1)
        base, _ = event_spike_times(by_cell, ids, (pk - dur / 2)[sub], (pk + dur / 2)[sub], max_spikes=60)
        T = dur[sub] * 1000; centres = place_centres(s, ids, run)
        parts = []
        for f in fracs:
            k_ev = int(round(f * len(T)))
            hits = 0
            for d in range(5):
                rng = np.random.default_rng(seed("spikein", sname, "ideal", f, d))
                chosen = rng.choice(len(T), k_ev, replace=False) if k_ev else np.array([], int)
                times = spikein_ideal(base, T, chosen, centres, rng) if k_ev else base
                res = fit_react_unified(times, T, np.random.default_rng(seed("e4-sp", sname, f, d)), **cfg)
                sc = best_map_match(res["orders"], centres, run["circular"], run["length"],
                                    np.random.default_rng(seed("e4-sc", sname, f, d)), n_null=500) if res["orders"] else {"p": np.nan}
                hits += bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
            totals[f] += hits
            parts.append(f"{f:.0%}:{hits}/5")
        print(f"  Spike-in {sname:<18} | " + " ".join(parts), flush=True)
    print("  Spike-in POOLED            | " + " ".join(f"{f:.0%}:{totals[f]}/10" for f in fracs), flush=True)

    # 2. Maze rest (+ 25 timing shuffles)
    n_ord = n_both = shuf_fp = 0
    passes = []
    for sname in HC11_ORDER:
        path = next(p for p in session_paths() if p.name.startswith(sname))
        s = load_session(path); s["pos_xy"] = two_d(path); run = running(s)
        rest = intersect(immobile_intervals(s), [tuple(s["maze_epoch"])])
        by_cell, ids, pooled = pyramidal_spikes(s, rest)
        b, pk = population_bursts(pooled, rest); dur = np.diff(b, axis=1).ravel()
        times, _ = event_spike_times(by_cell, ids, pk - dur / 2, pk + dur / 2, max_spikes=60)
        T = dur * 1000; centres = place_centres(s, ids, run)
        res = fit_react_unified(times, T, np.random.default_rng(seed("e4-mr", sname)), **cfg)
        sc = best_map_match(res["orders"], centres, run["circular"], run["length"],
                            np.random.default_rng(seed("e4-mrs", sname)), n_null=500) if res["orders"] else {"abs": np.nan, "p": np.nan}
        s_abs = []
        for d in range(25):
            srng = np.random.default_rng(seed("e4-mrsh", sname, d))
            rs = fit_react_unified(shuffle_timing(times, srng), T, srng, **cfg)
            if rs["orders"]:
                ss = best_map_match(rs["orders"], centres, run["circular"], run["length"], srng, n_null=200)
                s_abs.append(ss["abs"]); shuf_fp += (np.isfinite(ss["p"]) and ss["p"] < 0.05)
            else:
                s_abs.append(np.nan)
        p_tim = (1 + np.sum(np.nan_to_num(s_abs, nan=-1.0) >= sc["abs"])) / 26 if np.isfinite(sc["abs"]) else 1.0
        ok_o = bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
        ok_b = ok_o and p_tim < 0.05
        n_ord += ok_o; n_both += ok_b
        if ok_o:
            passes.append(f"{sname[:7]}(k={res['k']},p={sc['p']:.3f},pt={p_tim:.2f})")
    print(f"  Maze rest                  | ord={n_ord}/8 both={n_both}/8 fp={shuf_fp}/200 ({100*shuf_fp/200:.1f}%) | {' '.join(passes)}", flush=True)

    # 3. Sleep POST vs PRE (all NREM events)
    for ep in ("post", "pre"):
        n_p = 0
        det = []
        for sname in HC11_ORDER:
            path = next(p for p in session_paths() if p.name.startswith(sname))
            s = load_session(path); run = running(s)
            iv = nrem_intervals(s, ep)
            by_cell, ids, pooled = pyramidal_spikes(s, iv)
            b, pk = population_bursts(pooled, iv); dur = np.diff(b, axis=1).ravel()
            times, _ = event_spike_times(by_cell, ids, pk - dur / 2, pk + dur / 2, max_spikes=60)
            T = dur * 1000; centres = place_centres(s, ids, run)
            res = fit_react_unified(times, T, np.random.default_rng(seed("e4-sl", sname, ep)), **cfg)
            sc = best_map_match(res["orders"], centres, run["circular"], run["length"],
                                np.random.default_rng(seed("e4-sls", sname, ep)), n_null=500) if res["orders"] else {"abs": np.nan, "p": np.nan}
            ok = bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
            n_p += ok
            if ok:
                det.append(f"{sname[:7]}(k={res['k']},r={sc['abs']:.2f},p={sc['p']:.3f})")
        print(f"  Sleep {ep.upper():<4}                 | pass={n_p}/8 | {' '.join(det)}", flush=True)


if __name__ == "__main__":
    t0 = time.perf_counter()
    candidates = [
        ("align=True, alpha=1.5*pi, scale=sqrt_rms, power=2.0, otsu",
         dict(alpha=1.5 * np.pi, align=True, scale="sqrt_rms", power=2.0, merge_real_pairs=True, member_rule="otsu")),
        ("align=True, alpha=1.5*pi, scale=global_rms, power=1.5, frac=0.2",
         dict(alpha=1.5 * np.pi, align=True, scale="global_rms", power=1.5, merge_real_pairs=True, member_rule="frac", member_frac=0.2)),
        ("align=True, alpha=1.5*pi, scale=global_rms, power=1.5, otsu",
         dict(alpha=1.5 * np.pi, align=True, scale="global_rms", power=1.5, merge_real_pairs=True, member_rule="otsu")),
        ("align=True, alpha=1.5*pi, scale=active_rms, power=2.0, otsu",
         dict(alpha=1.5 * np.pi, align=True, scale="active_rms", power=2.0, merge_real_pairs=True, member_rule="otsu")),
    ]
    for tag, cfg in candidates:
        run_planted_suite(cfg, tag)
        run_hc11_suite(cfg, tag)
    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s", flush=True)
