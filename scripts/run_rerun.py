"""Matched rerun of the order-producing methods: cPCA, seqNMF (K x L grid), PP-Seq (corrected).

Every method gets exactly the same spikes: one fixed random subset of at most 1500 events per
dataset, each event a window of its own duration centred on the population peak (the cPCA
window). Scoring is identical for all three (cpca.evaluate.order_vs_place).

Usage:
    python scripts/run_rerun.py synthetic [A B ...] [--methods cpca,seqnmf,ppseq]
    python scripts/run_rerun.py hc11 [session ...]  [--methods cpca,seqnmf,ppseq]

Results: results/rerun/synthetic/<letter>.pkl and results/rerun/<session>.pkl; a method already
present in a result file is not recomputed (so PP-Seq and seqNMF can be run in separate passes).
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import seqnmf_baseline  # noqa: E402
from cpca import event_spike_times, fit, population_bursts  # noqa: E402
from cpca.evaluate import member_order_vs_place, order_vs_place  # noqa: E402
from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths  # noqa: E402
from scripts.run_baselines import auc, blocks_from_times  # noqa: E402
from sim.routes import CONDITIONS, generate, make_routes  # noqa: E402

OUT = ROOT / "results" / "rerun"
MAX_EVENTS = 1500
SEQ_KS = (5, 10, 15)
SEQ_LS_MS = (60, 120, 200)
SEQ_BIN_S = 0.010
METHODS = ("cpca", "seqnmf", "ppseq")


def parse_methods(argv):
    methods = METHODS
    rest = []
    for a in argv:
        if a.startswith("--methods"):
            methods = tuple(a.split("=", 1)[1].split(",")) if "=" in a else methods
        else:
            rest.append(a)
    return methods, rest


def fixed_subset(n_all, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_all, MAX_EVENTS, replace=False)) if n_all > MAX_EVENTS else np.arange(n_all)


def score_orders(orders, keys, circular, length, rng, n_null=300):
    """For each answer key: best (rho, null95, n_members, index) over the given cell-order vectors."""
    out = []
    for cen in keys:
        best = (0.0, np.nan, 0, -1)
        for j, lag in enumerate(orders):
            rho, n95, n = order_vs_place(lag, cen, circular, length, rng, n_null=n_null)
            if np.isfinite(rho) and abs(rho) > abs(best[0]):
                best = (rho, n95, n, j)
        out.append(best)
    return out


def map_tests(orders, centres, circular, length, rng, n_null=500):
    """hc-11: per order vector and running direction, (rho, null95, n)."""
    rows = []
    for j, lag in enumerate(orders):
        row = {"index": j, "map": {}}
        for sign in (1.0, -1.0):
            rho, n95, n = order_vs_place(lag, centres[sign], circular, length, rng, n_null=n_null)
            row["map"][sign] = {"rho": rho, "null95": n95, "n": n}
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- methods on one dataset
def run_cpca(times, T, rng):
    out = {}
    for label, settings in (("uncentred", dict(center=False, scale=False)), ("centred", dict(center=True, scale=True))):
        r = fit(times, T, rng=rng, ica=False, **settings)
        orders = []
        for j in range(r["n_kept"]):
            from cpca.decode import readout
            _, t, _, inc, _ = readout(r["patterns"][:, j], r["T_read_ms"], min_weight=0.2)
            orders.append(np.where(inc, t, np.nan))
        out[label] = {"orders": orders, "n_kept": r["n_kept"], "runtime_s": 0.0}
    return out


def run_seqnmf(times, T, rng):
    blocks_by_L = {}
    for L_ms in SEQ_LS_MS:
        L = int(round(L_ms / (SEQ_BIN_S * 1000)))
        blocks_by_L[L] = blocks_from_times(times, T, SEQ_BIN_S, pad=L)
    t0 = time.perf_counter()
    grid = seqnmf_baseline.run_grid(blocks_by_L, SEQ_BIN_S, rng, Ks=SEQ_KS, iters=80, restarts=2, lambda_iters=25, n_lambdas=7)
    out = []
    for g in grid:
        out.append({"K": g["K"], "L_ms": g["L_ms"], "lambda": g["lambda"], "runtime_s": g["runtime_s"],
                    "factors": [{"k": f["k"], "n_members": f["n_members"], "p": f["p"], "significant": f["significant"],
                                 "lag_span_ms": f["lag_span_ms"], "lag_ms": f["lag_ms"]} for f in g["factors"]]})
    return {"grid": out, "runtime_s": time.perf_counter() - t0}


def run_ppseq(times, T, rng):
    from baselines import ppseq_baseline as pp
    res = pp.fit(times, T, rng)
    return {"types": res["types"], "sequence_types": res["sequence_types"].tolist(), "best_setting": res["best_setting"], "sweep": res["sweep"],
            "p_event": res["p_event"], "p_event_sequence_types": res["p_event_sequence_types"],
            "n_events_fitted": res["n_events_fitted"], "n_spikes": res["n_spikes"], "runtime_s": res["runtime_s"]}


# ----------------------------------------------------------------------------- synthetic
def run_synthetic(only, methods):
    (OUT / "synthetic").mkdir(parents=True, exist_ok=True)
    for index, (name, (gen_kwargs, route_kwargs)) in enumerate(CONDITIONS.items()):
        if only and not any(name.startswith(o) for o in only):
            continue
        target_of = lambda m: OUT / "synthetic" / f"{name[0]}.{m}.pkl"
        out = {"condition": name}
        data_rng = np.random.default_rng(1000 * index)                 # identical draw to run_baselines.py
        routes = make_routes(data_rng, n_cells=80, **route_kwargs)
        data = generate(data_rng, n_cells=80, n_events=3000, routes=routes, **gen_kwargs)
        sub = fixed_subset(data["times"].shape[0], 777 + index)
        times, T = data["times"][sub], data["event_ms"]
        replay = (data["label"] > 0)[sub]
        out.update({"fraction_replay": float(replay.mean()), "n_events": int(len(sub))})
        rng = np.random.default_rng(index + 21)
        for method in methods:
            if target_of(method).exists():
                print(f"{name}: {method} cached", flush=True)
                continue
            t0 = time.perf_counter()
            print(f"{name}: {method}", flush=True)
            if method == "cpca":
                r = run_cpca(times, T, rng)
                for label in r:
                    r[label]["routes"] = score_orders(r[label]["orders"], data["centres"], False, 1.0, rng)
                out["cpca"] = r
                print("   " + " | ".join(f"{lab}: k={v['n_kept']} " + ", ".join(f"{abs(x[0]):.2f}[{x[1]:.2f}]" for x in v["routes"]) for lab, v in r.items()), flush=True)
            elif method == "seqnmf":
                r = run_seqnmf(times, T, rng)
                for g in r["grid"]:
                    g["routes"] = score_orders([f["lag_ms"] for f in g["factors"]], data["centres"], False, 1.0, rng)
                out["seqnmf"] = r
                print("   " + " | ".join(f"K={g['K']} L={g['L_ms']:.0f}: " + ", ".join(f"{abs(x[0]):.2f}[{x[1]:.2f}]" for x in g["routes"]) for g in r["grid"]) + f" ({r['runtime_s']:.0f}s)", flush=True)
            elif method == "ppseq":
                r = run_ppseq(times, T, rng)
                r["routes_type"] = score_orders([t["lag_ms"] for t in r["types"]], data["centres"], False, 1.0, rng)
                r["routes_package"] = score_orders([t["lag_ms_package"] for t in r["types"]], data["centres"], False, 1.0, rng)
                fitted = np.isfinite(r["p_event"])
                ps = r["p_event_sequence_types"]
                r["auc_replay"] = auc(ps[fitted], replay[fitted])
                r["auc_any_type"] = auc(r["p_event"][fitted], replay[fitted])
                r["flagged_replays"] = float(np.mean(ps[fitted & replay] > 0.5)) if (fitted & replay).any() else np.nan
                r["flagged_volleys"] = float(np.mean(ps[fitted & ~replay] > 0.5)) if (fitted & ~replay).any() else np.nan
                out["ppseq"] = r
                print(f"   R={r['best_setting']['num_sequence_types']} warp={'yes' if r['best_setting']['num_warp_values'] > 1 else 'no'} | type rule: " +
                      ", ".join(f"{abs(x[0]):.2f}[{x[1]:.2f}]" for x in r["routes_type"]) + " | package rule: " + ", ".join(f"{abs(x[0]):.2f}[{x[1]:.2f}]" for x in r["routes_package"]) +
                      f" | types: " + "; ".join(f"{t['n_members']} cells span {t['lag_span_ms']:.0f} ms ({t['mean_latent_events']:.0f} ev)" for t in r["types"]) +
                      f" | AUC {r['auc_replay']:.2f} ({r['runtime_s']:.0f}s)", flush=True)
            pickle.dump({k: v for k, v in out.items() if k not in METHODS or k == method}, open(target_of(method), "wb"))
        print(f"{name}: done", flush=True)


# ----------------------------------------------------------------------------- hc-11
def run_hc11(prefixes, methods):
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [p for p in session_paths() if not prefixes or any(p.name.startswith(x) for x in prefixes)]
    for path in paths:
        session = load_session(path)
        target_of = lambda m: OUT / f"{session['name']}.{m}.pkl"
        out = {"name": session["name"], "maze": session["maze"], "epochs": {}}
        done = {m: (pickle.load(open(target_of(m), "rb")) if target_of(m).exists() else None) for m in methods}
        run_info = running(session)
        out["circular"] = run_info["circular"]
        for ei, epoch in enumerate(("post", "pre")):
            intervals = nrem_intervals(session, epoch)
            by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
            bursts, peaks = population_bursts(pooled, intervals)
            duration = np.diff(bursts, axis=1).ravel()
            centres = place_centres(session, cell_ids, run_info)
            sub = fixed_subset(len(peaks), 555 + 2 * ORDER_INDEX.get(session["name"], 0) + ei)
            times, _ = event_spike_times(by_cell, cell_ids, (peaks - duration / 2)[sub], (peaks + duration / 2)[sub], max_spikes=60)
            T = duration[sub] * 1000
            block = out["epochs"].setdefault(epoch, {})
            block.update({"n_events_all": int(len(peaks)), "n_events": int(len(sub)), "n_cells": len(cell_ids),
                          "n_place": {s: int(np.isfinite(c).sum()) for s, c in centres.items()}})
            rng = np.random.default_rng(31 + ei)
            for method in methods:
                if done[method] is not None and method in done[method]["epochs"].get(epoch, {}):
                    print(f"[{session['name']}] {epoch}: {method} cached", flush=True)
                    continue
                print(f"[{session['name']}] {epoch}: {method} on {len(sub)} events", flush=True)
                if method == "cpca":
                    r = run_cpca(times, T, rng)
                    for label in r:
                        r[label]["tests"] = map_tests(r[label]["orders"], centres, run_info["circular"], run_info["length"], rng)
                        r[label].pop("orders")
                    block["cpca"] = r
                    print("   " + " | ".join(f"{lab}: k={v['n_kept']}, hits {sum(np.isfinite(m['rho']) and abs(m['rho']) > m['null95'] for c in v['tests'] for m in c['map'].values())}/{sum(np.isfinite(m['rho']) for c in v['tests'] for m in c['map'].values())}" for lab, v in r.items()), flush=True)
                elif method == "seqnmf":
                    r = run_seqnmf(times, T, rng)
                    for g in r["grid"]:
                        g["tests"] = map_tests([f["lag_ms"] for f in g["factors"]], centres, run_info["circular"], run_info["length"], rng)
                    block["seqnmf"] = r
                    print("   " + " | ".join(f"K={g['K']} L={g['L_ms']:.0f}: hits {sum(np.isfinite(m['rho']) and abs(m['rho']) > m['null95'] for c in g['tests'] for m in c['map'].values())}/{sum(np.isfinite(m['rho']) for c in g['tests'] for m in c['map'].values())}, sig {sum(f['significant'] for f in g['factors'])}" for g in r["grid"]) + f" ({r['runtime_s']:.0f}s)", flush=True)
                elif method == "ppseq":
                    r = run_ppseq(times, T, rng)
                    r["tests_type"] = map_tests([t["lag_ms"] for t in r["types"]], centres, run_info["circular"], run_info["length"], rng)
                    r["tests_package"] = map_tests([t["lag_ms_package"] for t in r["types"]], centres, run_info["circular"], run_info["length"], rng)
                    fitted = np.isfinite(r["p_event"])
                    r["flagged_fraction"] = float(np.mean(r["p_event_sequence_types"][fitted] > 0.5))
                    r.pop("p_event"); r.pop("p_event_sequence_types")
                    block["ppseq"] = r
                    hits = lambda tests: f"{sum(np.isfinite(m['rho']) and abs(m['rho']) > m['null95'] for c in tests for m in c['map'].values())}/{sum(np.isfinite(m['rho']) for c in tests for m in c['map'].values())}"
                    print(f"   R={r['best_setting']['num_sequence_types']} warp={'yes' if r['best_setting']['num_warp_values'] > 1 else 'no'} | hits type rule {hits(r['tests_type'])}, package rule {hits(r['tests_package'])} | types: " +
                          "; ".join(f"{t['n_members']}/{t['n_members_package']} cells span {t['lag_span_ms']:.0f} ms ({t['mean_latent_events']:.0f} ev)" for t in r["types"]) + f" ({r['runtime_s']:.0f}s)", flush=True)
        for method in methods:
            if done[method] is None:
                keep = {"name": out["name"], "maze": out["maze"], "circular": out["circular"],
                        "epochs": {ep: {k: v for k, v in blk.items() if k == method or k not in METHODS} for ep, blk in out["epochs"].items()}}
                pickle.dump(keep, open(target_of(method), "wb"))
        print(f"[{session['name']}] done", flush=True)


ORDER_INDEX = {n: i for i, n in enumerate(["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
                                           "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"])}

if __name__ == "__main__":
    methods, rest = parse_methods(sys.argv[1:])
    mode = rest[0] if rest else "synthetic"
    if mode == "synthetic":
        run_synthetic(rest[1:], methods)
    else:
        run_hc11(rest[1:], methods)
