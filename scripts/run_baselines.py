"""Baseline comparison: seqNMF, HMM (Maboudi), Bayesian decoding (map-using reference), count PCA/ICA, and cPCA.

Usage:
    python scripts/run_baselines.py synthetic            # planted routes, one draw per condition
    python scripts/run_baselines.py hc11 [session ...]   # all sessions, POST and PRE

Every method is scored on its own output with the same test:
- cPCA components and seqNMF factors give a cell order; it is compared with the
  answer key (planted positions or place-field centres) by Spearman against a
  cell-permutation null, per component/factor and (for hc-11) per running direction.
- The HMM gives per-event congruence (sequential vs not) and a latent map; it is
  scored by the fraction of congruent events (synthetic: AUC against the planted
  replay labels) and by latent-state place-field decoding error vs shuffled.
- Bayesian decoding uses the map as input and is reported as the reference ceiling.
Excluded, with reasons recorded in docs/baselines.md: PP-Seq (Julia not available),
convSeq (research code without a packaged API).
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import bayes_reference, hmm_baseline, seqnmf_baseline  # noqa: E402
from baselines.count_pca_ica import assemblies, event_counts  # noqa: E402
from cpca import event_spike_times, fit, population_bursts  # noqa: E402
from cpca.evaluate import member_order_vs_place, order_vs_place  # noqa: E402
from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths  # noqa: E402
from sim.routes import CONDITIONS, generate, make_routes  # noqa: E402

OUT = ROOT / "results" / "baselines"
SEQ_L = 12          # 120 ms at 10 ms bins


def blocks_from_times(times_ms, T_ms, bin_s, pad=0):
    """Binned counts per event from a (events, cells, spikes) array; T_ms per event or scalar."""
    n_events, n_cells, _ = times_ms.shape
    T_ms = np.broadcast_to(np.asarray(T_ms, dtype=float), (n_events,))
    blocks = []
    for i in range(n_events):
        nb = max(int(np.ceil(T_ms[i] / (bin_s * 1000))), 1)
        edges = np.linspace(0, T_ms[i], nb + 1)
        X = np.zeros((nb + pad, n_cells))
        for k in range(n_cells):
            t = times_ms[i, k]
            t = t[np.isfinite(t)]
            if t.size:
                X[:nb, k] = np.histogram(t, bins=edges)[0]
        blocks.append(X)
    return blocks


def auc(scores, labels):
    """Area under the ROC curve (rank statistic); larger scores = more replay-like."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.all() or not y.any():
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


# ----------------------------------------------------------------------------- synthetic
def run_synthetic():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for index, (name, (gen_kwargs, route_kwargs)) in enumerate(CONDITIONS.items()):
        t0 = time.perf_counter()
        data_rng = np.random.default_rng(1000 * index)
        rng = np.random.default_rng(index + 1)
        routes = make_routes(data_rng, n_cells=80, **route_kwargs)
        data = generate(data_rng, n_cells=80, n_events=3000, routes=routes, **gen_kwargs)
        T = data["event_ms"]
        replay = data["label"] > 0
        res = {"condition": name, "fraction_replay": float(replay.mean()), "methods": {}}
        # cPCA (both conventions), scored per route on its components
        for label, settings in (("cPCA uncentred", dict(center=False, scale=False)), ("cPCA centred", dict(center=True, scale=True))):
            r = fit(data["times"], T, rng=rng, ica=False, **settings)
            per_route = []
            for cen in data["centres"]:
                best = (0.0, np.nan)
                for j in range(r["n_kept"]):
                    rho, n95, n = member_order_vs_place(r["patterns"][:, j], r["T_read_ms"], cen, False, 1.0, rng, n_null=200)
                    if np.isfinite(rho) and abs(rho) > abs(best[0]):
                        best = (rho, n95)
                per_route.append(best)
            res["methods"][label] = {"k": r["n_kept"], "routes": per_route}
        # seqNMF
        blocks10 = blocks_from_times(data["times"], T, 0.010, pad=SEQ_L)
        sq = seqnmf_baseline.run(blocks10, 0.010, rng, L=SEQ_L)
        per_route = []
        for cen in data["centres"]:
            best = (0.0, np.nan, False)
            for f in sq["factors"]:
                rho, n95, n = order_vs_place(f["lag_ms"], cen, False, 1.0, rng, n_null=200)
                if np.isfinite(rho) and abs(rho) > abs(best[0]):
                    best = (rho, n95, f["significant"])
            per_route.append(best)
        res["methods"]["seqNMF"] = {"n_factors": len(sq["factors"]), "n_significant": sum(f["significant"] for f in sq["factors"]),
                                    "lambda": sq["lambda"], "routes": per_route, "runtime_s": sq["runtime_s"]}
        # HMM: congruence as a replay detector
        blocks20 = blocks_from_times(data["times"], T, 0.020)
        hm = hmm_baseline.run(blocks20, rng, n_states=30, n_shuffles=60)
        usable_mask = np.array([b.shape[0] >= 2 and b.sum() > 0 for b in blocks20])
        res["methods"]["HMM"] = {"congruent_99": hm["congruent_99"], "congruent_95": hm["congruent_95"],
                                 "auc_replay": auc(-hm["p_values"], replay[usable_mask]), "cv_gain": hm["cv_gain"], "runtime_s": hm["runtime_s"]}
        # count PCA/ICA: number of assemblies (no order)
        _, n_asm, _, _ = assemblies(event_counts(data["times"]), rng)
        res["methods"]["count PCA/ICA"] = {"n_assemblies": n_asm}
        results.append(res)
        print(f"{name}: " + " | ".join(
            [f"{m}: " + ", ".join(f"{abs(r[0]):.2f}[{r[1]:.2f}]" for r in v["routes"]) for m, v in res["methods"].items() if "routes" in v]
            + [f"HMM AUC {res['methods']['HMM']['auc_replay']:.2f}, congruent {100*res['methods']['HMM']['congruent_99']:.0f}%"])
            + f" | {time.perf_counter() - t0:.0f}s", flush=True)
    with open(OUT / "synthetic.pkl", "wb") as fh:
        pickle.dump(results, fh)


# ----------------------------------------------------------------------------- hc-11
def event_blocks(by_cell, cell_ids, bursts, bin_s, pad=0, max_spikes=60):
    starts, ends = bursts[:, 0], bursts[:, 1]
    times, _ = event_spike_times(by_cell, cell_ids, starts, ends, max_spikes=max_spikes)
    return blocks_from_times(times, (ends - starts) * 1000, bin_s, pad=pad)


def run_hc11(prefixes):
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [p for p in session_paths() if not prefixes or any(p.name.startswith(x) for x in prefixes)]
    for path in paths:
        t_all = time.perf_counter()
        session = load_session(path)
        run_info = running(session)
        out = {"name": session["name"], "maze": session["maze"], "circular": run_info["circular"], "epochs": {}}
        for epoch in ("post", "pre"):
            rng = np.random.default_rng(0)
            intervals = nrem_intervals(session, epoch)
            by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
            bursts, peaks = population_bursts(pooled, intervals)
            centres = place_centres(session, cell_ids, run_info)
            block = {"n_events": len(peaks), "n_cells": len(cell_ids), "n_place": {s: int(np.isfinite(c).sum()) for s, c in centres.items()}}
            # seqNMF on the full events
            blocks10 = event_blocks(by_cell, cell_ids, bursts, 0.010, pad=SEQ_L)
            sq = seqnmf_baseline.run(blocks10, 0.010, rng, L=SEQ_L)
            factors = []
            for f in sq["factors"]:
                row = {"k": f["k"], "n_members": f["n_members"], "p": f["p"], "significant": f["significant"], "lag_span_ms": f["lag_span_ms"],
                       "lag_ms": f["lag_ms"], "map": {}}
                for sign in (1.0, -1.0):
                    rho, n95, n = order_vs_place(f["lag_ms"], centres[sign], run_info["circular"], run_info["length"], rng, n_null=500)
                    row["map"][sign] = {"rho": rho, "null95": n95, "n": n}
                factors.append(row)
            block["seqNMF"] = {"factors": factors, "lambda": sq["lambda"], "sweep": sq["sweep"], "runtime_s": sq["runtime_s"]}
            # HMM (Maboudi settings)
            blocks20 = event_blocks(by_cell, cell_ids, bursts, 0.020)
            hm = hmm_baseline.run(blocks20, rng, n_states=30, session=session, run_info=run_info, cell_ids=cell_ids, n_shuffles=100)
            hm.pop("p_values")
            block["HMM"] = hm
            # Bayesian decoding reference (uses the map)
            maps, place, bin_centres = bayes_reference.rate_maps(session, cell_ids, run_info)
            block["Bayes"] = bayes_reference.run(blocks20, maps, place, bin_centres, rng)
            out["epochs"][epoch] = block
            sig = [f for f in factors if f["significant"]]
            hits = sum(np.isfinite(f["map"][s]["rho"]) and abs(f["map"][s]["rho"]) > f["map"][s]["null95"] for f in factors for s in f["map"])
            print(f"[{session['name']}] {epoch}: seqNMF {len(sig)}/{len(factors)} significant factors, map hits {hits}/{sum(np.isfinite(f['map'][s]['rho']) for f in factors for s in f['map'])} ({sq['runtime_s']:.0f}s) | "
                  f"HMM congruent 99%: {100*hm['congruent_99']:.1f}%, lsPF error {hm['lspf']['median_error_m']:.2f} vs shuffled {hm['lspf']['shuffled_median_error_m']:.2f} m ({hm['runtime_s']:.0f}s) | "
                  f"Bayes sig {100*block['Bayes']['frac_sig']:.1f}% strict {100*block['Bayes']['frac_strict']:.1f}%", flush=True)
        with open(OUT / f"{session['name']}.pkl", "wb") as fh:
            pickle.dump(out, fh)
        print(f"[{session['name']}] saved ({time.perf_counter() - t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    if mode == "synthetic":
        run_synthetic()
    else:
        run_hc11(sys.argv[2:])
