"""Add PP-Seq to the method comparison without re-running the other methods.

Usage:
    python scripts/run_ppseq.py synthetic [A B ...]     # same planted data as run_baselines.py (same seeds)
    python scripts/run_ppseq.py hc11 [session ...]      # same events as run_baselines.py, POST and PRE

Writes results/baselines/ppseq_synthetic/<letter>.pkl and results/baselines/ppseq_<session>.pkl.
Scoring mirrors the other order-producing methods: for each sequence type, the neuron
offsets of its members are compared with the answer key by order_vs_place; per event,
the posterior probability of containing a latent sequence gives a detection AUC against
planted replays (synthetic) or the fraction of events flagged (hc-11).
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import ppseq_baseline as pp  # noqa: E402
from cpca import event_spike_times, population_bursts  # noqa: E402
from cpca.evaluate import order_vs_place  # noqa: E402
from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths  # noqa: E402
from scripts.run_baselines import auc  # noqa: E402
from sim.routes import CONDITIONS, generate, make_routes  # noqa: E402

OUT = ROOT / "results" / "baselines"


def score_types(types, centres_list, circular, length, rng):
    """For each answer-key vector, the best (rho, null95, n, type index) over sequence types."""
    out = []
    for cen in centres_list:
        best = (0.0, np.nan, 0, -1)
        for t in types:
            rho, n95, n = order_vs_place(t["lag_ms"], cen, circular, length, rng, n_null=300)
            if np.isfinite(rho) and abs(rho) > abs(best[0]):
                best = (rho, n95, n, t["r"])
        out.append(best)
    return out


def run_synthetic(only):
    (OUT / "ppseq_synthetic").mkdir(parents=True, exist_ok=True)
    for index, (name, (gen_kwargs, route_kwargs)) in enumerate(CONDITIONS.items()):
        if only and not any(name.startswith(o) for o in only):
            continue
        target = OUT / "ppseq_synthetic" / f"{name[0]}.pkl"
        if target.exists():
            print(f"{name}: cached", flush=True)
            continue
        t0 = time.perf_counter()
        data_rng = np.random.default_rng(1000 * index)                # identical draw to run_baselines.py
        rng = np.random.default_rng(index + 11)
        routes = make_routes(data_rng, n_cells=80, **route_kwargs)
        data = generate(data_rng, n_cells=80, n_events=3000, routes=routes, **gen_kwargs)
        replay = data["label"] > 0
        print(f"{name}: fitting PP-Seq", flush=True)
        res = pp.fit(data["times"], data["event_ms"], rng)
        routes_scored = score_types(res["types"], data["centres"], False, 1.0, rng)
        fitted = np.isfinite(res["p_event"])
        ps = res["p_event_sequence_types"]
        out = {"condition": name, "fraction_replay": float(replay.mean()), "routes": routes_scored,
               "types": [{k: v for k, v in t.items() if k != "members"} for t in res["types"]], "sequence_types": res["sequence_types"].tolist(),
               "best_setting": res["best_setting"], "sweep": res["sweep"], "n_events_fitted": res["n_events_fitted"], "n_spikes": res["n_spikes"],
               "auc_any_type": auc(res["p_event"][fitted], replay[fitted]),
               "auc_replay": auc(ps[fitted], replay[fitted]),
               "flagged_replays": float(np.mean(ps[fitted & replay] > 0.5)) if (fitted & replay).any() else np.nan,
               "flagged_volleys": float(np.mean(ps[fitted & ~replay] > 0.5)) if (fitted & ~replay).any() else np.nan,
               "runtime_s": res["runtime_s"]}
        pickle.dump(out, open(target, "wb"))
        print(f"{name}: routes " + ", ".join(f"{abs(r[0]):.2f}[{r[1]:.2f}] (type {r[3] + 1})" for r in routes_scored) +
              f" | R={res['best_setting']['num_sequence_types']} | types: " + "; ".join(f"{t['n_members']} cells, span {t['lag_span_ms']:.0f} ms, {t['mean_latent_events']:.0f} events" for t in res["types"]) +
              f" | sequence-type detection AUC {out['auc_replay']:.2f} (any type {out['auc_any_type']:.2f}), flagged {100*out['flagged_replays']:.0f}% replays / {100*out['flagged_volleys']:.0f}% volleys | {time.perf_counter() - t0:.0f}s", flush=True)


def run_hc11(prefixes):
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [p for p in session_paths() if not prefixes or any(p.name.startswith(x) for x in prefixes)]
    for path in paths:
        session = load_session(path)
        target = OUT / f"ppseq_{session['name']}.pkl"
        if target.exists():
            print(f"[{session['name']}] cached", flush=True)
            continue
        t_all = time.perf_counter()
        run_info = running(session)
        out = {"name": session["name"], "maze": session["maze"], "circular": run_info["circular"], "epochs": {}}
        for epoch in ("post", "pre"):
            rng = np.random.default_rng(7)
            intervals = nrem_intervals(session, epoch)
            by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
            bursts, peaks = population_bursts(pooled, intervals)
            centres = place_centres(session, cell_ids, run_info)
            times, _ = event_spike_times(by_cell, cell_ids, bursts[:, 0], bursts[:, 1], max_spikes=60)
            duration_ms = (bursts[:, 1] - bursts[:, 0]) * 1000
            print(f"[{session['name']}] {epoch}: {len(peaks)} events, fitting PP-Seq", flush=True)
            res = pp.fit(times, duration_ms, rng)
            types = []
            for t in res["types"]:
                row = {k: v for k, v in t.items() if k != "members"}
                row["map"] = {}
                for sign in (1.0, -1.0):
                    rho, n95, n = order_vs_place(t["lag_ms"], centres[sign], run_info["circular"], run_info["length"], rng, n_null=500)
                    row["map"][sign] = {"rho": rho, "null95": n95, "n": n}
                types.append(row)
            fitted = np.isfinite(res["p_event"])
            out["epochs"][epoch] = {"n_events": len(peaks), "n_cells": len(cell_ids), "types": types, "sequence_types": res["sequence_types"].tolist(),
                                    "best_setting": res["best_setting"], "sweep": res["sweep"],
                                    "n_events_fitted": res["n_events_fitted"], "n_spikes": res["n_spikes"],
                                    "flagged_fraction_any": float(np.mean(res["p_event"][fitted] > 0.5)),
                                    "flagged_fraction": float(np.mean(res["p_event_sequence_types"][fitted] > 0.5)), "runtime_s": res["runtime_s"]}
            hits = sum(np.isfinite(t["map"][s]["rho"]) and abs(t["map"][s]["rho"]) > t["map"][s]["null95"] for t in types for s in t["map"])
            tests = sum(np.isfinite(t["map"][s]["rho"]) for t in types for s in t["map"])
            print(f"[{session['name']}] {epoch}: R={res['best_setting']['num_sequence_types']}, map hits {hits}/{tests}, types: " +
                  "; ".join(f"{t['n_members']} cells span {t['lag_span_ms']:.0f} ms ({t['mean_latent_events']:.0f} events)" for t in types) +
                  f", flagged {100*out['epochs'][epoch]['flagged_fraction']:.0f}% of events | {res['runtime_s']:.0f}s", flush=True)
        pickle.dump(out, open(target, "wb"))
        print(f"[{session['name']}] saved ({time.perf_counter() - t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    if mode == "synthetic":
        run_synthetic(sys.argv[2:])
    else:
        run_hc11(sys.argv[2:])
