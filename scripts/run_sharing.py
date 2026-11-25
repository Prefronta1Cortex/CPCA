"""Sequences that share cells: templates with event assignment (cpca/mixture.py) on planted data, the ladder and hc-11.

- synthetic: fresh draws (seeds not used in development), 10 per condition: A-H, the ladder rungs
  L0-L5, and the no-order control of scripts/run_direction.py; centred and uncentred complex PCA,
  and for L5 also centred with each event's own spike spread as the half-circle scale. Per planted
  map: the best order of the direction split (Step 2) and of the validated templates (Step 3),
  with coverage; for the template that matches the map best, the share of the map's replays
  assigned to it, the share of its events that are the map's replays, and the direction accuracy
  of its events. Per fit: the number of templates, validated templates, and validated templates
  that match no planted map;
- ladder: the difficulty ladder's own events (scripts/run_ladder.make_data, 1500 events, 3 draws),
  on which PP-Seq and seqNMF were run; scored like them (scripts/run_ladder.score);
- hc-11: 8 sessions, POST and PRE, centred complex PCA on all population bursts (as in
  scripts/run_hc11.py). Per template: validated or not, its order against both running
  directions' place maps, and its direction labels against each event's rank-order correlation
  with the map (scripts/run_direction.label_check).

Usage: python scripts/run_sharing.py [--workers=10] [--only=synthetic,ladder,hc11]
Results: results/sharing/<family>/<job>.pkl (existing files are skipped).
"""

import os
import pickle
import sys
import time
import traceback
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpca import fit  # noqa: E402
from cpca.direction import order_of, split_component  # noqa: E402
from cpca.evaluate import order_vs_place  # noqa: E402
from cpca.mixture import sequences  # noqa: E402

OUT = ROOT / "results" / "sharing"
N_SHUFFLES = 100
N_DRAWS = 10
MATCH = 0.5            # a validated template "matches" a planted map if |rho| >= MATCH on at least 6 of its members
HC11_ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
              "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pickle.dump(result, open(tmp, "wb"))
    tmp.replace(path)


def step2_orders(res, rng):
    out = []
    for j in range(res["n_kept"]):
        sp = split_component(res["encoded"], res["pca_components"][:, j], rng, test=False)
        out.append(sp["order"] if sp is not None else order_of(res["pca_components"][:, j]))
    return out


def run_step3(res, rng):
    t0 = time.perf_counter()
    sq = sequences(res, rng)
    runtime = time.perf_counter() - t0
    if sq is None or sq["templates"].shape[1] == 0:
        return None, [], np.zeros(0, bool), runtime
    keep = sq["check"]["keep"]
    orders = [order_of(sq["templates"][:, j]) for j in range(sq["templates"].shape[1])]
    return sq, orders, keep, runtime


def summary_of(sq, keep):
    if sq is None:
        return None
    return {"n_templates": int(sq["templates"].shape[1]), "n_valid": int(keep.sum()), "start": sq["start"], "n_iter": sq["n_iter"],
            "similarity": sq["check"]["similarity"], "p": sq["check"]["p"], "keep": keep, "energy": sq["energy"]}


# ----------------------------------------------------------------------------- synthetic
def synthetic_conditions():
    from scripts.run_direction import synthetic_conditions as sc
    return sc()


def synthetic_target(cond, draw):
    return OUT / "synthetic" / f"{cond.split(' ')[0]}_d{draw:02d}.pkl"


def per_map(orders, centres, rng):
    """Per map: (|rho|, coverage, index of the best output)."""
    out = []
    for cen in centres:
        best = (np.nan, np.nan, -1)
        n_map = int(np.isfinite(cen).sum())
        for j, o in enumerate(orders):
            rho, _, n = order_vs_place(o, cen, False, 1.0, rng, n_null=1)
            if np.isfinite(rho) and (best[2] < 0 or abs(rho) > best[0]):
                best = (abs(float(rho)), n / n_map, j)
        out.append(best)
    return out


def example_fit(cond, draw):
    """The centred fit of synthetic_job, recomputed (same seed and order of random draws): data, fit, Step 2 orders, Step 3."""
    from sim.routes import generate, make_routes
    gen_kwargs, route_kwargs = synthetic_conditions()[cond]
    rng = np.random.default_rng(seed("sharing-synthetic", cond, draw))
    data = generate(rng, n_cells=80, n_events=3000, routes=make_routes(rng, n_cells=80, **route_kwargs), **gen_kwargs)
    res = fit(data["times"], data["event_ms"], rng=rng, ica=False, center=True, scale=True, n_shuffles=N_SHUFFLES)
    s2 = step2_orders(res, rng)
    sq = run_step3(res, rng)[0]
    return data, res, s2, sq


def synthetic_job(cond, draw):
    from scripts.run_direction import NO_ORDER, shuffle_cells_within_events
    from sim.routes import generate, make_routes
    from cpca import event_scale
    t0 = time.perf_counter()
    gen_kwargs, route_kwargs = synthetic_conditions()[cond]
    rng = np.random.default_rng(seed("sharing-synthetic", cond, draw))
    data = generate(rng, n_cells=80, n_events=3000, routes=make_routes(rng, n_cells=80, **route_kwargs), **gen_kwargs)
    if cond == NO_ORDER:
        data["times"] = shuffle_cells_within_events(data["times"], rng)
    variants = [("centred", True, None), ("uncentred", False, None)]
    if cond.startswith("L5 "):
        variants.append(("centred, per-event scale", True, "spread"))
    out = {"family": "synthetic", "condition": cond, "draw": draw, "n_maps": len(data["centres"]), "variants": {}}
    for name, centred, scale in variants:
        if scale == "spread":
            T, lo = event_scale(data["times"])
            res = fit(data["times"] - lo[:, None, None], T, rng=rng, ica=False, center=centred, scale=centred, n_shuffles=N_SHUFFLES)
        else:
            res = fit(data["times"], data["event_ms"], rng=rng, ica=False, center=centred, scale=centred, n_shuffles=N_SHUFFLES)
        s2 = step2_orders(res, rng)
        sq, orders, keep, runtime = run_step3(res, rng)
        valid = [o for o, k in zip(orders, keep) if k]
        v = {"n_kept": res["n_kept"], "runtime_step3_s": runtime, "step3": summary_of(sq, keep),
             "step2_maps": per_map(s2, data["centres"], rng), "step3_maps": per_map(valid, data["centres"], rng),
             "all_templates_maps": per_map(orders, data["centres"], rng)}
        # validated templates that match no planted map
        n_false = 0
        for o in valid:
            r = [abs(np.nan_to_num(order_vs_place(o, cen, False, 1.0, rng, n_null=1)[0])) for cen in data["centres"]]
            n_false += int(max(r) < MATCH) if r else 1
        v["n_false"] = n_false
        # events of the best template per map
        events = []
        if sq is not None:
            valid_idx = np.flatnonzero(keep)
            lab = data["label"][sq["events"]]
            for r, (rho, cov, j) in enumerate(v["step3_maps"]):
                if j < 0:
                    events.append(None)
                    continue
                t = valid_idx[j]
                mine = sq["label"] == t
                own = lab == r + 1
                row = {"recall": float((mine & own).sum() / max(own.sum(), 1)), "precision": float((mine & own).sum() / max(mine.sum(), 1))}
                both = mine & own
                if both.sum() >= 10:
                    truth = data["direction"][sq["events"][both]] > 0
                    acc = float(np.mean(sq["forward"][both] == truth))
                    row["direction_accuracy"] = max(acc, 1 - acc)
                events.append(row)
        v["events"] = events
        out["variants"][name] = v
    out["runtime_s"] = time.perf_counter() - t0
    save(synthetic_target(cond, draw), out)
    c = out["variants"]["centred"]
    return " | ".join(f"m{r}: {a[0]:.2f}->{b[0]:.2f}" for r, (a, b) in enumerate(zip(c["step2_maps"], c["step3_maps"]))) + \
        f" | valid {c['step3']['n_valid'] if c['step3'] else 0}, false {c['n_false']}, {c['runtime_step3_s']:.1f}s"


# ----------------------------------------------------------------------------- ladder
def ladder_target(rung, draw):
    return OUT / "ladder" / f"{rung.split(' ')[0]}_d{draw}.pkl"


def ladder_job(rung, draw):
    from scripts.run_ladder import make_data, rung_code, score
    t0 = time.perf_counter()
    data = make_data(rung, draw)
    rng = np.random.default_rng(seed("sharing-ladder", rung_code(rung), draw))
    score_rng = np.random.default_rng(seed("score", rung_code(rung), draw, "cpca-sharing"))
    out = {"family": "ladder", "rung": rung, "draw": draw, "variants": {}}
    for name, centred in (("centred", True), ("uncentred", False)):
        t1 = time.perf_counter()
        res = fit(data["times"], data["event_ms"], rng=rng, ica=False, center=centred, scale=centred, n_shuffles=N_SHUFFLES)
        t_pca = time.perf_counter() - t1
        s2 = step2_orders(res, rng)
        sq, orders, keep, runtime = run_step3(res, rng)
        valid = [o for o, k in zip(orders, keep) if k]
        out["variants"][name] = {"n_kept": res["n_kept"], "step3": summary_of(sq, keep), "runtime_pca_s": t_pca, "runtime_step3_s": runtime,
                                 "step2": score(s2, data["centres"], score_rng), "step3": score(valid, data["centres"], score_rng) if valid else None}
    out["runtime_s"] = time.perf_counter() - t0
    save(ladder_target(rung, draw), out)
    c = out["variants"]["centred"]
    return "centred " + " ".join(f"{m['rho']:.2f}" for m in (c["step3"] or [])) + f" (step2 " + " ".join(f"{m['rho']:.2f}" for m in c["step2"]) + ")"


# ----------------------------------------------------------------------------- hc-11
def hc11_job(name, epoch):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    from scripts.run_direction import label_check, map_test
    t0 = time.perf_counter()
    session = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(session)
    intervals = nrem_intervals(session, epoch)
    by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
    bursts, peaks = population_bursts(pooled, intervals)
    duration = np.diff(bursts, axis=1).ravel()
    times, _ = event_spike_times(by_cell, cell_ids, peaks - duration / 2, peaks + duration / 2)
    centres = place_centres(session, cell_ids, run)
    rng = np.random.default_rng(seed("sharing-hc11", name, epoch))
    res = fit(times, duration * 1000, rng=rng, ica=False, n_shuffles=N_SHUFFLES)
    sq, orders, keep, runtime = run_step3(res, rng)
    comps = []
    for j in range(res["n_kept"]):
        o = order_of(res["pca_components"][:, j])
        comps.append({s: map_test(o, centres[s], run, rng) for s in (1.0, -1.0)})
    templates = []
    if sq is not None:
        S = res["encoded"][sq["events"]][:, sq["cells"]]
        for j, o in enumerate(orders):
            q = sq["templates"][sq["cells"], j]
            F, B = np.abs(S @ np.conj(q)), np.abs(S @ q)
            maps = {s: map_test(o, centres[s], run, rng) for s in (1.0, -1.0)}
            sp = {"events": sq["events"], "members": sq["cells"], "expresses": sq["label"] == j, "forward": sq["forward"],
                  "index": (F - B) / np.maximum(F + B, 1e-12)}
            fwd = sq["forward"][sq["label"] == j]
            templates.append({"index": j, "valid": bool(keep[j]), "n_events": int((sq["label"] == j).sum()),
                              "minority_fraction": float(min(fwd.mean(), 1 - fwd.mean())) if fwd.size else np.nan,
                              "n_members": int(np.isfinite(o).sum()), "map": maps, "order": o,
                              "labels": label_check(res["encoded"], sp, maps, centres, run, rng) if keep[j] else None})
    result = {"family": "hc11", "name": name, "epoch": epoch, "circular": run["circular"], "n_kept": res["n_kept"], "n_events": len(peaks),
              "n_cells": len(cell_ids), "n_fields": {s: int(np.isfinite(centres[s]).sum()) for s in (1.0, -1.0)},
              "centres": {s: centres[s] for s in (1.0, -1.0)},
              "step3": summary_of(sq, keep), "components": comps, "templates": templates, "runtime_step3_s": runtime,
              "runtime_s": time.perf_counter() - t0}
    save(OUT / "hc11" / f"{name}_{epoch}.pkl", result)
    hits = [t for t in templates if t["valid"] and any(np.isfinite(m["rho"]) and abs(m["rho"]) > m["null95"] for m in t["map"].values())]
    return f"k={res['n_kept']}, templates {len(templates)}, valid {int(keep.sum()) if sq else 0}, map hits {len(hits)}, {runtime:.0f}s"


# ----------------------------------------------------------------------------- scheduler
def all_jobs(families):
    from sim.routes import LADDER
    from scripts.run_ladder import N_DRAWS as LADDER_DRAWS
    jobs = []
    if "hc11" in families:
        jobs += [(hc11_job, (n, e)) for n in HC11_ORDER for e in ("post", "pre") if not (OUT / "hc11" / f"{n}_{e}.pkl").exists()]
    if "ladder" in families:
        jobs += [(ladder_job, (r, d)) for r in LADDER for d in range(LADDER_DRAWS) if not ladder_target(r, d).exists()]
    if "synthetic" in families:
        jobs += [(synthetic_job, (c, d)) for c in synthetic_conditions() for d in range(N_DRAWS) if not synthetic_target(c, d).exists()]
    return jobs


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), os.cpu_count() or 2)
    families = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--only=")), ["hc11", "ladder", "synthetic"])
    for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(n, "1")
    jobs = all_jobs(families)
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        running = {pool.submit(fn, *args): (fn.__name__, args) for fn, args in jobs}
        done = 0
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                label = running.pop(fut)
                done += 1
                try:
                    print(f"[{done}/{len(jobs)} {time.perf_counter() - t0:6.0f}s] {label[0]} {label[1]}: {fut.result()}", flush=True)
                except Exception:
                    print(f"[{done}/{len(jobs)}] FAILED {label}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
