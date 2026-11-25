"""Forward and backward plays: the direction split (cpca/direction.py) on planted data, hc-11 and hc-5.

- synthetic: fresh draws (seeds not used while developing the method), 10 per condition, the
  planted conditions A-H, the ladder rungs L0-L5, and a no-order control (L0 with each event's
  spike trains handed to a random permutation of the cells that fired); centred and uncentred
  complex PCA. Per planted route: the best component's order before the split and after it (the
  guarded order), the direction labels of its expressing events against the planted direction,
  the estimated against the planted reverse fraction; per component: the held-out test
  (cpca.direction.heldout_test) and whether the guard accepted the split. A, D, E and L1 play forward only;
- hc-11: 8 sessions, POST and PRE, centred complex PCA on all population bursts (as in
  scripts/run_hc11.py). Per component and running direction: member order against the place map
  before and after the split, and the template's own order (cpca.evaluate.order_vs_place); the
  direction labels against each event's rank-order correlation with the map (label_check); the
  templates of the forward and the backward group fitted separately (the same route played both
  ways should match the map with opposite signs). The same for one template fitted on the whole
  population (split_members on all cells, no component and so no guard).

hc-5 is not used: its sessions have 7-25 wheel runs, fewer than the 20 events the split needs.

Usage: python scripts/run_direction.py [--workers=10] [--only=synthetic,hc11]
Results: results/direction/<family>/<job>.pkl (existing files are skipped).
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
from cpca.direction import group_templates, order_of, split_component, split_members  # noqa: E402
from cpca.evaluate import order_vs_place  # noqa: E402

OUT = ROOT / "results" / "direction"
N_SHUFFLES = 100
N_SHUFFLES_HC11 = 100
HC11_ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
              "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pickle.dump(result, open(tmp, "wb"))
    tmp.replace(path)


def split_all(res, rng):
    """Direction split of every kept component; list of summaries (and the full split objects)."""
    rows, splits = [], []
    for j in range(res["n_kept"]):
        sp = split_component(res["encoded"], res["pca_components"][:, j], rng)
        splits.append(sp)
        if sp is None:
            rows.append(None)
            continue
        rows.append({k: sp[k] for k in ("accepted", "agreement", "minority_fraction", "backward_fraction", "n_expressing",
                                         "heldout_share", "heldout_p", "n_iter")} |
                    {"n_members": int(sp["members"].sum()), "n_events": int(sp["events"].size)})
    return rows, splits


# ----------------------------------------------------------------------------- synthetic
NO_ORDER = "S0 L0 with each event's spike trains shuffled among the cells that fired (no order)"


def synthetic_conditions():
    from sim.routes import CONDITIONS, LADDER
    keep = ["L0 one map on 60 cells, both directions", "L1 two maps on the same 60 cells, forward only",
            "L2 two maps on the same cells, both directions", "L3 L2 + partial replays (40% segments)",
            "L4 L2 + assemblies", "L5 L2 + variable pacing (30-90 ms)"]
    return {**CONDITIONS, **{k: LADDER[k] for k in keep}, NO_ORDER: LADDER[keep[0]]}


def synthetic_target(cond, draw):
    return OUT / "synthetic" / f"{cond.split(' ')[0]}_d{draw:02d}.pkl"


def shuffle_cells_within_events(times, rng):
    """Give each event's spike trains to a random permutation of the cells that fired in it."""
    out = times.copy()
    for e in range(times.shape[0]):
        active = np.flatnonzero(np.isfinite(times[e]).any(axis=1))
        out[e, active] = times[e, rng.permutation(active)]
    return out


def synthetic_job(cond, draw):
    from sim.routes import generate, make_routes
    t0 = time.perf_counter()
    gen_kwargs, route_kwargs = synthetic_conditions()[cond]
    rng = np.random.default_rng(seed("direction-synthetic", cond, draw))
    routes = make_routes(rng, n_cells=80, **route_kwargs)
    data = generate(rng, n_cells=80, n_events=3000, routes=routes, **gen_kwargs)
    if cond == NO_ORDER:
        data["times"] = shuffle_cells_within_events(data["times"], rng)
    out = {"family": "synthetic", "condition": cond, "draw": draw, "reverse_frac": gen_kwargs.get("reverse_frac", 0.5), "variants": {}}
    for conv, centred in (("centred", True), ("uncentred", False)):
        res = fit(data["times"], data["event_ms"], rng=rng, ica=False, center=centred, scale=centred, n_shuffles=N_SHUFFLES)
        rows, splits = split_all(res, rng)
        before = [order_of(res["pca_components"][:, j]) for j in range(res["n_kept"])]
        after = [s["order"] if s is not None else before[j] for j, s in enumerate(splits)]
        routes_out = []
        for r, cen in enumerate(data["centres"]):
            def best(orders):
                vals = [abs(np.nan_to_num(order_vs_place(o, cen, False, 1.0, rng, n_null=1)[0])) for o in orders]
                return (int(np.argmax(vals)), float(max(vals))) if vals else (-1, np.nan)
            jb, rb = best(before)
            ja, ra = best(after)
            row = {"route": r, "before": rb, "after": ra, "component_before": jb, "component_after": ja}
            if ja >= 0 and splits[ja] is not None:
                sp = splits[ja]
                ev = sp["events"][sp["expresses"]]
                own = data["label"][ev] == r + 1
                if own.sum() >= 10:
                    truth = data["direction"][ev][own] > 0
                    acc = float(np.mean(sp["forward"][sp["expresses"]][own] == truth))
                    true_minor = float(min(np.mean(truth), 1 - np.mean(truth)))
                    row.update(accuracy=max(acc, 1 - acc), true_minority=true_minor,
                               caught=float(np.isin(np.flatnonzero(data["label"] == r + 1), ev).mean()), precision=float(own.mean()),
                               minority=sp["minority_fraction"], accepted=sp["accepted"])
            routes_out.append(row)
        out["variants"][conv] = {"n_kept": res["n_kept"], "components": rows, "routes": routes_out}
    out["runtime_s"] = time.perf_counter() - t0
    save(synthetic_target(cond, draw), out)
    c = out["variants"]["centred"]["routes"]
    return " | ".join(f"r{x['route']}: {x['before']:.2f}->{x['after']:.2f}" for x in c)


# ----------------------------------------------------------------------------- hc-11
def map_test(order, centres, run, rng):
    rho, null95, n = order_vs_place(order, centres, run["circular"], run["length"], rng, n_null=500)
    return {"rho": rho, "null95": null95, "n": n}


def label_check(encoded, sp, template_maps, centres, run, rng, n_null=200, n_other=300):
    """Direction labels against the map, event by event (the map is used only here, to score).

    The map the template matches best (if it beats its null) fixes the template's sign; a forward
    label then predicts that sign for the event's own rank-order correlation with the map, a
    backward label the opposite. Scored on events whose rank-order correlation beats its own
    null (cell-identity shuffle): all expressing events, and for comparison up to n_other
    randomly chosen non-expressing ones.
    """
    hits = {s: m for s, m in template_maps.items() if np.isfinite(m["rho"]) and abs(m["rho"]) > m["null95"]}
    if not hits:
        return None
    s = max(hits, key=lambda k: abs(hits[k]["rho"]))
    sign = np.sign(hits[s]["rho"])
    E = encoded[sp["events"]][:, sp["members"]]
    cen = centres[s][sp["members"]]
    other = np.flatnonzero(~sp["expresses"])
    scored = np.sort(np.concatenate([np.flatnonzero(sp["expresses"]), rng.choice(other, min(n_other, other.size), replace=False)]))
    rows = []
    for i in scored:
        t = np.where(E[i] != 0, np.angle(E[i]), np.nan)
        rho, null95, n = order_vs_place(t, cen, run["circular"], run["length"], rng, n_null=n_null)
        if np.isfinite(rho) and abs(rho) > null95:
            rows.append((i, rho))
    idx = np.array([i for i, _ in rows], int)
    rho = np.array([r for _, r in rows])
    predicted = np.where(sp["forward"][idx], sign, -sign)
    correct = np.sign(rho) == predicted
    expr = sp["expresses"][idx]
    return {"map": s, "template_rho": hits[s]["rho"], "n_events": int(E.shape[0]), "n_expressing": int(sp["expresses"].sum()),
            "n_other_scored": int(scored.size - sp["expresses"].sum()),
            "n_significant": int(idx.size), "n_significant_expressing": int(expr.sum()),
            "correct_expressing": int(correct[expr].sum()), "correct_other": int(correct[~expr].sum()),
            "reverse_expressing": int((np.sign(rho[expr]) != sign).sum()), "event_rho": rho, "event_expresses": expr, "event_correct": correct,
            "event_index": sp["events"][idx], "event_forward": sp["forward"][idx], "event_strength": sp["index"][idx]}


def hc11_job(name, epoch):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    t0 = time.perf_counter()
    session = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(session)
    intervals = nrem_intervals(session, epoch)
    by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
    bursts, peaks = population_bursts(pooled, intervals)
    duration = np.diff(bursts, axis=1).ravel()
    times, _ = event_spike_times(by_cell, cell_ids, peaks - duration / 2, peaks + duration / 2)
    centres = place_centres(session, cell_ids, run)
    rng = np.random.default_rng(seed("direction-hc11", name, epoch))
    res = fit(times, duration * 1000, rng=rng, ica=False, n_shuffles=N_SHUFFLES_HC11)
    rows, splits = split_all(res, rng)
    comps = []
    for j in range(res["n_kept"]):
        before = order_of(res["pca_components"][:, j])
        sp = splits[j]
        row = {"index": j, "split": rows[j], "map": {}}
        for sign in (1.0, -1.0):
            row["map"][sign] = {"before": map_test(before, centres[sign], run, rng)}
            if sp is not None:
                row["map"][sign]["after"] = map_test(sp["order"], centres[sign], run, rng)
                row["map"][sign]["template"] = map_test(order_of(sp["template"]), centres[sign], run, rng)
        if sp is not None:
            row["labels"] = label_check(res["encoded"], sp, {s: row["map"][s]["template"] for s in row["map"]}, centres, run, rng)
            g = group_templates(res["encoded"], sp)
            if g["forward"] is not None and g["backward"] is not None:
                row["groups"] = {s: {k: map_test(order_of(g[k]), centres[s], run, rng) for k in ("forward", "backward")} for s in (1.0, -1.0)}
        comps.append(row)
    # one direction-invariant template on the whole population (no component, so no guard)
    sp = split_members(res["encoded"], np.ones(res["encoded"].shape[1], bool), rng)
    population = None
    if sp is not None:
        population = {k: sp[k] for k in ("minority_fraction", "backward_fraction", "n_expressing", "heldout_share",
                                         "heldout_p", "n_iter")} | {"n_events": int(sp["events"].size)}
        population["map"] = {s: map_test(order_of(sp["template"]), centres[s], run, rng) for s in (1.0, -1.0)}
        population["labels"] = label_check(res["encoded"], sp, population["map"], centres, run, rng)
        population["template"] = sp["template"]
        if population["labels"] is not None:
            # spike times of the scored replays, for the example rasters
            population["labels"]["event_times_ms"] = times[population["labels"]["event_index"]]
            population["labels"]["event_duration_ms"] = duration[population["labels"]["event_index"]] * 1000
        g = group_templates(res["encoded"], sp)
        if g["forward"] is not None and g["backward"] is not None:
            population["groups"] = {s: {k: map_test(order_of(g[k]), centres[s], run, rng) for k in ("forward", "backward")} for s in (1.0, -1.0)}
    result_centres = {s: centres[s] for s in (1.0, -1.0)}
    result = {"family": "hc11", "name": name, "epoch": epoch, "circular": run["circular"], "n_kept": res["n_kept"], "n_events": len(peaks),
              "components": comps, "population": population, "centres": result_centres, "runtime_s": time.perf_counter() - t0}
    save(OUT / "hc11" / f"{name}_{epoch}.pkl", result)
    n_acc = sum(1 for r in rows if r is not None and r["accepted"])
    lab = population["labels"] if population else None
    pop = (f"population: reverse {population['minority_fraction']:.2f}, map rho " +
           "/".join(f"{population['map'][s]['rho']:+.2f}" for s in (1.0, -1.0)) +
           (f", labels {lab['correct_expressing']}/{lab['n_significant_expressing']}" if lab else ", no map hit")) if population else "population: none"
    return f"k={res['n_kept']}, accepted {n_acc}; {pop}"


# ----------------------------------------------------------------------------- scheduler
def all_jobs(families):
    jobs = []
    if "hc11" in families:
        jobs += [(hc11_job, (n, e)) for n in HC11_ORDER for e in ("post", "pre") if not (OUT / "hc11" / f"{n}_{e}.pkl").exists()]
    if "synthetic" in families:
        jobs += [(synthetic_job, (c, d)) for c in synthetic_conditions() for d in range(10) if not synthetic_target(c, d).exists()]
    return jobs


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), os.cpu_count() or 2)
    families = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--only=")), ["hc11", "synthetic"])
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
