"""Steps 4 and 5 of docs/adoption.md: false positives of active centring on timing-shuffled events, and the arc
(pi against 1.5 pi) with the current pipeline.

Usage: python scripts/run_arc_and_fp.py [--workers=9]
Output: results/adoption/arc_fp/rows.pkl, summary.md
"""

import os
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_adoption_check import clean  # noqa: E402
from scripts.run_separation_check import HC11, hc5_job, hc11_job, laps_job, orders_of, score, seed  # noqa: E402

OUT = ROOT / "results" / "adoption" / "arc_fp"
ARCS = {"pi": np.pi, "1.5pi": 1.5 * np.pi}
CENTRES = {"column": True, "active": "active"}


def _events(name, kind):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    from cpca.replay import immobile_intervals
    from scripts.run_replay_hc11 import intersect, two_d
    from scripts.run_rerun import ORDER_INDEX, fixed_subset
    path = next(p for p in session_paths() if p.name.startswith(name))
    s = load_session(path)
    run = running(s)
    if kind == "pre":
        iv = nrem_intervals(s, "pre")
    else:
        s["pos_xy"] = two_d(path)
        iv = intersect(immobile_intervals(s), [tuple(s["maze_epoch"])])
    by_cell, ids, pooled = pyramidal_spikes(s, iv)
    b, pk = population_bursts(pooled, iv)
    dur = np.diff(b, axis=1).ravel()
    sub = fixed_subset(len(pk), 555 + 2 * ORDER_INDEX[name] + 1) if kind == "pre" else np.arange(len(pk))
    times, _ = event_spike_times(by_cell, ids, (pk - dur / 2)[sub], (pk + dur / 2)[sub], max_spikes=60)
    return times, dur[sub] * 1000, place_centres(s, ids, run), run


def _best_p(r, centres, run, rng):
    from scripts.plot_hc11_sequences import best_map_match
    orders = [rd["times_ms"] for rd in r["readouts"]]
    return best_map_match(orders, centres, run["circular"], run["length"], rng, n_null=500)["p"] if orders else np.nan


def fp_job(name, kind):
    from cpca import fit
    from scripts.run_maze_rest import shuffle_timing
    times, T, centres, run = _events(name, kind)
    rows = []
    for i in range(20):
        sh = shuffle_timing(times, np.random.default_rng(seed("fp-shuffle", name, kind, i)))
        for c, center in CENTRES.items():
            r = fit(sh, T, rng=np.random.default_rng(seed("fp-fit", name, kind, i)), center=center, scale=True, ica=False)
            p = _best_p(r, centres, run, np.random.default_rng(seed("fp-score", name, kind, i)))
            rows.append({"set": "fp", "case": name, "kind": kind, "draw": i, "center": c, "k": r["n_kept"],
                         "hit": bool(np.isfinite(p) and p < 0.05)})
    return rows


def spike_job(name, f, draw):
    from cpca import fit
    from scripts.run_spikein_hc11 import ideal
    base, T, centres, run = _events(name, "pre")
    rng = np.random.default_rng(seed("adopt-spike", name, f, draw))
    k = int(round(f * len(T)))
    times = ideal(base, T, rng.choice(len(T), k, replace=False), centres, rng) if k else base
    rows = []
    for a, arc in ARCS.items():
        for c, center in CENTRES.items():
            r = fit(times, T, rng=np.random.default_rng(seed("adopt-spikefit", name, f, draw)), center=center, scale=True,
                    ica=False, arc=arc)
            p = _best_p(r, centres, run, np.random.default_rng(seed("adopt-spikesc", name, f, draw)))
            rows.append({"set": "spike-in", "case": name, "share": f, "draw": draw, "arc": a, "center": c, "k": r["n_kept"],
                         "hit": bool(np.isfinite(p) and p < 0.05)})
    return rows


def react_job(cond, draw, phase):
    import experimental.scripts.dev_unmixing as dv
    from cpca import fit
    from cpca.separate import separate
    d, kw = dv.get_data(cond, draw, phase)
    perm = np.random.default_rng(seed("split", phase, cond, draw)).permutation(d["times"].shape[0])
    halves = (perm[: perm.size // 2], perm[perm.size // 2:])
    maps = d["maps"] if sum(kw["rates"]) > 0 else []
    rows = []
    for fold, (tr, te) in enumerate((halves, halves[::-1])):
        for a, arc in ARCS.items():
            for c, center in CENTRES.items():
                rng = np.random.default_rng(seed("fit", phase, cond, draw, fold))
                r = fit(d["times"][tr], d["T"][tr], rng=rng, ica=False, center=center, scale=True, n_shuffles=50, arc=arc)
                outs, _ = separate(r, d["times"][tr], d["T"][tr], d["times"][te], d["T"][te], rng, reversal=True,
                                   min_events=20, arc=arc)
                rows.append({"set": "planted reactivations", "case": cond, "phase": phase, "draw": draw, "fold": fold,
                             "arc": a, "center": c, "n_maps": len(maps),
                             "verdicts": score(orders_of(outs, r["T_read_ms"]), maps, kw["shared"],
                                               np.random.default_rng(seed("judge", phase, cond, draw, fold)))})
    return rows


def waking_job(fn, args, a):
    import cpca
    import cpca.pipeline
    import cpca.separate
    cpca.fit = partial(cpca.pipeline.fit, arc=ARCS[a])
    original = cpca.separate.__dict__.setdefault("_separate_pi", cpca.separate.separate)
    cpca.separate.separate = partial(original, arc=ARCS[a])
    rows = fn(*args)
    for r in rows:
        r["arc"] = a
    return rows


def summarize(rows):
    L = ["# Steps 4 and 5 (docs/adoption.md)", ""]
    fp = [r for r in rows if r["set"] == "fp"]
    if fp:
        L += ["## Step 4: timing-shuffled events, fits whose best output passes the order test (arc pi)", "",
              "| events | column | active |", "|---|---|---|"]
        for kind in ("pre", "maze"):
            cells = []
            for c in CENTRES:
                rs = [r for r in fp if r["kind"] == kind and r["center"] == c]
                cells.append(f"{sum(r['hit'] for r in rs)}/{len(rs)} (median k {np.median([r['k'] for r in rs]):.0f})")
            L.append(f"| {'PRE, matched 1500' if kind == 'pre' else 'rest on the maze'} | " + " | ".join(cells) + " |")
        L.append("")
    sp = [r for r in rows if r["set"] == "spike-in"]
    if sp:
        shares = sorted({r["share"] for r in sp})
        L += ["## Step 5: spike-in hits, all 8 sessions", "", "| arc | centring | " + " | ".join(f"{f:.0%}" for f in shares) + " |",
              "|---|---|" + "---|" * len(shares)]
        for a in ARCS:
            for c in CENTRES:
                L.append(f"| {a} | {c} | " + " | ".join(
                    f"{sum(r['hit'] for r in sp if r['arc'] == a and r['center'] == c and r['share'] == f)}/"
                    f"{sum(1 for r in sp if r['arc'] == a and r['center'] == c and r['share'] == f)}" for f in shares) + " |")
        L.append("")
    pr = [r for r in rows if r["set"] == "planted reactivations"]
    if pr:
        keys = [(a, c) for c in CENTRES for a in ARCS]
        L += ["## Step 5: planted reactivations, clean fits of 10 (variance share for shared maps; original in brackets)", "",
              "| condition | " + " | ".join(f"{c}, {a}" for a, c in keys) + " |", "|---|" + "---|" * len(keys)]
        tot = {k: [0, 0] for k in keys}
        for case in sorted({r["case"] for r in pr}):
            cells = []
            for k in keys:
                rs = [r for r in pr if r["case"] == case and (r["arc"], r["center"]) == k]
                x, y = sum(clean(r, "share") for r in rs), sum(clean(r, "original") for r in rs)
                tot[k][0] += x
                tot[k][1] += y
                cells.append(f"{x} ({y})")
            L.append(f"| {case[:44]} | " + " | ".join(cells) + " |")
        L.append("| total | " + " | ".join(f"{x} ({y})" for x, y in tot.values()) + " |")
        L.append("")
    wk = [r for r in rows if r["set"] in ("planted laps", "hc-11 laps (linear)", "hc-11 laps (circular)", "hc-5 wheel")]
    if wk:
        L += ["## Step 5: waking, clean fits (original / variance share)", "", "| data | pi | 1.5pi |", "|---|---|---|"]
        for key in sorted({(r["set"], r["case"] if r["set"] == "planted laps" else "") for r in wk}):
            cells = []
            for a in ARCS:
                rs = [r for r in wk if (r["set"], r["case"] if r["set"] == "planted laps" else "") == key and r["arc"] == a]
                cells.append(" / ".join(f"{sum(clean(r, c) for r in rs)}/{len(rs)}" for c in ("original", "share")))
            L.append(f"| {key[0]} {key[1][:40]} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    import cpca.hc5 as hc5
    import experimental.scripts.dev_unmixing as dv
    from experimental.scripts.dev_waking import CONDITIONS as LAPS
    from scripts.run_separation import CONDITIONS as REACT
    jobs = [(fp_job, (n, kind)) for n in HC11 for kind in ("pre", "maze")]
    jobs += [(spike_job, (n, f, d)) for n in HC11 for f in (0.0, 0.03, 0.05, 0.10) for d in range(5)]
    jobs += [(react_job, (c, d, "confirm")) for c in REACT for d in range(5)]
    jobs += [(react_job, (c, d, "stress_confirm")) for c in dv.STRESS for d in range(5)]
    for a in ARCS:
        jobs += [(waking_job, (laps_job, (c, d, "confirm", ["full"]), a)) for c in LAPS for d in range(5)]
        jobs += [(waking_job, (hc11_job, (n, f), a)) for n in HC11 for f in (0, 1)]
        jobs += [(waking_job, (hc5_job, (n, f), a)) for n in hc5.session_names() for f in (0, 1)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, *a): (fn.__name__, a) for fn, a in jobs}
        for fut in as_completed(futs):
            try:
                rows += fut.result()
            except Exception:
                print(f"FAILED {futs[fut]}:\n{traceback.format_exc()[-1500:]}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(rows, open(OUT / "rows.pkl", "wb"))
    text = summarize(rows)
    (OUT / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
