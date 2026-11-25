"""Simplified separation step (cpca/separate.py): which stages are essential, and does it keep the confirmed results.

Planted reactivations (sim/two_routes.py, conditions A-F of scripts/run_separation.py) and planted laps
(sim/laps.py, W1-W5), with the data, halves and seeds of the earlier runs, so the numbers compare directly:
development draws for removing one stage at a time, confirmation draws for the full step. Real waking data
(hc-11 laps, hc-5 wheel) once, in waking mode.

Usage: python scripts/run_separation_check.py [--workers=9]
Output: results/separation_check/rows.pkl, summary.md
"""

import os
import pickle
import sys
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "separation_check"
STAGES = ["full", "rotation", "refine", "explain", "split", "select"]      # 'full', or the stage switched off
VARIANTS = {"full": {}}
REACT_BASE = {"dev": 7000, "confirm": 9000}
LAPS_BASE = {"dev": 20000, "confirm": 21000}
HC11 = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
        "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def orders_of(outs, T_read):
    from cpca.decode import readout
    res = []
    for t in outs:
        _, times, _, inc, _ = readout(t["template"], T_read, min_weight=1e-9)
        res.append(np.where(inc & t["members"], times, np.nan))
    return res


def judge_all(order, maps, shared, rng):
    """(original, variance share, partial correlation) verdicts: (passes, map)."""
    from scripts.run_pure_capabilities import judge_share
    from scripts.run_separation import judge
    from experimental.scripts.dev_unmixing import judge_partial
    if not maps:
        return (False, -1), (False, -1), (False, -1)
    ok, k, _ = judge(order, maps, shared, rng)
    if not shared:
        return (ok, k), (ok, k), (ok, k)
    return (ok, k), judge_share(order, maps, rng), judge_partial(order, maps, rng)


def score(orders, maps, shared, rng):
    return [dict(zip(("original", "share", "partial"), judge_all(o, maps, shared, rng))) for o in orders]


def react_job(cond, draw, phase, stages):
    from cpca import fit
    from cpca.separate import separate
    from scripts.run_separation import CONDITIONS
    from sim.two_routes import generate
    kw = CONDITIONS[cond]
    d = generate(np.random.default_rng(REACT_BASE[phase] + 97 * list(CONDITIONS).index(cond) + draw), **kw)
    perm = np.random.default_rng(seed("split", phase, cond, draw)).permutation(d["times"].shape[0])
    halves = (perm[: perm.size // 2], perm[perm.size // 2:])
    maps = d["maps"] if sum(kw["rates"]) > 0 else []
    rows = []
    for fold, (tr, te) in enumerate((halves, halves[::-1])):
        for st in stages:
            rng = np.random.default_rng(seed("fit", phase, cond, draw, fold))
            r = fit(d["times"][tr], d["T"][tr], rng=rng, ica=False, center=True, scale=True, n_shuffles=50)
            outs, _ = separate(r, d["times"][tr], d["T"][tr], d["times"][te], d["T"][te], rng, reversal=True,
                               min_events=20, stages=None if st == "full" else [st])
            rows.append({"set": "planted reactivations", "case": cond, "phase": phase, "draw": draw, "fold": fold,
                         "stage": st, "n_maps": len(maps), "verdicts": score(orders_of(outs, r["T_read_ms"]), maps, kw["shared"],
                                                                             np.random.default_rng(seed("judge", phase, cond, draw, fold)))})
    return rows


def laps_job(cond, draw, phase, stages):
    from cpca import fit
    from cpca.separate import separate
    from experimental.scripts.dev_waking import CONDITIONS
    from sim.laps import generate
    kw = CONDITIONS[cond]
    d = generate(np.random.default_rng(LAPS_BASE[phase] + 97 * list(CONDITIONS).index(cond) + draw), **kw)
    n_dir = kw.get("laps_per_direction", (30, 30))
    maps = [] if kw.get("shuffle") else [m for m, n in zip(d["maps"], n_dir) if n > 0 and np.isfinite(m).sum() >= 6]
    rng0 = np.random.default_rng(seed("half", phase, cond, draw))
    dirs = d["direction"]
    half = np.zeros(dirs.size, bool)
    for s in np.unique(dirs):
        idx = rng0.permutation(np.flatnonzero(dirs == s))
        half[idx[: idx.size // 2]] = True
    rows = []
    for fold, train in enumerate((half, ~half)):
        tr, te = np.flatnonzero(train), np.flatnonzero(~train)
        for st in stages:
            rng = np.random.default_rng(seed("fit", phase, cond, draw, fold, "simplified"))
            r = fit(d["times"][tr], d["T"][tr], rng=rng, ica=False, center=False, scale=False, n_shuffles=50)
            outs, _ = separate(r, d["times"][tr], d["T"][tr], d["times"][te], d["T"][te], rng, reversal=False,
                               min_events=4, stages=None if st in VARIANTS else [st], **VARIANTS.get(st, {}))
            rows.append({"set": "planted laps", "case": cond, "phase": phase, "draw": draw, "fold": fold, "stage": st,
                         "n_maps": len(maps), "verdicts": score(orders_of(outs, r["T_read_ms"]), maps, True,
                                                                np.random.default_rng(seed("judge", phase, cond, draw, fold)))})
    return rows


def hc11_job(name, fold, variant="full"):
    from cpca import event_spike_times, fit
    from cpca.hc11 import laps, load_session, place_centres, pyramidal_spikes, running, session_paths
    from cpca.separate import separate
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_laps_hc11 import heldout_run
    s = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(s)
    L = laps(run)
    starts, ends, dirs = (np.array([x[i] for x in L]) for i in range(3))
    rng0 = np.random.default_rng(seed("waking-sep", name))
    half = np.zeros(len(L), bool)
    for dd in np.unique(dirs):
        idx = rng0.permutation(np.flatnonzero(dirs == dd))
        half[idx[: idx.size // 2]] = True
    train = half if fold == 0 else ~half
    tr, te = np.flatnonzero(train), np.flatnonzero(~train)
    by_cell, ids, _ = pyramidal_spikes(s, np.column_stack([starts, ends]))
    X, _ = event_spike_times(by_cell, ids, starts[tr], ends[tr], max_spikes=400)
    Y, _ = event_spike_times(by_cell, ids, starts[te], ends[te], max_spikes=400)
    TX, TY = (ends[tr] - starts[tr]) * 1000, (ends[te] - starts[te]) * 1000
    c = place_centres(s, ids, heldout_run(run, np.column_stack([starts[te], ends[te]])), min_spikes=20)
    maps = [c[sg] for sg in (1.0, -1.0) if np.isfinite(c[sg]).sum() >= 6 and np.sum(dirs[te] == sg) >= 3]
    rng = np.random.default_rng(seed("simplified-hc11", name, fold))
    r = fit(X, TX, rng=rng, ica=False, center=False, scale=False, n_shuffles=50)
    outs, _ = separate(r, X, TX, Y, TY, rng, reversal=False, min_events=4, **VARIANTS[variant])
    orders = orders_of(outs, r["T_read_ms"])
    jrng = np.random.default_rng(seed("simplified-hc11-judge", name, fold))
    if run["circular"]:
        verdicts = []
        for o in orders:
            ok = bool(maps and np.sum(np.isfinite(o) & np.isfinite(maps[0])) >= 6 and
                      best_map_match([o], {1.0: maps[0]}, True, run["length"], jrng, n_null=500)["p"] < 0.05)
            verdicts.append({k: (ok, 0 if ok else -1) for k in ("original", "share", "partial")})
    else:
        verdicts = score(orders, maps, len(maps) > 1, jrng)
    return [{"set": f"hc-11 laps ({'circular' if run['circular'] else 'linear'})", "case": name, "phase": "real", "draw": 0,
             "fold": fold, "stage": variant, "n_maps": len(maps), "verdicts": verdicts}]


def hc5_job(name, fold, variant="full"):
    import cpca.hc5 as hc5
    from cpca import fit
    from cpca.separate import separate
    from scripts.run_hc5 import training_data
    sess = hc5.load_session(name)
    X, T, train, test, _ = training_data(sess, "aligned", 0, fold)
    Y, TY, _, _, _ = training_data(sess, "aligned", 0, 1 - fold)
    TX = np.full(X.shape[0], T) if np.ndim(T) == 0 else np.asarray(T)
    TY = np.full(Y.shape[0], TY) if np.ndim(TY) == 0 else np.asarray(TY)
    f = hc5.time_fields(sess, test)
    maps = [f] if np.isfinite(f).sum() >= 6 else []
    rng = np.random.default_rng(seed("simplified-hc5", name, fold))
    r = fit(X, TX, rng=rng, ica=False, center=False, scale=False, n_shuffles=50)
    outs, _ = separate(r, X, TX, Y, TY, rng, reversal=False, min_events=4, **VARIANTS[variant])
    return [{"set": "hc-5 wheel", "case": name, "phase": "real", "draw": 0, "fold": fold, "stage": variant, "n_maps": len(maps),
             "verdicts": score(orders_of(outs, r["T_read_ms"]), maps, True, np.random.default_rng(seed("simplified-hc5-judge", name, fold)))}]


def summarize(rows):
    def line(rs, crit):
        outs = [[v[crit] for v in r["verdicts"]] for r in rs]
        n = sum(len(o) for o in outs)
        p = sum(ok for o in outs for ok, _ in o)
        if rs[0]["n_maps"] == 0:
            clean = sum(len(o) == 0 for o in outs)
        else:
            clean = sum(len({k for ok, k in o if ok}) == r["n_maps"] and all(ok for ok, _ in o) for r, o in zip(rs, outs))
        return f"{p}/{n} ({p / max(n, 1):.0%}), {clean}/{len(rs)}"

    lines = ["# Simplified separation step: stage check and confirmation", "",
             "Entries: outputs that are one route, and clean fits (every route its own output and every output one route; "
             "with no routes, clean = no output). Criteria: original / variance share / partial correlation for maps on shared cells.", ""]
    for setname, phase in (("planted reactivations", "dev"), ("planted laps", "dev")):
        lines += [f"## {setname}, development draws: one stage switched off at a time (original criterion)", "",
                  "| condition | " + " | ".join("full" if s == "full" else f"without {s}" for s in STAGES) + " |",
                  "|---|" + "---|" * len(STAGES)]
        for case in sorted({r["case"] for r in rows if r["set"] == setname}):
            cells = [line([r for r in rows if r["set"] == setname and r["phase"] == phase and r["case"] == case and r["stage"] == st], "original")
                     if any(r["set"] == setname and r["phase"] == phase and r["case"] == case and r["stage"] == st for r in rows) else "–"
                     for st in STAGES]
            lines.append(f"| {case[:34]} | " + " | ".join(cells) + " |")
        lines.append("")
    for setname in ("planted reactivations", "planted laps"):
        lines += [f"## {setname}, confirmation draws, full step", "", "| condition | original | variance share | partial |", "|---|---|---|---|"]
        for case in sorted({r["case"] for r in rows if r["set"] == setname}):
            rs = [r for r in rows if r["set"] == setname and r["phase"] == "confirm" and r["case"] == case]
            if rs:
                lines.append(f"| {case[:34]} | {line(rs, 'original')} | {line(rs, 'share')} | {line(rs, 'partial')} |")
        lines.append("")
    lines += ["## Real waking data, full step (waking mode)", "", "| data | original | variance share | partial |", "|---|---|---|---|"]
    for setname in ("hc-11 laps (linear)", "hc-11 laps (circular)", "hc-5 wheel"):
        rs = [r for r in rows if r["set"] == setname]
        if rs:
            lines.append(f"| {setname} | {line(rs, 'original')} | {line(rs, 'share')} | {line(rs, 'partial')} |")
    return "\n".join(lines)


def rerun_summary(rows):
    def line(rs, crit):
        outs = [[v[crit] for v in r["verdicts"]] for r in rs]
        n = sum(len(o) for o in outs)
        p = sum(ok for o in outs for ok, _ in o)
        clean = sum(len(o) == 0 for o in outs) if rs[0]["n_maps"] == 0 else \
            sum(len({k for ok, k in o if ok}) == r["n_maps"] and all(ok for ok, _ in o) for r, o in zip(rs, outs))
        return f"{p}/{n} ({p / max(n, 1):.0%}), {clean}/{len(rs)}"
    lines = ["| data | phase | variant | original | variance share | partial |", "|---|---|---|---|---|---|"]
    for key in sorted({(r["set"], r["case"] if r["set"].startswith("planted") else "", r["phase"], r["stage"]) for r in rows}):
        rs = [r for r in rows if (r["set"], r["case"] if r["set"].startswith("planted") else "", r["phase"], r["stage"]) == key]
        lines.append(f"| {key[0]} {key[1][:30]} | {key[2]} | {key[3]} | {line(rs, 'original')} | {line(rs, 'share')} | {line(rs, 'partial')} |")
    return "\n".join(lines)


def main_rerun(workers):
    import cpca.hc5 as hc5
    from experimental.scripts.dev_waking import CONDITIONS as LAPS
    from scripts.run_separation import CONDITIONS as REACT
    jobs = [(react_job, (c, d, "confirm", ["full"])) for c in REACT for d in range(5)]
    jobs += [(laps_job, (c, d, ph, list(VARIANTS))) for c in LAPS for d in range(5) for ph in ("dev", "confirm")]
    jobs += [(hc11_job, (n, f, v)) for n in HC11 for f in (0, 1) for v in VARIANTS]
    jobs += [(hc5_job, (n, f, v)) for n in hc5.session_names() for f in (0, 1) for v in VARIANTS]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, *a): a for fn, a in jobs}
        for fut in as_completed(futs):
            try:
                rows += fut.result()
            except Exception:
                print(f"FAILED {futs[fut]}:\n{traceback.format_exc()[-1500:]}", flush=True)
    pickle.dump(rows, open(OUT / "rows_restored.pkl", "wb"))
    text = rerun_summary(rows)
    (OUT / "summary_restored.md").write_text(text + "\n")
    print(text)


def main(argv):
    if "--restored" in argv:
        workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[v] = "1"
        return main_rerun(workers)
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    import cpca.hc5 as hc5
    from experimental.scripts.dev_waking import CONDITIONS as LAPS
    from scripts.run_separation import CONDITIONS as REACT
    jobs = [(react_job, (c, d, "dev", STAGES)) for c in REACT for d in range(5)]
    jobs += [(react_job, (c, d, "confirm", ["full"])) for c in REACT for d in range(5)]
    jobs += [(laps_job, (c, d, "dev", STAGES)) for c in LAPS for d in range(5)]
    jobs += [(laps_job, (c, d, "confirm", ["full"])) for c in LAPS for d in range(5)]
    jobs += [(hc11_job, (n, f)) for n in HC11 for f in (0, 1)]
    jobs += [(hc5_job, (n, f)) for n in hc5.session_names() for f in (0, 1)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, *a): a for fn, a in jobs}
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
