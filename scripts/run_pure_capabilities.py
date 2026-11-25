"""Pure capabilities of complex PCA (docs/unmixing.md, "Pure capabilities"): cpca.fit only, no un-mixing step.

Per fit (training half) and held-out answer key: (1) components kept, (2) is each route's order present among the
components (best-of-outputs null), (3) is each route inside the kept span, (4) is each component one route,
(5) for reactivations and planted data, which components pass the latency check on the held-out half.

Usage: python scripts/run_pure_capabilities.py [--workers=9]
Output: results/pure/rows.pkl, summary.md
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

OUT = ROOT / "results" / "pure"
HC11 = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
        "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]
LAPS = {
    "W1 both ways rho 0.2": dict(both=0.7, rho=0.2),
    "W2 both ways rho 0.7": dict(both=0.7, rho=0.7),
    "W3 10 laps each": dict(both=0.7, rho=0.2, laps_per_direction=(10, 10)),
    "W4 one direction": dict(both=0.0, laps_per_direction=(40, 0)),
    "W5 control, no sequence": dict(both=0.7, rho=0.2, shuffle=True),
}
N_SPAN_NULL = 200


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def halves_by(labels, rng):
    half = np.zeros(len(labels), bool)
    for v in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == v))
        half[idx[: idx.size // 2]] = True
    return half


# ----------------------------------------------------------------------------- scoring
def judge_share(order, maps, rng, max_gain=0.1):
    """Maps on shared cells: one map if its order matches it (p < 0.05) and the other maps add less than max_gain to the
    explained variance of the order in a rank regression on it; exactly one map must qualify."""
    from scipy.stats import rankdata
    from scripts.run_separation import spearman_p
    ok = []
    for k, m in enumerate(maps):
        on = np.isfinite(order) & np.isfinite(m)
        if on.sum() < 6:
            ok.append(False)
            continue
        good = spearman_p(order[on], m[on], rng)[1] < 0.05
        others = [o for j, o in enumerate(maps) if j != k]
        both = on & np.all([np.isfinite(o) for o in others], axis=0) if others else on
        if good and others and both.sum() >= 6:
            a = rankdata(order[both]) - rankdata(order[both]).mean()
            X1 = (rankdata(m[both]) - rankdata(m[both]).mean())[:, None]
            X2 = np.column_stack([X1] + [rankdata(o[both]) - rankdata(o[both]).mean() for o in others])
            r2 = lambda X: 1 - np.sum((a - X @ np.linalg.lstsq(X, a, rcond=None)[0]) ** 2) / max(np.sum(a ** 2), 1e-300)
            good = r2(X2) - r2(X1) < max_gain
        ok.append(bool(good))
    return sum(ok) == 1, (ok.index(True) if sum(ok) == 1 else -1)


def span_test(V, maps, rng):
    """Per map: share of its pattern (phase pi x normalised position on its cells) inside span(V, conj V), and whether it
    beats the 99th percentile of random cell sets of the same size with random phases."""
    from cpca.containment import basis, containment
    n = V.shape[0]
    B = basis(V) if V.shape[1] else np.zeros((n, 0))
    out = []
    for m in maps:
        on = np.isfinite(m)
        if on.sum() < 6 or not B.shape[1]:
            out.append({"share": 0.0, "in_span": False})
            continue
        x = (m[on] - np.nanmin(m)) / max(np.nanmax(m) - np.nanmin(m), 1e-12)
        u = np.zeros(n, complex)
        u[on] = np.exp(1j * np.pi * x)
        share = float(containment(u[:, None], B)[0])
        null = []
        for _ in range(N_SPAN_NULL):
            w = np.zeros(n, complex)
            w[rng.choice(n, int(on.sum()), replace=False)] = np.exp(1j * np.pi * rng.random(int(on.sum())))
            null.append(containment(w[:, None], B)[0])
        out.append({"share": share, "in_span": bool(share > np.percentile(null, 99))})
    return out


def latency_ps(r, te_times, te_T, rng, n_sur=50):
    from cpca.decompose import prepare
    from cpca.encode import encode
    from cpca.readout_check import _expression, redraw_latency
    s = r["settings"]
    P = r["pca_components"]
    Z = prepare(encode(te_times, te_T), center=s["center"], scale=s["scale"])
    null = np.zeros((n_sur, P.shape[1]))
    for i in range(n_sur):
        Zs = prepare(encode(redraw_latency(te_times, te_T, rng), te_T), center=s["center"], scale=s["scale"])
        null[i] = [_expression(Zs, P[:, j]) for j in range(P.shape[1])]
    e = np.array([_expression(Z, P[:, j]) for j in range(P.shape[1])])
    return ((1 + (null >= e).sum(axis=0)) / (1 + n_sur)).tolist()


def evaluate(r, maps, kind, rng, circular=False, length=1.0, te=None, judge=None):
    """Questions 1-5 for one fit. kind: 'different' (planted maps on different cells), 'shared' (maps on shared cells,
    running directions), 'one' (one map), 'figure8'."""
    from cpca.decode import readout
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_separation import judge as judge_sep
    k = r["n_kept"]
    orders = [np.where(inc, t, np.nan) for _, t, _, inc, _ in
              (readout(r["pca_components"][:, j], r["T_read_ms"], min_weight=0.2) for j in range(k))]
    row = {"k": k, "n_maps": len(maps), "routes": [], "components": []}
    for m in maps:
        present = bool(orders) and best_map_match(orders, {1.0: m}, circular, length, rng, n_null=500)["p"] < 0.05
        row["routes"].append({"order_present": bool(present)})
    for rt, sp in zip(row["routes"], span_test(r["pca_components"], maps, rng)):
        rt.update(sp)
    lat = latency_ps(r, *te, rng) if (te is not None and k) else [None] * k
    for o, p in zip(orders, lat):
        if not maps:
            c = {"one": False, "one_original": False, "route": -1}
        elif kind == "figure8":
            v = judge(o, maps[0], maps[1], rng)
            c = {"one": v in ("left", "right", "stem"), "one_original": v in ("left", "right", "stem"),
                 "route": {"left": 0, "right": 1}.get(v, -1), "verdict": v}
        elif circular:
            ok = bool(np.sum(np.isfinite(o) & np.isfinite(maps[0])) >= 6 and
                      best_map_match([o], {1.0: maps[0]}, True, length, rng, n_null=500)["p"] < 0.05)
            c = {"one": ok, "one_original": ok, "route": 0 if ok else -1}
        elif kind == "different":
            ok, kk, _ = judge_sep(o, maps, False, rng)
            c = {"one": ok, "one_original": ok, "route": kk}
        else:
            ok, kk = judge_share(o, maps, rng)
            ok0, _, _ = judge_sep(o, maps, True, rng)
            c = {"one": ok, "one_original": ok0, "route": kk}
        c["latency_p"] = p
        row["components"].append(c)
    return row


# ----------------------------------------------------------------------------- data
def planted_routes(cond, draw):
    from cpca import fit
    from scripts.run_separation import CONDITIONS
    from sim.two_routes import generate
    kw = CONDITIONS[cond]
    d = generate(np.random.default_rng(30000 + 97 * list(CONDITIONS).index(cond) + draw), **kw)
    perm = np.random.default_rng(seed("pure-split", cond, draw)).permutation(d["times"].shape[0])
    tr, te = perm[: perm.size // 2], perm[perm.size // 2:]
    rng = np.random.default_rng(seed("pure-fit", cond, draw))
    r = fit(d["times"][tr], d["T"][tr], rng=rng, ica=False, center=True, scale=True, n_shuffles=50)
    maps = d["maps"] if sum(kw["rates"]) > 0 else []
    row = evaluate(r, maps, "shared" if kw["shared"] else "different", rng, te=(d["times"][te], d["T"][te]))
    return {"group": "planted reactivations", "case": cond, "draw": draw, **row}


def planted_laps(cond, draw):
    from cpca import fit
    from sim.laps import generate
    kw = LAPS[cond]
    d = generate(np.random.default_rng(30000 + 97 * list(LAPS).index(cond) + draw), **kw)
    half = halves_by(d["direction"], np.random.default_rng(seed("pure-laps", cond, draw)))
    tr = np.flatnonzero(half)
    rng = np.random.default_rng(seed("pure-laps-fit", cond, draw))
    r = fit(d["times"][tr], d["T"][tr], rng=rng, ica=False, center=False, scale=False, n_shuffles=50)
    n_dir = kw.get("laps_per_direction", (30, 30))
    maps = [] if kw.get("shuffle") else [m for m, n in zip(d["maps"], n_dir) if n > 0]
    row = evaluate(r, maps, "shared", rng)
    return {"group": "planted laps", "case": cond, "draw": draw, **row}


def hc5_job(name, fold):
    import cpca.hc5 as hc5
    from cpca import fit
    from scripts.run_hc5 import training_data
    s = hc5.load_session(name)
    times, T, train, test, _ = training_data(s, "aligned", 0, fold)
    T = np.full(times.shape[0], T) if np.ndim(T) == 0 else np.asarray(T)
    f = hc5.time_fields(s, test)
    rng = np.random.default_rng(seed("pure-hc5", name, fold))
    r = fit(times, T, rng=rng, ica=False, center=False, scale=False, n_shuffles=50)
    row = evaluate(r, [f] if np.isfinite(f).sum() >= 6 else [], "one", rng)
    return {"group": "hc-5 wheel", "case": name, "draw": fold, **row}


def lap_job(source, key, fold):
    from cpca import event_spike_times, fit
    from cpca.hc11 import laps, place_centres, pyramidal_spikes, running
    from scripts.run_laps_hc11 import heldout_run
    if source == "hc-11":
        from cpca.hc11 import load_session, session_paths
        s = load_session(next(p for p in session_paths() if p.name.startswith(key)))
    else:
        from cpca.dandi001695 import load_session
        s = load_session(key)
    run = running(s)
    L = laps(run)
    starts, ends, dirs = (np.array([x[i] for x in L]) for i in range(3))
    half = halves_by(dirs, np.random.default_rng(seed("pure-laps-half", source, str(key))))
    train = half if fold == 0 else ~half
    tr, te = np.flatnonzero(train), np.flatnonzero(~train)
    by_cell, cell_ids, _ = pyramidal_spikes(s, np.column_stack([starts, ends]))
    X, _ = event_spike_times(by_cell, cell_ids, starts[tr], ends[tr], max_spikes=400)
    rng = np.random.default_rng(seed("pure-laps-fit", source, str(key), fold))
    r = fit(X, (ends[tr] - starts[tr]) * 1000, rng=rng, ica=False, center=False, scale=False, n_shuffles=50)
    c = place_centres(s, cell_ids, heldout_run(run, np.column_stack([starts[te], ends[te]])), min_spikes=20)
    maps = [c[sg] for sg in (1.0, -1.0) if np.isfinite(c[sg]).sum() >= 6 and np.sum(dirs[te] == sg) >= 3]
    row = evaluate(r, maps, "shared" if len(maps) > 1 else "one", rng, circular=bool(run["circular"]), length=run["length"])
    track = "circular" if run["circular"] else "linear"
    return {"group": f"{source} laps ({track})", "case": key if source == "hc-11" else Path(key).name, "draw": fold, **row}


def fig8_job(path, fold):
    import cpca.huszar as hz
    from cpca import fit
    from experimental.scripts.run_initial_real import figure8_events, judge_figure8
    s = hz.load_session(path)
    w = hz.task_window(s)[1]
    trials = hz.task_trials(s, w)
    arms = s["arm"][trials]
    cells = hz.putative_pyramidal(s)
    half = halves_by(arms, np.random.default_rng(seed("pure-fig8", s["name"])))
    train = half if fold == 0 else ~half
    times, T = figure8_events(s, trials, cells)
    held = dict(s)
    held["arm"] = s["arm"].copy()
    held["arm"][trials[train]] = "none"
    cen = hz.place_centres(*hz.arm_maps(held, cells, w))
    rng = np.random.default_rng(seed("pure-fig8-fit", s["name"], fold))
    tr = np.flatnonzero(train)
    r = fit(times[tr], T[tr], rng=rng, ica=False, center=False, scale=False, n_shuffles=50)
    row = evaluate(r, [cen[1.0], cen[-1.0]], "figure8", rng, length=hz.LENGTH_CM, judge=judge_figure8)
    return {"group": "000552 figure-eight", "case": s["name"], "draw": fold, **row}


def react_job(dataset, key, epoch, fold):
    from cpca import event_spike_times, fit, population_bursts
    from cpca.hc11 import laps, nrem_intervals, place_centres, pyramidal_spikes, running
    from cpca.replay import immobile_intervals
    from scripts.run_replay_hc11 import intersect
    if dataset == "hc-11":
        from cpca.hc11 import load_session, session_paths
        from scripts.run_replay_hc11 import two_d
        path = next(p for p in session_paths() if p.name.startswith(key))
        s = load_session(path)
        s["pos_xy"] = two_d(path)
    else:
        from cpca.dandi001695 import load_session
        s = load_session(key)
    run = running(s)

    def bursts(iv):
        by_cell, cell_ids, pooled = pyramidal_spikes(s, iv)
        b, pk = population_bursts(pooled, iv)
        return by_cell, cell_ids, pk, np.diff(b, axis=1).ravel()

    rng = np.random.default_rng(seed("pure-react", dataset, str(key), epoch))
    if epoch == "maze":
        by_cell, cell_ids, pk, d = bursts(intersect(immobile_intervals(s), [tuple(s["maze_epoch"])]))
    else:
        n = min(len(bursts(nrem_intervals(s, e))[2]) for e in ("pre", "post"))
        by_cell, cell_ids, pk, d = bursts(nrem_intervals(s, epoch))
        pick = np.sort(rng.choice(len(pk), n, replace=False))
        pk, d = pk[pick], d[pick]
    times, _ = event_spike_times(by_cell, cell_ids, pk - d / 2, pk + d / 2, max_spikes=60)
    T = d * 1000
    perm = rng.permutation(len(pk))
    halves = (perm[: perm.size // 2], perm[perm.size // 2:])
    tr, te = halves if fold == 0 else halves[::-1]
    r = fit(times[tr], T[tr], rng=rng, ica=False, center=True, scale=True, n_shuffles=50)
    dirs = np.array([x[2] for x in laps(run)])
    c = place_centres(s, cell_ids, run)
    maps = [c[sg] for sg in (1.0, -1.0) if np.isfinite(c[sg]).sum() >= 6 and np.sum(dirs == sg) >= 3]
    row = evaluate(r, maps, "shared" if len(maps) > 1 else "one", rng, circular=bool(run["circular"]), length=run["length"],
                   te=(times[te], T[te]))
    return {"group": f"{dataset} {epoch}", "case": key if dataset == "hc-11" else Path(key).name, "draw": fold, **row}


# ----------------------------------------------------------------------------- summary
def summarize(rows):
    order = ["planted reactivations", "planted laps", "hc-5 wheel", "hc-11 laps (circular)", "hc-11 laps (linear)",
             "001695 laps (linear)", "000552 figure-eight", "hc-11 maze", "hc-11 post", "hc-11 pre",
             "001695 maze", "001695 post", "001695 pre"]
    lines = ["# Pure capabilities of complex PCA", "", "Design: `docs/unmixing.md` (pure capabilities). One fit per row of the source; "
             "components read at 20% of the largest weight; no rotation, templates or split.", "",
             "| data | case | fits | components kept (median) | routes with order present | routes inside the span | components that are one route | clean fits | latency-passing components that are one route |",
             "|---|---|---|---|---|---|---|---|---|"]
    for g in order:
        rs = [r for r in rows if r["group"] == g]
        if not rs:
            continue
        cases = sorted({r["case"] for r in rs}) if g.startswith("planted") else ["all"]
        for case in cases:
            sub = [r for r in rs if case == "all" or r["case"] == case]
            routes = [x for r in sub for x in r["routes"]]
            comps = [x for r in sub for x in r["components"]]
            clean = sum(r["n_maps"] > 0 and len({c["route"] for c in r["components"] if c["one"]}) == r["n_maps"]
                        and all(c["one"] for c in r["components"]) for r in sub) if sub[0]["n_maps"] else sum(r["k"] == 0 for r in sub)
            lat = [c for c in comps if c["latency_p"] is not None and c["latency_p"] < 0.05]
            lat_txt = f"{sum(c['one'] for c in lat)}/{len(lat)}" if any(c["latency_p"] is not None for c in comps) else "–"
            pct = lambda a, b: f"{a}/{b}" if b else "–"
            lines.append(f"| {g} | {case[:34]} | {len(sub)} | {np.median([r['k'] for r in sub]):.0f} | "
                         f"{pct(sum(x['order_present'] for x in routes), len(routes))} | {pct(sum(x['in_span'] for x in routes), len(routes))} | "
                         f"{pct(sum(c['one'] for c in comps), len(comps))} | {clean}/{len(sub)} | {lat_txt} |")
    lines += ["", "For data without a sequence (planted no-replay and no-sequence controls), clean = no component kept.",
              "Components that are one route: variance-share criterion where maps share cells (the original criterion is in rows.pkl)."]
    return "\n".join(lines)


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    import cpca.dandi001695 as dd
    import cpca.hc5 as hc5
    import cpca.huszar as hz
    from scripts.run_separation import CONDITIONS
    jobs = [(planted_routes, (c, d)) for c in CONDITIONS for d in range(5)]
    jobs += [(planted_laps, (c, d)) for c in LAPS for d in range(5)]
    jobs += [(hc5_job, (n, f)) for n in hc5.session_names() for f in (0, 1)]
    jobs += [(lap_job, ("hc-11", n, f)) for n in HC11 for f in (0, 1)]
    jobs += [(lap_job, ("001695", p, f)) for p in dd.session_paths() for f in (0, 1)]
    jobs += [(fig8_job, (p, f)) for p in hz.session_paths() for f in (0, 1)]
    jobs += [(react_job, ("hc-11", n, e, f)) for n in HC11 for e in ("maze", "post", "pre") for f in (0, 1)]
    jobs += [(react_job, ("001695", p, e, f)) for p in dd.session_paths() for e in ("maze", "post", "pre") for f in (0, 1)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, *a): a for fn, a in jobs}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception:
                print(f"FAILED {futs[fut]}:\n{traceback.format_exc()[-1200:]}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(rows, open(OUT / "rows.pkl", "wb"))
    text = summarize(rows)
    (OUT / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
