"""Steps 2 and 3 of docs/adoption.md: detection with active centring, and the separation variants V0-V3 (arc pi).

Step 2: spike-in on all 8 hc-11 sessions (matched 1500 PRE events, ideal replays, 5 draws per share), column
and active centring on the same draws, best output's order against the map.
Step 3: planted reactivations A-F (confirmation draws) and G-M (stress confirmation draws), 5 draws x 2 folds,
variants V0-V3; and the waking re-check of the current step after the ICA fix.

Usage: python scripts/run_adoption_check.py [--workers=9]
Output: results/adoption/rows.pkl, summary.md
"""

import os
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_separation_check import HC11, hc5_job, hc11_job, laps_job, orders_of, score, seed  # noqa: E402

OUT = ROOT / "results" / "adoption"
SHARES = (0.0, 0.02, 0.03, 0.05, 0.10, 0.20)
VARIANTS = ("V0", "V1", "V2", "V3")


def spikein_job(name, f, draw):
    from cpca import event_spike_times, fit, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_rerun import ORDER_INDEX, fixed_subset
    from scripts.run_spikein_hc11 import ideal
    s = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(s)
    iv = nrem_intervals(s, "pre")
    by_cell, ids, pooled = pyramidal_spikes(s, iv)
    b, pk = population_bursts(pooled, iv)
    dur = np.diff(b, axis=1).ravel()
    sub = fixed_subset(len(pk), 555 + 2 * ORDER_INDEX[name] + 1)
    base, _ = event_spike_times(by_cell, ids, (pk - dur / 2)[sub], (pk + dur / 2)[sub], max_spikes=60)
    T = dur[sub] * 1000
    centres = place_centres(s, ids, run)
    rng = np.random.default_rng(seed("adopt-spike", name, f, draw))
    k = int(round(f * len(T)))
    times = ideal(base, T, rng.choice(len(T), k, replace=False), centres, rng) if k else base
    rows = []
    for center in (True, "active"):
        r = fit(times, T, rng=np.random.default_rng(seed("adopt-spikefit", name, f, draw)), center=center, scale=True, ica=False)
        orders = [rd["times_ms"] for rd in r["readouts"]]
        p = best_map_match(orders, centres, run["circular"], run["length"], np.random.default_rng(seed("adopt-spikesc", name, f, draw)),
                           n_null=500)["p"] if orders else np.nan
        rows.append({"set": "spike-in", "case": name, "share": f, "draw": draw, "center": "active" if center == "active" else "column",
                     "k": r["n_kept"], "hit": bool(np.isfinite(p) and p < 0.05)})
    return rows


def _direct(r):
    from cpca.separate import MERGE, similarity
    kept = []
    for j in range(r["n_kept"]):
        if all(similarity(r["patterns"][:, j], r["patterns"][:, i]) < MERGE for i in kept):
            kept.append(j)
    return [r["readouts"][j]["times_ms"] for j in kept]


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
        args = (d["times"][tr], d["T"][tr], d["times"][te], d["T"][te])
        for v in VARIANTS:
            rng = np.random.default_rng(seed("fit", phase, cond, draw, fold))
            r = fit(d["times"][tr], d["T"][tr], rng=rng, ica=v in ("V2", "V3"), center=True if v == "V0" else "active",
                    scale=True, n_shuffles=50)
            if v == "V3":
                orders = _direct(r)
            else:
                outs, _ = separate(r, *args, rng, reversal=True, min_events=20, rotation="ica" if v == "V2" else "quartimax")
                orders = orders_of(outs, r["T_read_ms"])
            rows.append({"set": "planted reactivations", "case": cond, "phase": phase, "draw": draw, "fold": fold, "stage": v,
                         "k": r["n_kept"], "n_maps": len(maps),
                         "verdicts": score(orders, maps, kw["shared"], np.random.default_rng(seed("judge", phase, cond, draw, fold)))})
    return rows


def clean(r, crit):
    o = [v[crit] for v in r["verdicts"]]
    if r["n_maps"] == 0:
        return len(o) == 0
    return len({k for ok, k in o if ok}) == r["n_maps"] and all(ok for ok, _ in o)


def summarize(rows):
    L = ["# Adoption check (docs/adoption.md), arc pi", ""]
    sp = [r for r in rows if r["set"] == "spike-in"]
    if sp:
        L += ["## Step 2: spike-in, hits (best output's order p < 0.05) and median components kept", "",
              "| sessions | centring | " + " | ".join(f"{f:.0%}" for f in SHARES) + " |", "|---|---|" + "---|" * len(SHARES)]
        for grp, names in (("two largest", HC11[:2]), ("other six", HC11[2:])):
            for c in ("column", "active"):
                cells = []
                for f in SHARES:
                    rs = [r for r in sp if r["case"] in names and r["center"] == c and r["share"] == f]
                    cells.append(f"{sum(r['hit'] for r in rs)}/{len(rs)} (k {np.median([r['k'] for r in rs]):.0f})")
                L.append(f"| {grp} | {c} | " + " | ".join(cells) + " |")
        L.append("")
    pr = [r for r in rows if r["set"] == "planted reactivations"]
    if pr:
        L += ["## Step 3: planted reactivations, clean fits of 10 (variance share for shared maps; original in brackets)", "",
              "| condition | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
        tot = {v: [0, 0] for v in VARIANTS}
        for case in sorted({r["case"] for r in pr}):
            cells = []
            for v in VARIANTS:
                rs = [r for r in pr if r["case"] == case and r["stage"] == v]
                a, b = sum(clean(r, "share") for r in rs), sum(clean(r, "original") for r in rs)
                tot[v][0] += a
                tot[v][1] += b
                cells.append(f"{a} ({b})")
            L.append(f"| {case[:44]} | " + " | ".join(cells) + " |")
        L.append("| total | " + " | ".join(f"{a} ({b})" for a, b in tot.values()) + " |")
        L.append("")
    wk = [r for r in rows if r["set"] in ("planted laps", "hc-11 laps (linear)", "hc-11 laps (circular)", "hc-5 wheel")]
    if wk:
        L += ["## Waking re-check, current step after the ICA fix: clean fits (original / variance share / partial)", "",
              "| data | clean fits |", "|---|---|"]
        for key in sorted({(r["set"], r["case"] if r["set"] == "planted laps" else "") for r in wk}):
            rs = [r for r in wk if (r["set"], r["case"] if r["set"] == "planted laps" else "") == key]
            L.append(f"| {key[0]} {key[1][:40]} | " + " / ".join(f"{sum(clean(r, c) for r in rs)}/{len(rs)}"
                                                           for c in ("original", "share", "partial")) + " |")
    return "\n".join(L)


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    import cpca.hc5 as hc5
    import experimental.scripts.dev_unmixing as dv
    from experimental.scripts.dev_waking import CONDITIONS as LAPS
    from scripts.run_separation import CONDITIONS as REACT
    jobs = [(spikein_job, (n, f, d)) for n in HC11 for f in SHARES for d in range(5)]
    jobs += [(react_job, (c, d, "confirm")) for c in REACT for d in range(5)]
    jobs += [(react_job, (c, d, "stress_confirm")) for c in dv.STRESS for d in range(5)]
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
