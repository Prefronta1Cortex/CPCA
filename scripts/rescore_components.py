"""Per-component re-scoring of hc-5, hc-11 laps and hc-11 sleep (docs/rescore.md).

Usage: python scripts/rescore_components.py [--workers=9]
Output: results/rescore/*.pkl, results/rescore/summary.md
"""

import os
import pickle
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "rescore"
HC11 = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
        "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]


def comp_orders(r, patterns=None):
    from cpca.decode import readout
    P = r["pca_components"] if patterns is None else patterns
    out = []
    for j in range(P.shape[1]):
        _, t, _, inc, _ = readout(P[:, j], r["T_read_ms"], min_weight=0.2)
        out.append(np.where(inc, t, np.nan))
    return out


def one_map(order, centres, circular, length, rng):
    """Single-output test per map; passes if exactly one map (of those with at least 6 member fields) matches."""
    from scripts.plot_hc11_sequences import best_map_match
    ps = {}
    for s, c in centres.items():
        if np.sum(np.isfinite(order) & np.isfinite(c)) >= 6:
            ps[s] = best_map_match([order], {s: c}, circular, length, rng, n_null=500)["p"]
    sig = [s for s, p in ps.items() if np.isfinite(p) and p < 0.05]
    return len(sig) == 1, (sig[0] if len(sig) == 1 else None), ps


# ----------------------------------------------------------------------------- hc-5
def hc5_job(name, condition, repeat, fold):
    from cpca import fit, hc5
    from scipy.stats import spearmanr
    from scripts.run_hc5 import training_data
    s = hc5.load_session(name)
    times, T, train, test, _ = training_data(s, condition, repeat, fold)
    key = hc5.time_fields(s, test)
    rng = np.random.default_rng(hash((name, condition, repeat, fold)) % (2 ** 32))
    res = {}
    for conv, c in (("uncentred", False), ("centred", True)):
        r = fit(times, T, rng=rng, ica=False, center=c, scale=c)
        comps = []
        for o in comp_orders(r):
            m = np.isfinite(o) & np.isfinite(key)
            if m.sum() < 6:
                comps.append({"passes": False, "n": int(m.sum())})
                continue
            rho = spearmanr(o[m], key[m]).statistic
            null = np.array([spearmanr(o[m], rng.permutation(key[m])).statistic for _ in range(1000)])
            p = (1 + np.sum(np.abs(null) >= abs(rho))) / 1001
            comps.append({"passes": bool(p < 0.05), "rho": float(rho), "p": float(p), "n": int(m.sum())})
        res[conv] = comps
    return ("hc5", name, condition, repeat, fold), res


# ----------------------------------------------------------------------------- hc-11 laps
def laps_job(name, split, fold):
    from cpca import event_spike_times, fit
    from cpca.hc11 import laps, load_session, place_centres, pyramidal_spikes, running, session_paths
    from scripts.run_laps_hc11 import heldout_run, seed
    s = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(s)
    L = laps(run)
    starts, ends, dirs = (np.array([x[i] for x in L]) for i in range(3))
    rng = np.random.default_rng(seed("laps", name, split))
    half = np.zeros(len(L), bool)
    for d in np.unique(dirs):
        idx = rng.permutation(np.flatnonzero(dirs == d))
        half[idx[: idx.size // 2]] = True
    train = half if fold == 0 else ~half
    by_cell, cell_ids, _ = pyramidal_spikes(s, np.column_stack([starts, ends]))
    times, _ = event_spike_times(by_cell, cell_ids, starts[train], ends[train], max_spikes=400)
    T = (ends[train] - starts[train]) * 1000
    centres = place_centres(s, cell_ids, heldout_run(run, np.column_stack([starts[~train], ends[~train]])), min_spikes=20)
    centres = {k: v for k, v in centres.items() if np.isfinite(v).sum() >= 6}
    frng = np.random.default_rng(seed("laps-fit", name, split, fold))
    res = {}
    for conv, c in (("uncentred", False), ("centred", True)):
        r = fit(times, T, rng=frng, ica=False, center=c, scale=c)
        comps = []
        for o in comp_orders(r):
            ok, which, ps = one_map(o, centres, run["circular"], run["length"], frng)
            comps.append({"passes": ok, "map": which, "p": ps})
        res[conv] = {"components": comps, "maps": list(centres)}
    return ("laps", name, split, fold), res


# ----------------------------------------------------------------------------- hc-11 sleep
def sleep_job(name, epoch, fold):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    from scripts.run_separation import method_outputs
    s = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(s)
    iv = nrem_intervals(s, epoch)
    by_cell, cell_ids, pooled = pyramidal_spikes(s, iv)
    b, pk = population_bursts(pooled, iv)
    d = np.diff(b, axis=1).ravel()
    times, _ = event_spike_times(by_cell, cell_ids, pk - d / 2, pk + d / 2, max_spikes=60)
    T = d * 1000
    perm = np.random.default_rng(hash((name, epoch)) % (2 ** 32)).permutation(len(pk))
    halves = (perm[: perm.size // 2], perm[perm.size // 2:])
    tr, te = halves if fold == 0 else halves[::-1]
    centres = {k: v for k, v in place_centres(s, cell_ids, run).items() if np.isfinite(v).sum() >= 6}
    rng = np.random.default_rng(hash((name, epoch, fold)) % (2 ** 32))
    res = {}
    for m in ("centred PCA", "templates"):
        outs = []
        for order, p in method_outputs(m, times[tr], T[tr], times[te], T[te], rng):
            if p >= 0.05:
                continue
            ok, which, ps = one_map(order, centres, run["circular"], run["length"], rng)
            outs.append({"passes": ok, "map": which, "p": ps, "p_timing": p})
        res[m] = outs
    return ("sleep", name, epoch, fold), res


def summary(R):
    lines = ["# Per-component re-scoring", "", "Produced by `scripts/rescore_components.py`. Design: `docs/rescore.md`. Every entry counts output "
             "components (not the best one).", ""]
    hc = {k: v for k, v in R.items() if k[0] == "hc5"}
    lines += ["## hc-5 wheel runs (6 sessions x 2 window conditions x 3 repeats x 2 folds)", "",
              "| windows | convention | components per fit (median) | components that are the sequence | fits where every component is |", "|---|---|---|---|---|"]
    for cond in ("aligned", "unaligned"):
        for conv in ("uncentred", "centred"):
            fits = [v[conv] for k, v in hc.items() if k[2] == cond]
            n = [len(f) for f in fits]
            p = sum(sum(c["passes"] for c in f) for f in fits)
            allp = sum(len(f) > 0 and all(c["passes"] for c in f) for f in fits)
            lines.append(f"| {cond} | {conv} | {np.median(n):.0f} | {p}/{sum(n)} ({p / max(sum(n), 1):.0%}) | {allp}/{len(fits)} |")
    lp = {k: v for k, v in R.items() if k[0] == "laps"}
    lines += ["", "## hc-11 waking laps (8 sessions x 3 splits x 2 folds)", "",
              "| session | convention | components per fit (median) | components matching exactly one direction | fits with every direction represented | fits where every component passes |",
              "|---|---|---|---|---|---|"]
    for n_ in HC11:
        for conv in ("uncentred", "centred"):
            fits = [v[conv] for k, v in lp.items() if k[1] == n_]
            if not fits:
                continue
            n = [len(f["components"]) for f in fits]
            p = sum(sum(c["passes"] for c in f["components"]) for f in fits)
            rep = sum(set(f["maps"]) <= {c["map"] for c in f["components"] if c["passes"]} for f in fits)
            allp = sum(len(f["components"]) > 0 and all(c["passes"] for c in f["components"]) for f in fits)
            lines.append(f"| {n_} | {conv} | {np.median(n):.0f} | {p}/{sum(n)} ({p / max(sum(n), 1):.0%}) | {rep}/{len(fits)} | {allp}/{len(fits)} |")
    sl = {k: v for k, v in R.items() if k[0] == "sleep"}
    lines += ["", "## hc-11 sleep (all NREM bursts, 2 folds; outputs = components passing the latency check)", "",
              "| session | epoch | method | outputs (fold 1 + fold 2) | outputs matching exactly one direction |", "|---|---|---|---|---|"]
    tot = {}
    for n_ in HC11:
        for ep in ("post", "pre"):
            for m in ("centred PCA", "templates"):
                fits = [v[m] for k, v in sl.items() if k[1] == n_ and k[2] == ep]
                if not fits:
                    continue
                n = sum(len(f) for f in fits)
                p = sum(sum(o["passes"] for o in f) for f in fits)
                t = tot.setdefault((ep, m), [0, 0, 0])
                t[0] += n
                t[1] += p
                t[2] += p > 0
                lines.append(f"| {n_} | {ep.upper()} | {m} | {n} | {p} |")
    lines += ["", "| epoch | method | outputs | outputs matching exactly one direction | sessions with at least one |", "|---|---|---|---|---|"]
    for (ep, m), (n, p, s_) in sorted(tot.items()):
        lines.append(f"| {ep.upper()} | {m} | {n} | {p} ({p / max(n, 1):.0%}) | {s_}/8 |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv):
    from cpca import hc5
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    warnings.filterwarnings("ignore")
    jobs = [(hc5_job, (n, c, r, f)) for n in hc5.session_names() for c in ("aligned", "unaligned") for r in range(3) for f in (0, 1)]
    jobs += [(laps_job, (n, s, f)) for n in HC11 for s in range(3) for f in (0, 1)]
    jobs += [(sleep_job, (n, e, f)) for n in HC11 for e in ("post", "pre") for f in (0, 1)]
    R = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *a): a for fn, a in jobs}
        for fut in as_completed(futures):
            try:
                k, v = fut.result()
                R[k] = v
            except Exception:
                print(f"FAILED {futures[fut]}:\n{traceback.format_exc()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(R, open(OUT / "results.pkl", "wb"))
    summary(R)


if __name__ == "__main__":
    main(sys.argv[1:])
