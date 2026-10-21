"""hc-5 wheel benchmark: complex PCA, seqNMF and PP-Seq on the same wheel-run windows.

Question: from the spikes of a few wheel runs, without being told which cell fires when,
does a method return an output whose member cells are ordered as their time fields are
on runs it never saw?

Data (cpca/hc5.py): six alternation sessions of one rat, CA1 principal cells, one window
per wheel run. Every method gets the same windows and the same spikes; trial type and
time fields are withheld.

Cross-fitting: the runs of a session are split at random into two halves (3 repeats);
each method is fit on one half and scored against time fields computed from the other
half, then the halves swap. Conditions:

- aligned:    every window starts at wheel onset (length W);
- unaligned:  every window starts at its own random time in [0, W/3) before onset and lasts
              W + W/3, so it always contains the whole run but the runs no longer share a
              clock; the offsets are fixed per session and repeat and identical for every method;
- shuffled:   aligned windows, each cell's spikes circularly shifted by an independent
              random amount within each training run (destroys the sequence, keeps rates);
              every method should fall to chance. One repeat.

Methods, settings fixed before scoring:

- cPCA: uncentred (the wheel sequence occurs in every run, the regime where the plan's rule
  is to leave the columns uncentred) and centred; T = window length (half-circle rule);
  components above the Marchenko-Pastur bound; order = read-out times of members.
- seqNMF: lambda by the authors' crossover rule (2 x lambda_0), 200 ms bins, K = 5 and
  L = one window (the paper's guidance: K above the expected number of sequences, L about one
  sequence); a K x L sensitivity grid K in {3, 5, 10}, L in {W/2, W}. Order = lag of each
  member's peak loading.
- PP-Seq: the authors' package, time unit 1 s (the paper's hippocampal analysis is in
  seconds), 12-candidate random sweep scored on one shared held-out mask, best refit; order =
  neuron offsets of members (type rule; the package's rule also kept).
- reference (not a method): each cell's peak time in the training windows' average, i.e.
  what you get by assuming the windows are aligned. It is the ceiling when they are, and
  shows what alignment is worth when they are not.

Scoring (cpca.evaluate.best_order): signed Spearman between member order and held-out time
field, best output, against a max-over-outputs permutation null; coverage; precision.

Usage:
    python scripts/run_hc5.py [--workers=9] [--methods=cpca,reference,seqnmf,ppseq] [--redo-empty-ppseq] [session ...]
Results: results/hc5/<session>/<condition>_r<repeat>_f<fold>.<method>.pkl (existing files are skipped).
"""

import os
import pickle
import sys
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpca import fit, hc5  # noqa: E402
from cpca.decode import readout  # noqa: E402
from cpca.evaluate import best_order  # noqa: E402

OUT = ROOT / "results" / "hc5"
CONDITIONS = {"aligned": 3, "unaligned": 3, "shuffled": 1}      # condition -> repeats
JITTER = 1 / 3
METHODS = ("cpca", "reference", "seqnmf", "ppseq")
SEQ_BIN_S = 0.2
SEQ_KS = (3, 5, 10)
SEQ_HEADLINE_K = 5
PP_UNIT_MS = 1000.0
PP_GAP_UNITS = 3.0
PP_SWEEP = 20
# the paper's schedule for the final fit (20 anneal stages x 100 sweeps from temperature 500, then 100 sweeps
# with 1000 split-merge moves); a shorter one for the 20 masked sweep candidates
PP_FINAL_SCHEDULE = {"num_anneals": 20, "samples_per_anneal": 100, "max_temperature": 500.0, "save_every_during_anneal": 100,
                     "samples_after_anneal": 100, "save_every_after_anneal": 5,
                     "split_merge_moves_during_anneal": 100, "split_merge_moves_after_anneal": 1000}
PP_SWEEP_SCHEDULE = {"num_anneals": 10, "samples_per_anneal": 50, "max_temperature": 500.0, "save_every_during_anneal": 50,
                     "samples_after_anneal": 50, "save_every_after_anneal": 5,
                     "split_merge_moves_during_anneal": 100, "split_merge_moves_after_anneal": 100,
                     "num_spike_resamples_per_anneal": 5, "num_spike_resamples": 10, "samples_per_resample": 5}
PP_REFITS_IF_EMPTY = 2         # if the best candidate's refit has no latent events, try the next-best (in PP-Seq's favour)
N_NULL = 1000


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def split(session, repeat):
    n = len(session["onsets_s"])
    perm = np.random.default_rng(seed(session["name"], "split", repeat)).permutation(n)
    return np.sort(perm[: n // 2]), np.sort(perm[n // 2:])


def offsets(session, repeat):
    """Window starts relative to wheel onset for the unaligned condition: uniform in (-W/3, 0]."""
    rng = np.random.default_rng(seed(session["name"], "offsets", repeat))
    return -rng.uniform(0, JITTER * session["W_s"], len(session["onsets_s"]))


def training_data(session, condition, repeat, fold):
    """(times_ms, T_ms, train runs, test runs, window offsets) for one fit."""
    a, b = split(session, repeat)
    train, test = (a, b) if fold == 0 else (b, a)
    W = session["W_s"]
    if condition == "unaligned":
        off = offsets(session, repeat)[train]
        times, T = hc5.window_times(session, train, off, W * (1 + JITTER))
    else:
        off = np.zeros(len(train))
        times, T = hc5.window_times(session, train, off, W)
    if condition == "shuffled":
        rng = np.random.default_rng(seed(session["name"], "shuffle", repeat, fold))
        shift = rng.uniform(0, T, size=times.shape[:2])
        times = np.sort((times + shift[..., None]) % T, axis=-1)
    return times, T, train, test, off


# ----------------------------------------------------------------------------- methods
def run_cpca(times, T, rng):
    out = {}
    for label, kw in (("uncentred", dict(center=False, scale=False)), ("centred", dict(center=True, scale=True))):
        t0 = time.perf_counter()
        r = fit(times, T, rng=rng, ica=False, **kw)
        orders = []
        for j in range(r["n_kept"]):
            _, t, _, inc, _ = readout(r["patterns"][:, j], r["T_read_ms"], min_weight=0.2)
            orders.append(np.where(inc, t, np.nan))
        out[label] = {"orders": orders, "n_kept": r["n_kept"], "eigenvalues": r["eigenvalues"], "noise_bound": r["noise_bound"],
                      "runtime_s": time.perf_counter() - t0}
    return out


def run_reference(times, T, rng, bin_ms=250.0, smooth_bins=2.0, min_peak_hz=2.0):
    from scipy.ndimage import gaussian_filter1d
    t0 = time.perf_counter()
    edges = np.arange(0, T + 1e-9, bin_ms)
    counts = np.array([[np.histogram(c[np.isfinite(c)], edges)[0] for c in run] for run in times]).sum(0)
    rate = gaussian_filter1d(counts / (times.shape[0] * bin_ms / 1000), smooth_bins, axis=1, mode="nearest")
    peak = 0.5 * (edges[:-1] + edges[1:])[np.argmax(rate, axis=1)]
    order = np.where(rate.max(1) >= min_peak_hz, peak, np.nan)
    return {"average": {"orders": [order], "runtime_s": time.perf_counter() - t0}}


def run_seqnmf(times, T, rng):
    from baselines import seqnmf_baseline
    from scripts.run_baselines import blocks_from_times
    out = {}
    L_full = int(round(T / (SEQ_BIN_S * 1000)))
    for L in (L_full // 2, L_full):
        blocks = blocks_from_times(times, T, SEQ_BIN_S, pad=L)
        for K in SEQ_KS:
            t0 = time.perf_counter()
            r = seqnmf_baseline.run(blocks, SEQ_BIN_S, np.random.default_rng(int(rng.integers(1 << 30))), K=K, L=L, iters=100, restarts=3,
                                    lambda_iters=30, n_lambdas=9, test_frac=0.25)
            out[f"K{K}_L{'full' if L == L_full else 'half'}"] = {
                "orders": [f["lag_ms"] for f in r["factors"]], "K": K, "L_ms": r["L_ms"], "lambda": r["lambda"],
                "significant": [f["significant"] for f in r["factors"]], "p": [f["p"] for f in r["factors"]],
                "runtime_s": time.perf_counter() - t0}
    return out


def run_ppseq(times, T, rng):
    from baselines import ppseq_baseline as pp
    t0 = time.perf_counter()
    r = pp.fit(times, T, rng, sweep_size=PP_SWEEP, verbose=False, unit_ms=PP_UNIT_MS, gap_units=PP_GAP_UNITS, sweep_workers=1, threads=0,
               sweep_schedule=PP_SWEEP_SCHEDULE, final_schedule=PP_FINAL_SCHEDULE, refits_if_empty=PP_REFITS_IF_EMPTY)
    failed = sum(not np.isfinite(s["heldout_log_like"]) for s in r["sweep"])
    if failed > PP_SWEEP // 2:
        raise RuntimeError(f"{failed}/{PP_SWEEP} PP-Seq sweep candidates failed: {r['sweep'][0].get('error', '')}")
    # a type that explains no spikes and has no latent events in the posterior is not an output of the fit
    active = [t["spikes_final"] > 0 or t["mean_latent_events"] > 0 for t in r["types"]]
    common = {"best_setting": r["best_setting"], "sweep": r["sweep"], "n_spikes": r["n_spikes"], "runtime_s": time.perf_counter() - t0,
              "types": [{k: v for k, v in t.items() if not k.startswith("lag_ms")} for t in r["types"]], "active": active,
              "p_event": r["p_event"], "n_failed_candidates": failed, "refit_attempts": r["refit_attempts"]}
    return {"type_rule": {"orders": [t["lag_ms"] for t, a in zip(r["types"], active) if a], **common},
            "package_rule": {"orders": [t["lag_ms_package"] for t, a in zip(r["types"], active) if a], **common}}


RUNNERS = {"cpca": run_cpca, "reference": run_reference, "seqnmf": run_seqnmf, "ppseq": run_ppseq}


# ----------------------------------------------------------------------------- one job
def target(name, condition, repeat, fold, method):
    return OUT / name / f"{condition}_r{repeat}_f{fold}.{method}.pkl"


def job(name, condition, repeat, fold, method):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))
    session = hc5.load_session(name)
    times, T, train, test, off = training_data(session, condition, repeat, fold)
    key = hc5.time_fields(session, test)
    rng = np.random.default_rng(seed(name, condition, repeat, fold, method))
    t0 = time.perf_counter()
    variants = RUNNERS[method](times, T, rng)
    score_rng = np.random.default_rng(seed(name, condition, repeat, fold, method, "score"))
    for v in variants.values():
        v["score"] = best_order(v["orders"], key, score_rng, n_null=N_NULL)
    result = {"name": name, "condition": condition, "repeat": repeat, "fold": fold, "method": method,
              "train": train, "test": test, "offsets_s": off, "T_ms": T, "key": key, "n_cells": len(session["cell_ids"]),
              "variants": variants, "runtime_s": time.perf_counter() - t0}
    path = target(name, condition, repeat, fold, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pickle.dump(result, open(tmp, "wb"))
    tmp.replace(path)
    return {v: (round(x["score"]["rho"], 2) if np.isfinite(x["score"]["rho"]) else None, round(x["score"]["p"], 3) if np.isfinite(x["score"]["p"]) else None)
            for v, x in variants.items()}, result["runtime_s"]


def all_jobs(names, methods):
    jobs = []
    for name in names:
        for condition, repeats in CONDITIONS.items():
            for repeat in range(repeats):
                for fold in (0, 1):
                    for method in methods:
                        if not target(name, condition, repeat, fold, method).exists():
                            jobs.append((name, condition, repeat, fold, method))
    # longest first so the pool drains evenly
    cost = {"ppseq": 0, "seqnmf": 1, "cpca": 2, "reference": 3}
    return sorted(jobs, key=lambda j: (cost[j[4]], j[0], j[1], j[2], j[3]))


def main(argv):
    workers = max(1, (os.cpu_count() or 2) - 1)
    methods = METHODS
    names = []
    for a in argv:
        if a.startswith("--workers="):
            workers = int(a.split("=", 1)[1])
        elif a.startswith("--methods="):
            methods = tuple(a.split("=", 1)[1].split(","))
        elif not a.startswith("--"):
            names.append(a)
    names = [n for n in hc5.session_names() if not names or any(n.startswith(x) for x in names)]
    for n in names:
        hc5.load_session(n)                       # build caches before forking
    if "--redo-empty-ppseq" in argv:
        # PP-Seq fits made before the next-best-candidate fallback existed, whose refit had no latent events
        for path in OUT.glob("*/*.ppseq.pkl"):
            v = pickle.load(open(path, "rb"))["variants"]["type_rule"]
            if "refit_attempts" not in v and not any(v["active"]):
                print(f"redo {path.relative_to(OUT)}", flush=True)
                path.unlink()
    jobs = all_jobs(names, methods)
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, *j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            j = futures[fut]
            try:
                scores, runtime = fut.result()
                print(f"[{i}/{len(jobs)} {time.perf_counter() - t_start:7.0f}s] {j[0]} {j[1]} r{j[2]} f{j[3]} {j[4]}: {scores} ({runtime:.0f}s)", flush=True)
            except Exception:
                print(f"[{i}/{len(jobs)}] FAILED {j}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
