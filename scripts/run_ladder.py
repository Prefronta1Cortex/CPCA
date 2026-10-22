"""The difficulty ladder: cPCA, seqNMF and PP-Seq on condition H taken apart one difficulty at a time.

Every rung of sim.routes.LADDER adds one difficulty to "two maps on the same cells, both
directions" (L2): partial replays, assemblies, variable pacing, and all of H. L0 (one map) and
L1 (two maps, forward only) are the steps below. The question is which difficulty opens the gap
between cPCA and PP-Seq seen on condition H.

Matched data: for each rung and draw one set of 1500 planted events is generated and every method
gets exactly those events. Settings are the ones of the matched rerun (scripts/run_rerun.py):
cPCA uncentred and centred (centred is the regime of a minority of replays, 40% here);
seqNMF with the lambda-crossover rule on a K x L grid (headline K = 15, L = 120 ms); PP-Seq with
the 20-candidate held-out sweep, plus the refit-if-empty fallback (in PP-Seq's favour).

Score per planted map: the best output's |Spearman| between member order and the map, with a
p-value against a null that takes the best of the method's outputs (both signs) under each
permutation of the map (cpca.evaluate.best_order on the orders and their reverses).

Usage: python scripts/run_ladder.py [--workers=9] [--methods=cpca,seqnmf,ppseq]
Results: results/ladder/<rung>_d<draw>.<method>.pkl (existing files are skipped).
"""

import os
import pickle
import sys
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpca.evaluate import best_order  # noqa: E402
from sim.routes import LADDER, generate, make_routes  # noqa: E402

OUT = ROOT / "results" / "ladder"
N_DRAWS = 3
N_EVENTS = 1500
N_CELLS = 80
METHODS = ("cpca", "seqnmf", "ppseq")


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def rung_code(name):
    return name.split(" ", 1)[0]


def make_data(name, draw):
    gen_kwargs, route_kwargs = LADDER[name]
    rng = np.random.default_rng(seed("ladder", rung_code(name), draw))
    routes = make_routes(rng, n_cells=N_CELLS, **route_kwargs)
    return generate(rng, n_cells=N_CELLS, n_events=N_EVENTS, routes=routes, **gen_kwargs)


def score(orders, centres, rng, n_null=500):
    """Per planted map: best |rho| over outputs and its max-over-outputs p (both signs allowed)."""
    both = [np.asarray(o, float) for o in orders] + [-np.asarray(o, float) for o in orders]
    out = []
    for cen in centres:
        s = best_order(both, cen, rng, n_null=n_null)
        out.append({"rho": s["rho"], "p": s["p"], "null95": s["null95"], "n": s["n"], "coverage": s["coverage"], "n_outputs": len(orders)})
    return out


def run_cpca(times, T, rng):
    from scripts.run_rerun import run_cpca as rr
    t0 = time.perf_counter()
    r = rr(times, T, rng)
    for v in r.values():
        v["runtime_s"] = (time.perf_counter() - t0) / 2
    return r


def run_seqnmf(times, T, rng):
    from scripts.run_rerun import run_seqnmf as rr
    r = rr(times, T, rng)
    return {f"K{g['K']}_L{g['L_ms']:.0f}": {"orders": [f["lag_ms"] for f in g["factors"]], "K": g["K"], "L_ms": g["L_ms"],
                                           "lambda": g["lambda"], "runtime_s": g["runtime_s"]} for g in r["grid"]}


RUNNERS = {"cpca": run_cpca, "seqnmf": run_seqnmf}


# ----------------------------------------------------------------------------- PP-Seq as fine-grained jobs
# A PP-Seq fit is 20 masked sweep candidates and one refit. Run as one job per fit, the last wave
# leaves most cores idle; here every candidate is its own (cached) job and the refit starts as soon
# as its candidates are done. The plan (settings, mask, seeds) is rebuilt identically in every job,
# so the result is the same as baselines.ppseq_baseline.fit with the same rng.
SWEEP_DIR = OUT / "ppseq_sweep"
PP_SWEEP_SIZE = 20


def ppseq_plan(name, draw):
    from baselines import ppseq_baseline as pp
    data = make_data(name, draw)
    rng = np.random.default_rng(seed("fit", rung_code(name), draw, "ppseq"))
    return data, pp.plan_fit(data["times"], data["event_ms"], rng, sweep_size=PP_SWEEP_SIZE)


def candidate_path(name, draw, i):
    return SWEEP_DIR / f"{rung_code(name)}_d{draw}_c{i:02d}.json"


def ppseq_candidate_job(name, draw, i):
    import json
    import tempfile
    from baselines import ppseq_baseline as pp
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _, plan = ppseq_plan(name, draw)
    with tempfile.TemporaryDirectory() as wd:
        row = pp.sweep_candidate(plan, i, Path(wd))
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = candidate_path(name, draw, i).with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    tmp.replace(candidate_path(name, draw, i))
    return row["heldout_log_like"], row["runtime_s"]


def ppseq_final_job(name, draw):
    import json
    import tempfile
    from baselines import ppseq_baseline as pp
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.perf_counter()
    data, plan = ppseq_plan(name, draw)
    sweep = [json.loads(candidate_path(name, draw, i).read_text()) for i in range(PP_SWEEP_SIZE)]
    failed = sum(not np.isfinite(s["heldout_log_like"]) for s in sweep)
    if failed > PP_SWEEP_SIZE // 2:
        raise RuntimeError(f"{failed} PP-Seq sweep candidates failed")
    with tempfile.TemporaryDirectory() as wd:
        parsed, best, attempts = pp.refit_best(plan, sweep, Path(wd), threads=0, refits_if_empty=2)
    common = {"best_setting": best, "refit_attempts": attempts, "n_failed_candidates": failed, "sweep": sweep,
              "runtime_s": time.perf_counter() - t0 + float(np.nansum([s["runtime_s"] for s in sweep])), "n_spikes": int(plan["spikes"].shape[0])}
    variants = {"type_rule": {"orders": [t["lag_ms"] for t in parsed["types"]], **common},
                "package_rule": {"orders": [t["lag_ms_package"] for t in parsed["types"]], **common}}
    return finish(name, draw, "ppseq", data, variants, t0)


def target(name, draw, method):
    return OUT / f"{rung_code(name)}_d{draw}.{method}.pkl"


def finish(name, draw, method, data, variants, t0):
    score_rng = np.random.default_rng(seed("score", rung_code(name), draw, method))
    for v in variants.values():
        v["maps"] = score(v["orders"], data["centres"], score_rng)
    result = {"rung": name, "draw": draw, "method": method, "fraction_replay": float(np.mean(data["label"] > 0)),
              "variants": variants, "runtime_s": time.perf_counter() - t0}
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = target(name, draw, method).with_suffix(".tmp")
    pickle.dump(result, open(tmp, "wb"))
    tmp.replace(target(name, draw, method))
    return {k: [round(m["rho"], 2) if np.isfinite(m["rho"]) else None for m in v["maps"]] for k, v in variants.items()
            if k in ("centred", "uncentred", "K15_L120", "type_rule")}, result["runtime_s"]


def job(name, draw, method):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))
    data = make_data(name, draw)
    rng = np.random.default_rng(seed("fit", rung_code(name), draw, method))
    t0 = time.perf_counter()
    variants = RUNNERS[method](data["times"], data["event_ms"], rng)
    return finish(name, draw, method, data, variants, t0)


def main(argv):
    """Priority scheduler: ready PP-Seq refits, then PP-Seq candidates, then seqNMF, then cPCA; every core stays busy."""
    from concurrent.futures import FIRST_COMPLETED, wait
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), os.cpu_count() or 2)
    methods = next((tuple(a.split("=", 1)[1].split(",")) for a in argv if a.startswith("--methods=")), METHODS)
    fits = [(name, d) for name in LADDER for d in range(N_DRAWS)]
    candidates, finals_waiting, finals_ready, others = [], set(), [], []
    if "ppseq" in methods:
        for name, d in fits:
            if target(name, d, "ppseq").exists():
                continue
            todo = [i for i in range(PP_SWEEP_SIZE) if not candidate_path(name, d, i).exists()]
            candidates += [(name, d, i) for i in todo]
            (finals_waiting.add((name, d)) if todo else finals_ready.append((name, d)))
    for m in ("seqnmf", "cpca"):
        if m in methods:
            others += [(name, d, m) for name, d in fits if not target(name, d, m).exists()]
    total = len(candidates) + len(finals_waiting) + len(finals_ready) + len(others)
    print(f"{total} jobs ({len(candidates)} PP-Seq candidates, {len(finals_waiting) + len(finals_ready)} PP-Seq refits, {len(others)} other) on {workers} workers", flush=True)
    t0, done = time.perf_counter(), 0

    def next_job():
        if finals_ready:
            nd = finals_ready.pop(0)
            return ("final",) + nd, ppseq_final_job, nd
        if candidates:
            c = candidates.pop(0)
            return ("cand",) + c, ppseq_candidate_job, c
        if others:
            o = others.pop(0)
            return ("other",) + o, job, o
        return None

    with ProcessPoolExecutor(max_workers=workers) as pool:
        running = {}
        while True:
            while len(running) < workers:
                nxt = next_job()
                if nxt is None:
                    break
                key, fn, args = nxt
                running[pool.submit(fn, *args)] = key
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                key = running.pop(fut)
                done += 1
                stamp = f"[{done}/{total} {time.perf_counter() - t0:6.0f}s]"
                try:
                    res = fut.result()
                except Exception:
                    print(f"{stamp} FAILED {key}:\n{traceback.format_exc()}", flush=True)
                    continue
                if key[0] == "cand":
                    _, name, d, i = key
                    if all(candidate_path(name, d, j).exists() for j in range(PP_SWEEP_SIZE)) and (name, d) in finals_waiting:
                        finals_waiting.discard((name, d))
                        finals_ready.append((name, d))
                    if i == PP_SWEEP_SIZE - 1 or done % 20 == 0:
                        print(f"{stamp} PP-Seq candidate {rung_code(name)} d{d} c{i}: held-out {res[0]:.3f} ({res[1]:.0f}s)", flush=True)
                else:
                    name, d = key[1], key[2]
                    label = "ppseq" if key[0] == "final" else key[3]
                    print(f"{stamp} {rung_code(name)} d{d} {label}: {res[0]} ({res[1]:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
