"""Per-component separation benchmark (docs/separation.md): do routes separate onto components?

Usage: python scripts/run_separation.py [--workers=9]
Output: results/separation/<condition>_<draw>.pkl, results/separation/summary.md
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

OUT = ROOT / "results" / "separation"
CONDITIONS = {
    "A different cells, 10% + 10%, both ways": dict(rates=(0.10, 0.10), shared=False),
    "B different cells, 13% + 7%, both ways": dict(rates=(0.13, 0.07), shared=False),
    "C same cells, 10% + 10%, forward": dict(rates=(0.10, 0.10), shared=True, reverse_frac=0.0),
    "D same cells, 10% + 10%, both ways": dict(rates=(0.10, 0.10), shared=True),
    "E one map, 20%, both ways": dict(rates=(0.20,), shared=False),
    "F no replays": dict(rates=(0.0,), shared=False),
}
METHODS = ["centred PCA", "centred ICA", "uncentred ICA", "templates"]


def spearman_p(order, pos, rng, n_perm=1000):
    from scipy.stats import rankdata
    a, b = rankdata(order), rankdata(pos)
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    if den == 0:
        return 0.0, 1.0
    r = abs(a @ b) / den
    null = np.abs(np.array([a @ rng.permutation(b) for _ in range(n_perm)])) / den
    return float(r), float((1 + np.sum(null >= r)) / (1 + n_perm))


def judge(order, maps, shared, rng):
    """Per-output criterion of docs/separation.md. Returns (passes, map index or -1, details)."""
    mem = np.isfinite(order)
    on = [mem & np.isfinite(m) for m in maps]
    if shared:
        res = []
        for m, o in zip(maps, on):
            res.append(spearman_p(order[o], m[o], rng) if o.sum() >= 6 else (0.0, 1.0))
        sig = [p < 0.05 for _, p in res]
        if sum(sig) == 1:
            k = sig.index(True)
            return True, k, {"rho": res[k][0], "rhos": [r for r, _ in res]}
        return False, -1, {"rhos": [r for r, _ in res]}
    counts = np.array([o.sum() for o in on])
    if counts.sum() == 0:
        return False, -1, {"purity": 0.0}
    k = int(np.argmax(counts))
    purity = counts[k] / counts.sum()
    if counts[k] < 6 or purity < 0.8:
        return False, -1, {"purity": float(purity)}
    r, p = spearman_p(order[on[k]], maps[k][on[k]], rng)
    return bool(p < 0.05), (k if p < 0.05 else -1), {"purity": float(purity), "rho": r}


def method_outputs(method, tr_times, tr_T, te_times, te_T, rng):
    """Candidate outputs (order, latency-check p) of one method fitted on the training half."""
    from cpca.readout_check import timing_check
    if method != "templates":
        centred = method.startswith("centred")
        _, comps = timing_check(tr_times, tr_T, te_times, te_T, rng, center=centred, scale=centred, ica=method.endswith("ICA"))
        return [(o["order"], o["p_timing"]) for o in comps]
    from cpca import fit
    from cpca.decode import readout
    from cpca.decompose import prepare
    from cpca.encode import encode
    from cpca.mixture import sequences
    from cpca.readout_check import _expression, redraw_latency
    r = fit(tr_times, tr_T, rng=rng, ica=False, center=True, scale=True, n_shuffles=50)
    if not r["n_kept"]:
        return []
    fitted = sequences(r, rng, test=False)
    if fitted is None or not fitted["templates"].shape[1]:
        return []
    Q = fitted["templates"]
    Z = prepare(encode(te_times, te_T), center=False, scale=False)
    null = np.array([[_expression(prepare(encode(redraw_latency(te_times, te_T, rng), te_T), center=False, scale=False), Q[:, j])
                      for j in range(Q.shape[1])] for _ in range(50)])
    out = []
    for j in range(Q.shape[1]):
        e = _expression(Z, Q[:, j])
        _, t, _, inc, _ = readout(Q[:, j], r["T_read_ms"], min_weight=0.2)
        out.append((np.where(inc, t, np.nan), float((1 + np.sum(null[:, j] >= e)) / 51)))
    return out


def job(cond, draw):
    from sim.two_routes import generate
    kw = CONDITIONS[cond]
    rng = np.random.default_rng(5000 + 97 * list(CONDITIONS).index(cond) + draw)
    d = generate(rng, **kw)
    perm = np.random.default_rng(draw + 1).permutation(d["times"].shape[0])
    halves = (perm[: perm.size // 2], perm[perm.size // 2:])
    maps = d["maps"] if sum(kw["rates"]) > 0 else []
    rows = []
    for fold, (tr, te) in enumerate((halves, halves[::-1])):
        for m in METHODS:
            frng = np.random.default_rng(draw * 31 + fold)
            try:
                outs = method_outputs(m, d["times"][tr], d["T"][tr], d["times"][te], d["T"][te], frng)
            except Exception as err:                                             # a method that fails to fit has no outputs
                outs, note = [], repr(err)[:200]
            else:
                note = ""
            judged = []
            for order, p in outs:
                ok, k, det = judge(order, maps, kw["shared"], np.random.default_rng(draw + len(judged))) if maps else (False, -1, {})
                judged.append({"p_timing": p, "passes_check": p < 0.05, "passes": ok, "map": k, **det})
            rows.append({"condition": cond, "draw": draw, "fold": fold, "method": m, "outputs": judged, "n_maps": len(maps), "note": note})
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(rows, open(OUT / f"{list(CONDITIONS).index(cond)}_{draw}.pkl", "wb"))
    return " | ".join(f"{r['method']}: {sum(o['passes_check'] for o in r['outputs'])} outputs, {sum(o['passes_check'] and o['passes'] for o in r['outputs'])} pass"
                      for r in rows if r["fold"] == 0)


def summary():
    rows = [r for f in sorted(OUT.glob("*_*.pkl")) for r in pickle.load(open(f, "rb"))]
    lines = ["# Per-component separation benchmark", "", "Produced by `scripts/run_separation.py`. Design: `docs/separation.md`. "
             "Outputs: components (or templates) that pass the latency check on held-out events. 10 fits per cell (5 draws x 2 folds).", "",
             "| condition | method | outputs per fit (median; before the check) | outputs passing the criterion | maps represented (fits with all maps) | clean fits |",
             "|---|---|---|---|---|---|"]
    for cond in CONDITIONS:
        for m in METHODS:
            rs = [r for r in rows if r["condition"] == cond and r["method"] == m]
            if not rs:
                continue
            outs = [[o for o in r["outputs"] if o["passes_check"]] for r in rs]
            n_out = [len(o) for o in outs]
            n_raw = [len(r["outputs"]) for r in rs]
            n_pass = sum(sum(o["passes"] for o in oo) for oo in outs)
            total = sum(n_out)
            all_maps = sum(rs[i]["n_maps"] > 0 and len({o["map"] for o in outs[i] if o["passes"]}) == rs[i]["n_maps"] for i in range(len(rs)))
            clean = sum(rs[i]["n_maps"] > 0 and len({o["map"] for o in outs[i] if o["passes"]}) == rs[i]["n_maps"] and all(o["passes"] for o in outs[i])
                        for i in range(len(rs)))
            if rs[0]["n_maps"] == 0:
                clean = sum(n == 0 for n in n_out)
            share = f"{n_pass}/{total} ({n_pass / total:.0%})" if total else "0/0"
            lines.append(f"| {cond} | {m} | {np.median(n_out):.0f} ({np.median(n_raw):.0f}) | {share} | {all_maps}/{len(rs)} | {clean}/{len(rs)} |")
    lines += ["", "No replays (F): a clean fit is one with no outputs."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv):
    if "--summary" in argv:
        return summary()
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    todo = [(c, d) for c in CONDITIONS for d in range(5) if not (OUT / f"{list(CONDITIONS).index(c)}_{d}.pkl").exists()]
    print(f"{len(todo)} jobs on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, *a): a for a in todo}
        for fut in as_completed(futures):
            try:
                print(f"{futures[fut]}: {fut.result()}", flush=True)
            except Exception:
                print(f"FAILED {futures[fut]}:\n{traceback.format_exc()}", flush=True)
    summary()


if __name__ == "__main__":
    main(sys.argv[1:])
