"""Development harness for un-mixing (docs/unmixing.md, "Development"): development draws only (seed base 7000).

Usage: python scripts/dev_unmixing.py <variant> [--workers=9] [--draws=5] [--conditions=ABCDEF] [--phase=dev|confirm]
Prints the per-output table; results/unmixing/dev_<variant>.pkl
"""

import os
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import experimental.scripts.run_unmixing as ru  # noqa: E402
from scripts.run_separation import CONDITIONS, judge  # noqa: E402

ru.SEED_BASE["dev"] = 7000
STRESS = {
    "G three routes, different cells, 7% each": dict(rates=(0.07, 0.07, 0.07), shared=False),
    "H different cells, 5% + 5%": dict(rates=(0.05, 0.05), shared=False),
    "I different cells, 3% + 3%": dict(rates=(0.03, 0.03), shared=False),
    "J different cells, 25 cells per route, 10% + 10%": dict(rates=(0.10, 0.10), shared=False, n_route=25),
    "K same cells, 5% + 5%, both ways": dict(rates=(0.05, 0.05), shared=True),
    "L three maps, same cells, 7% each": dict(rates=(0.07, 0.07, 0.07), shared=True),
    "M one map, 5%": dict(rates=(0.05,), shared=False),
}


def get_data(cond, draw, phase):
    if phase not in ("stress", "stress_dev", "stress_confirm"):
        return ru.make_data(cond, draw, phase)
    from sim.two_routes import generate
    kw = STRESS[cond]
    base = {"stress": 11000, "stress_dev": 13000, "stress_confirm": 15000}[phase]
    rng = np.random.default_rng(base + 97 * list(STRESS).index(cond) + draw)
    return generate(rng, **kw), kw
OUT = ROOT / "results" / "unmixing"

VARIANTS = {
    "sparse_templates": dict(),
    "sparse_templates_norefine": dict(refine_iter=0),
    "sparse_templates_gap": dict(rule="gap"),
    "sparse_explain": dict(explain=True),
    "sparse_explain_split": dict(explain=True, split=True),
    "sparse_split_validated": dict(explain=True, split=True, validate=True),
    "sparse_full": dict(explain=True, split=True, validate=True, fwer=True, reproduce=0.8),
    "final": dict(explain=True, split=True, validate=True, fwer=True, reproduce=0.97),
    "final_k": dict(explain=True, split="k", validate=True, fwer=True, reproduce=0.97),
    "final_k_now": dict(explain=True, split="k", validate=True, fwer=True, reproduce=0.97),
    "final_k_floor20": dict(explain=True, split="k", validate=True, fwer=True, reproduce=0.97, min_events=20),
    "sparse_explain_fwer": dict(explain=True, fwer=True),
    "global_templates": dict(),
    "global_sparse_start": dict(),
    "sparse_global_split": dict(explain=True, split="global", validate=True, fwer=True),
}


def outputs(variant, tr_times, tr_T, te_times, te_T, rng):
    if variant == "global_sparse_start":
        return global_sparse(tr_times, tr_T, te_times, te_T, rng)
    if variant == "global_templates":
        from scripts.run_separation import method_outputs
        return [(o, p, np.isfinite(o)) for o, p in method_outputs("templates", tr_times, tr_T, te_times, te_T, rng)]
    from cpca import fit
    from cpca.decode import readout
    from cpca.decompose import prepare
    from cpca.encode import encode
    from cpca.readout_check import _expression, redraw_latency
    import experimental.separate as um
    kw = dict(VARIANTS[variant])
    fwer = kw.pop("fwer", False)
    if kw.pop("validate", False):
        kw["validate"] = encode(te_times, te_T)
    if kw.get("rule") == "gap":
        kw["rule"] = um.largest_gap_mask
    r = fit(tr_times, tr_T, rng=rng, ica=False, center=True, scale=True, n_shuffles=50)
    temps = um.separate(r, encode(tr_times, tr_T), rng, **kw)
    if not temps:
        return []
    Z = prepare(encode(te_times, te_T), center=False, scale=False)
    n_sur = 200 if fwer else ru.N_SURROGATES
    sur = [prepare(encode(redraw_latency(te_times, te_T, rng), te_T), center=False, scale=False) for _ in range(n_sur)]
    out = []
    for t in temps:
        q = t["template"]
        e = _expression(Z, q)
        p = float((1 + sum(_expression(Zs, q) >= e for Zs in sur)) / (1 + n_sur))
        p = min(1.0, p * len(temps)) if fwer else p
        _, times, _, inc, _ = readout(q, r["T_read_ms"], min_weight=1e-9)
        out.append((np.where(inc, times, np.nan), p, t["members"]))
    return out


def global_sparse(tr_times, tr_T, te_times, te_T, rng, n_sur=200):
    from cpca import fit
    from cpca.decode import readout
    from cpca.decompose import prepare
    from cpca.encode import encode
    from cpca.readout_check import _expression, redraw_latency
    import experimental.separate as um
    r = fit(tr_times, tr_T, rng=rng, ica=False, center=True, scale=True, n_shuffles=50)
    enc = encode(tr_times, tr_T)
    sparse = um.separate(r, enc, rng, explain=True)
    got = um.global_with_sparse_starts(r, enc, rng, sparse)
    if not isinstance(got, tuple) or not got[0].shape[1]:
        return []
    T, start = got
    Z = prepare(encode(te_times, te_T), center=False, scale=False)
    sur = [prepare(encode(redraw_latency(te_times, te_T, rng), te_T), center=False, scale=False) for _ in range(n_sur)]
    out = []
    for j in range(T.shape[1]):
        q = T[:, j]
        e = _expression(Z, q)
        p = min(1.0, float((1 + sum(_expression(Zs, q) >= e for Zs in sur)) / (1 + n_sur)) * T.shape[1])
        _, times, _, inc, _ = readout(q, r["T_read_ms"], min_weight=0.2)
        out.append((np.where(inc, times, np.nan), p, inc))
    return out


def judge_partial(order, maps, rng, n_perm=1000):
    """Shared cells: passes for map k if its order matches k (|Spearman| permutation p < 0.05) and its partial
    Spearman correlation with the other map, controlling for k, is not significant (p >= 0.05). Passes overall if
    exactly one map qualifies. Added after the development results (docs/unmixing.md)."""
    from scipy.stats import rankdata
    from scripts.run_separation import spearman_p
    ok = []
    for k, m in enumerate(maps):
        on = np.isfinite(order) & np.isfinite(m)
        if on.sum() < 6:
            ok.append(False)
            continue
        _, p = spearman_p(order[on], m[on], rng, n_perm)
        good = p < 0.05
        for j, other in enumerate(maps):
            if j == k or not good:
                continue
            both = on & np.isfinite(other)
            if both.sum() < 6:
                continue
            a, b, c = rankdata(order[both]), rankdata(other[both]), rankdata(m[both])
            c = (c - c.mean()) / c.std()
            ra = a - a.mean() - (a - a.mean()) @ c / (c @ c) * c
            rb = b - b.mean() - (b - b.mean()) @ c / (c @ c) * c
            den = np.sqrt((ra @ ra) * (rb @ rb))
            r = abs(ra @ rb) / den if den > 0 else 0.0
            null = np.array([abs(ra @ rng.permutation(rb)) / den for _ in range(n_perm)]) if den > 0 else np.zeros(1)
            good = good and (1 + np.sum(null >= r)) / (1 + null.size) >= 0.05
        ok.append(good)
    return sum(ok) == 1, (ok.index(True) if sum(ok) == 1 else -1)


def judge_share(order, maps, rng, n_perm=1000, max_gain=0.1):
    """Shared cells: passes for map k if its order matches k (|Spearman| permutation p < 0.05) and adding the other
    maps to a rank regression on k raises the explained variance of the order by less than max_gain. Passes overall
    if exactly one map qualifies. Added after the waking development results (docs/unmixing.md)."""
    from scipy.stats import rankdata
    from scripts.run_separation import spearman_p
    ok = []
    for k, m in enumerate(maps):
        on = np.isfinite(order) & np.isfinite(m)
        if on.sum() < 6:
            ok.append(False)
            continue
        good = spearman_p(order[on], m[on], rng, n_perm)[1] < 0.05
        others = [o for j, o in enumerate(maps) if j != k]
        both = on & np.all([np.isfinite(o) for o in others], axis=0) if others else on
        if good and others and both.sum() >= 6:
            a = rankdata(order[both])
            a = a - a.mean()
            X1 = rankdata(m[both])[:, None] - rankdata(m[both]).mean()
            X2 = np.column_stack([X1] + [rankdata(o[both]) - rankdata(o[both]).mean() for o in others])
            r2 = lambda X: 1 - np.sum((a - X @ np.linalg.lstsq(X, a, rcond=None)[0]) ** 2) / max(np.sum(a ** 2), 1e-300)
            good = r2(X2) - r2(X1) < max_gain
        ok.append(bool(good))
    return sum(ok) == 1, (ok.index(True) if sum(ok) == 1 else -1)


def job(variant, cond, draw, phase):
    d, kw = get_data(cond, draw, phase)
    perm = np.random.default_rng(ru.seed("split", phase, cond, draw)).permutation(d["times"].shape[0])
    halves = (perm[: perm.size // 2], perm[perm.size // 2:])
    maps = d["maps"] if sum(kw["rates"]) > 0 else []
    rows = []
    for fold, (tr, te) in enumerate((halves, halves[::-1])):
        rng = np.random.default_rng(ru.seed("fit", phase, cond, draw, fold))
        outs = outputs(variant, d["times"][tr], d["T"][tr], d["times"][te], d["T"][te], rng)
        jrng = np.random.default_rng(ru.seed("judge", phase, cond, draw, fold))
        judged = []
        for order, p, mask in outs:
            ok, k, det = judge(order, maps, kw["shared"], jrng) if maps else (False, -1, {})
            cover = float((mask & np.isfinite(maps[k])).sum() / np.isfinite(maps[k]).sum()) if ok else np.nan
            if maps and kw["shared"]:
                ok2, k2 = judge_partial(order, maps, jrng)
            else:
                ok2, k2 = ok, k
            cover2 = float((mask & np.isfinite(maps[k2])).sum() / np.isfinite(maps[k2]).sum()) if ok2 else np.nan
            judged.append({"p_latency": p, "passes_check": p < 0.05, "passes": ok, "map": k, "coverage": cover,
                           "passes_partial": ok2, "map_partial": k2, "coverage_partial": cover2,
                           "n_members": int(mask.sum()), "purity": det.get("purity", np.nan)})
        rows.append({"condition": cond, "draw": draw, "fold": fold, "outputs": judged, "n_maps": len(maps)})
    return rows


def table(rows, check=True, criterion=""):
    lines = ["| condition | outputs per fit | outputs passing | clean fits | coverage | fits with outputs |", "|---|---|---|---|---|---|"]
    for cond in list(CONDITIONS) + list(STRESS):
        rs = [r for r in rows if r["condition"] == cond]
        if not rs:
            continue
        outs = [[o for o in r["outputs"] if (o["passes_check"] or not check)] for r in rs]
        n = [len(o) for o in outs]
        P, K, C = "passes" + criterion, "map" + criterion, "coverage" + criterion
        p = sum(o[P] for oo in outs for o in oo)
        if rs[0]["n_maps"] == 0:
            clean = sum(x == 0 for x in n)
        else:
            clean = sum(len({o[K] for o in oo if o[P]}) == rs[0]["n_maps"] and all(o[P] for o in oo) for oo in outs)
        covs = [o[C] for oo in outs for o in oo if o[P]]
        cov = f"{np.median(covs):.2f}" if covs else "–"
        lines.append(f"| {cond[:40]} | {np.median(n):.0f} | {p}/{sum(n)} ({p / max(sum(n), 1):.0%}) | {clean}/{len(rs)} | {cov} | {sum(x > 0 for x in n)}/{len(rs)} |")
    return "\n".join(lines)


def main(argv):
    variant = argv[0]
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    draws = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--draws=")), 5)
    letters = next((a.split("=", 1)[1] for a in argv if a.startswith("--conditions=")), "ABCDEF")
    phase = next((a.split("=", 1)[1] for a in argv if a.startswith("--phase=")), "dev")
    conds = list(STRESS) if phase.startswith("stress") else [c for c in CONDITIONS if c[0] in letters]
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(job, variant, c, d, phase): (c, d) for c in conds for d in range(draws)}
        for f in as_completed(futs):
            try:
                rows += f.result()
            except Exception:
                print(f"FAILED {futs[f]}:\n{traceback.format_exc()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(rows, open(OUT / f"{phase}_{variant}.pkl", "wb"))
    print(f"## {variant} ({phase}), after the latency check\n" + table(rows))
    print(f"\n## {variant} ({phase}), after the latency check, partial-correlation criterion for shared cells\n" + table(rows, criterion="_partial"))
    print(f"\n## {variant} ({phase}), all outputs\n" + table(rows, check=False))


if __name__ == "__main__":
    main(sys.argv[1:])
