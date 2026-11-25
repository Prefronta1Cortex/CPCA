"""Un-mixing complex PCA's components (docs/unmixing.md): span check, candidate un-mixing steps, ladder per output.

Usage: python scripts/run_unmixing.py [span|candidates|ladder|summary] [--workers=9] [--confirm]
Output: results/unmixing/*.pkl, results/unmixing/summary.md
"""

import os
import pickle
import sys
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run_separation import CONDITIONS, judge  # noqa: E402

OUT = ROOT / "results" / "unmixing"
CANDIDATES = ["PCA", "ICA", "Quartimax", "Count ICA + phase"]
RULES = ["20% of max", "mean + 2 SD", "two-group split"]
N_SURROGATES = 50
SEED_BASE = {"screen": 5000, "dev": 7000, "confirm": 9000}


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def make_data(cond, draw, phase="screen"):
    from sim.two_routes import generate
    kw = CONDITIONS[cond]
    rng = np.random.default_rng(SEED_BASE[phase] + 97 * list(CONDITIONS).index(cond) + draw)
    return generate(rng, **kw), kw


# ----------------------------------------------------------------------------- membership rules
def otsu_mask(a):
    s = np.sort(a)
    n = s.size
    if n < 3 or s[-1] <= 0:
        return a >= s[-1]
    c1, c2 = np.cumsum(s), np.cumsum(s ** 2)
    best, cut = np.inf, s[-1]
    for i in range(1, n):
        lo_n, hi_n = i, n - i
        lo_var = c2[i - 1] - c1[i - 1] ** 2 / lo_n
        hi_var = (c2[-1] - c2[i - 1]) - (c1[-1] - c1[i - 1]) ** 2 / hi_n
        if lo_var + hi_var < best:
            best, cut = lo_var + hi_var, s[i]
    return a >= cut


def member_mask(a, rule):
    a = np.asarray(a, float)
    if a.max() <= 0:
        return np.zeros(a.shape, bool)
    if rule == "20% of max":
        return a >= 0.2 * a.max()
    if rule == "mean + 2 SD":
        return a > a.mean() + 2 * a.std()
    return otsu_mask(a)


# ----------------------------------------------------------------------------- part 1: span
def count_pca(times, T):
    from baselines import count_assemblies as ca
    X = ca.binned_events(times, T)
    sd = X.std(0)
    act = sd > 0
    Z = (X[:, act] - X[:, act].mean(0)) / sd[act]
    v, V = np.linalg.eigh(np.corrcoef(Z.T))
    o = np.argsort(v)[::-1]
    v, V = v[o], V[:, o]
    k = int(np.sum(v > (1 + np.sqrt(act.sum() / X.shape[0])) ** 2))
    W = np.zeros((X.shape[1], k))
    W[act] = V[:, :k]
    return W


def span_job(cond, draw):
    from cpca import fit
    from cpca.containment import basis, containment
    d, kw = make_data(cond, draw)
    if sum(kw["rates"]) == 0:
        return ("span", cond, draw), []
    rng = np.random.default_rng(seed("span", cond, draw))
    n = d["times"].shape[1]
    spaces = {}
    for label, c in (("complex PCA, centred", True), ("complex PCA, uncentred", False)):
        r = fit(d["times"], d["T"], rng=rng, ica=False, center=c, scale=c, n_shuffles=50)
        spaces[label] = (basis(r["pca_components"]) if r["n_kept"] else np.zeros((n, 0)), r["n_kept"])
    Wc = count_pca(d["times"], d["T"])
    spaces["count PCA"] = (basis(Wc, with_conjugates=False) if Wc.shape[1] else np.zeros((n, 0)), Wc.shape[1])
    rows = []
    for label, (B, k) in spaces.items():
        targets = ["membership"] + ([] if label == "count PCA" else ["order"])
        for t in targets:
            null = []
            for _ in range(500):
                cells = rng.choice(n, 40, replace=False)
                u = np.zeros(n, complex)
                u[cells] = 1.0 if t == "membership" else np.exp(1j * np.pi * rng.random(40))
                null.append(containment(u[:, None], B)[0] if B.shape[1] else 0.0)
            null = np.array(null)
            for ri, m in enumerate(d["maps"]):
                on = np.isfinite(m)
                u = np.zeros(n, complex)
                u[on] = 1.0 if t == "membership" else np.exp(1j * np.pi * m[on])
                share = containment(u[:, None], B)[0] if B.shape[1] else 0.0
                rows.append({"condition": cond, "draw": draw, "space": label, "k": k, "target": t, "route": ri,
                             "share": float(share), "null_median": float(np.median(null)), "null99": float(np.percentile(null, 99))})
    return ("span", cond, draw), rows


# ----------------------------------------------------------------------------- part 2: candidates
def candidate_outputs(tr_times, tr_T, te_times, te_T, rng):
    """{(candidate, rule): [(order, p_latency, members)]} for one training/held-out split."""
    from baselines import count_assemblies as ca
    from cpca import fit
    from cpca.decode import readout
    from cpca.decompose import prepare
    from cpca.direction import split_members
    from cpca.encode import encode
    from cpca.readout_check import _expression, redraw_latency
    from experimental.rotation import quartimax

    r = fit(tr_times, tr_T, rng=rng, ica=True, center=True, scale=True, n_shuffles=50)
    T_read = r["T_read_ms"]
    vectors = {}
    if r["n_kept"]:
        vectors["PCA"] = r["pca_components"]
        vectors["ICA"] = r["patterns"]
        vectors["Quartimax"] = quartimax(r["pca_components"])[0]
    else:
        vectors.update({c: np.zeros((tr_times.shape[1], 0), complex) for c in ("PCA", "ICA", "Quartimax")})
    assemblies = ca.fit(tr_times, tr_T, rng)
    raw_train = encode(tr_times, tr_T)

    Z = {"centred": prepare(encode(te_times, te_T), center=True, scale=True),
         "raw": prepare(encode(te_times, te_T), center=False, scale=False)}
    surrogates = {"centred": [], "raw": []}
    for _ in range(N_SURROGATES):
        e = encode(redraw_latency(te_times, te_T, rng), te_T)
        surrogates["centred"].append(prepare(e, center=True, scale=True))
        surrogates["raw"].append(prepare(e, center=False, scale=False))

    def score(q, convention):
        e = _expression(Z[convention], q)
        null = np.array([_expression(Zs, q) for Zs in surrogates[convention]])
        return float((1 + np.sum(null >= e)) / (1 + N_SURROGATES))

    def order_of(q):
        _, t, _, inc, _ = readout(q, T_read, min_weight=1e-9)
        return np.where(inc, t, np.nan)

    out = {}
    for rule in RULES:
        for cand in ("PCA", "ICA", "Quartimax"):
            rows = []
            for w in vectors[cand].T:
                mask = member_mask(np.abs(w), rule)
                if mask.sum() < 2:
                    continue
                q = np.where(mask, w, 0)
                rows.append((order_of(q), score(q, "centred"), mask))
            out[(cand, rule)] = rows
        rows = []
        for w in assemblies["weights"].T:
            mask = member_mask(np.maximum(w, 0), rule)
            if mask.sum() < 3:
                continue
            sp = split_members(raw_train, mask, rng, test=False)
            if sp is None:
                continue
            q = sp["template"]
            rows.append((order_of(q), score(q, "raw"), mask))
        out[("Count ICA + phase", rule)] = rows
    return out


def candidates_job(cond, draw, phase="screen"):
    d, kw = make_data(cond, draw, phase)
    perm = np.random.default_rng(seed("split", phase, cond, draw)).permutation(d["times"].shape[0])
    halves = (perm[: perm.size // 2], perm[perm.size // 2:])
    maps = d["maps"] if sum(kw["rates"]) > 0 else []
    rows = []
    for fold, (tr, te) in enumerate((halves, halves[::-1])):
        rng = np.random.default_rng(seed("fit", phase, cond, draw, fold))
        outs = candidate_outputs(d["times"][tr], d["T"][tr], d["times"][te], d["T"][te], rng)
        jrng = np.random.default_rng(seed("judge", phase, cond, draw, fold))
        for (cand, rule), items in outs.items():
            judged = []
            for order, p, mask in items:
                ok, k, det = judge(order, maps, kw["shared"], jrng) if maps else (False, -1, {})
                cover = float((mask & np.isfinite(maps[k])).sum() / np.isfinite(maps[k]).sum()) if ok else np.nan
                judged.append({"p_latency": p, "passes_check": p < 0.05, "passes": ok, "map": k, "coverage": cover,
                               "n_members": int(mask.sum()), **{key: v for key, v in det.items() if key != "rhos"}})
            rows.append({"phase": phase, "condition": cond, "draw": draw, "fold": fold, "candidate": cand, "rule": rule,
                         "outputs": judged, "n_maps": len(maps)})
    return ("candidates", phase, cond, draw), rows


# ----------------------------------------------------------------------------- part 3: ladder per output
def ladder_job(rung):
    from scripts.run_ladder import make_data as ladder_data
    from sim.routes import LADDER
    name = next(n for n in LADDER if n.startswith(rung + " "))
    rows = []
    for draw in range(3):
        maps = ladder_data(name, draw)["centres"]
        rng = np.random.default_rng(seed("ladder-rescore", rung, draw))
        for method, variants in (("cpca", ("centred", "uncentred")), ("ppseq", ("type_rule",)), ("seqnmf", ("K15_L120",))):
            f = ROOT / "results" / "ladder" / f"{rung}_d{draw}.{method}.pkl"
            if not f.exists():
                continue
            stored = pickle.load(open(f, "rb"))["variants"]
            for v in variants:
                judged = []
                for o in stored[v]["orders"]:
                    o = np.asarray(o, float)
                    ok, k, _ = judge(o, maps, True, rng)
                    judged.append({"passes": ok, "map": k, "n_members": int(np.isfinite(o).sum())})
                rows.append({"rung": rung, "draw": draw, "method": method, "variant": v, "outputs": judged, "n_maps": len(maps)})
    return ("ladder", rung), rows


# ----------------------------------------------------------------------------- summary
def summarize():
    lines = ["# Un-mixing: results", "", "Produced by `scripts/run_unmixing.py`. Design: `docs/unmixing.md`.", ""]
    f = OUT / "span.pkl"
    if f.exists():
        rows = pickle.load(open(f, "rb"))
        lines += ["## Part 1: are the routes inside the kept span?", "",
                  "| condition | space | components kept (median) | target | routes above the null's 99th percentile | share (median) | null (median) |",
                  "|---|---|---|---|---|---|---|"]
        for cond in CONDITIONS:
            for space in ("complex PCA, centred", "complex PCA, uncentred", "count PCA"):
                for t in ("membership", "order"):
                    rs = [r for r in rows if r["condition"] == cond and r["space"] == space and r["target"] == t]
                    if not rs:
                        continue
                    hit = sum(r["share"] > r["null99"] for r in rs)
                    lines.append(f"| {cond} | {space} | {np.median([r['k'] for r in rs]):.0f} | {t} | {hit}/{len(rs)} | "
                                 f"{np.median([r['share'] for r in rs]):.2f} | {np.median([r['null_median'] for r in rs]):.2f} |")
        lines.append("")
    for phase in ("screen", "confirm"):
        f = OUT / f"candidates_{phase}.pkl"
        if not f.exists():
            continue
        rows = pickle.load(open(f, "rb"))
        lines += [f"## Part 2: candidates ({phase})", "",
                  "Outputs pass the latency check on held-out events; clean = every map represented and every output passes; "
                  "coverage = members on the route / route cells, median over passing outputs.", "",
                  "| condition | candidate | rule | outputs per fit (median) | outputs passing | clean fits | coverage | solves |",
                  "|---|---|---|---|---|---|---|---|"]
        for cond in CONDITIONS:
            for cand in CANDIDATES:
                for rule in RULES:
                    rs = [r for r in rows if r["condition"] == cond and r["candidate"] == cand and r["rule"] == rule]
                    if not rs:
                        continue
                    outs = [[o for o in r["outputs"] if o["passes_check"]] for r in rs]
                    n_out = [len(o) for o in outs]
                    total = sum(n_out)
                    n_pass = sum(o["passes"] for oo in outs for o in oo)
                    if rs[0]["n_maps"] == 0:
                        clean = sum(n == 0 for n in n_out)
                        cov, solves = "–", "–"
                    else:
                        clean = sum(len({o["map"] for o in oo if o["passes"]}) == rs[0]["n_maps"] and all(o["passes"] for o in oo)
                                    for oo in outs)
                        covs = [o["coverage"] for oo in outs for o in oo if o["passes"]]
                        cov = f"{np.median(covs):.2f}" if covs else "–"
                        solves = "yes" if total and n_pass / total >= 0.8 and clean >= 8 and covs and np.median(covs) >= 0.5 else "no"
                    share = f"{n_pass}/{total} ({n_pass / total:.0%})" if total else "0/0"
                    lines.append(f"| {cond} | {cand} | {rule} | {np.median(n_out):.0f} | {share} | {clean}/{len(rs)} | {cov} | {solves} |")
        lines += ["", "No replays (F): clean = no outputs; a candidate must also have outputs in at most 2 of 10 F fits.", ""]
    f = OUT / "ladder.pkl"
    if f.exists():
        rows = pickle.load(open(f, "rb"))
        lines += ["## Part 3: ladder outputs per output (stored fits, no latency check)", "",
                  "| rung | method | outputs per fit (median) | outputs matching exactly one map | clean fits |", "|---|---|---|---|---|"]
        for rung in ("L0", "L1", "L2", "L3", "L4", "L5", "L6"):
            for method, v in (("cpca", "centred"), ("cpca", "uncentred"), ("ppseq", "type_rule"), ("seqnmf", "K15_L120")):
                rs = [r for r in rows if r["rung"] == rung and r["method"] == method and r["variant"] == v]
                if not rs:
                    continue
                n = [len(r["outputs"]) for r in rs]
                p = sum(o["passes"] for r in rs for o in r["outputs"])
                clean = sum(len({o["map"] for o in r["outputs"] if o["passes"]}) == r["n_maps"] and all(o["passes"] for o in r["outputs"]) for r in rs)
                label = {"cpca": f"complex PCA, {v}", "ppseq": "PP-Seq", "seqnmf": "seqNMF (K 15, L 120 ms)"}[method]
                lines.append(f"| {rung} | {label} | {np.median(n):.0f} | {p}/{sum(n)} ({p / max(sum(n), 1):.0%}) | {clean}/{len(rs)} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def run(jobs, workers, out_name):
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *args): args for fn, args in jobs}
        for fut in as_completed(futures):
            try:
                key, rows = fut.result()
                results += rows
                print(f"done {key}", flush=True)
            except Exception:
                print(f"FAILED {futures[fut]}:\n{traceback.format_exc()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(results, open(OUT / out_name, "wb"))


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    part = argv[0] if argv and not argv[0].startswith("--") else "summary"
    if part == "span":
        run([(span_job, (c, d)) for c in CONDITIONS for d in range(5)], workers, "span.pkl")
    elif part == "candidates":
        phase = "confirm" if "--confirm" in argv else "screen"
        run([(candidates_job, (c, d, phase)) for c in CONDITIONS for d in range(5)], workers, f"candidates_{phase}.pkl")
    elif part == "ladder":
        run([(ladder_job, (r,)) for r in ("L0", "L1", "L2", "L3", "L4", "L5", "L6")], workers, "ladder.pkl")
    summarize()


if __name__ == "__main__":
    main(sys.argv[1:])
