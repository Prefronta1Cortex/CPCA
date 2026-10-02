"""Run the pipeline on synthetic events with planted routes.

Four variants are compared on every condition in sim.routes.CONDITIONS:
  1. uncentred arrows, fixed T = event length       (the original formulation)
  2. centred + scaled, PCA only, fixed T
  3. centred + scaled, PCA only, T = each event's own spike spread
  4. centred + scaled, PCA + ICA, fixed T
For each planted route the best pattern is scored by the Spearman correlation
between its members' relative times and their positions on the route, against
a null that permutes positions among cells. Each pattern also gets a
label-free timing-consistency z; the rank of the route's pattern by that z
says whether the route could have been picked without the answer key.
Results: results/synthetic/.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpca import event_scale, fit  # noqa: E402
from cpca.evaluate import member_order_vs_place, members, timing_consistency  # noqa: E402
from sim.routes import CONDITIONS, generate, make_routes  # noqa: E402

OUT = ROOT / "results" / "synthetic"
VARIANTS = {
    "uncentred, fixed T": dict(center=False, scale=False, ica=False, per_event_T=False),
    "centred, PCA only": dict(center=True, scale=True, ica=False, per_event_T=False),
    "centred, T per event": dict(center=True, scale=True, ica=False, per_event_T=True),
    "centred, PCA + ICA": dict(center=True, scale=True, ica=True, per_event_T=False),
}


def sequence_rank(result, rng, min_span=np.pi / 3):
    """Label-free ranking of patterns: by timing-consistency z, among patterns whose members spread over
    at least min_span of the circle (a synchronous volley is consistent but has no spread). Returns z and ranks
    (patterns below the span threshold get rank k, i.e. last)."""
    k = result["n_kept"]
    if k == 0:
        return np.array([]), np.array([])
    z = np.array([timing_consistency(result["encoded"], result["patterns"][:, j], rng, n_shuffles=20)[0] for j in range(k)])
    span = np.array([r["span_rad"] for r in result["readouts"]])
    key = np.where(span >= min_span, z, -np.inf)
    ranks = np.argsort(np.argsort(-key)) + 1
    return z, ranks


def score_routes(result, data, rng, T_read):
    """Per planted route: best |rho| over patterns, its null, purity, and the pattern's label-free rank."""
    k = result["n_kept"]
    z, z_rank = sequence_rank(result, rng)
    rows = []
    for r, cen in enumerate(data["centres"]):
        best = (np.nan, np.nan, 0, np.nan, np.nan)
        for j in range(k):
            rho, null95, n = member_order_vs_place(result["patterns"][:, j], T_read, cen, False, 1.0, rng, n_null=200)
            if np.isfinite(rho) and (not np.isfinite(best[0]) or abs(rho) > abs(best[0])):
                m = members(result["patterns"][:, j])
                purity = float(np.isfinite(cen[m]).mean()) if m.any() else np.nan
                best = (rho, null95, n, purity, int(z_rank[j]))
        rows.append({"route": r, "rho": best[0], "null95": best[1], "n_members_on_route": best[2], "purity": best[3], "z_rank": best[4]})
    return rows, z


N_REPEATS = 3


def run_condition(index, name, gen_kwargs, route_kwargs, repeat):
    data_rng = np.random.default_rng(1000 * index + repeat)   # stable across runs
    rng = np.random.default_rng(index + 7 * repeat)
    routes = make_routes(data_rng, n_cells=80, **route_kwargs)
    data = generate(data_rng, n_cells=80, n_events=3000, routes=routes, **gen_kwargs)
    out = {"condition": name, "repeat": repeat, "n_routes": len(routes), "fraction_replay": float(np.mean(data["label"] > 0)), "variants": {}}
    for vname, v in VARIANTS.items():
        t0 = time.perf_counter()
        if v["per_event_T"]:
            T, start = event_scale(data["times"])
            times = data["times"] - start[:, None, None]
        else:
            T, times = data["event_ms"], data["times"]
        res = fit(times, T, rng=rng, center=v["center"], scale=v["scale"], ica=v["ica"])
        rows, z = score_routes(res, data, rng, res["T_read_ms"])
        out["variants"][vname] = {"n_kept": res["n_kept"], "eigen_share": (res["eigenvalues"][:5] / res["eigenvalues"].sum()).tolist(),
                                  "routes": rows, "timing_z": z.tolist(), "runtime_s": time.perf_counter() - t0}
    return out


def aggregate(results):
    """Per condition and variant: median best |rho| per route, fraction of (route, repeat) above chance, median k, median z-rank."""
    table = {}
    for res in results:
        for vname, v in res["variants"].items():
            slot = table.setdefault((res["condition"], vname), {"rho": [], "hit": [], "k": [], "rank": [], "replay": []})
            slot["k"].append(v["n_kept"])
            slot["replay"].append(res["fraction_replay"])
            for r in v["routes"]:
                if np.isfinite(r["rho"]):
                    slot["rho"].append(abs(r["rho"]))
                    slot["hit"].append(abs(r["rho"]) > r["null95"])
                    slot["rank"].append(r["z_rank"])
                else:
                    slot["rho"].append(0.0)
                    slot["hit"].append(False)
    return table


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for index, (name, (gen_kwargs, route_kwargs)) in enumerate(CONDITIONS.items()):
        for repeat in range(N_REPEATS):
            results.append(run_condition(index, name, gen_kwargs, route_kwargs, repeat))
        print(f"done: {name}", flush=True)
    with open(OUT / "results.pkl", "wb") as fh:
        pickle.dump(results, fh)
    write_summary(results)


def write_summary(results):
    table = aggregate(results)
    lines = ["# Synthetic routes: results", "",
             f"{N_REPEATS} independent draws per condition, 80 cells, 3000 events. Per planted route and draw, the best pattern is the one whose members' "
             "relative times correlate most with the route positions (|Spearman|). Reported: median |rho| over routes and draws; "
             "fraction of (route, draw) pairs above the 95th percentile of the cell-permutation null; median number of components kept (Marchenko-Pastur); "
             "median label-free rank of the route's pattern (1 = would be picked first without the answer key; patterns whose members span under 60 degrees rank last).", "",
             "| condition | replay | " + " | ".join(VARIANTS) + " |", "|---|---|" + "---|" * len(VARIANTS)]
    for name in CONDITIONS:
        cells = []
        replay = None
        for vname in VARIANTS:
            s = table[(name, vname)]
            replay = np.mean(s["replay"])
            rank = f"{np.median(s['rank']):.0f}" if s["rank"] else "—"
            cells.append(f"ρ {np.median(s['rho']):.2f}, {100*np.mean(s['hit']):.0f}% above chance, k {np.median(s['k']):.0f}, rank {rank}")
        lines.append(f"| {name} | {100*replay:.0f}% | " + " | ".join(cells) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
