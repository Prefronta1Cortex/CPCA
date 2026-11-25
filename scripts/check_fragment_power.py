"""Power of the fragment criterion at hc-11 size: planted data with 25 cells on the route (docs/fragment.md).

Usage: python scripts/check_fragment_power.py
"""
import sys, os; sys.path.insert(0, '.')
import numpy as np
from concurrent.futures import ProcessPoolExecutor
def job(args):
    cond, draw = args
    from sim.routes import generate, make_routes
    from scripts.run_fragment import criterion
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_fragment import cpca_orders
    rng = np.random.default_rng(9000 + draw + 100 * ["whole", "segment", "assemblies"].index(cond))
    routes = make_routes(rng, n_cells=80, n_routes=1, cells_per_route=25)
    extra, gen = {}, dict(participation=0.15, jitter_ms=8.0, replay_frac=0.2, reverse_frac=0.5)
    if cond == "segment": gen["segment"] = 0.4
    if cond == "assemblies":
        gen["replay_frac"] = 0.0
        m = rng.integers(0, 6, size=80); m[routes[0]["cells"]] = np.minimum((routes[0]["pos"] * 6).astype(int), 5); extra["assembly_membership"] = m
    d = generate(rng, n_cells=80, n_events=2000, routes=routes, **gen, **extra)
    T = np.full(2000, d["event_ms"]); key = {1.0: d["centres"][0] * 100}
    r = criterion(d["times"], T, key, 100.0, rng, n_shuffles=30, n_null=300)
    old = best_map_match(cpca_orders(d["times"], T, rng), key, False, 100.0, rng, n_null=300)
    return cond, r["cpca"]["p"], r["p_timing"], old["p"]
if __name__ == "__main__":
    jobs = [(c, k) for c in ("whole", "segment", "assemblies") for k in range(4)]
    with ProcessPoolExecutor(9) as pool:
        res = list(pool.map(job, jobs))
    for c in ("whole", "segment", "assemblies"):
        rs = [r for r in res if r[0] == c]
        f = lambda x: bool(np.isfinite(x) and x < 0.05)
        print(f"25 route cells, {c}: fragment bar1 {sum(f(r[1]) for r in rs)}/4, both bars {sum(f(r[1]) and f(r[2]) for r in rs)}/4, old whole-track test {sum(f(r[3]) for r in rs)}/4")
