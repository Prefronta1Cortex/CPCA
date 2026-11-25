import numpy as np
from scipy.stats import spearmanr

from cpca import sparse
from sim.routes import generate, make_routes


def _best_rho(orders, positions):
    best = 0.0
    for o in orders:
        m = np.isfinite(o) & np.isfinite(positions)
        if m.sum() >= 6:
            best = max(best, abs(spearmanr(o[m], positions[m]).statistic))
    return best


def test_event_sparse_templates_find_a_rare_route_played_both_ways():
    rng = np.random.default_rng(3)
    routes = make_routes(rng, n_cells=80, n_routes=1, cells_per_route=60)
    data = generate(rng, n_cells=80, n_events=1500, routes=routes, replay_frac=0.05, reverse_frac=0.5,
                    participation=0.15, seq_participation=0.25)
    res = sparse.fit(data["times"], np.full(1500, data["event_ms"]), np.random.default_rng(0), normalize=False)
    assert _best_rho(res["orders"], data["centres"][0]) > 0.8


def test_event_sparse_templates_return_distinct_outputs():
    rng = np.random.default_rng(4)
    data = generate(rng, n_cells=60, n_events=800, routes=make_routes(rng, n_cells=60), replay_frac=0.0, participation=0.15)
    res = sparse.fit(data["times"], np.full(800, data["event_ms"]), np.random.default_rng(0), fracs=(0.05,), n_seeds=100)
    qs = [t["template"] for t in res["templates"]]
    for i in range(len(qs)):
        for j in range(i):
            assert max(abs(np.vdot(qs[i], qs[j])), abs(np.vdot(np.conj(qs[i]), qs[j]))) <= 0.8 + 1e-9
