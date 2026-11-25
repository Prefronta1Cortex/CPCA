import numpy as np
from scipy.stats import spearmanr

from cpca import fit
from cpca.subspace import phase_sync, reduced_covariance, seriate
from sim.routes import generate, make_routes


def _rho(order, positions):
    m = np.isfinite(order) & np.isfinite(positions)
    return abs(spearmanr(order[m], positions[m]).statistic)


def test_subspace_readouts_recover_a_route_only_timing_can_order():
    rng = np.random.default_rng(0)
    routes = make_routes(rng, n_cells=60, n_routes=1, cells_per_route=60)
    data = generate(rng, n_cells=60, n_events=2500, routes=routes, replay_frac=0.4, reverse_frac=0.5, participation=0.15)
    res = fit(data["times"], data["event_ms"], rng=rng, ica=False, n_shuffles=30)
    k = res["n_kept"]
    C = reduced_covariance(res["eigenvalues"], res["pca_components"], k)
    d = np.sqrt(np.real(np.diag(C)))
    pos = data["centres"][0]
    assert _rho(seriate(np.real(C) / np.outer(d, d)), pos) > 0.8
    assert _rho(phase_sync(C), pos) > 0.8
    # spike counts alone (who fires together, no timing) carry no order here
    counts = np.isfinite(data["times"]).sum(axis=-1).astype(float)
    Z = (counts - counts.mean(0)) / np.where(counts.std(0) > 0, counts.std(0), 1)
    R = np.corrcoef(Z.T)
    assert _rho(seriate(np.nan_to_num(R)), pos) < 0.5


def test_ring_seriation_recovers_a_circle():
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    S = np.cos(theta[:, None] - theta[None, :])
    order = seriate(S, ring=True)
    # circular agreement up to rotation and reflection
    z = np.exp(1j * (order - theta))
    w = np.exp(1j * (order + theta))
    assert max(abs(z.mean()), abs(w.mean())) > 0.95
