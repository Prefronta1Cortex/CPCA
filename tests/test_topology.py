import numpy as np

from cpca import fit
from cpca.topology import classical_mds, mantel, map_similarity, reduced_similarity, similarity_to_distance
from sim.routes import generate, make_routes


def bump_maps(positions, n_bins=40, width=0.08):
    """Gaussian rate maps at given positions in [0, 1] (NaN position: no field -> flat zero map)."""
    x = (np.arange(n_bins) + 0.5) / n_bins
    pos = np.asarray(positions, float)
    maps = np.exp(-0.5 * ((x[None, :] - pos[:, None]) / width) ** 2)
    maps[~np.isfinite(pos)] = 0
    return maps


def random_symmetric(rng, n):
    A = rng.normal(size=(n, n))
    return (A + A.T) / 2


def test_mantel_is_calibrated_and_detects_a_relation():
    rng = np.random.default_rng(0)
    ps = [mantel(random_symmetric(rng, 30), random_symmetric(rng, 30), rng, n_perm=200)["p"] for _ in range(80)]
    assert 0.35 < np.mean(ps) < 0.65 and np.mean(np.array(ps) < 0.05) < 0.15
    A = random_symmetric(rng, 30)
    out = mantel(A + 0.5 * random_symmetric(rng, 30), A, rng, n_perm=500)
    assert out["rho"] > 0.5 and out["p"] < 0.01 and out["n_pairs"] == 30 * 29 // 2


def test_reduced_similarity_ignores_each_components_global_phase():
    rng = np.random.default_rng(1)
    V = np.linalg.qr(rng.normal(size=(20, 20)) + 1j * rng.normal(size=(20, 20)))[0]
    lam = np.sort(rng.uniform(0.5, 3, 20))[::-1]
    R1 = reduced_similarity(lam, V, k=5)
    R2 = reduced_similarity(lam, V * np.exp(1j * rng.uniform(0, 2 * np.pi, 20))[None, :], k=5)
    np.testing.assert_allclose(R1, R2, atol=1e-12)
    np.testing.assert_allclose(np.diag(R1), 1.0)
    assert np.all(np.abs(reduced_similarity(lam, V, k=5, kind="magnitude")) <= 1 + 1e-12)


def test_mds_recovers_a_line():
    x = np.linspace(0, 1, 15)
    D = np.abs(x[:, None] - x[None, :])
    Y = classical_mds(D, dim=1)[:, 0]
    assert abs(np.corrcoef(Y, x)[0, 1]) > 0.999
    assert np.allclose(similarity_to_distance(np.ones((2, 2))), 0)


def test_reduced_space_preserves_the_topology_of_a_planted_route_and_not_of_noise():
    rng = np.random.default_rng(3)
    routes = make_routes(rng, n_cells=60, n_routes=1, cells_per_route=60)
    pos = np.full(60, np.nan)
    pos[routes[0]["cells"]] = routes[0]["pos"]
    M = map_similarity(bump_maps(pos))
    data = generate(rng, n_cells=60, n_events=2500, routes=routes, replay_frac=0.4, reverse_frac=0.5, participation=0.15)
    res = fit(data["times"], data["event_ms"], rng=rng, ica=False, n_shuffles=30)
    R = reduced_similarity(res["eigenvalues"], res["pca_components"], k=res["n_kept"])
    assert mantel(R, M, rng, n_perm=500)["p"] < 0.01
    noise = generate(rng, n_cells=60, n_events=2500, routes=routes, replay_frac=0.0, participation=0.15)
    res0 = fit(noise["times"], noise["event_ms"], rng=rng, ica=False, n_shuffles=30)
    R0 = reduced_similarity(res0["eigenvalues"], res0["pca_components"], k=res0["n_kept"])
    out0 = mantel(R0, M, rng, n_perm=500)
    assert res0["n_kept"] == 0 and not np.isfinite(out0["p"])       # nothing above the noise bound, nothing to test
    # the full space (no reduction) has no topology either
    from cpca.decompose import prepare
    from cpca.theory import lag_matrix
    C = lag_matrix(prepare(res0["encoded"]))
    full = np.real(C) / np.sqrt(np.outer(np.real(np.diag(C)), np.real(np.diag(C))))
    assert mantel(full, M, rng, n_perm=500)["p"] > 0.01
