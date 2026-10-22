"""The shuffle-null noise bound against the Marchenko-Pastur bound."""

import numpy as np

from cpca import encode, fit
from cpca.decompose import decompose, n_above_null, n_significant, prepare, shuffle_null
from sim.routes import generate, make_routes


def independent_noise(seed, M=2000, N=60):
    """Cells firing independently at different rates, at uniform times on the half-circle."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.05, 0.5, N)
    return np.where(rng.random((M, N)) < q, rng.uniform(0, 100, (M, N)), np.nan)


def test_shuffle_preserves_total_variance():
    rng = np.random.default_rng(0)
    X = prepare(encode(independent_noise(0), 100.0), center=False, scale=False)
    values = decompose(encode(independent_noise(0), 100.0))[0]
    null = shuffle_null(X, rng, n_shuffles=5)
    np.testing.assert_allclose(null.sum(axis=1), values.sum(), rtol=1e-10)


def test_marchenko_pastur_overcounts_uncentred_noise_and_the_shuffle_does_not():
    S = encode(independent_noise(1), 100.0)
    M, N = S.shape
    values = decompose(S)[0]
    k_mp, _ = n_significant(values, M, N)
    k_null, _ = n_above_null(values, shuffle_null(prepare(S, center=False, scale=False), np.random.default_rng(1), 50), centred=False)
    assert k_mp >= 4          # unequal variances alone push eigenvalues over the textbook edge
    assert k_null == 1        # only the mean pattern, which uncentred PCA always has


def test_bounds_agree_for_centred_scaled_noise():
    S = encode(independent_noise(2), 100.0)
    M, N = S.shape
    values = decompose(S, center=True, scale=True)[0]
    k_mp, mp = n_significant(values, M, N)
    k_null, ceiling = n_above_null(values, shuffle_null(prepare(S), np.random.default_rng(2), 50), centred=True)
    assert k_mp == 0 and k_null == 0
    assert abs(ceiling / mp - 1) < 0.05


def test_shuffle_bound_keeps_a_planted_route():
    rng = np.random.default_rng(3)
    routes = make_routes(rng, n_cells=60, n_routes=1)
    data = generate(rng, n_cells=60, n_events=2000, routes=routes, replay_frac=0.3, reverse_frac=0.0, participation=0.15)
    centred = fit(data["times"], data["event_ms"], rng=rng, ica=False, n_shuffles=30)
    uncentred = fit(data["times"], data["event_ms"], rng=rng, ica=False, center=False, scale=False, n_shuffles=30)
    assert centred["n_kept"] >= 1 and uncentred["n_kept"] >= 2
    assert centred["bound_rule"] == "shuffle" and np.isfinite(centred["mp_bound"])
