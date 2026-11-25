import numpy as np

from cpca.ica import unmix
from cpca.theory import lag_matrix


def _mixed_patterns(rng, M=4000, N=30):
    """Two sparse sources on the same cells with different timing, equally often (PCA mixes them)."""
    phase = np.stack([np.linspace(0, np.pi, N), np.linspace(np.pi, 0, N)], axis=1)
    A = np.exp(1j * phase) / np.sqrt(N)
    on = rng.random((M, 2)) < 0.1
    S = on * np.exp(1j * rng.uniform(0, 2 * np.pi, (M, 2))) * rng.uniform(0.5, 1.5, (M, 2))
    X = S @ A.T + 0.02 * (rng.normal(size=(M, N)) + 1j * rng.normal(size=(M, N)))
    return X, A


def _overlap(P, A):
    return np.abs(P.conj().T @ A)


def test_unmix_recovers_known_sources():
    rng = np.random.default_rng(1)
    X, A = _mixed_patterns(rng)
    vals, vecs = np.linalg.eigh(lag_matrix(X))
    vals, vecs = vals[::-1][:2], vecs[:, ::-1][:, :2]
    P, _, _ = unmix(X, vals, vecs, np.random.default_rng(2))
    best = _overlap(P, A).max(axis=0)
    assert np.all(best > 0.95)


def test_principal_components_alone_mix_the_sources():
    rng = np.random.default_rng(1)
    X, A = _mixed_patterns(rng)
    _, vecs = np.linalg.eigh(lag_matrix(X))
    vecs = vecs[:, ::-1][:, :2]
    assert _overlap(vecs, A).max() < 0.95
