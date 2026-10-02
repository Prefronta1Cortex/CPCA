"""Complex independent component analysis for un-mixing shared subspaces.

PCA returns the subspace that patterns occupy. When two patterns share cells
with different timing, or occur equally often, PCA returns mixtures of them.
ICA rotates inside the kept subspace so each output is as non-Gaussian as
possible, which for sparse patterns (each present in a minority of events)
recovers the patterns themselves. This is the complex fixed-point algorithm
of Bingham and Hyvarinen (2000), applied to the whitened PCA scores.
"""

import numpy as np

from cpca.theory import anchor_phase


def _symmetric_orthogonalize(W):
    # W <- W (W^H W)^(-1/2)
    gram = W.conj().T @ W
    values, vectors = np.linalg.eigh(gram)
    inv_sqrt = vectors @ np.diag(1 / np.sqrt(np.maximum(values, 1e-12))) @ vectors.conj().T
    return W @ inv_sqrt


def complex_fastica(whitened, rng, max_iter=500, tol=1e-6, a=0.1, n_init=5):
    """Unitary unmixing matrix W for whitened complex data (events x k).

    Sources are y = whitened @ W. Contrast g(u) = 1 / (a + u) on u = |y|^2.
    The best of n_init random starts by total contrast is returned.
    """
    X = np.asarray(whitened)
    M, k = X.shape
    best = None
    for _ in range(n_init):
        W = rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k))
        W = _symmetric_orthogonalize(W)
        for _ in range(max_iter):
            Y = X @ W                                         # (M, k)
            U = np.abs(Y) ** 2
            g = 1 / (a + U)
            dg = -1 / (a + U) ** 2
            # w_j <- E[x conj(y_j) g(u_j)] - E[g(u_j) + u_j g'(u_j)] w_j
            W_new = (X.T @ (np.conj(Y) * g)) / M - W * np.mean(g + U * dg, axis=0)[None, :]
            W_new = _symmetric_orthogonalize(W_new)
            change = 1 - np.min(np.abs(np.sum(np.conj(W_new) * W, axis=0)))
            W = W_new
            if change < tol:
                break
        Y = X @ W
        contrast = float(np.sum(np.mean(np.log(a + np.abs(Y) ** 2), axis=0)))
        if best is None or contrast < best[0]:                # smaller log(a+u) mean = sparser sources
            best = (contrast, W)
    return best[1]


def unmix(prepared, eigenvalues, eigenvectors, rng, **kwargs):
    """Rotate the kept PCA components into independent components.

    prepared: the centred/scaled arrow matrix used for the PCA (events x cells).
    eigenvalues, eigenvectors: the kept components (columns).
    Returns cell-space patterns (cells x k, unit norm, anchored), their event
    scores (events x k), and the unitary rotation applied.
    """
    k = eigenvectors.shape[1]
    if k == 0:
        return eigenvectors, np.zeros((prepared.shape[0], 0), dtype=complex), np.zeros((0, 0))
    if k == 1:
        return eigenvectors, prepared @ np.conj(eigenvectors), np.ones((1, 1))
    whitened = (prepared @ np.conj(eigenvectors)) / np.sqrt(np.maximum(eigenvalues, 1e-12))[None, :]
    W = complex_fastica(whitened, rng, **kwargs)
    # prepared ~ Y W^H Lambda^(1/2) V^H  ->  pattern j in cell space is column j of V Lambda^(1/2) W
    patterns = eigenvectors @ (np.sqrt(eigenvalues)[:, None] * W)
    norms = np.linalg.norm(patterns, axis=0)
    patterns = patterns / np.maximum(norms, 1e-12)
    for j in range(k):
        patterns[:, j] = anchor_phase(patterns[:, j])
    ic_scores = prepared @ np.conj(patterns)
    # order by the energy each pattern explains
    order = np.argsort(-np.mean(np.abs(ic_scores) ** 2, axis=0))
    return patterns[:, order], ic_scores[:, order], W[:, order]
