"""One order of the cells from the whole reduced space, instead of one per component.

Two read-outs of the kept components V_K (eigenvalues Lambda_K), reduced covariance C_K = V_K Lambda_K V_K^H:

- spectral seriation of a pairwise closeness S (cells x cells): cells ordered along the leading non-trivial
  direction of the normalised graph Laplacian of max(S, 0) (linear order), or by the angle of the two
  leading non-trivial directions (ring order, for circular tracks);
- angular synchronisation of C_K: the leading eigenvector of D^{-1/2} C_K D^{-1/2}, with D each cell's total
  coupling sum_l |C_kl|, gives each cell a phase; the phases are read out like a component.
"""

import numpy as np

from cpca.direction import order_of


def reduced_covariance(eigenvalues, vectors, k):
    V = np.asarray(vectors)[:, :k]
    return (V * np.clip(np.asarray(eigenvalues, float)[:k], 0, None)[None, :]) @ V.conj().T


def seriate(S, ring=False, min_degree_frac=1e-3):
    """Order (one value per cell, NaN for isolated cells) from a symmetric closeness matrix."""
    A = np.clip(np.asarray(S, float), 0, None)
    np.fill_diagonal(A, 0)
    deg = A.sum(axis=1)
    use = deg > min_degree_frac * deg.max() if deg.max() > 0 else np.zeros(len(deg), bool)
    out = np.full(len(deg), np.nan)
    if use.sum() < 3:
        return out
    A = A[np.ix_(use, use)]
    d = A.sum(axis=1)
    L = np.eye(len(d)) - A / np.sqrt(np.outer(d, d))
    values, vectors = np.linalg.eigh(L)
    if ring:
        u = vectors[:, 1:3] / np.sqrt(d)[:, None]
        out[use] = np.mod(np.arctan2(u[:, 1], u[:, 0]), 2 * np.pi)
    else:
        out[use] = vectors[:, 1] / np.sqrt(d)
    return out


def phase_sync(C, min_degree_frac=1e-3):
    """Order from angular synchronisation of a Hermitian coupling matrix (one value per cell, NaN outside)."""
    C = np.asarray(C, complex).copy()
    np.fill_diagonal(C, 0)
    deg = np.abs(C).sum(axis=1)
    use = deg > min_degree_frac * deg.max() if deg.max() > 0 else np.zeros(len(deg), bool)
    if use.sum() < 3:
        return np.full(len(deg), np.nan)
    Cu = C[np.ix_(use, use)] / np.sqrt(np.outer(deg[use], deg[use]))
    values, vectors = np.linalg.eigh(Cu)
    z = np.zeros(len(deg), complex)
    z[use] = np.exp(1j * np.angle(vectors[:, -1]))
    return order_of(z, member_frac=0.5)
