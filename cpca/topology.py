"""Does the reduced space preserve the topology of the map? Pairwise tests over all kept components.

For every pair of cells two numbers are compared:

- closeness in the reduced space: the reduced covariance C_K = V_K Lambda_K V_K^H (components
  above the noise bound only), normalised like a correlation, r_kl = Re(C_K,kl) / sqrt(C_K,kk C_K,ll).
  Two cells are close when they load on the same components *and* at similar phases (similar
  firing times). r_kl is unchanged by each component's arbitrary global phase. The magnitude-only
  version |C_K,kl| / sqrt(...) keeps co-membership and ignores timing; with all components instead
  of K it is the full-space (complex) correlation of the arrows;
- closeness on the map: the correlation between the two cells' direction-resolved rate maps
  (running directions concatenated, so heading is a dimension of its own and cells with several
  fields are handled).

The statistic is the Spearman correlation between the two over all pairs; its null permutes the
cells of the map (Mantel test), which respects the dependence between pairs that share a cell.
"""

import numpy as np
from scipy.stats import rankdata


def reduced_similarity(eigenvalues, vectors, k=None, kind="phase"):
    """Pairwise closeness of cells in the space of the first k components (all if k is None).

    kind "phase": Re(C_K) normalised (co-membership and timing); "magnitude": |C_K| normalised
    (co-membership only). Returns an (N, N) matrix with ones on the diagonal.
    """
    k = len(eigenvalues) if k is None else int(k)
    V = np.asarray(vectors)[:, :k]
    lam = np.clip(np.asarray(eigenvalues, float)[:k], 0, None)
    S = (V * lam[None, :]) @ V.conj().T
    d = np.sqrt(np.clip(np.real(np.diag(S)), 1e-300, None))
    num = np.real(S) if kind == "phase" else np.abs(S)
    return num / np.outer(d, d)


def map_similarity(maps):
    """Correlation between cells' rate maps; maps is (cells, features). Cells with a flat map get NaN rows."""
    X = np.asarray(maps, float).reshape(len(maps), -1)
    X = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(X, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = X / norm[:, None]
    return Z @ Z.T


def mantel(model, reference, rng, n_perm=2000, chunk=250):
    """Spearman correlation between two similarity matrices over cell pairs, with a cell-permutation null.

    model, reference: (n, n) symmetric matrices over the same cells. The null permutes the cells
    of `reference`. Returns rho, one-sided p (rho larger than chance), null95, null mean, n cells, n pairs.
    """
    model, reference = np.asarray(model, float), np.asarray(reference, float)
    n = model.shape[0]
    iu = np.triu_indices(n, 1)
    a = rankdata(model[iu])
    if n < 4 or np.ptp(a) == 0 or np.ptp(reference[iu]) == 0:
        # no variation on one side (e.g. no component above the noise bound): nothing to test
        return {"rho": np.nan, "p": np.nan, "null95": np.nan, "null_mean": np.nan, "n_cells": n, "n_pairs": int(iu[0].size)}
    a = (a - a.mean()) / np.linalg.norm(a - a.mean())
    R = np.zeros((n, n))
    R[iu] = rankdata(reference[iu])
    R = R + R.T
    b = R[iu]
    mu, sd = b.mean(), np.linalg.norm(b - b.mean())          # a permutation of cells permutes the pairs: same multiset
    rho = float(a @ (b - mu) / sd)
    null = np.empty(n_perm)
    for s in range(0, n_perm, chunk):
        m = min(chunk, n_perm - s)
        perms = np.array([rng.permutation(n) for _ in range(m)])                 # (m, n)
        Bp = R[perms[:, iu[0]], perms[:, iu[1]]]                                  # (m, pairs)
        null[s:s + m] = (Bp - mu) @ a / sd
    return {"rho": rho, "p": float((1 + np.sum(null >= rho)) / (n_perm + 1)), "null95": float(np.percentile(null, 95)),
            "null_mean": float(null.mean()), "n_cells": n, "n_pairs": int(iu[0].size)}


def classical_mds(distance, dim=2):
    """Classical multidimensional scaling of a distance matrix: (n, dim) coordinates."""
    D2 = np.asarray(distance, float) ** 2
    n = D2.shape[0]
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ D2 @ J
    values, vectors = np.linalg.eigh((B + B.T) / 2)
    order = np.argsort(values)[::-1][:dim]
    return vectors[:, order] * np.sqrt(np.clip(values[order], 0, None))


def similarity_to_distance(similarity):
    """Distance between unit-normalised embeddings: sqrt(2 - 2 r)."""
    return np.sqrt(np.clip(2 - 2 * np.asarray(similarity, float), 0, None))
