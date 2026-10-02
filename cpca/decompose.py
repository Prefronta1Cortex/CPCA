"""Complex PCA of the event-by-cell arrow matrix.

Two conventions, chosen by the clock of the events.

- Events with an arbitrary clock (no shared reference): leave the columns
  uncentred. The lag matrix C = S^T conj(S) / M is then invariant to each
  event's clock start, and its first eigenvector is dominated by which cells
  tend to fire (participation).
- Events aligned on a shared reference (the population peak): centre each
  cell on its mean arrow and divide by its root-mean-square length, as in
  count-based assembly analysis (Lopes-dos-Santos et al., 2013). Components
  then describe how events differ from each other, which is where different
  routes live.

The number of components to keep is set by the Marchenko-Pastur bound, the
largest eigenvalue a matrix of the same size made of independent noise would
produce.
"""

import numpy as np

from cpca.theory import anchor_phase, eigh_desc, lag_matrix, scores


def prepare(encoded, center=True, scale=True):
    """Centre each cell on its mean arrow and/or scale it to unit root-mean-square length."""
    X = np.array(encoded, dtype=complex, copy=True)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    if scale:
        rms = np.sqrt(np.mean(np.abs(X) ** 2, axis=0))
        np.divide(X, rms, out=X, where=rms > 0)
    return X


def marchenko_pastur_bound(n_events, n_cells, variance=1.0):
    """Largest eigenvalue of the covariance of pure noise with the given per-cell variance."""
    return variance * (1 + np.sqrt(n_cells / n_events)) ** 2


def n_significant(eigenvalues, n_events, n_cells):
    """How many eigenvalues exceed the Marchenko-Pastur bound (variance = mean eigenvalue)."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    bound = marchenko_pastur_bound(n_events, n_cells, variance=eigenvalues.mean())
    return int((eigenvalues > bound).sum()), float(bound)


def decompose(encoded, n_components=None, center=False, scale=False):
    """Eigen-decomposition of the (optionally centred and scaled) arrow matrix.

    Returns eigenvalues (largest first), anchored eigenvectors as columns,
    complex scores of shape (events, components), and the covariance matrix.
    With center=False and scale=False this is the uncentred lag matrix of the
    original formulation.
    """
    X = prepare(encoded, center=center, scale=scale)
    cov = lag_matrix(X)
    eigenvalues, eigenvectors = eigh_desc(cov)
    n_cells = eigenvectors.shape[1]
    if n_components is None:
        n_components = n_cells
    n_components = min(int(n_components), n_cells)
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = np.array(eigenvectors[:, :n_components], copy=True)
    for component in range(n_components):
        eigenvectors[:, component] = anchor_phase(eigenvectors[:, component])
    return eigenvalues, eigenvectors, scores(X, eigenvectors), cov
