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

The number of components to keep is set against a shuffle null (permutation
parallel analysis, Horn 1965; Buja & Eyuboglu 1992): each cell's arrows are
permuted across events independently, which keeps every cell's participation,
arrow lengths, mean arrow and phase distribution and destroys only the
coordination between cells. A component is kept if its eigenvalue exceeds the
95th percentile of the largest *noise* eigenvalue of the shuffled matrix.

The Marchenko-Pastur bound, the largest eigenvalue of independent noise with
equal variance in every cell, gives the same number for centred and scaled
arrows (zero mean, unit variance per cell). It does not hold for uncentred
arrows, whose cells have very different variances (participation) and a
non-zero mean: there it keeps components even on pure noise. It is kept here
for comparison.
"""

import numpy as np

from cpca.theory import anchor_phase, eigh_desc, lag_matrix, scores


def prepare(encoded, center=True, scale=True):
    """Centre each cell on its mean arrow and/or scale it to unit root-mean-square length.

    center="active": align each event and centre on active entries only (cpca.active).
    """
    X = np.array(encoded, dtype=complex, copy=True)
    if isinstance(center, str) and center == "active":
        from cpca.active import align_events, centre_active
        X = centre_active(align_events(X))
    elif center:
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


def shuffle_null(prepared, rng, n_shuffles=100):
    """Eigenvalues (n_shuffles, cells), largest first, of the matrix with each cell's column permuted across events.

    Centring and scaling act on each column separately and are unchanged by permuting the
    rows of that column, so shuffling the prepared matrix equals shuffling the arrows and
    preparing them again.
    """
    X = np.asarray(prepared)
    M, N = X.shape
    rows = np.broadcast_to(np.arange(M)[:, None], (M, N))
    null = np.empty((n_shuffles, N))
    for s in range(n_shuffles):
        P = np.take_along_axis(X, rng.permuted(rows, axis=0), axis=0)
        null[s] = eigh_desc(lag_matrix(P))[0]
    return null


def n_above_null(eigenvalues, null, centred, quantile=95.0):
    """Components to keep against a shuffle null; returns (k, ceiling).

    The shuffle keeps each cell's mean arrow, so for uncentred arrows its first eigenvalue is
    the mean pattern, not noise, and is present in the data for the same reason (in the hc-5
    wheel runs that mean pattern is the sequence itself). There the first component is always
    kept and the noise ceiling is the shuffle's second eigenvalue. Centred arrows have no mean,
    and the ceiling is the shuffle's first eigenvalue. Components are counted in order until
    the first one below the ceiling.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    structural = 0 if centred else 1
    ceiling = float(np.percentile(null[:, structural], quantile))
    k = structural
    while k < eigenvalues.size and eigenvalues[k] > ceiling:
        k += 1
    return min(k, eigenvalues.size), ceiling


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
