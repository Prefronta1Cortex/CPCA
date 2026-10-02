"""Lag matrix, eigenvectors, and event scores.

Eigenvectors are anchored so the strongest cell points at angle 0.
"""

import numpy as np

from cpca.theory import anchor_phase, eigh_desc, lag_matrix, scores


def decompose(encoded, n_components=None):
    """Top components of the uncentered lag matrix.

    Returns eigenvalues (largest first), anchored eigenvectors as columns,
    complex scores of shape (events, components), and the lag matrix.
    """
    encoded = np.asarray(encoded)
    lag = lag_matrix(encoded)
    eigenvalues, eigenvectors = eigh_desc(lag)
    n_cells = eigenvectors.shape[1]
    if n_components is None:
        n_components = n_cells
    n_components = min(int(n_components), n_cells)
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = np.array(eigenvectors[:, :n_components], copy=True)
    for component in range(n_components):
        eigenvectors[:, component] = anchor_phase(eigenvectors[:, component])
    return eigenvalues, eigenvectors, scores(encoded, eigenvectors), lag
