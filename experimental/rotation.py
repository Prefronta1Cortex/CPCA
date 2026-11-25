"""Sparse rotation of complex components within their span (complex quartimax).

Given orthonormal components V (cells x k), find the unitary rotation R that maximises sum |(V R)_ij|^4, which
concentrates each rotated component on as few cells as possible. The span is unchanged, so the rotation only decides
how the kept subspace is cut into components. The optimisation runs Jacobi sweeps over pairs of columns: each pair is
rotated by [[c, -s e^{-j phi}], [s e^{j phi}, c]] with the (theta, phi) on a grid that maximises the pair's objective.
"""

import numpy as np

_THETA = np.linspace(0.0, np.pi / 2, 33)
_PHI = np.linspace(0.0, 2 * np.pi, 32, endpoint=False)


def _best_pair(a, b):
    c = np.cos(_THETA)[:, None, None]
    s = np.sin(_THETA)[:, None, None]
    e = np.exp(1j * _PHI)[None, :, None]
    new_a = c * a + s * e * b
    new_b = -s * np.conj(e) * a + c * b
    f = (np.abs(new_a) ** 4).sum(-1) + (np.abs(new_b) ** 4).sum(-1)
    i, j = np.unravel_index(np.argmax(f), f.shape)
    return new_a[i, j], new_b[i, j], f[i, j]


def quartimax(V, max_sweeps=30, tol=1e-9):
    """Rotated components (cells x k, unit norm) and the objective sum |w|^4 after each sweep."""
    W = np.array(V, dtype=complex, copy=True)
    k = W.shape[1]
    history = [float((np.abs(W) ** 4).sum())]
    for _ in range(max_sweeps):
        for p in range(k - 1):
            for q in range(p + 1, k):
                W[:, p], W[:, q], _ = _best_pair(W[:, p], W[:, q])
        history.append(float((np.abs(W) ** 4).sum()))
        if history[-1] - history[-2] < tol * max(history[-2], 1e-300):
            break
    return W, history
