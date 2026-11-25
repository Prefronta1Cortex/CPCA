"""Is the sleep fit contained in the waking fit? An asymmetric (subset) comparison of two complex PCA fits.

Sleep replays part of what was experienced, so each kept sleep component should lie inside the space spanned by
the kept waking components and their conjugates (a reversed replay is the conjugate), while the waking fit may
hold more than sleep replays. containment(V, B) is, per sleep component (unit-norm columns of V), the share of it
inside the orthonormal basis B: ||B^H v||^2, between 0 and 1. The cell-identity null permutes the rows (cells) of
B, which keeps its dimension.
"""

import numpy as np


def basis(W, with_conjugates=True):
    """Orthonormal basis (cells x d) of the span of the columns of W (and of their conjugates)."""
    W = np.asarray(W, complex)
    if W.size == 0:
        return np.zeros((W.shape[0], 0), complex)
    M = np.hstack([W, np.conj(W)]) if with_conjugates else W
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, s > 1e-10 * s.max()]


def containment(V, B):
    """Share of each column of V (normalised) inside span(B)."""
    V = np.asarray(V, complex)
    V = V / np.maximum(np.linalg.norm(V, axis=0, keepdims=True), 1e-300)
    return np.sum(np.abs(B.conj().T @ V) ** 2, axis=0)


def best_match(V, W):
    """Per column of V: the best squared overlap with one column of W or its conjugate (both normalised)."""
    V = np.asarray(V, complex) / np.maximum(np.linalg.norm(V, axis=0, keepdims=True), 1e-300)
    W = np.asarray(W, complex) / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-300)
    return np.maximum(np.abs(W.conj().T @ V) ** 2, np.abs(W.T @ V) ** 2).max(axis=0)


def null_containment(V, B, rng, n_perm=1000, W=None):
    """Per permutation of B's cells (rows): containment of each column of V, (n_perm, columns); and, if W is given,
    the best single-component match against W permuted the same way."""
    span = np.empty((n_perm, V.shape[1]))
    match = np.empty((n_perm, V.shape[1])) if W is not None else None
    for i in range(n_perm):
        perm = rng.permutation(B.shape[0])
        span[i] = containment(V, B[perm])
        if W is not None:
            match[i] = best_match(V, W[perm])
    return span, match
