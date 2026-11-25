"""Complex independent component analysis for un-mixing shared subspaces.

PCA returns the subspace that patterns occupy. When two patterns share cells
with different timing, or occur equally often, PCA returns mixtures of them.
ICA rotates inside the kept subspace so each output is as non-Gaussian as
possible, which for sparse patterns (each present in a minority of events)
recovers the patterns themselves. The default is gradient ascent of
sum_j E|y_j|^3 over unitary rotations of the whitened PCA scores; the complex
fixed-point algorithm of Bingham and Hyvarinen (2000) is kept as
method="fastica", but it does not split routes that each take part in many
events (tests/test_pipeline.py).
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
            # sources are y = W^T x, so column j of W is conj(w_j) of the paper:
            # conj(w_j) <- E[conj(x) y_j g(u_j)] - E[g(u_j) + u_j g'(u_j)] conj(w_j)
            W_new = (X.conj().T @ (Y * g)) / M - W * np.mean(g + U * dg, axis=0)[None, :]
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


def riemannian_ica(whitened, rng, power=1.5, n_init=8, max_iter=200, tol=1e-6):
    """Unitary unmixing matrix W maximising sum_j mean |y_j|^(2 power), y = whitened @ W.

    Gradient ascent on the unitary group (skew-Hermitian direction, polar retraction,
    backtracking line search). The identity and n_init - 1 random unitary starts are
    tried; the best by contrast is returned.
    """
    Z = np.asarray(whitened)
    M, k = Z.shape
    if k <= 1:
        return np.eye(k, dtype=complex)

    def loss_and_grad(W):
        Y = Z @ W
        U = np.abs(Y) ** 2
        Up = np.power(U + 1e-12, power - 1)
        # gradient of the loss with respect to conj(W)
        return -float(np.sum(np.mean(U * Up, axis=0))), (Z.conj().T @ (Y * (-power * Up))) / M

    starts = [np.eye(k, dtype=complex)]
    starts += [_symmetric_orthogonalize(rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k))) for _ in range(n_init - 1)]
    best = None
    for W in starts:
        loss, grad = loss_and_grad(W)
        step = 0.5
        for _ in range(max_iter):
            G = W.conj().T @ grad
            A = G.conj().T - G
            norm2 = float(np.real(np.sum(A.conj() * A)))
            if norm2 < tol ** 2:
                break
            for _ in range(20):
                W_try = _symmetric_orthogonalize(W + step * (W @ A))
                loss_try, grad_try = loss_and_grad(W_try)
                if loss_try <= loss - 1e-4 * step * norm2:
                    W, loss, grad = W_try, loss_try, grad_try
                    step = min(step * 1.5, 2.0)
                    break
                step *= 0.5
            else:
                break
        if best is None or loss < best[0]:
            best = (loss, W)
    return best[1]


def unmix(prepared, eigenvalues, eigenvectors, rng, method="riemannian", **kwargs):
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
    W = (riemannian_ica if method == "riemannian" else complex_fastica)(whitened, rng, **kwargs)
    # prepared ~ Y W^H Lambda^(1/2) V^T  ->  pattern j in cell space is column j of V Lambda^(1/2) conj(W)
    patterns = eigenvectors @ (np.sqrt(eigenvalues)[:, None] * np.conj(W))
    norms = np.linalg.norm(patterns, axis=0)
    patterns = patterns / np.maximum(norms, 1e-12)
    for j in range(k):
        patterns[:, j] = anchor_phase(patterns[:, j])
    ic_scores = prepared @ np.conj(patterns)
    # order by the energy each pattern explains
    order = np.argsort(-np.mean(np.abs(ic_scores) ** 2, axis=0))
    return patterns[:, order], ic_scores[:, order], W[:, order]
