"""Count-based assembly detection (Lopes-dos-Santos, Ribeiro & Tort, 2013).

The real-valued version of the same pipeline: spike counts per cell per
event, z-scored, PCA, components above the Marchenko-Pastur bound, FastICA.
It finds which cells fire together (assemblies) and cannot see order. Used to
show what the complex version adds and that both find the same groups.
"""

import numpy as np
from sklearn.decomposition import FastICA

from cpca.decompose import marchenko_pastur_bound


def event_counts(times_ms):
    """Spike counts per event and cell from a (events, cells, spikes) time array."""
    return np.isfinite(times_ms).sum(axis=-1).astype(float)


def assemblies(counts, rng, max_components=20):
    """Assembly weight vectors (k x cells, unit norm, largest weight positive) and the kept count.

    Returns weights, number of assemblies, eigenvalues, bound.
    """
    M, N = counts.shape
    keep = counts.std(axis=0) > 0
    Z = (counts[:, keep] - counts[:, keep].mean(axis=0)) / counts[:, keep].std(axis=0)
    cov = Z.T @ Z / M
    values, vectors = np.linalg.eigh(cov)
    values, vectors = values[::-1], vectors[:, ::-1]
    bound = marchenko_pastur_bound(M, int(keep.sum()), variance=values.mean())
    k = int(min((values > bound).sum(), max_components))
    weights = np.full((k, N), np.nan)
    if k >= 1:
        if k == 1:
            W = vectors[:, :1].T
        else:
            ica = FastICA(n_components=k, random_state=int(rng.integers(1 << 31)), max_iter=2000, tol=1e-4)
            ica.fit(Z @ vectors[:, :k])
            W = (vectors[:, :k] @ ica.components_.T).T
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
        W = W * np.sign(W[np.arange(k), np.argmax(np.abs(W), axis=1)])[:, None]
        weights[:, keep] = W
    return weights, k, values, bound


def members(weights, frac=0.2):
    """Cells whose weight is at least frac of the assembly's largest weight."""
    w = np.nan_to_num(weights)
    return w >= frac * w.max() if w.max() > 0 else np.zeros(weights.shape, bool)


def best_jaccard(mask_sets_a, mask_sets_b):
    """For each member set in a, the best Jaccard overlap with any set in b."""
    out = []
    for a in mask_sets_a:
        best = 0.0
        for b in mask_sets_b:
            union = np.logical_or(a, b).sum()
            if union:
                best = max(best, np.logical_and(a, b).sum() / union)
        out.append(best)
    return np.array(out)
