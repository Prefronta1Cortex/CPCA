"""Centring on active entries, for reactivation events (center="active").

Column centring (cpca.decompose.prepare) subtracts each cell's mean arrow from every event,
including events where the cell is silent. The silent entries then carry -mean, so which
cells fire together (co-firing) enters the lag matrix even when their timing is unrelated.

Here each event is first rotated so its summed arrow points along 0 (each event's own clock
start is arbitrary), then each cell's mean arrow over the events where it fires is
subtracted from those entries only; silent entries stay 0. If a cell's phase, when it fires,
does not depend on which other cells fire with it, every cross term then has zero mean:
co-firing and fixed per-cell latencies add nothing, and only coordinated timing remains.

The matching noise bound permutes each cell's entries among the events where it fires,
which keeps co-firing exactly and destroys only the coordination of timing.
"""

import numpy as np

from cpca.theory import eigh_desc, lag_matrix


def align_events(encoded):
    """Rotate each event so the sum of its arrows has phase 0."""
    S = np.asarray(encoded, dtype=complex)
    return S * np.exp(-1j * np.angle(S.sum(axis=1)))[:, None]


def centre_active(encoded):
    """Subtract each cell's mean arrow over its active events from those entries only."""
    X = np.array(encoded, dtype=complex, copy=True)
    active = X != 0
    count = active.sum(axis=0)
    mean = np.zeros(X.shape[1], complex)
    np.divide(X.sum(axis=0), count, out=mean, where=count > 0)
    return np.where(active, X - mean[None, :], 0)


def shuffle_null_active(prepared, rng, n_shuffles=100):
    """Eigenvalues (n_shuffles, cells), largest first, with each cell's active entries permuted among its active events."""
    X = np.asarray(prepared)
    M, N = X.shape
    active = [np.flatnonzero(X[:, c] != 0) for c in range(N)]
    null = np.empty((n_shuffles, N))
    P = np.zeros_like(X)
    for s in range(n_shuffles):
        for c, idx in enumerate(active):
            P[idx, c] = X[rng.permutation(idx), c]
        null[s] = eigh_desc(lag_matrix(P))[0]
    return null
