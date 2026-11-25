"""Cell assemblies from binned spike counts: PCA and ICA (Lopes-dos-Santos, Ribeiro & Tort, 2013).

The method complex PCA extends, used as its authors describe it: spike counts in short bins, z-scored per
cell; the number of assemblies is the number of eigenvalues of the correlation matrix above the
Marchenko-Pastur bound (1 + sqrt(N/B))^2; FastICA in the subspace of those components gives one weight
vector per assembly (unit length, sign fixed so that the largest weight is positive). Members of an
assembly are the cells whose weight exceeds the mean weight by more than 2 standard deviations, the rule of
the applications that followed the paper (e.g. van de Ven et al., 2016).

On reactivation events the input is the events' activity binned at 25 ms and concatenated, so the method
sees exactly the spikes the other methods see. It gives groups of cells, not orders.
"""

import numpy as np
from sklearn.decomposition import FastICA

BIN_MS = 25.0


def binned_events(times_ms, T_ms, bin_ms=BIN_MS):
    """(bins, cells) spike counts: each event cut into bins of bin_ms (the last one shorter), events concatenated."""
    rows = []
    for e in range(times_ms.shape[0]):
        n_bins = max(int(np.ceil(T_ms[e] / bin_ms)), 1)
        edges = np.arange(n_bins + 1) * bin_ms
        t = times_ms[e]
        counts = np.zeros((n_bins, t.shape[0]))
        for c in range(t.shape[0]):
            s = t[c][np.isfinite(t[c])]
            if s.size:
                counts[:, c] = np.histogram(np.clip(s, 0, edges[-1] - 1e-9), bins=edges)[0]
        rows.append(counts)
    return np.concatenate(rows, axis=0)


def fit(times_ms, T_ms, rng, bin_ms=BIN_MS, member_sd=2.0):
    """Assemblies of the events: weights (cells x k), member masks (list of boolean arrays), k."""
    X = binned_events(times_ms, np.asarray(T_ms, float), bin_ms)
    B, N = X.shape
    sd = X.std(axis=0)
    active = sd > 0
    Z = np.zeros_like(X)
    Z[:, active] = (X[:, active] - X[:, active].mean(axis=0)) / sd[active]
    C = np.corrcoef(Z[:, active].T)
    values, vectors = np.linalg.eigh(C)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    bound = (1 + np.sqrt(active.sum() / B)) ** 2
    k = int(np.sum(values > bound))
    weights = np.zeros((N, k))
    if k == 0:
        return {"weights": weights, "members": [], "k": 0, "bound": bound, "n_bins": B}
    P = vectors[:, :k]
    projected = Z[:, active] @ P
    if k == 1:
        W = P
    else:
        ica = FastICA(n_components=k, random_state=int(rng.integers(2 ** 31 - 1)), whiten="unit-variance", max_iter=2000)
        ica.fit(projected)
        W = P @ ica.components_.T          # back to cell space (unmixing weights, as in the paper)
    W = W / np.linalg.norm(W, axis=0, keepdims=True)
    W = W * np.sign(W[np.argmax(np.abs(W), axis=0), np.arange(k)])[None, :]
    weights[active] = W
    members = [weights[:, j] > weights[:, j].mean() + member_sd * weights[:, j].std() for j in range(k)]
    return {"weights": weights, "members": members, "k": k, "bound": bound, "n_bins": B}
