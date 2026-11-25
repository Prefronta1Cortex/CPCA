"""SPOTDisClust (Grossberger, Battaglia & Vinck, 2018) on reactivation events, with the authors' code.

The dissimilarity between two events is the SPOTDis of the authors' package (baselines/third_party/spot,
github.com/LGro/spot), in its pooled form ('SPOTD_xcorr_pooled'): for every cell active in both events, the
earth mover's distance between its spike-time cross-correlations with all other cells' spikes pooled,
averaged over those cells. The pairwise form ('SPOTD_xcorr', each pair of cells active in both events) cannot
separate patterns when each pair of events shares only a few active cells, as reactivation events do (about
4 on planted data, where replays of one route were nearer to volleys than to each other); the pooled form
separates them (scripts/validate_methods.py). Events
are then clustered with HDBSCAN on the precomputed dissimilarities, with the paper's setting
(minimum cluster size = minimum samples = 10). Spike times are expressed as a fraction of each event's
duration, so that events of different lengths are comparable (the paper uses epochs of one length; this is
the analogue for events, and the same normalisation complex PCA uses). Event pairs with no cell pair active
in both have no SPOTDis; they get the largest dissimilarity observed (maximally unlike).

The method clusters events, not cells. Each cluster is turned into a sequence template the usual way:
members are the cells that fire more often in the cluster's events than in the other events (one-sided
Fisher exact test, p < 0.01); each member's position is its median relative first-spike time over the
cluster's events in which it fires.
"""

import sys
from pathlib import Path

import numpy as np

SPOT = Path(__file__).resolve().parent / "third_party" / "spot"
if str(SPOT) not in sys.path:
    sys.path.insert(0, str(SPOT))

MIN_CLUSTER = 10


def _pooled_distances():
    """The authors' 'SPOTD_xcorr_pooled' (spot.spot.xcorr_pooled_spotdis_cpu_) with one fix.

    Their function calls signature_emd_(a, b) with the shorter list first when the lists differ in length,
    which signature_emd_ rejects (it raises for any unequal lengths; the plain 'SPOTD_xcorr' metric orders
    them correctly). This copy orders them the same way as 'SPOTD_xcorr' and leaves event pairs without any
    comparison as NaN instead of dividing by zero; nothing else is changed.
    """
    from numba import jit, prange
    from spot import tools
    from spot.spot import signature_emd_

    @jit(nopython=True, parallel=True)
    def pooled(spike_times, ii, pairs):
        n_epochs, n_channels = ii.shape[0], ii.shape[1]
        D = np.full((n_epochs, n_epochs), np.nan)
        for i in prange(pairs.shape[0]):
            e1, e2 = pairs[i, 0], pairs[i, 1]
            total, n = 0.0, 0
            for c in range(n_channels):
                n1 = ii[e1, c, 1] - ii[e1, c, 0]
                n2 = ii[e2, c, 1] - ii[e2, c, 0]
                if n1 == 0 or n2 == 0:
                    continue
                p1 = np.empty(0)
                p2 = np.empty(0)
                m1 = 0
                m2 = 0
                for t in range(n_channels):
                    if t != c:
                        m1 += ii[e1, t, 1] - ii[e1, t, 0]
                        m2 += ii[e2, t, 1] - ii[e2, t, 0]
                if m1 == 0 or m2 == 0:
                    continue
                p1 = np.empty(m1)
                p2 = np.empty(m2)
                k1 = 0
                k2 = 0
                for t in range(n_channels):
                    if t != c:
                        a, b = ii[e1, t, 0], ii[e1, t, 1]
                        p1[k1:k1 + b - a] = spike_times[a:b]
                        k1 += b - a
                        a, b = ii[e2, t, 0], ii[e2, t, 1]
                        p2[k2:k2 + b - a] = spike_times[a:b]
                        k2 += b - a
                x1 = np.array(tools.xcorr_list(spike_times[ii[e1, c, 0]:ii[e1, c, 1]], p1))
                x2 = np.array(tools.xcorr_list(spike_times[ii[e2, c, 0]:ii[e2, c, 1]], p2))
                if len(x1) >= len(x2):
                    total += signature_emd_(x1, x2)
                else:
                    total += signature_emd_(x2, x1)
                n += 1
            if n > 0:
                D[e1, e2] = total / n
                D[e2, e1] = total / n
        return D

    return pooled


def dissimilarity(spikes, ii, metric):
    """SPOTDis matrix (events x events) with the authors' scaling (divided by twice the epoch length, here 1)."""
    import itertools
    from spot import spot
    if metric == "SPOTD_xcorr":
        return spot.distances(spikes, ii, epoch_length=1.0, metric="SPOTD_xcorr")
    pairs = np.array(list(itertools.combinations(range(ii.shape[0]), 2)), dtype=np.int64)
    D = _pooled_distances()(spikes, ii, pairs) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def spot_arrays(times_ms, T_ms):
    """The authors' input format: one flat array of spike times (fraction of the event) and (events, cells, 2) index ranges."""
    M, N, _ = times_ms.shape
    flat, ii = [], np.zeros((M, N, 2), dtype=np.int64)
    k = 0
    for e in range(M):
        for c in range(N):
            s = np.sort(times_ms[e, c][np.isfinite(times_ms[e, c])]) / T_ms[e]
            ii[e, c] = (k, k + s.size)
            flat.append(s)
            k += s.size
    return np.concatenate(flat).astype(np.float64) if flat else np.zeros(0), ii


def fit(times_ms, T_ms, rng=None):
    import hdbscan
    from scipy.stats import fisher_exact
    T_ms = np.asarray(T_ms, float)
    spikes, ii = spot_arrays(times_ms, T_ms)
    D = dissimilarity(spikes, ii, "SPOTD_xcorr_pooled")
    D = np.where(np.isfinite(D), D, np.nanmax(D[np.isfinite(D)]) if np.isfinite(D).any() else 1.0)
    np.fill_diagonal(D, 0.0)
    labels = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=MIN_CLUSTER, min_samples=MIN_CLUSTER).fit_predict(D)
    active = np.isfinite(times_ms).any(axis=-1)                         # (events, cells)
    with np.errstate(invalid="ignore"):
        rel = np.nanmin(times_ms, axis=-1) / T_ms[:, None]          # first spike, fraction of the event
    orders, members = [], []
    for c in sorted(set(labels) - {-1}):
        ev = labels == c
        a_in, a_out = active[ev].sum(axis=0), active[~ev].sum(axis=0)
        m = np.array([fisher_exact([[a_in[k], ev.sum() - a_in[k]], [a_out[k], (~ev).sum() - a_out[k]]], alternative="greater")[1] < 0.01
                      for k in range(active.shape[1])])
        o = np.full(times_ms.shape[1], np.nan)
        for cell in np.flatnonzero(m):
            o[cell] = np.nanmedian(rel[ev, cell])
        members.append(m)
        orders.append(o)
    return {"labels": labels, "n_clusters": len(orders), "orders": orders, "members": members,
            "noise_fraction": float(np.mean(labels == -1))}
