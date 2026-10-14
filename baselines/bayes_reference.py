"""Bayesian replay decoding (Zhang et al., 1998; Davidson et al., 2009; Grosmark & Buzsaki, 2016).

This is the reference that USES the waking map: place-field rate maps are the
input, each event is decoded to a position posterior per 20 ms bin, and the
weighted correlation of the posterior with time is compared with a column-cycle
shuffle (circular shift of each time bin's posterior). It is not a competitor
under the label-free rule; it anchors how much replay the events contain.

Reported per epoch: fraction of events significant at p < 0.05, and under the
strict criterion of Grosmark & Buzsaki (|R| > 0.5, at least five bins, p < 0.05).
"""

import time

import numpy as np
from scipy.ndimage import gaussian_filter1d


def rate_maps(session, cell_ids, run, n_bins=40, min_spikes=40, min_peak_hz=3.0, min_occupancy_s=0.2):
    """Direction-agnostic place maps (cells x bins) and the mask of place cells."""
    maze = session["maze_epoch"]
    on = (session["times"] >= maze[0]) & (session["times"] < maze[1])
    st, sid = session["times"][on], session["ids"][on]
    pos_t, pos_x, direction, length = run["t"], run["x"], run["direction"], run["length"]
    j = np.clip(np.searchsorted(pos_t, st), 1, pos_t.size - 1)
    j = np.where(np.abs(pos_t[j - 1] - st) <= np.abs(pos_t[j] - st), j - 1, j)
    keep = (np.abs(pos_t[j] - st) < 0.04) & (direction[j] != 0)
    sid, sx = sid[keep], pos_x[j[keep]]
    dt = float(np.median(np.diff(pos_t)))
    edges = np.linspace(0, length, n_bins + 1)
    occupancy = np.histogram(pos_x[direction != 0], bins=edges)[0] * dt
    valid = occupancy > min_occupancy_s
    maps = np.zeros((len(cell_ids), n_bins))
    place = np.zeros(len(cell_ids), bool)
    for k, cell in enumerate(cell_ids):
        m = sid == cell
        if m.sum() < min_spikes:
            continue
        rate = np.zeros(n_bins)
        rate[valid] = np.histogram(sx[m], bins=edges)[0][valid] / occupancy[valid]
        maps[k] = gaussian_filter1d(rate, 1.0, mode="nearest")
        place[k] = maps[k].max() >= min_peak_hz
    centres = 0.5 * (edges[:-1] + edges[1:])
    return maps, place, centres


def weighted_corr(post, centres):
    T, X = post.shape
    t = np.arange(T)[:, None]
    x = centres[None, :]
    w = post.sum()
    mt, mx = (post * t).sum() / w, (post * x).sum() / w
    cov = (post * (t - mt) * (x - mx)).sum()
    vt, vx = (post * (t - mt) ** 2).sum(), (post * (x - mx) ** 2).sum()
    return cov / np.sqrt(vt * vx) if vt > 0 and vx > 0 else np.nan


def run(blocks, maps, place, centres, rng, bin_s=0.02, min_cells=5, n_shuffles=500):
    t0 = time.perf_counter()
    log_l = np.log(maps[place] + 1e-3)
    expo = bin_s * maps[place].sum(axis=0)
    n_bins = maps.shape[1]
    R = np.full(len(blocks), np.nan)
    P = np.full(len(blocks), np.nan)
    nb = np.zeros(len(blocks), int)
    for i, counts in enumerate(blocks):
        c = counts[:, place]
        nb[i] = c.shape[0]
        if (c.sum(0) > 0).sum() < min_cells or c.shape[0] < 3:
            continue
        lp = c @ log_l - expo[None, :]
        lp -= lp.max(axis=1, keepdims=True)
        post = np.exp(lp)
        post /= post.sum(axis=1, keepdims=True)
        r = weighted_corr(post, centres)
        if not np.isfinite(r):
            continue
        shifts = rng.integers(0, n_bins, size=(n_shuffles, post.shape[0]))
        idx = (np.arange(n_bins)[None, :] - shifts[:, :, None]) % n_bins
        null = np.array([weighted_corr(np.take_along_axis(post, idx[s], axis=1), centres) for s in range(n_shuffles)])
        R[i] = r
        P[i] = (np.sum(np.abs(null) >= abs(r)) + 1) / (n_shuffles + 1)
    tested = np.isfinite(R)
    sig = tested & (P < 0.05)
    strict = sig & (np.abs(R) > 0.5) & (nb >= 5)
    return {"n_tested": int(tested.sum()), "frac_sig": float(sig.sum() / max(tested.sum(), 1)),
            "frac_strict": float(strict.sum() / max(tested.sum(), 1)), "mean_absR": float(np.nanmean(np.abs(R))),
            "n_forward": int((sig & (R > 0)).sum()), "n_reverse": int((sig & (R < 0)).sum()), "runtime_s": time.perf_counter() - t0}
