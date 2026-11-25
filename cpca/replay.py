"""Bayesian replay analysis of Grosmark & Buzsaki (2016, Supplementary Materials), for hc-11.

Reproduces the paper's procedure as written, so that our reading of the data can be checked against the
original result (sequence content higher in POST than in PRE sleep).

  events      population synchrony: all pyramidal spikes in 1 ms bins, Gaussian kernel (SD 15 ms); trigger at
              mean + 3 SD of the bins inside NREM of PRE and POST combined; an event runs from where the rate
              rises above the mean to where it returns to it; 50-500 ms; at least 5 pyramidal cells fire.
              The paper also required an LFP-detected ripple inside each event (50.6% of synchrony events
              passed); the hc-11 files contain no LFP, so this condition cannot be applied.
  fields      per running direction (linear tracks; the one running direction on the circular track), speed at
              least 5 cm/s: spikes and occupancy in 2 cm bins, each smoothed with a Gaussian (SD 5 cm), rate =
              smoothed spikes / smoothed occupancy. Place cell: at least 5 consecutive bins above the 99th
              percentile of 5000 null rate maps (the cell's spikes redrawn from the unsmoothed occupancy) and
              an in-field peak of at least 1 Hz.
  decoding    events of at least 100 ms in which at least max(5, 10% of the place cells) fire; 20 ms bins;
              flat prior, Poisson likelihood with all place cells of the direction; on linear tracks the
              event's direction is the one with the larger Pr(direction | spikes) from all pyramidal cells'
              running rates in each direction.
  score       r = posterior-weighted correlation of position with time; 1000 null templates, each place
              cell's unsmoothed rate map shifted circularly by its own random number of bins and re-smoothed;
              rZ = (|r| - mean |r_null|) / SD |r_null|; p = share of |r_null| >= |r|. Secondary null: the
              event's 20 ms bins permuted (1000 times).
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from cpca.events import in_intervals

BIN_M = 0.02
SMOOTH_M = 0.05
TAU_S = 0.02
EPS_HZ = 1e-3


# ----------------------------------------------------------------------------- events
def synchrony_events(pooled_times, baseline_intervals, allowed_intervals, by_cell, cell_ids,
                     kernel_sd_s=0.015, n_sd=3.0, min_s=0.05, max_s=0.5, min_cells=5):
    """(n, 2) start/end of population-synchrony events lying inside allowed_intervals."""
    t0 = float(min(np.min(baseline_intervals), np.min(allowed_intervals))) - 1.0
    t1 = float(max(np.max(baseline_intervals), np.max(allowed_intervals))) + 1.0
    edges_n = int(np.ceil((t1 - t0) / 0.001))
    idx = np.floor((pooled_times[(pooled_times >= t0) & (pooled_times < t1)] - t0) / 0.001).astype(np.int64)
    rate = np.bincount(idx, minlength=edges_n).astype(np.float32)
    rate = gaussian_filter1d(rate, kernel_sd_s / 0.001, mode="constant")
    centres = t0 + (np.arange(edges_n) + 0.5) * 0.001
    base = rate[in_intervals(centres, baseline_intervals)]
    mean, trigger = float(base.mean()), float(base.mean() + n_sd * base.std())
    above = rate > mean
    d = np.diff(above.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends, edges_n]
    out = []
    for a, b in zip(starts, ends):
        if rate[a:b].max() < trigger:
            continue
        s, e = t0 + a * 0.001, t0 + b * 0.001
        if not (min_s <= e - s <= max_s):
            continue
        out.append((s, e))
    ev = np.array(out) if out else np.zeros((0, 2))
    if ev.size:
        inside = np.zeros(len(ev), bool)
        for lo, hi in allowed_intervals:
            inside |= (ev[:, 0] >= lo) & (ev[:, 1] <= hi)
        ev = ev[inside]
    if ev.size:
        n_active = np.zeros(len(ev), int)
        for c in cell_ids:
            sp = by_cell[c]
            n_active += (np.searchsorted(sp, ev[:, 1]) - np.searchsorted(sp, ev[:, 0])) > 0
        ev = ev[n_active >= min_cells]
    return ev


def immobile_intervals(session, max_speed=0.05, smooth_s=0.25, min_s=0.5):
    """Intervals of the maze epoch in which the tracked head (2-D position) moves slower than max_speed."""
    t, xy = session["pos_t"], session["pos_xy"]
    ok = np.isfinite(xy).all(axis=0)
    t, xy = t[ok], xy[:, ok]
    dt = float(np.median(np.diff(t)))
    xs = gaussian_filter1d(xy, smooth_s / dt, axis=1, mode="nearest")
    speed = np.r_[0.0, np.hypot(*np.diff(xs, axis=1)) / np.maximum(np.diff(t), 1e-6)]
    still = (speed < max_speed) & np.r_[True, np.diff(t) < 0.2]
    d = np.diff(still.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1
    if still[0]:
        starts = np.r_[0, starts]
    if still[-1]:
        ends = np.r_[ends, still.size]
    iv = np.array([(t[a], t[b - 1]) for a, b in zip(starts, ends) if t[b - 1] - t[a] >= min_s])
    return iv if iv.size else np.zeros((0, 2))


# ----------------------------------------------------------------------------- place fields
def _smooth(x, axis=-1):
    return gaussian_filter1d(np.asarray(x, float), SMOOTH_M / BIN_M, axis=axis, mode="nearest")


def place_fields(session, cell_ids, run, direction, rng, n_null=5000, min_run=5, min_peak_hz=1.0):
    """Per cell, in one running direction: smoothed rate map, unsmoothed rate map, place-cell flag, mean running rate."""
    maze = session["maze_epoch"]
    on = (session["times"] >= maze[0]) & (session["times"] < maze[1])
    st, sid = session["times"][on], session["ids"][on]
    t, x, dirn, L = run["t"], run["x"], run["direction"], run["length"]
    j = np.clip(np.searchsorted(t, st), 1, t.size - 1)
    j = np.where(np.abs(t[j - 1] - st) <= np.abs(t[j] - st), j - 1, j)
    keep = (np.abs(t[j] - st) < 0.04) & (dirn[j] == direction)
    sid, sx = sid[keep], x[j[keep]]
    dt = float(np.median(np.diff(t)))
    n_bins = int(round(L / BIN_M))
    edges = np.linspace(0, L, n_bins + 1)
    occ = np.histogram(x[dirn == direction], bins=edges)[0] * dt
    occ_s = _smooth(occ)
    visited = occ_s > 1e-9
    pdf = occ / occ.sum()
    run_time = occ.sum()
    n = len(cell_ids)
    rate, raw, place, mean_rate = np.zeros((n, n_bins)), np.zeros((n, n_bins)), np.zeros(n, bool), np.zeros(n)
    for k, c in enumerate(cell_ids):
        cnt = np.histogram(sx[sid == c], bins=edges)[0].astype(float)
        mean_rate[k] = cnt.sum() / run_time
        rate[k] = np.where(visited, _smooth(cnt) / np.where(visited, occ_s, 1), 0.0)
        raw[k] = np.where(occ > 0, cnt / np.where(occ > 0, occ, 1), 0.0)
        m = int(cnt.sum())
        if m == 0:
            continue
        null = rng.multinomial(m, pdf, size=n_null).astype(float)
        null = np.where(visited, _smooth(null, axis=1) / np.where(visited, occ_s, 1), 0.0)
        above = rate[k] > np.percentile(null, 99, axis=0)
        best = 0.0
        d = np.diff(np.r_[0, above.astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a >= min_run:
                best = max(best, rate[k, a:b].max())
        place[k] = best >= min_peak_hz
    centres = 0.5 * (edges[:-1] + edges[1:])
    return {"rate": rate, "raw": raw, "place": place, "mean_rate": mean_rate, "x": centres, "occupancy": occ}


def null_templates(raw, rng, n_null=1000):
    """(n_null, cells, bins): each cell's unsmoothed map shifted circularly by its own random number of bins, re-smoothed."""
    n, P = raw.shape
    shifts = rng.integers(0, P, size=(n_null, n))
    idx = (np.arange(P)[None, None, :] - shifts[:, :, None]) % P
    return np.clip(_smooth(raw[np.arange(n)[None, :, None], idx], axis=2), 0, None)


# ----------------------------------------------------------------------------- decoding
def binned(by_cell, cell_ids, start, n_bins, tau=TAU_S):
    """Spike counts (n_bins, cells) of an event in tau-second bins from its start."""
    out = np.zeros((n_bins, len(cell_ids)))
    for k, c in enumerate(cell_ids):
        sp = by_cell[c]
        a, b = np.searchsorted(sp, start), np.searchsorted(sp, start + n_bins * tau)
        if b > a:
            np.add.at(out[:, k], np.minimum(((sp[a:b] - start) / tau).astype(int), n_bins - 1), 1)
    return out


def _posterior_moments(counts, logF, sumF, x, tau=TAU_S):
    """Posterior-weighted first and second position moments per time bin, for a stack of templates.

    counts (M, k) for the k active cells; logF (S, k, P); sumF (S, P) = sum over all template cells.
    Returns A, B (S, M): sum_p post * x and sum_p post * x^2.
    """
    S, k, P = logF.shape
    L = (counts @ logF.transpose(1, 0, 2).reshape(k, S * P)).reshape(-1, S, P).transpose(1, 0, 2)
    L = L - tau * sumF[:, None, :]
    L -= L.max(axis=2, keepdims=True)
    post = np.exp(L)
    post /= post.sum(axis=2, keepdims=True)
    return post @ x, post @ (x * x)


def _wcorr(A, B, M):
    tm = np.arange(M) - (M - 1) / 2.0
    mx = A.mean(axis=-1, keepdims=True)
    cov = (A * tm).sum(axis=-1)
    vx = B.sum(axis=-1) - M * mx[..., 0] ** 2
    vt = (tm ** 2).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt(vt * np.maximum(vx, 0))


def decoder(rate, nulls):
    """The observed template and its null templates stacked, as the decoder uses them."""
    F = np.concatenate([rate[None], nulls], axis=0)                       # (1 + S, cells, P)
    return {"logF": np.log(F + EPS_HZ), "sumF": F.sum(axis=1)}


def score_event(counts, dec, x, rng, n_time=1000, tau=TAU_S):
    """r, rZ and both p-values of one event (counts: bins x place cells of the template)."""
    M = counts.shape[0]
    active = np.flatnonzero(counts.sum(axis=0) > 0)
    c = counts[:, active]
    A, B = _posterior_moments(c, dec["logF"][:, active], dec["sumF"], x, tau)
    r_all = _wcorr(A, B, M)
    r, rn = float(r_all[0]), np.abs(r_all[1:])
    perm = np.argsort(rng.random((n_time, M)), axis=1)
    rt = np.abs(_wcorr(A[0][perm], B[0][perm], M))
    sd = rn.std()
    return {"r": r, "rZ": float((abs(r) - rn.mean()) / sd) if sd > 0 else np.nan,
            "p": float((1 + np.sum(rn >= abs(r))) / (1 + rn.size)),
            "p_time": float((1 + np.sum(rt >= abs(r))) / (1 + n_time))}


def direction_of(counts_all, mean_rates, duration_s):
    """Index of the running direction with the larger Pr(direction | spikes) (all pyramidal cells, whole event)."""
    sp = counts_all.sum(axis=0)
    ll = [float(sp @ np.log(m + EPS_HZ) - duration_s * m.sum()) for m in mean_rates]
    return int(np.argmax(ll))
