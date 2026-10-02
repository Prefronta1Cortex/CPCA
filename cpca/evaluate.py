"""Label-free quality measures for components, and the post-hoc map check.

Label-free (no behaviour used):
- how many components exceed the noise bound;
- split-half reproducibility of the kept components;
- timing consistency of each component: its event scores against the same
  scores after each event's arrows are given random angles.

Answer key (behaviour used once, at the end, per component):
- do the member cells' relative times follow their place-field centres?
- are the members' fields clustered on the track (a segment) rather than spread?
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from cpca.decode import readout
from cpca.theory import randomize_phases


def members(component, frac=0.2):
    """Cells whose weight is at least frac of the component's largest weight."""
    weight = np.abs(component)
    return weight >= frac * weight.max() if weight.max() > 0 else np.zeros(component.shape, bool)


def match_components(P, Q):
    """Mean |<p, q>| over the best one-to-one pairing of two sets of unit components."""
    if P.shape[1] == 0 or Q.shape[1] == 0:
        return np.nan
    overlap = np.nan_to_num(np.abs(P.conj().T @ Q))
    rows, cols = linear_sum_assignment(-overlap)
    return float(overlap[rows, cols].mean())


def timing_consistency(encoded, component, rng, n_shuffles=50):
    """z-score of the component's mean squared event score against phase-randomised events."""
    observed = float(np.mean(np.abs(encoded @ np.conj(component)) ** 2))
    null = np.array([np.mean(np.abs(randomize_phases(encoded, rng) @ np.conj(component)) ** 2) for _ in range(n_shuffles)])
    return (observed - null.mean()) / max(null.std(), 1e-12), observed, float(null.mean())


def member_order_vs_place(component, T, centres, circular, length, rng, frac=0.2, min_members=6, n_null=500, n_rotations=12):
    """Spearman between members' relative times and their place-field centres, with a null.

    centres: place-field centre per cell (NaN = no field). For circular tracks
    the best of n_rotations rotations is taken for data and null alike. The
    null permutes field centres among all cells that have one. Returns
    (rho, null 95th percentile of |rho|, number of members with a field).
    """
    _, times, _, included, _ = readout(component, T, min_weight=frac)
    use = included & np.isfinite(times) & np.isfinite(centres)
    n = int(use.sum())
    if n < min_members:
        return np.nan, np.nan, n
    rotations = range(n_rotations) if circular else [0]

    def rank_corr(a, b):
        a = a - a.mean(axis=-1, keepdims=True)
        b = b - b.mean(axis=-1, keepdims=True)
        denom = np.sqrt((a ** 2).sum(axis=-1) * (b ** 2).sum(axis=-1))
        return (a * b).sum(axis=-1) / np.where(denom > 0, denom, np.nan)

    rank_t = np.argsort(np.argsort(times[use])).astype(float)
    # ranks of the members' centres under each rotation (rotation only matters for circular tracks)
    rotated = np.array([np.argsort(np.argsort((centres[use] + s * length / n_rotations) % length)).astype(float) for s in rotations])
    observed = rank_corr(rank_t[None, :], rotated)                               # (rotations,)
    rho = float(observed[np.nanargmax(np.abs(observed))])
    # null: permute field centres among all cells with a field, keep the members fixed
    has = np.flatnonzero(np.isfinite(centres))
    member_idx = np.flatnonzero(use)
    perms = np.array([rng.permutation(has) for _ in range(n_null)])                # (n_null, n_has)
    pos_in_has = np.searchsorted(has, member_idx)
    permuted_centres = centres[perms[:, pos_in_has]]                              # (n_null, n)
    null = np.full(n_null, -np.inf)
    for s in rotations:
        moved = (permuted_centres + s * length / n_rotations) % length
        ranks = np.argsort(np.argsort(moved, axis=1), axis=1).astype(float)
        null = np.maximum(null, np.abs(rank_corr(rank_t[None, :], ranks)))
    return rho, float(np.nanpercentile(null, 95)), n


def member_field_compactness(component, centres, circular, length, rng, frac=0.2, n_null=500):
    """Mean pairwise distance between members' field centres, and the same for random cells with fields.

    Smaller than the null means the component is a segment of the track.
    Returns (observed, null 5th percentile, null mean, number of members with a field).
    """
    use = members(component, frac) & np.isfinite(centres)
    c = centres[use]
    n = c.size
    if n < 3:
        return np.nan, np.nan, np.nan, n

    def spread(values):
        d = np.abs(values[:, None] - values[None, :])
        if circular:
            d = np.minimum(d, length - d)
        return float(d[np.triu_indices(values.size, 1)].mean())

    pool = centres[np.isfinite(centres)]
    null = np.array([spread(rng.choice(pool, n, replace=False)) for _ in range(n_null)])
    return spread(c), float(np.percentile(null, 5)), float(null.mean()), n
