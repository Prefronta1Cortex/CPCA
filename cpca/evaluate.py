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
    """Spearman between a component's members' relative times and their place-field centres, with a null.

    centres: place-field centre per cell (NaN = no field). For circular tracks
    the best of n_rotations rotations is taken for data and null alike. The
    null permutes field centres among all cells that have one. Returns
    (rho, null 95th percentile of |rho|, number of members with a field).
    """
    _, times, _, included, _ = readout(component, T, min_weight=frac)
    return order_vs_place(np.where(included, times, np.nan), centres, circular, length, rng, min_members, n_null, n_rotations)


def order_vs_place(times, centres, circular, length, rng, min_members=6, n_null=500, n_rotations=12):
    """Same test for any method's cell order: `times` is one value per cell (NaN = not a member).

    Used for complex-PCA components, seqNMF factor lags, or any other per-cell ordering,
    so every method is scored identically.
    """
    times = np.asarray(times, dtype=float)
    use = np.isfinite(times) & np.isfinite(centres)
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


def best_order(orders, key, rng, min_members=6, n_null=1000):
    """Score a method's outputs against a one-dimensional answer key, with a null that is fair across output counts.

    orders: list of per-cell times (NaN = not a member), one per output (component, factor, type).
    key: per-cell true time (NaN = no field). The statistic is the signed Spearman correlation
    between an output's member times and their key, over members with a key (at least
    `min_members`); a sequence recovered backwards counts as wrong. Recovery is the best output.
    The null permutes the key among all cells that have one, scores every output under the same
    permutation and keeps the best, so a method gains nothing from producing more outputs.

    Returns rho (best output), p (against the max-over-outputs null), null95 (of that null),
    index of the best output, n (its members with a key), coverage (n / cells with a key),
    and per output: rho, its own null 95th percentile, n; `precision` is the fraction of testable
    outputs above their own null95.
    """
    from scipy.stats import rankdata

    key = np.asarray(key, float)
    has = np.isfinite(key)
    n_key = int(has.sum())
    rows, tests = [], []
    for j, times in enumerate(orders):
        times = np.asarray(times, float)
        use = np.isfinite(times) & has
        n = int(use.sum())
        rows.append({"index": j, "n": n, "rho": np.nan, "null95": np.nan})
        if n >= min_members:
            tests.append((j, np.flatnonzero(use), rankdata(times[use])))
    out = {"rho": np.nan, "p": np.nan, "null95": np.nan, "index": -1, "n": 0, "coverage": 0.0, "n_key": n_key,
           "n_outputs": len(orders), "n_testable": len(tests), "outputs": rows, "precision": np.nan}
    if not tests:
        return out
    key_cells = np.flatnonzero(has)
    perms = np.array([rng.permutation(key_cells) for _ in range(n_null)])        # (n_null, n_key)
    lookup = np.full(key.size, -1)
    lookup[key_cells] = np.arange(n_key)

    def corr(a, b):
        a = a - a.mean(-1, keepdims=True)
        b = b - b.mean(-1, keepdims=True)
        d = np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1))
        return (a * b).sum(-1) / np.where(d > 0, d, np.nan)

    null_max = np.full(n_null, -np.inf)
    for j, cells, rank_t in tests:
        rho = float(corr(rank_t, rankdata(key[cells])))
        permuted = key[perms[:, lookup[cells]]]                                    # (n_null, n)
        null = corr(rank_t[None, :], np.apply_along_axis(rankdata, 1, permuted))
        null = np.nan_to_num(null, nan=-1.0)
        null_max = np.maximum(null_max, null)
        rows[j].update(rho=rho, null95=float(np.percentile(null, 95)))
    best = max((r for r in rows if np.isfinite(r["rho"])), key=lambda r: r["rho"])
    out.update(rho=best["rho"], index=best["index"], n=best["n"], coverage=best["n"] / max(n_key, 1),
               p=float((1 + np.sum(null_max >= best["rho"])) / (n_null + 1)), null95=float(np.percentile(null_max, 95)),
               precision=float(np.mean([r["rho"] > r["null95"] for r in rows if np.isfinite(r["rho"])])))
    return out


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
