"""Is a component a sequence? A read-out check that rejects components that only contrast groups of cells.

A component's members (weight at least member_frac of the largest) each have a phase. A sequence spreads its members'
phases along an arc; a component that only contrasts two co-firing groups has phases near 0 and pi (a real vector up
to a common phase), which the read-out (cpca.decode.readout) would still turn into a two-step "order".
circularity(q) = |sum_k w_k^2| / sum_k |w_k|^2 over the members: 1 for a real vector (a pure group contrast or a
single group), small when the phases spread. The threshold is calibrated on planted data (docs/readout_check.md).
"""

import numpy as np

from cpca.decode import readout


def members(q, member_frac=0.2):
    w = np.abs(q)
    return w >= member_frac * w.max() if w.max() > 0 else np.zeros(q.shape, bool)


def circularity(q, member_frac=0.2):
    q = np.asarray(q, complex)
    m = members(q, member_frac)
    w = q[m]
    den = np.sum(np.abs(w) ** 2)
    return float(abs(np.sum(w * w)) / den) if den > 0 else np.nan


def shuffle_timing(times, rng):
    """Within each event, give the active cells each other's spike trains (which cells fire is kept)."""
    out = times.copy()
    for e in range(times.shape[0]):
        act = np.flatnonzero(np.isfinite(times[e]).any(axis=-1))
        out[e, act] = times[e, act[rng.permutation(act.size)]]
    return out


def redraw_latency(times, T, rng):
    """Each active cell keeps its spike count; its relative times (t / T) are redrawn from its own pooled relative times."""
    rel = times / T[:, None, None]
    out = np.full_like(times, np.nan)
    for c in range(times.shape[1]):
        pool = rel[:, c][np.isfinite(rel[:, c])]
        if pool.size == 0:
            continue
        fin = np.isfinite(times[:, c])
        out[:, c][fin] = rng.choice(pool, int(fin.sum())) * np.broadcast_to(T[:, None], fin.shape)[fin]
    out.sort(axis=-1)
    return out


def _expression(Z, q):
    q = q / max(np.linalg.norm(q), 1e-300)
    return float(np.mean(np.maximum(np.abs(np.conj(Z) @ q), np.abs(Z @ q)) ** 2))


def timing_check(train_times, train_T, test_times, test_T, rng, center=True, scale=True, ica=False, n_shuffles=50,
                 member_frac=0.2, null="latency"):
    """Fit on the training events; per kept component, its expression on the held-out events against surrogates of the
    same events: null="latency" (default; each cell's relative times redrawn from its own pooled times) or "shuffle"
    (spike trains shuffled among each event's active cells). Returns the fit and one dict per component."""
    from cpca import fit
    from cpca.decompose import prepare
    from cpca.encode import encode
    r = fit(train_times, train_T, rng=rng, ica=ica, center=center, scale=scale, n_shuffles=50)
    if not r["n_kept"]:
        return r, []
    P = r["patterns"] if ica else r["pca_components"]
    Z = prepare(encode(test_times, test_T), center=center, scale=scale)
    null_e = np.zeros((n_shuffles, P.shape[1]))
    for s in range(n_shuffles):
        surrogate = redraw_latency(test_times, test_T, rng) if null == "latency" else shuffle_timing(test_times, rng)
        Zs = prepare(encode(surrogate, test_T), center=center, scale=scale)
        null_e[s] = [_expression(Zs, P[:, j]) for j in range(P.shape[1])]
    comps = []
    for o in outputs(P, r["T_read_ms"], member_frac=member_frac):
        e = _expression(Z, P[:, o["index"]])
        o.update(expression=e, null_mean=float(null_e[:, o["index"]].mean()),
                 p_timing=float((1 + np.sum(null_e[:, o["index"]] >= e)) / (1 + n_shuffles)))
        comps.append(o)
    return r, comps


def outputs(patterns, T_read, max_circularity=None, member_frac=0.2):
    """Per pattern (cells x k): order (NaN for non-members), members, circularity, and whether it passes the check."""
    out = []
    for j in range(patterns.shape[1]):
        q = patterns[:, j]
        _, t, _, inc, _ = readout(q, T_read, min_weight=member_frac)
        c = circularity(q, member_frac)
        out.append({"index": j, "order": np.where(inc, t, np.nan), "members": inc, "circularity": c,
                    "passes": True if max_circularity is None else bool(np.isfinite(c) and c <= max_circularity)})
    return out
