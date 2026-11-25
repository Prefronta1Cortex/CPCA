"""The recurring-fragment criterion: does some output order the cells of one stretch of the track like the map?

For each output (a per-cell order, NaN = not a member) and each map (field centres per cell, NaN = no field), the
statistic is |Spearman| between the members' order and their field centres, restricted to the members whose
centres lie in one contiguous stretch of fixed length (`frac` of the track) and only where at least `min_cells`
members do. The best is taken over outputs, maps, senses (absolute value) and stretch positions (starts every
`step` of the track). The null permutes each map's centres among the cells that have a field in it and takes the
same best, so the stretch search, like the output search, gains nothing. Tracks are treated as linear.

`timing_p` compares the real statistic with the same statistic after refitting on data whose spike trains were
shuffled among each event's active cells (who fires kept, who fires when destroyed).
"""

import numpy as np
from scipy.stats import rankdata

FRAC = 0.4
MIN_CELLS = 8
STEP = 0.05


def _nancorr(A, B):
    m = np.isfinite(A) & np.isfinite(B)
    n = m.sum(axis=1)
    A0, B0 = np.where(m, A, 0.0), np.where(m, B, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = A0.sum(axis=1) / n, B0.sum(axis=1) / n
        da, db = np.where(m, A - ma[:, None], 0.0), np.where(m, B - mb[:, None], 0.0)
        return (da * db).sum(axis=1) / np.sqrt((da ** 2).sum(axis=1) * (db ** 2).sum(axis=1))


def _best_rows(o, P, length, frac, min_cells, step):
    """Per row of P (centres of the member cells, one row per permutation): best |rho| over stretch positions."""
    L = frac * length
    best = np.zeros(P.shape[0])
    where = np.full(P.shape[0], np.nan)
    O = np.broadcast_to(o, P.shape)
    for a in np.arange(0.0, length - L + 1e-9, step * length):
        W = (P >= a) & (P <= a + L)
        ok = W.sum(axis=1) >= min_cells
        if not ok.any():
            continue
        r = np.abs(_nancorr(rankdata(np.where(W, O, np.nan), axis=1, nan_policy="omit"),
                            rankdata(np.where(W, P, np.nan), axis=1, nan_policy="omit")))
        r = np.where(ok, np.nan_to_num(r), 0.0)
        better = r > best
        best, where = np.where(better, r, best), np.where(better, a, where)
    return best, where


def best_fragment(orders, centres, length, rng=None, n_null=1000, frac=FRAC, min_cells=MIN_CELLS, step=STEP):
    """Best fragment match over outputs, maps and stretches; p against the same best under permuted centres (n_null=0: no null)."""
    has = {s: np.flatnonzero(np.isfinite(c)) for s, c in centres.items()}
    perms = {s: np.array([rng.permutation(h) for _ in range(n_null)]) for s, h in has.items()} if n_null else {}
    out = {"abs": 0.0, "index": -1, "map": np.nan, "start": np.nan, "n_tested": 0}
    null_max = np.zeros(n_null)
    for j, o in enumerate(orders):
        o = np.asarray(o, float)
        for s, cen in centres.items():
            idx = np.flatnonzero(np.isfinite(o) & np.isfinite(cen))
            if idx.size < min_cells:
                continue
            out["n_tested"] += 1
            P = cen[idx][None, :]
            if n_null:
                P = np.vstack([P, cen[perms[s][:, np.searchsorted(has[s], idx)]]])
            best, where = _best_rows(o[idx], P, length, frac, min_cells, step)
            if best[0] > out["abs"]:
                out.update(abs=float(best[0]), index=j, map=s, start=float(where[0]))
            if n_null:
                null_max = np.maximum(null_max, best[1:])
    if out["index"] < 0:
        return {**out, "abs": np.nan, "p": np.nan}
    out["p"] = float((1 + np.sum(null_max >= out["abs"])) / (n_null + 1)) if n_null else np.nan
    return out


def timing_p(real_abs, shuffled_abs):
    a = np.nan_to_num(np.asarray(shuffled_abs, float), nan=0.0)
    return float((1 + np.sum(a >= real_abs)) / (1 + a.size)) if np.isfinite(real_abs) else np.nan
