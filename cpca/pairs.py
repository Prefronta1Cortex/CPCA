"""Conjugate-pair read-out: the order of a route played both ways, read from a pair of components.

When a route is replayed forward and backward in equal numbers, the replay part of the lag matrix is
Re(q q^H) = c c^T + s s^T with q = c + js: the order is not lost but split over a near-degenerate pair of real
components, each carrying about half of the eigenvalue, and the phases of either component alone are 0 or pi.
q itself is still in the pair's span. This module finds, for a pair of components (v1, v2), the unit combination
u = a v1 + b v2 whose phases are most spread around the circle, meaning the one least like a real vector: the one
that zeroes the doubled-phase resultant sum_k u_k^2 (a quadratic equation in b/a; for real orthonormal v1, v2 the
answer is (v1 + j v2) / sqrt 2). For a degenerate pair this is q or its conjugate, either of which gives the order up
to reversal. The read-out is label-free and costs no extra fitting; with best-of-outputs scoring the extra outputs are
accounted for by the null (docs/segments.md, addendum).
"""

import numpy as np

from cpca.direction import order_of


def circular_combination(v1, v2):
    """Unit combination of v1 and v2 (orthonormalised within their span) with the smallest |sum_k u_k^2|."""
    v1 = np.asarray(v1, complex)
    v2 = np.asarray(v2, complex)
    v1 = v1 / max(np.linalg.norm(v1), 1e-300)
    v2 = v2 - np.vdot(v1, v2) * v1
    n2 = np.linalg.norm(v2)
    if n2 < 1e-12:
        return v1
    v2 = v2 / n2
    p11, p22, p12 = np.sum(v1 * v1), np.sum(v2 * v2), np.sum(v1 * v2)       # pseudo-covariance (complex symmetric)
    # u = v1 + r v2 (up to scale) with p11 + 2 p12 r + p22 r^2 = 0
    if abs(p22) > 1e-12:
        disc = np.sqrt(p12 ** 2 - p11 * p22 + 0j)
        roots = [(-p12 + disc) / p22, (-p12 - disc) / p22]
    elif abs(p12) > 1e-12:
        roots = [-p11 / (2 * p12)]
    else:
        roots = [1j]                                                        # v1 already circular; any partner works
    best = None
    for r in roots:
        u = v1 + r * v2
        u = u / max(np.linalg.norm(u), 1e-300)
        resultant = abs(np.sum(u * u))
        if best is None or resultant < best[0]:
            best = (resultant, u)
    return best[1]


def pair_orders(patterns, member_frac=0.2, how="adjacent"):
    """Orders (one per pair of components; NaN for non-members) from the circular combination of each pair.

    patterns: (cells, k) complex components. how: "adjacent" pairs (j, j + 1) in the given order, or "all" pairs.
    """
    P = np.asarray(patterns)
    k = P.shape[1] if P.ndim == 2 else 0
    if k < 2:
        return []
    pairs = [(j, j + 1) for j in range(k - 1)] if how == "adjacent" else [(a, b) for a in range(k) for b in range(a + 1, k)]
    return [order_of(circular_combination(P[:, a], P[:, b]), member_frac=member_frac) for a, b in pairs]
