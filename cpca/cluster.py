"""Mixture complex PCA: several templates, each averaged over its own events.

The single lag matrix C averages the lag arrows s_ik * conj(s_il) over every
event. When events come from more than one route (or one route played
forward and backward) the arrows of different routes point different ways
and cancel. Here events are first assigned to K templates and each template
is the top eigenvector of the lag matrix of its own events only. Assignment
uses |<s_i, p_k>|^2 / |s_i|^2, which ignores the event's arbitrary clock
start (a common phase) and the event's size. This is k-means on the complex
unit sphere with a phase-invariant distance; each group's template is refit
as its own top eigenvector, so the two steps alternate like ordinary k-means.

Do not subtract each event's mean arrow before clustering. The residuals of
a forward event and of the same route played backward differ only by a
global sign, and a global phase is exactly what the assignment ignores, so
centering erases the direction (checked on synthetic data: the groups then
contain equal numbers of forward and reverse events).
"""

import numpy as np

from cpca.stats import coactivation_null, timing_centered
from cpca.theory import anchor_phase, eigh_desc, lag_matrix


def unit_rows(encoded, min_cells=3):
    """Rows divided by their length. Rows with fewer active cells than min_cells become 0."""
    encoded = np.asarray(encoded)
    active = (encoded != 0).sum(axis=1)
    norm = np.linalg.norm(encoded, axis=1)
    out = np.zeros_like(encoded)
    ok = (norm > 0) & (active >= min_cells)
    out[ok] = encoded[ok] / norm[ok, None]
    return out, ok


def top_eigenvector(rows):
    _, vectors = eigh_desc(lag_matrix(rows))
    return anchor_phase(vectors[:, 0])


def _init_templates(U, K, rng):
    """k-means++ style seeding with the phase-invariant distance 1 - |<u, p>|^2."""
    M = U.shape[0]
    chosen = [int(rng.integers(M))]
    dist = 1 - np.abs(U @ np.conj(U[chosen[0]])) ** 2
    for _ in range(1, K):
        prob = np.maximum(dist, 0)
        prob = prob / prob.sum() if prob.sum() > 0 else np.full(M, 1 / M)
        pick = int(rng.choice(M, p=prob))
        chosen.append(pick)
        dist = np.minimum(dist, 1 - np.abs(U @ np.conj(U[pick])) ** 2)
    return np.array([U[c] for c in chosen]).T  # (N, K)


def phase_only(encoded):
    """Every nonzero arrow set to length 1. Which cells fired still matters; how much they fired does not."""
    encoded = np.asarray(encoded)
    out = np.zeros_like(encoded)
    nz = encoded != 0
    out[nz] = encoded[nz] / np.abs(encoded[nz])
    return out


def fit_mixture(encoded, K, rng, n_init=10, max_iter=60, min_cells=3, tol=1e-6, similarity="cosine"):
    """Assign events to K templates and refit each template from its own events.

    similarity="cosine" uses |<s_i, p_k>|^2 / |s_i|^2 on the arrows as given, so
    an event's arrow lengths (burst size, co-activation) enter the assignment.
    similarity="phase" first sets every nonzero event arrow to length 1, so the
    event's burst sizes no longer matter; the template's lengths still do.
    similarity="phase_both" also sets every template arrow to length 1 during
    assignment, so only the relative timing of the cells that fired decides
    the group: which cells fired, and how often, is ignored. Use this when
    different events involve different subsets of cells (segments of a route).

    Returns a dict with assignments (-1 for events with fewer than min_cells
    active cells), the templates read from C of each group (columns), the
    templates read from C - C0 of each group, group sizes, and the objective
    (mean best squared cosine). The best of n_init restarts is kept.
    """
    encoded = np.asarray(encoded)
    if similarity not in ("cosine", "phase", "phase_both"):
        raise ValueError("similarity must be 'cosine', 'phase' or 'phase_both'")
    work = phase_only(encoded) if similarity in ("phase", "phase_both") else encoded
    U, ok = unit_rows(work, min_cells=min_cells)
    rows = np.flatnonzero(ok)
    if rows.size < K:
        raise ValueError("fewer usable events than templates")
    Uo = U[rows]
    best = None
    for _ in range(n_init):
        P = _init_templates(Uo, K, rng)
        assign = np.full(rows.size, -1)
        objective = -np.inf
        for _ in range(max_iter):
            if similarity == "phase_both":
                # Rows of Uo are unit arrows divided by sqrt(n_i). With unit template arrows Q,
                # |Uo_i . conj(Q_k)| * sqrt(n_i) / n_ik is the resultant length of the phase
                # differences over the n_ik cells shared by event i and template k: 1 when every
                # shared cell agrees on its relative time, ~0 when the differences are random.
                Q = phase_only(P.T).T
                active = (Uo != 0).astype(float)
                n_i = active.sum(axis=1, keepdims=True)
                n_ik = np.maximum(active @ (np.abs(Q) ** 2), 1e-12)
                sim = np.abs(Uo @ np.conj(Q)) ** 2 * n_i / n_ik ** 2
            else:
                sim = np.abs(Uo @ np.conj(P)) ** 2        # (M_ok, K)
            new_assign = np.argmax(sim, axis=1)
            new_objective = float(sim[np.arange(rows.size), new_assign].mean())
            if np.array_equal(new_assign, assign) or new_objective - objective < tol:
                assign, objective = new_assign, new_objective
                break
            assign, objective = new_assign, new_objective
            for k in range(K):
                members = Uo[assign == k]
                if members.shape[0] >= 2:
                    P[:, k] = top_eigenvector(members)
                else:
                    P[:, k] = Uo[int(rng.integers(rows.size))]
        if best is None or objective > best["objective"]:
            best = {"assign_ok": assign.copy(), "P": P.copy(), "objective": objective}
    assign = np.full(encoded.shape[0], -1)
    assign[rows] = best["assign_ok"]
    templates_c = np.zeros((encoded.shape[1], K), dtype=complex)
    templates_delta = np.zeros((encoded.shape[1], K), dtype=complex)
    sizes = np.zeros(K, dtype=int)
    for k in range(K):
        members = encoded[assign == k]
        sizes[k] = members.shape[0]
        if members.shape[0] < 2:
            templates_c[:, k] = np.nan
            templates_delta[:, k] = np.nan
            continue
        lag = lag_matrix(members)
        templates_c[:, k] = anchor_phase(eigh_desc(lag)[1][:, 0])
        delta = timing_centered(lag, coactivation_null(members, members != 0))
        templates_delta[:, k] = anchor_phase(eigh_desc(delta)[1][:, 0])
    return {
        "assign": assign,
        "templates_c": templates_c,
        "templates_delta": templates_delta,
        "sizes": sizes,
        "objective": best["objective"],
        "similarity": similarity,
    }


def match_templates(P, Q):
    """Mean |<p, q>| over the best one-to-one pairing of two template sets."""
    from scipy.optimize import linear_sum_assignment

    overlap = np.abs(P.conj().T @ Q)
    overlap = np.nan_to_num(overlap)
    rows, cols = linear_sum_assignment(-overlap)
    return float(overlap[rows, cols].mean())


def split_half_stability(encoded, K, rng, **kwargs):
    """Fit on two random halves and report how well the template sets match.

    Label-free model selection: K with the highest stability is preferred.
    """
    encoded = np.asarray(encoded)
    order = rng.permutation(encoded.shape[0])
    mid = encoded.shape[0] // 2
    a = fit_mixture(encoded[order[:mid]], K, rng, **kwargs)
    b = fit_mixture(encoded[order[mid:]], K, rng, **kwargs)
    return match_templates(a["templates_c"], b["templates_c"])
