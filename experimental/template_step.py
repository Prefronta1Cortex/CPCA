"""Sequences that share cells: one template per sequence, each event assigned to one of them.

When two orders play over the same cells (two running directions' maps on one track) and occur
about equally often, they are not orthogonal, and PCA returns their sum and their differences:
on planted data the first centred component follows both maps at once (|rho| about 0.7 with
each) and the later ones each map at 0.75-0.9. The direction split of one component
(cpca/direction.py) then fits the blend.

The repair keeps PCA as the first stage and adds one step after it:

1. cells: the union of the members of the kept components (weight >= 20% of each component's
   largest); events: those in which at least 3 of them fire; raw (uncentred) arrows, as in the
   direction split.
2. templates: K = the number of kept components, started from their complex ICA rotation
   (cpca/ica.py), which is sparser than the components and already close to single sequences,
   and separately from the components themselves.
3. alternate: every event goes to the template it matches best either way round,
   max(|<s, q>|, |<s, conj q>|), and gets that template's direction; it counts only if this
   best match beats the 95th percentile of the same best match after shuffling which cell got
   which arrow in that event (cpca.direction.permute_within_events; 20 shuffles drawn once, so
   the alternation can settle); each template is refitted on its events with the backward ones
   conjugated (leading eigenvector). Templates with fewer than MIN_EVENTS events are dropped;
   two templates at least MERGE alike (up to conjugation) are one, and the one with fewer events
   is dropped. Repeat until nothing changes. Each step can only increase the energy the templates
   capture, sum over events of the best |<s, q>|^2; of the two starts the one with more energy
   is kept.
4. held-out check: steps 1-3 (PCA with the same K, ICA, alternation) are rerun on each half of
   the events; a template is kept only if in both halves a template at least SIMILAR to it was
   found and that template finds expressing events in the other half above chance (each
   held-out event judged by its own permutation p-value, binomial count against 5%, Bonferroni
   over templates and halves; cpca.direction.expression_p).

A phase-invariant mixture of templates was tried before PCA existed in this project and was at
chance on real sessions (docs/lessons_learned.md). The differences are the PCA start, the
conjugation (direction), the per-event gate, and the held-out check.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest

from cpca.decompose import decompose, prepare
from cpca.direction import _leading, expression_p, permute_within_events
from cpca.ica import unmix

MERGE = 0.8
MIN_EVENTS = 20
SIMILAR = 0.8
N_ICA_STARTS = 3


def similarity(a, b, conjugate=True):
    """How alike two templates are, up to a common phase and (conjugate=True) to conjugation (time reversal): in [0, 1]."""
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    if not conjugate:
        return float(abs(np.vdot(a, b)))
    return float(max(abs(np.vdot(a, b)), abs(np.vdot(np.conj(a), b))))


def members(component, frac=0.2):
    w = np.abs(component)
    return w >= frac * w.max() if w.max() > 0 else np.zeros(w.shape, bool)


def _assign(S, Q, shuffled, conjugate=True):
    F = np.abs(S @ np.conj(Q))
    B = np.abs(S @ Q) if conjugate else np.zeros_like(F)
    best = np.maximum(F, B)
    k = np.argmax(best, axis=1)
    rows = np.arange(S.shape[0])
    top = best[rows, k]
    null = np.array([(np.maximum(np.abs(R @ np.conj(Q)), np.abs(R @ Q)) if conjugate else np.abs(R @ np.conj(Q))).max(axis=1)
                     for R in shuffled])
    express = top > np.percentile(null, 95, axis=0)
    return k, F[rows, k] >= B[rows, k], express, top


def alternate(S, Q0, rng, n_shuffles=20, max_iter=100, min_events=MIN_EVENTS, conjugate=True):
    """Step 3 on events S (events x cells) from initial templates Q0 (cells x K). Returns a dict."""
    Q = Q0 / np.maximum(np.linalg.norm(Q0, axis=0, keepdims=True), 1e-12)
    shuffled = [permute_within_events(S, rng) for _ in range(n_shuffles)]
    state = None
    for it in range(max_iter):
        k, fwd, express, top = _assign(S, Q, shuffled, conjugate)
        label = np.where(express, k, -1)
        counts = np.bincount(label[label >= 0], minlength=Q.shape[1])
        keep = np.ones(Q.shape[1], bool)
        # settled: at most 0.5% of events (and at least 2) still change template or direction, or out of iterations
        changed = np.inf if state is None else int(np.sum((label != state[0]) | (fwd != state[1])))
        settled = changed <= max(2, 0.005 * S.shape[0]) or it == max_iter - 1
        if settled:
            # only once the assignment has settled: early on, a template that will become one sequence starts
            # with few events and close to the blend it is being separated from
            for i in range(Q.shape[1]):
                for j in range(i + 1, Q.shape[1]):
                    if keep[i] and keep[j] and similarity(Q[:, i], Q[:, j], conjugate) >= MERGE:
                        keep[i if counts[i] < counts[j] else j] = False
            keep &= counts >= min_events
        if not keep.all():
            Q, state = Q[:, keep], None
            if Q.shape[1] == 0:
                break
            continue
        if settled:
            break
        state = (label, fwd)
        for j in range(Q.shape[1]):
            m = label == j
            if m.sum() >= 5:
                Q[:, j] = _leading(np.where(fwd[m][:, None], S[m], np.conj(S[m])))
    if Q.shape[1] == 0:
        return {"templates": Q, "label": np.full(S.shape[0], -1), "forward": np.ones(S.shape[0], bool),
                "energy": 0.0, "n_iter": it + 1}
    k, fwd, express, top = _assign(S, Q, shuffled, conjugate)
    return {"templates": Q, "label": np.where(express, k, -1), "forward": fwd, "energy": float(np.sum(top[express] ** 2)),
            "n_iter": it + 1}


def _pca(encoded, K, center, scale):
    eigenvalues, components, _, _ = decompose(encoded, n_components=K, center=center, scale=scale)
    return eigenvalues, components, prepare(encoded, center=center, scale=scale)


def fit_templates(encoded, eigenvalues, components, prepared, rng, member_frac=0.2, min_active=3, min_events=MIN_EVENTS,
                  conjugate=True, extra_starts=None):
    """Steps 1-3. encoded: raw arrows (events x cells); eigenvalues, components: the kept PCA components."""
    K = components.shape[1]
    n_cells = encoded.shape[1]
    if K == 0:
        return None
    U = np.any([members(components[:, j], member_frac) for j in range(K)], axis=0)
    S_all = encoded[:, U]
    active = (S_all != 0).sum(axis=1) >= min_active
    if active.sum() < 2 * min_events:
        return None
    S = S_all[active]
    starts = {"components": components[U]}
    if K > 1:
        for s in range(N_ICA_STARTS):
            starts[f"ica{s}"] = unmix(prepared, eigenvalues, components, rng)[0][U]
    if extra_starts is not None:
        for name, make in extra_starts.items():
            Q0 = make(S)
            if Q0 is not None and Q0.shape[1]:
                starts[name] = Q0
    runs = {name: alternate(S, Q0.copy(), rng, min_events=min_events, conjugate=conjugate) for name, Q0 in starts.items()}
    start = max(runs, key=lambda n: runs[n]["energy"])
    r = runs[start]
    T = np.zeros((n_cells, r["templates"].shape[1]), complex)
    T[U] = r["templates"]
    return {"templates": T, "cells": U, "events": np.flatnonzero(active), "label": r["label"], "forward": r["forward"],
            "start": start, "energy": {n: v["energy"] for n, v in runs.items()}, "n_iter": r["n_iter"]}


def heldout_check(encoded, fitted, K, rng, center=True, scale=True, member_frac=0.2, min_active=3, alpha=0.05):
    """Step 4. Per template of `fitted`: (similarity of the best match, p) in each half, and whether it is kept."""
    n = encoded.shape[0]
    perm = rng.permutation(n)
    halves = perm[: n // 2], perm[n // 2:]
    T = fitted["templates"]
    nT = T.shape[1]
    sims = np.zeros((2, nT))
    ps = np.ones((2, nT))
    for f, (train, held) in enumerate((halves, halves[::-1])):
        vals, comps, prep = _pca(encoded[train], K, center, scale)
        fold = fit_templates(encoded[train], vals, comps, prep, rng, member_frac, min_active)
        if fold is None or fold["templates"].shape[1] == 0:
            continue
        Tf = fold["templates"]
        S_held = encoded[held][:, fold["cells"]]
        S_held = S_held[(S_held != 0).sum(axis=1) >= min_active]
        fold_p = []
        for j in range(Tf.shape[1]):
            hits = int((expression_p(S_held, Tf[fold["cells"], j], rng) <= alpha).sum())
            fold_p.append(binomtest(hits, S_held.shape[0], alpha, alternative="greater").pvalue if S_held.shape[0] else 1.0)
        # one-to-one: a template cannot borrow the evidence of a fold template that stands for another one
        sim = np.array([[similarity(T[:, i], Tf[:, j]) for j in range(Tf.shape[1])] for i in range(nT)])
        rows, cols = linear_sum_assignment(-sim)
        for i, j in zip(rows, cols):
            sims[f, i], ps[f, i] = sim[i, j], fold_p[j]
    keep = (sims >= SIMILAR).all(axis=0) & (ps * 2 * max(nT, 1) < alpha).all(axis=0)
    return {"similarity": sims, "p": ps, "keep": keep}


def sequences(result, rng, test=True, member_frac=0.2, min_active=3):
    """Steps 1-4 on a cpca.fit result (uses its encoded arrows, kept components and conventions)."""
    K = result["n_kept"]
    vals, comps = result["eigenvalues"][:K], result["pca_components"]
    fitted = fit_templates(result["encoded"], vals, comps, result["prepared"], rng, member_frac, min_active)
    if fitted is None:
        return None
    if test and fitted["templates"].shape[1]:
        s = result["settings"]
        fitted["check"] = heldout_check(result["encoded"], fitted, K, rng, center=s["center"], scale=s["scale"],
                                        member_frac=member_frac, min_active=min_active)
    return fitted
