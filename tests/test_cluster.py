"""Mixture complex PCA separates routes that cancel in a single lag matrix."""

import numpy as np
from scipy.stats import spearmanr

from cpca import decompose, encode, readout
from cpca.cluster import fit_mixture, split_half_stability, unit_rows

T = 100.0


def _mixture_events(rng, n_cells=40, n_events=1200, participation=0.3, jitter=4.0, reverse_frac=0.5, replay_frac=1.0):
    """Forward (label 0) and reverse (label 1) plays of one route, plus synchronous background (label 2)."""
    order = np.linspace(5.0, 85.0, n_cells)
    times = np.full((n_events, n_cells), np.nan)
    label = np.zeros(n_events, dtype=int)
    for i in range(n_events):
        fire = rng.random(n_cells) < participation
        if rng.random() >= replay_frac:
            label[i] = 2
            times[i, fire] = rng.normal(45.0, jitter, size=int(fire.sum()))
            continue
        rev = rng.random() < reverse_frac
        label[i] = int(rev)
        base = order[::-1] if rev else order
        times[i, fire] = base[fire] + rng.normal(0.0, jitter, size=int(fire.sum()))
    return np.clip(times, 0.0, T), order, label


def _order_corr(component, order):
    _, t, _, inc, _ = readout(component, T, min_weight=0.1)
    return spearmanr(t[inc], order[inc]).statistic


def test_single_template_cancels_but_mixture_separates():
    rng = np.random.default_rng(0)
    times, order, label = _mixture_events(rng)
    S = encode(times, T)
    _, comps, _, _ = decompose(S, n_components=1)
    single = abs(_order_corr(comps[:, 0], order))
    fit = fit_mixture(S, K=2, rng=rng, n_init=5)
    both = [abs(_order_corr(fit["templates_c"][:, k], order)) for k in range(2)]
    signs = [np.sign(_order_corr(fit["templates_c"][:, k], order)) for k in range(2)]
    assert single < 0.5
    assert min(both) > 0.9
    assert signs[0] != signs[1]
    ok = fit["assign"] >= 0
    agreement = np.mean(fit["assign"][ok] == label[ok])
    assert max(agreement, 1 - agreement) > 0.9


def test_unit_rows_drop_small_events():
    S = np.array([[1, 1j, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    U, ok = unit_rows(S, min_cells=2)
    assert ok.tolist() == [True, False, False]
    assert np.isclose(np.linalg.norm(U[0]), 1)
    assert np.all(U[1:] == 0)


def test_stability_prefers_true_number_of_routes():
    rng = np.random.default_rng(2)
    times, _, _ = _mixture_events(rng, n_events=1600, participation=0.35)
    S = encode(times, T)
    stab = {K: split_half_stability(S, K, rng, n_init=4) for K in (2, 4)}
    assert stab[2] > stab[4]


def test_mixture_with_background_events_still_finds_both_orders():
    """60% of events are synchronous volleys; the two routes still get their own templates."""
    rng = np.random.default_rng(3)
    times, order, label = _mixture_events(rng, n_events=2000, participation=0.2, replay_frac=0.4)
    S = encode(times, T)
    fit = fit_mixture(S, K=2, rng=np.random.default_rng(0), n_init=6)
    corr = [_order_corr(fit["templates_delta"][:, k], order) for k in range(2)]
    assert max(corr) > 0.9 and min(corr) < -0.9
    # forward and reverse plays land in different groups
    for route in (0, 1):
        in_groups = np.bincount(fit["assign"][(label == route) & (fit["assign"] >= 0)], minlength=2)
        assert in_groups.max() > 0.85 * in_groups.sum()
