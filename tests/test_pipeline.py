"""The two decisive design facts, the ICA step, event detection, and the full pipeline on planted routes."""

import numpy as np
from scipy.stats import spearmanr

from cpca import decompose, encode, event_spike_times, fit, population_bursts
from cpca.evaluate import match_components, member_order_vs_place, members, timing_consistency
from cpca.ica import unmix
from cpca.decompose import prepare
from sim.routes import generate, make_routes


def _route_events(rng, span_ms, reverse_frac=0.4, n_cells=30, n_events=3000):
    """One route, forward/backward mix, random clock per event. Returns arrows and the route template."""
    T = 100.0
    order = np.linspace(0, span_ms, n_cells)
    S = np.zeros((n_events, n_cells), complex)
    for i in range(n_events):
        fire = rng.random(n_cells) < 0.2
        base = order[::-1] if rng.random() < reverse_frac else order
        t = base[fire] + rng.normal(0, 6, fire.sum())
        S[i, fire] = np.exp(1j * np.pi * t / T) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    p = np.exp(1j * np.pi * order / T)
    return S, p / np.linalg.norm(p)


def test_fact_1_routes_separate_only_when_they_fill_the_half_circle():
    rng = np.random.default_rng(0)
    # small arc: forward and backward are nearly parallel and PCA merges them
    S, p = _route_events(rng, span_ms=25.0)
    _, V, _, _ = decompose(S)
    assert abs(np.vdot(p, np.conj(p))) > 0.85
    assert abs(np.vdot(V[:, 0], p)) > 0.9 and abs(np.vdot(V[:, 0], np.conj(p))) > 0.9
    # full half-circle: they are orthogonal and land on separate components
    S, p = _route_events(rng, span_ms=100.0)
    _, V, _, _ = decompose(S)
    assert abs(np.vdot(p, np.conj(p))) < 0.1
    assert abs(np.vdot(V[:, 0], p)) > 0.95 and abs(np.vdot(V[:, 0], np.conj(p))) < 0.3
    assert abs(np.vdot(V[:, 1], np.conj(p))) > 0.95 and abs(np.vdot(V[:, 1], p)) < 0.3


def test_fact_2_centring_removes_the_participation_component():
    """Aligned volleys with unequal participation: uncentred PC1 is participation; centring removes it."""
    rng = np.random.default_rng(1)
    M, N = 3000, 40
    participation = np.linspace(0.05, 0.6, N)
    fire = rng.random((M, N)) < participation[None, :]
    times = np.where(fire, rng.normal(50.0, 8.0, size=(M, N)), np.nan)
    S = encode(times, 100.0)
    values_raw, V_raw, _, _ = decompose(S)
    values_c, _, _, _ = decompose(S, center=True, scale=True)
    assert spearmanr(np.abs(V_raw[:, 0]), participation).statistic > 0.95
    assert values_raw[0] / values_raw.sum() > 0.3
    assert values_c[0] / values_c.sum() < 0.1


def test_pca_gives_the_union_of_shared_cell_routes_and_ica_splits_it():
    rng = np.random.default_rng(2)
    routes = make_routes(rng, n_cells=60, n_routes=2, shared=0.5, cells_per_route=40)
    data = generate(rng, n_cells=60, n_events=4000, routes=routes, replay_frac=0.6, reverse_frac=0.0, participation=0.1)
    S = encode(data["times"], data["event_ms"])
    X = prepare(S)
    k = 3  # two routes plus the volley-versus-replay direction
    values, V, _, _ = decompose(S, n_components=k, center=True, scale=True)
    patterns, _, _ = unmix(X, values, V, rng)

    def table(P):
        return np.abs([[member_order_vs_place(P[:, j], 100.0, cen, False, 1.0, rng, n_null=5)[0] for j in range(k)] for cen in data["centres"]])

    pca, ica = table(V), table(patterns)
    # PCA: one component carries both routes at once (a union)
    assert np.argmax(pca[0]) == np.argmax(pca[1]) and pca.max(axis=1).min() > 0.85
    # ICA: each route has its own component
    assert np.argmax(ica[0]) != np.argmax(ica[1]) and ica.max(axis=1).min() > 0.9


def test_pipeline_recovers_two_routes_in_both_directions_from_a_minority_of_events():
    rng = np.random.default_rng(3)
    routes = make_routes(rng, n_cells=80, n_routes=2)
    data = generate(rng, n_cells=80, n_events=4000, routes=routes, replay_frac=0.3, reverse_frac=0.5, participation=0.15, jitter_ms=6.0)
    result = fit(data["times"], data["event_ms"], rng=rng)
    assert result["n_kept"] >= 2
    for cen in data["centres"]:
        rhos = [member_order_vs_place(result["patterns"][:, j], 100.0, cen, False, 1.0, rng, n_null=20)[0] for j in range(result["n_kept"])]
        rhos = [r for r in rhos if np.isfinite(r)]
        assert max(abs(r) for r in rhos) > 0.85


def test_timing_consistency_detects_a_real_pattern():
    rng = np.random.default_rng(4)
    data = generate(rng, n_cells=40, n_events=1500, replay_frac=0.5, reverse_frac=0.0, participation=0.2)
    S = encode(data["times"], data["event_ms"])
    result = fit(data["times"], data["event_ms"], rng=rng, ica=False)
    z, _, _ = timing_consistency(S, result["patterns"][:, 0], rng, n_shuffles=20)
    assert z > 5


def test_match_components_is_one_when_sets_are_equal():
    rng = np.random.default_rng(5)
    P = rng.normal(size=(10, 3)) + 1j * rng.normal(size=(10, 3))
    P /= np.linalg.norm(P, axis=0)
    assert np.isclose(match_components(P, P[:, ::-1] * np.exp(1j * 0.7)), 1.0)
    assert members(np.array([1.0, 0.1, 0.5]), frac=0.2).tolist() == [True, False, True]


def test_population_bursts_and_event_spike_times():
    rng = np.random.default_rng(6)
    background = np.sort(rng.uniform(0, 100, size=2000))
    bursts = np.concatenate([rng.normal(centre, 0.015, size=60) for centre in (20.0, 50.0, 80.0)])
    spikes = np.sort(np.concatenate([background, bursts]))
    events, peaks = population_bursts(spikes, [(0.0, 100.0)])
    assert len(events) == 3
    assert np.allclose(np.sort(peaks), [20.0, 50.0, 80.0], atol=0.03)
    by_cell = {0: spikes, 1: np.array([20.0, 20.004])}
    times, truncated = event_spike_times(by_cell, [0, 1], events[:, 0], events[:, 1], max_spikes=10)
    assert times.shape == (3, 2, 10) and truncated >= 1
    first_event = np.argmin(np.abs(peaks - 20.0))
    assert np.isfinite(times[first_event, 1]).sum() == 2
