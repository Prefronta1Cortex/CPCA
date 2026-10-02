"""Library behavior for encoding, nulls, readout, and direction splits."""

import time

import numpy as np
from scipy.stats import spearmanr

from cpca import (
    coactivation_null,
    decompose,
    encode,
    lag_matrix,
    readout,
    direction_split_null,
    split_by_direction,
    split_half,
    timing_centered,
    timing_excess,
)
from cpca.stats import null_statistics, shuffle_times
from cpca.theory import encode_matrix, timing_share


def _quadratic(lag, component):
    return float(np.real(np.vdot(component, lag @ component)))


def test_one_spike_encoding_matches_theory():
    times = np.array([[0.0, 20.0, np.nan], [5.0, np.nan, 40.0]])
    T = 80.0
    assert np.allclose(encode(times, T), encode_matrix(times, T))


def test_sum_mean_and_first_on_a_burst():
    T = 100.0
    coincident = np.array([[[0.0, 0.0]]])
    assert np.isclose(encode(coincident, T, how="sum")[0, 0], 2)
    assert np.isclose(encode(coincident, T, how="mean")[0, 0], 1)
    assert np.isclose(encode(coincident, T, how="first")[0, 0], 1)
    opposite = np.array([[[0.0, T]]])
    assert np.isclose(encode(opposite, T, how="sum")[0, 0], 0)
    assert np.isclose(encode(opposite, T, how="first")[0, 0], 1)
    silent = np.array([[[np.nan, np.nan]]])
    assert encode(silent, T, how="first")[0, 0] == 0


def test_rescale_does_not_subtract_the_mean():
    times = np.zeros((20, 2))
    times[:, 1] = 0.0
    times[::2, 1] = np.nan
    encoded = encode(times, T=50.0, rescale=True)
    assert np.allclose(encoded[:, 0], 1)
    assert np.nanmax(np.real(encoded)) > 0
    assert np.count_nonzero(encoded[:, 1] == 0) == 10


def test_null_matches_time_shuffles():
    rng = np.random.default_rng(0)
    times = rng.uniform(0.0, 40.0, size=(100, 5))
    times[rng.random(times.shape) < 0.3] = np.nan
    encoded = encode(times, 100.0)
    expected = coactivation_null(encoded)
    accumulated = np.zeros((times.shape[1], times.shape[1]), dtype=complex)
    draws = 400
    for _ in range(draws):
        shuffled = encode(shuffle_times(times, rng), 100.0)
        accumulated += lag_matrix(shuffled)
    accumulated /= draws
    assert np.max(np.abs(accumulated - expected)) < 0.03
    assert np.max(np.abs(expected.imag)) < 1e-12


def test_timing_excess_is_the_centered_quadratic_form():
    rng = np.random.default_rng(1)
    times = rng.uniform(0.0, 30.0, size=(50, 6))
    times[rng.random(times.shape) < 0.25] = np.nan
    encoded = encode(times, 80.0)
    component = rng.normal(size=6) + 1j * rng.normal(size=6)
    excess = timing_excess(encoded, component)
    centered = timing_centered(lag_matrix(encoded), coactivation_null(encoded))
    assert np.allclose(excess.mean(), _quadratic(centered, component))
    assert np.allclose(centered.imag, lag_matrix(encoded).imag)


def test_synchronous_event_has_no_timing_excess():
    encoded = np.ones((1, 5), dtype=complex)
    component = np.exp(1j * np.linspace(0.0, 1.2, 5))
    assert np.allclose(timing_excess(encoded, component), 0)
    delta, gamma = null_statistics(encoded)
    assert np.isclose(delta[0], 1)
    assert np.isclose(gamma[0], 1)


def test_readout_of_a_short_sequence():
    times = np.array([0.0, 20.0, 40.0])
    T = 100.0
    component = np.exp(1j * np.pi * times / T) * np.array([3.0, 2.0, 1.0])
    component = np.append(component, 0.01)
    weights, recovered, delays, included, span = readout(component, T)
    assert included.tolist() == [True, True, True, False]
    assert weights[3] == 0
    assert np.allclose(recovered[:3], times)
    assert np.isnan(recovered[3])
    assert np.all(delays[:3] >= -1e-10)
    assert np.isclose(delays[2], 0)
    assert np.allclose(recovered[:3] + delays[:3], times.max())
    assert span < np.pi


def test_readout_unwraps_past_half_a_turn():
    T = 100.0
    truth = np.linspace(0.0, 150.0, 16)
    component = np.exp(1j * np.pi * truth / T)
    _, recovered, _, included, span = readout(component, T, min_weight=0.0)
    assert included.all()
    assert span > np.pi
    assert spearmanr(truth, recovered).statistic > 0.999
    assert np.allclose(np.diff(recovered), np.diff(truth))


def test_direction_split_recovers_both_orders():
    rng = np.random.default_rng(2)
    step = 15.0
    truth = np.arange(6) * step
    T = 200.0
    forward = _events(rng, truth, 500, reverse=False)
    backward = _events(rng, truth, 500, reverse=True)
    encoded = encode(np.vstack([forward, backward]), T)
    result = split_by_direction(encoded)
    assert set(result["groups"]) == {1.0, -1.0}
    forward_sign = result["signs"][0]
    for sign, expected in ((forward_sign, 1.0), (-forward_sign, -1.0)):
        _, recovered, _, included, _ = readout(result["groups"][sign]["component"], T)
        steps = np.diff(recovered[included])
        assert np.all(np.sign(steps) == expected)
        assert np.allclose(np.abs(steps), step, atol=1.5)


def test_direction_split_on_a_real_pair_beats_phase_noise():
    rng = np.random.default_rng(5)
    step = 15.0
    truth = np.arange(6) * step
    T = 200.0
    forward = _events(rng, truth, 400, reverse=False)
    backward = _events(rng, truth, 400, reverse=True)
    encoded = encode(np.vstack([forward, backward]), T)
    observed, nulls = direction_split_null(encoded, rng, n_draws=8)
    assert observed < -0.9
    assert observed < np.nanpercentile(nulls, 50)


def test_split_half_components_agree():
    rng = np.random.default_rng(3)
    encoded = encode(_events(rng, np.arange(12) * 8.0, 800), T=200.0)
    overlap, energy = split_half(encoded, rng)
    assert overlap > 0.95
    assert energy > 0


def test_large_run_recovers_order_and_respects_the_nulls():
    rng = np.random.default_rng(4)
    n_cells, n_events = 100, 5000
    truth = np.arange(n_cells) * 5.0
    T = 800.0
    times = _events(rng, truth, n_events, drop=0.7, jitter=2.0)
    started = time.perf_counter()
    encoded = encode(times, T)
    eigenvalues, components, _, lag = decompose(encoded, n_components=1)
    weights, recovered, _, included, _ = readout(components[:, 0], T)
    null = coactivation_null(encoded)
    centered = timing_centered(lag, null)
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0
    assert spearmanr(truth[included], recovered[included]).statistic > 0.95
    assert eigenvalues[0] > 0
    assert np.count_nonzero(weights) == included.sum()

    shifted = encode(times + rng.uniform(0.0, 40.0, size=(n_events, 1)), T)
    assert np.max(np.abs(lag_matrix(shifted) - lag)) < 1e-8

    shuffled = encode(shuffle_times(times, rng), T)
    shuffled_lag = lag_matrix(shuffled)
    shuffled_centered = timing_centered(shuffled_lag, coactivation_null(shuffled))
    assert timing_share(shuffled_lag) < 0.5 * timing_share(lag)
    shuffled_top = decompose(shuffled, n_components=1)[0][0]
    centered_top = float(np.linalg.eigvalsh(centered)[-1])
    shuffled_centered_top = float(np.linalg.eigvalsh(shuffled_centered)[-1])
    assert shuffled_centered_top < centered_top
    assert shuffled_top < eigenvalues[0]


def _events(rng, truth, n_events, drop=0.2, jitter=2.0, reverse=False):
    base = truth.max() - truth if reverse else truth
    observed = np.tile(base, (n_events, 1))
    observed = observed + rng.normal(0.0, jitter, size=observed.shape)
    observed[rng.random(observed.shape) < drop] = np.nan
    return observed
