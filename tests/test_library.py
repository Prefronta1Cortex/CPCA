"""Library behaviour: encoding variants, per-event scale, nulls, read-out, decomposition options."""

import numpy as np
from scipy.stats import spearmanr

from cpca import (
    coactivation_null,
    decompose,
    encode,
    event_scale,
    lag_matrix,
    n_significant,
    prepare,
    readout,
    timing_centered,
    timing_excess,
)
from cpca.stats import null_statistics, shuffle_times, split_half
from cpca.theory import encode_matrix, timing_share


def _quadratic(lag, component):
    return float(np.real(np.vdot(component, lag @ component)))


def _events(rng, truth, n_events, drop=0.2, jitter=2.0, reverse=False):
    base = truth.max() - truth if reverse else truth
    observed = np.tile(base, (n_events, 1)) + rng.normal(0.0, jitter, size=(n_events, len(truth)))
    observed[rng.random(observed.shape) < drop] = np.nan
    return observed


def test_one_spike_encoding_matches_theory():
    times = np.array([[0.0, 20.0, np.nan], [5.0, np.nan, 40.0]])
    assert np.allclose(encode(times, 80.0), encode_matrix(times, 80.0))


def test_sum_mean_and_first_on_a_burst():
    T = 100.0
    coincident = np.array([[[0.0, 0.0]]])
    assert np.isclose(encode(coincident, T, how="sum")[0, 0], 2)
    assert np.isclose(encode(coincident, T, how="mean")[0, 0], 1)
    assert np.isclose(encode(coincident, T, how="first")[0, 0], 1)
    opposite = np.array([[[0.0, T]]])
    assert np.isclose(encode(opposite, T, how="sum")[0, 0], 0)
    assert np.isclose(encode(opposite, T, how="first")[0, 0], 1)
    assert encode(np.array([[[np.nan, np.nan]]]), T, how="first")[0, 0] == 0


def test_per_event_scale_maps_each_event_onto_the_half_circle():
    times = np.array([[0.0, 10.0, 20.0], [0.0, 40.0, 80.0]])
    T, t0 = event_scale(times)
    assert np.allclose(T, [20.0, 80.0])
    encoded = encode(times, T, t0=t0)
    # last spike of each event sits at angle pi, whatever its absolute spread
    assert np.allclose(np.angle(encoded[:, 2]), np.pi)
    assert np.allclose(np.angle(encoded[:, 1]), np.pi / 2)


def test_per_event_T_broadcasts_like_scalar():
    rng = np.random.default_rng(0)
    times = rng.uniform(0, 50, size=(5, 4))
    same = encode(times, np.full(5, 100.0))
    assert np.allclose(same, encode(times, 100.0))


def test_rescale_does_not_subtract_the_mean():
    times = np.zeros((20, 2))
    times[::2, 1] = np.nan
    encoded = encode(times, T=50.0, rescale=True)
    assert np.allclose(encoded[:, 0], 1)
    assert np.count_nonzero(encoded[:, 1] == 0) == 10


def test_prepare_centres_and_scales_columns():
    rng = np.random.default_rng(1)
    S = rng.normal(size=(200, 5)) + 1j * rng.normal(size=(200, 5)) + 3.0
    X = prepare(S, center=True, scale=True)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(np.sqrt(np.mean(np.abs(X) ** 2, axis=0)), 1)
    assert np.allclose(prepare(S, center=False, scale=False), S)


def test_null_matches_time_shuffles():
    rng = np.random.default_rng(0)
    times = rng.uniform(0.0, 40.0, size=(100, 5))
    times[rng.random(times.shape) < 0.3] = np.nan
    encoded = encode(times, 100.0)
    expected = coactivation_null(encoded)
    accumulated = np.zeros((5, 5), dtype=complex)
    for _ in range(400):
        accumulated += lag_matrix(encode(shuffle_times(times, rng), 100.0))
    accumulated /= 400
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


def test_synchronous_event_has_no_timing_excess():
    encoded = np.ones((1, 5), dtype=complex)
    component = np.exp(1j * np.linspace(0.0, 1.2, 5))
    assert np.allclose(timing_excess(encoded, component), 0)
    delta, gamma = null_statistics(encoded)
    assert np.isclose(delta[0], 1) and np.isclose(gamma[0], 1)


def test_readout_of_a_short_sequence():
    times = np.array([0.0, 20.0, 40.0])
    T = 100.0
    component = np.append(np.exp(1j * np.pi * times / T) * np.array([3.0, 2.0, 1.0]), 0.01)
    weights, recovered, delays, included, span = readout(component, T)
    assert included.tolist() == [True, True, True, False]
    assert weights[3] == 0
    assert np.allclose(recovered[:3], times)
    assert np.isnan(recovered[3])
    assert np.all(delays[:3] >= -1e-10)
    assert np.allclose(recovered[:3] + delays[:3], times.max())
    assert span < np.pi


def test_readout_unwraps_past_half_a_turn():
    T = 100.0
    truth = np.linspace(0.0, 150.0, 16)
    component = np.exp(1j * np.pi * truth / T)
    _, recovered, _, included, span = readout(component, T, min_weight=0.0)
    assert included.all() and span > np.pi
    assert spearmanr(truth, recovered).statistic > 0.999
    assert np.allclose(np.diff(recovered), np.diff(truth))


def test_split_half_components_agree():
    rng = np.random.default_rng(3)
    encoded = encode(_events(rng, np.arange(12) * 8.0, 800), T=200.0)
    overlap, energy = split_half(encoded, rng)
    assert overlap > 0.95 and energy > 0


def test_marchenko_pastur_counts_only_structure():
    rng = np.random.default_rng(4)
    M, N = 2000, 30
    noise = rng.normal(size=(M, N)) + 1j * rng.normal(size=(M, N))
    values, _, _, _ = decompose(noise, center=True, scale=True)
    k_noise, _ = n_significant(values, M, N)
    assert k_noise <= 1
    signal = noise.copy()
    pattern = np.exp(1j * np.linspace(0, np.pi, N))
    signal[: M // 3] += 3.0 * pattern[None, :] * np.exp(1j * rng.uniform(0, 2 * np.pi, size=(M // 3, 1)))
    values, _, _, _ = decompose(signal, center=True, scale=True)
    k_signal, _ = n_significant(values, M, N)
    assert k_signal >= 1


def test_uncentred_decompose_is_invariant_to_event_clocks_but_centred_is_not():
    rng = np.random.default_rng(5)
    times = _events(rng, np.array([0.0, 15.0, 30.0, 60.0]), 300)
    S = encode(times, 100.0)
    spun = S * np.exp(1j * rng.uniform(0, 2 * np.pi, size=(S.shape[0], 1)))
    assert np.max(np.abs(decompose(S)[3] - decompose(spun)[3])) < 1e-10
    assert np.max(np.abs(decompose(S, center=True)[3] - decompose(spun, center=True)[3])) > 1e-2
