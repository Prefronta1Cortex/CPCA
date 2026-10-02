"""Mathematical claims for complex PCA of spike phase.

Tolerances are for a fixed random seed. They encode the claim, not a fit to
one unpublished script: the quoted eigenvalues in the research plan are not
reproduced here because that script's draw order was not recorded.
"""

import numpy as np

from cpca.theory import (
    aligned_phase_error,
    anchor_phase,
    eigh_desc,
    encode_matrix,
    encode_times,
    lag_matrix,
    permute_observed_times,
    randomize_phases,
    relative_times,
    scores,
    timing_share,
)


def _sequence(n_events, times, drop_prob, jitter_std, rng, reverse=False, shift=0.0):
    times = np.asarray(times, dtype=float)
    observed = np.tile(times.max() - times if reverse else times, (n_events, 1))
    if jitter_std:
        observed = observed + rng.normal(0.0, jitter_std, size=observed.shape)
    if drop_prob:
        observed = observed.astype(float)
        observed[rng.random(observed.shape) < drop_prob] = np.nan
    if shift:
        observed = observed + rng.uniform(0.0, shift, size=(n_events, 1))
    return observed


def _partial_sequence(n_events, n_cells, step, window, fire_prob, jitter_std, rng):
    """Contiguous windows on a line. The two endpoint cells are never covered."""
    times = np.arange(n_cells, dtype=float) * step
    span = times[-1]
    starts = (np.arange(n_events) + 0.5) / n_events * (span - window)
    covered = (times[None, :] >= starts[:, None]) & (
        times[None, :] <= starts[:, None] + window
    )
    fire = covered & (rng.random((n_events, n_cells)) < fire_prob)
    observed = np.full((n_events, n_cells), np.nan)
    observed[fire] = np.broadcast_to(times, (n_events, n_cells))[fire]
    if jitter_std:
        observed[fire] = observed[fire] + rng.normal(0.0, jitter_std, size=int(fire.sum()))
    return observed, times


def test_lag_angle_equals_time_difference():
    T = 100.0
    early, late = 10.0, 35.0
    lag = encode_times([late], T) * np.conj(encode_times([early], T))
    assert np.isclose(np.angle(lag), np.pi * (late - early) / T)


def test_burst_sum_concentrates_and_can_cancel():
    T = 100.0
    coincident = encode_times([20.0, 20.0], T)
    assert np.isclose(coincident, 2 * encode_times([20.0], T))
    spread = encode_times([0.0, 50.0], T)
    assert np.isclose(spread, 1 + 1j)
    assert np.isclose(np.abs(spread), np.sqrt(2))
    assert np.isclose(encode_times([0.0, T], T), 0)


def test_full_turn_is_two_windows():
    T = 100.0
    assert np.isclose(encode_times([0.0], T), encode_times([2 * T], T))
    assert np.isclose(encode_times([T], T), -1)
    assert not np.isclose(encode_times([0.0], T), encode_times([T], T))


def test_property_a_event_shift_leaves_lag_matrix_unchanged():
    rng = np.random.default_rng(0)
    times = _sequence(300, [0.0, 20.0, 40.0, 70.0], 0.3, 3.0, rng, shift=50.0)
    encoded = encode_matrix(times, T=100.0)
    phases = rng.uniform(0, 2 * np.pi, size=encoded.shape[0])
    shifted = encoded * np.exp(1j * phases)[:, None]
    assert np.max(np.abs(lag_matrix(encoded) - lag_matrix(shifted))) < 1e-12


def test_shifting_spikes_and_reference_together_leaves_encoding_unchanged():
    times = np.array([[0.0, 20.0, np.nan], [5.0, np.nan, 40.0]])
    shift = np.array([3.0, -7.0])
    T = 80.0
    original = encode_matrix(times, T, t0=0.0)
    moved = encode_matrix(times + shift[:, None], T, t0=shift)
    assert np.allclose(original, moved)


def test_column_centering_breaks_shift_invariance():
    rng = np.random.default_rng(1)
    times = _sequence(200, [0.0, 15.0, 30.0, 60.0], 0.2, 0.0, rng)
    encoded = encode_matrix(times, T=100.0)
    phases = rng.uniform(0, 2 * np.pi, size=encoded.shape[0])
    shifted = encoded * np.exp(1j * phases)[:, None]

    def _centered(matrix):
        return matrix - matrix.mean(axis=0)

    raw_change = np.max(np.abs(lag_matrix(encoded) - lag_matrix(shifted)))
    centered_change = np.max(
        np.abs(lag_matrix(_centered(encoded)) - lag_matrix(_centered(shifted)))
    )
    assert raw_change < 1e-12
    assert centered_change > 1e-2


def test_exact_sequence_is_rank_one():
    times = np.array([0.0, 20.0, 40.0])
    T = 100.0
    pattern = np.exp(1j * np.pi * times / T)
    pattern = pattern / np.linalg.norm(pattern)
    rng = np.random.default_rng(2)
    phases = rng.uniform(0, 2 * np.pi, size=(150, 1))
    encoded = np.exp(1j * phases) * pattern
    eigenvalues, eigenvectors = eigh_desc(lag_matrix(encoded))
    assert eigenvalues[0] == np.max(eigenvalues)
    assert eigenvalues[1] < 1e-10
    # Equal |p_k| makes the anchor cell arbitrary, so compare phases only.
    error = aligned_phase_error(
        np.angle(eigenvectors[:, 0]),
        np.angle(pattern),
        np.ones(len(pattern)),
    )
    assert np.max(np.abs(error)) < 1e-8


def test_property_b_order_survives_jitter_and_dropout():
    T = 100.0
    truth = np.array([0.0, 20.0, 40.0])
    rng = np.random.default_rng(0)
    times = _sequence(400, truth, drop_prob=0.3, jitter_std=3.0, rng=rng, shift=25.0)
    eigenvalues, eigenvectors = eigh_desc(lag_matrix(encode_matrix(times, T)))
    component = anchor_phase(eigenvectors[:, 0])
    recovered = relative_times(component, T)
    anchor = int(np.argmax(np.abs(component)))
    assert eigenvalues[0] > 5 * eigenvalues[1]
    assert np.allclose(recovered, truth - truth[anchor], atol=1.5)
    assert np.max(np.abs(component)) - np.min(np.abs(component)) < 0.05


def test_property_c_large_window_reduces_to_spike_counts():
    rng = np.random.default_rng(3)
    n_events, n_cells = 40, 6
    layers = []
    counts = np.zeros((n_events, n_cells))
    for _ in range(4):
        fire = rng.random((n_events, n_cells)) < 0.4
        when = rng.uniform(0.0, 80.0, size=(n_events, n_cells))
        when[~fire] = np.nan
        layers.append(encode_matrix(when, T=1e9))
        counts += fire
    encoded = np.sum(layers, axis=0)
    assert np.max(np.abs(encoded - counts)) < 1e-6
    count_lag = (counts.T @ counts) / n_events
    assert np.max(np.abs(lag_matrix(encoded) - count_lag)) < 1e-6
    assert timing_share(lag_matrix(encoded)) < 1e-6


def test_timing_share_shrinks_as_window_grows_even_when_order_is_real():
    truth = np.array([0.0, 20.0, 40.0, 60.0])
    rng = np.random.default_rng(4)
    times = _sequence(500, truth, drop_prob=0.2, jitter_std=2.0, rng=rng)
    short = lag_matrix(encode_matrix(times, T=120.0))
    long = lag_matrix(encode_matrix(times, T=1e6))
    shuffled = lag_matrix(encode_matrix(permute_observed_times(times, rng), T=120.0))
    assert timing_share(short) > timing_share(shuffled)
    assert timing_share(long) < 0.1 * timing_share(short)


def test_property_d_partial_coverage_recovers_phase_on_the_circle():
    T = 150.0
    rng = np.random.default_rng(5)
    observed, truth = _partial_sequence(
        n_events=4000,
        n_cells=30,
        step=12.0,
        window=100.0,
        fire_prob=0.6,
        jitter_std=3.0,
        rng=rng,
    )
    fired = np.isfinite(observed).sum(axis=0)
    assert fired[0] == 0
    assert fired[-1] == 0
    assert np.count_nonzero(fired) == 28

    eigenvalues, eigenvectors = eigh_desc(lag_matrix(encode_matrix(observed, T)))
    component = eigenvectors[:, 0]
    # One long sequence seen in pieces is not rank-one: coverage still varies.
    assert eigenvalues[1] > 0.5 * eigenvalues[0]
    assert np.max(np.abs(component[[0, -1]])) < 1e-8

    live = fired > 0
    error = aligned_phase_error(
        np.angle(component[live]),
        np.pi * truth[live] / T,
        np.abs(component[live]),
    )
    assert np.max(np.abs(error)) < 0.02

    raw_order = np.argsort(np.angle(anchor_phase(component[live])))
    assert not np.array_equal(raw_order, np.arange(live.sum()))


def test_property_d_zero_noise_phases_match_to_numerical_precision():
    """Consistent lags are exactly rank-one in phase.

    C_kl = w_kl exp(j(theta_k - theta_l)) with real w factors as a diagonal
    phase matrix times a real matrix. Eigenvector phases therefore match the
    spike phases exactly; magnitudes absorb how often each cell was covered.
    """
    T = 150.0
    rng = np.random.default_rng(6)
    observed, truth = _partial_sequence(
        n_events=3000,
        n_cells=30,
        step=12.0,
        window=100.0,
        fire_prob=1.0,
        jitter_std=0.0,
        rng=rng,
    )
    live = np.isfinite(observed).any(axis=0)
    _, eigenvectors = eigh_desc(lag_matrix(encode_matrix(observed, T)))
    error = aligned_phase_error(
        np.angle(eigenvectors[live, 0]),
        np.pi * truth[live] / T,
        np.abs(eigenvectors[live, 0]),
    )
    assert np.max(np.abs(error)) < 1e-8


def test_adjacent_lags_reconstruct_unwrapped_order():
    T = 150.0
    rng = np.random.default_rng(7)
    observed, truth = _partial_sequence(
        n_events=2000,
        n_cells=30,
        step=12.0,
        window=100.0,
        fire_prob=1.0,
        jitter_std=0.0,
        rng=rng,
    )
    lag = lag_matrix(encode_matrix(observed, T))
    cells = np.flatnonzero(np.isfinite(observed).any(axis=0))
    recovered = np.zeros(len(cells))
    for index in range(1, len(cells)):
        later, earlier = cells[index], cells[index - 1]
        recovered[index] = recovered[index - 1] + (T / np.pi) * np.angle(lag[later, earlier])
    assert np.allclose(np.diff(recovered), np.diff(truth[cells]), atol=1e-8)


def test_forward_and_reverse_split_across_two_components():
    times = np.arange(6) * 15.0
    T = 200.0
    rng = np.random.default_rng(8)
    forward = _sequence(600, times, 0.2, 2.0, rng, reverse=False, shift=30.0)
    reverse = _sequence(600, times, 0.2, 2.0, rng, reverse=True, shift=30.0)
    forward_only = encode_matrix(forward, T)
    mixed = encode_matrix(np.vstack([forward, reverse]), T)
    labels = np.array([1] * 600 + [-1] * 600)

    forward_lag = lag_matrix(forward_only)
    mixed_lag = lag_matrix(mixed)
    forward_eigenvalues, _ = eigh_desc(forward_lag)
    mixed_eigenvalues, mixed_vectors = eigh_desc(mixed_lag)
    assert forward_eigenvalues[0] > 5 * forward_eigenvalues[1]
    assert timing_share(mixed_lag) < timing_share(forward_lag) / 5
    assert mixed_eigenvalues[1] > 3 * mixed_eigenvalues[2]

    mixed_scores = scores(mixed, mixed_vectors[:, :2])
    direction = np.sign(np.imag(mixed_scores[:, 0] * np.conj(mixed_scores[:, 1])))
    accuracy = max(np.mean(direction == labels), np.mean(direction == -labels))
    assert accuracy > 0.99

    # Event 0 is forward, so its sign marks the forward group.
    for mask, expected_sign in (
        (direction == direction[0], 1.0),
        (direction != direction[0], -1.0),
    ):
        _, vectors = eigh_desc(lag_matrix(mixed[mask]))
        steps = np.diff(relative_times(anchor_phase(vectors[:, 0]), T))
        assert np.all(np.sign(steps) == expected_sign)
        assert np.allclose(np.abs(steps), 15.0, atol=1.5)


def test_time_shuffle_removes_timing_and_keeps_participation():
    truth = np.arange(8) * 12.0
    T = 200.0
    rng = np.random.default_rng(9)
    times = _sequence(800, truth, drop_prob=0.4, jitter_std=2.0, rng=rng)
    original = lag_matrix(encode_matrix(times, T))
    shuffled = lag_matrix(encode_matrix(permute_observed_times(times, rng), T))
    original_eigenvalues, _ = eigh_desc(original)
    shuffled_eigenvalues, _ = eigh_desc(shuffled)
    assert np.allclose(np.diag(original), np.diag(shuffled))
    assert timing_share(shuffled) < 0.25 * timing_share(original)
    assert shuffled_eigenvalues[0] < original_eigenvalues[0]


def test_phase_randomization_is_the_multi_spike_null():
    """Random phases keep each |s_ik| and destroy a real burst sequence.

    Permuting raw spike times is not well defined once a cell fires twice.
    Randomizing the encoded phasor is the null that preserves participation.
    """
    rng = np.random.default_rng(10)
    n_events, n_cells = 500, 5
    order = np.arange(n_cells) * 15.0
    base = np.tile(order, (n_events, 1)) + rng.uniform(0.0, 50.0, size=(n_events, 1))
    first = base + rng.normal(0.0, 0.4, size=base.shape)
    second = base + rng.normal(0.0, 0.4, size=base.shape)
    silent = rng.random(base.shape) < 0.3
    first[silent] = np.nan
    second[silent] = np.nan
    encoded = encode_matrix(first, T=120.0) + encode_matrix(second, T=120.0)
    randomized_encoded = randomize_phases(encoded, rng)
    original = lag_matrix(encoded)
    randomized = lag_matrix(randomized_encoded)
    assert np.allclose(np.abs(randomized_encoded), np.abs(encoded))
    assert np.allclose(np.diag(original), np.diag(randomized))
    assert timing_share(original) > 0.2
    assert timing_share(randomized) < 0.25 * timing_share(original)


def test_detector_weights_synchronize_the_template():
    times = np.array([0.0, 12.0, 27.0, 45.0])
    T = 100.0
    component = anchor_phase(np.exp(1j * np.pi * times / T))
    event = np.exp(1j * np.pi * relative_times(component, T) / T)
    score = np.sum(event * np.conj(component))
    assert score.real > 0
    assert abs(score.imag) < 1e-10


def test_empty_events_scale_eigenvalues_without_rotating_components():
    times = np.array([[0.0, 10.0, 25.0], [0.0, 10.0, 25.0], [np.nan, np.nan, np.nan]])
    T = 80.0
    encoded = encode_matrix(times[:2], T)
    with_empty = encode_matrix(times, T)
    eigenvalues, vectors = eigh_desc(lag_matrix(encoded))
    eigenvalues_empty, vectors_empty = eigh_desc(lag_matrix(with_empty))
    assert np.allclose(eigenvalues_empty, eigenvalues * 2 / 3)
    error = aligned_phase_error(
        np.angle(vectors_empty[:, 0]),
        np.angle(vectors[:, 0]),
        np.ones(vectors.shape[0]),
    )
    assert np.max(np.abs(error)) < 1e-8


def test_lag_matrix_is_hermitian_with_nonnegative_eigenvalues():
    rng = np.random.default_rng(11)
    times = _sequence(100, [0.0, 8.0, 19.0, 33.0], 0.5, 1.0, rng, shift=10.0)
    lag = lag_matrix(encode_matrix(times, T=50.0))
    assert np.allclose(lag, lag.conj().T)
    eigenvalues, _ = eigh_desc(lag)
    assert np.all(eigenvalues >= -1e-10)
