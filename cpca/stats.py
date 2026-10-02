"""Co-activation null, timing excess, and split-half checks.

C0 is the lag matrix expected when each event's arrows are reassigned at
random among the cells that fired. It is real. C - C0 keeps every lead and lag.
"""

import numpy as np

from cpca.decompose import decompose
from cpca.theory import lag_matrix, permute_observed_times, randomize_phases


def null_statistics(encoded, active=None):
    """Per-event power delta and pair agreement gamma of the co-activation null.

    active is a boolean mask of cells that fired. Missing entries of encoded
    (exact zeros) are treated as silent when active is omitted.
    """
    encoded = np.asarray(encoded)
    if active is None:
        active = encoded != 0
    active = np.asarray(active, dtype=bool)
    power = np.abs(encoded) ** 2
    count = active.sum(axis=1)
    sum_power = (power * active).sum(axis=1)
    sum_arrows = (encoded * active).sum(axis=1)
    delta = np.zeros(encoded.shape[0], dtype=float)
    gamma = np.zeros(encoded.shape[0], dtype=float)
    fired = count > 0
    delta[fired] = sum_power[fired] / count[fired]
    paired = count > 1
    gamma[paired] = (
        np.abs(sum_arrows[paired]) ** 2 - sum_power[paired]
    ) / (count[paired] * (count[paired] - 1))
    return delta, gamma


def coactivation_null(encoded, active=None):
    """Closed-form shuffle expectation C0, real and symmetric."""
    encoded = np.asarray(encoded)
    if active is None:
        active = encoded != 0
    active_f = np.asarray(active, dtype=float)
    delta, gamma = null_statistics(encoded, active)
    outer = active_f.T @ (active_f * gamma[:, None])
    cells = np.arange(encoded.shape[1])
    outer[cells, cells] = active_f.T @ delta
    return outer / encoded.shape[0]


def timing_centered(lag, null):
    """C - C0. The imaginary part equals that of lag."""
    return np.asarray(lag) - np.asarray(null)


def timing_excess(encoded, component, active=None):
    """Per-event |a|^2 minus the within-event shuffle expectation.

    The mean of this vector equals component^H (C - C0) component.
    """
    encoded = np.asarray(encoded)
    component = np.asarray(component)
    if active is None:
        active = encoded != 0
    active = np.asarray(active, dtype=bool)
    delta, gamma = null_statistics(encoded, active)
    sum_template = (active * component).sum(axis=1)
    sum_weight = (active * (np.abs(component) ** 2)).sum(axis=1)
    expected = gamma * (np.abs(sum_template) ** 2 - sum_weight) + delta * sum_weight
    observed = np.abs(encoded @ np.conj(component)) ** 2
    return observed - expected


def shuffle_times(times, rng):
    """Permute spike times among cells that fired. One spike per cell."""
    return permute_observed_times(times, rng)


def shuffle_phases(encoded, rng):
    """Randomize arrow angles and keep lengths. Use this when cells burst."""
    return randomize_phases(encoded, rng)


def split_half(encoded, rng):
    """Fit the top component on each half of the events.

    Returns the absolute overlap of the two components, and the held-out
    energy p^H C_test p of the first half's component.
    """
    encoded = np.asarray(encoded)
    order = rng.permutation(encoded.shape[0])
    mid = encoded.shape[0] // 2
    first, second = encoded[order[:mid]], encoded[order[mid:]]
    _, components, _, _ = decompose(first, n_components=1)
    _, other, _, _ = decompose(second, n_components=1)
    component = components[:, 0]
    overlap = np.abs(np.vdot(component, other[:, 0]))
    held_out = lag_matrix(second)
    energy = float(np.real(np.vdot(component, held_out @ component)))
    return overlap, energy
