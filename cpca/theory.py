"""Reference equations for complex PCA of spike phase.

Encoding, lag matrix, and scores follow the research plan. A full turn of
phase is 2T of time, because the angle is pi * t / T.
"""

import numpy as np


def encode_times(times, T, t0=0.0):
    """Sum one cell's spikes in one event into a single phasor.

    Silent cells return 0. Several spikes add. A full turn is 2T.
    """
    times = np.asarray(times, dtype=float).ravel()
    times = times[np.isfinite(times)]
    if times.size == 0:
        return np.complex128(0)
    return np.exp(1j * np.pi * (times - t0) / T).sum(dtype=np.complex128)


def encode_matrix(times, T, t0=0.0):
    """Encode a one-spike-per-cell matrix. NaN means the cell did not fire.

    times has shape (events, cells). t0 is a scalar or one reference per event.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 2:
        raise ValueError("times must have shape (events, cells)")
    t0 = np.asarray(t0, dtype=float)
    if t0.ndim == 0:
        dt = times - t0
    elif t0.shape == (times.shape[0],):
        dt = times - t0[:, None]
    else:
        raise ValueError("t0 must be a scalar or one value per event")
    with np.errstate(invalid="ignore"):
        encoded = np.exp(1j * np.pi * dt / T)
    encoded[~np.isfinite(times)] = 0
    return encoded


def lag_matrix(encoded):
    """Uncentered Hermitian lag matrix, shape (cells, cells)."""
    encoded = np.asarray(encoded)
    n_events = encoded.shape[0]
    if n_events == 0:
        raise ValueError("encoded must contain at least one event")
    return (encoded.T @ encoded.conj()) / n_events


def eigh_desc(lag):
    """Eigenpairs of a Hermitian matrix, largest eigenvalue first.

    Eigenvalues are real. Eigenvectors are columns.
    """
    hermitian = (lag + lag.conj().T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order].real, eigenvectors[:, order]


def anchor_phase(component):
    """Rotate a component so its largest entry is real and non-negative."""
    component = np.asarray(component)
    strongest = np.argmax(np.abs(component))
    return component * np.exp(-1j * np.angle(component[strongest]))


def relative_times(component, T):
    """Map anchored component angles to times in (-T, T]."""
    return (T / np.pi) * np.angle(component)


def scores(encoded, components):
    """Complex scores of each event on each component (columns)."""
    return encoded @ np.asarray(components).conj()


def timing_share(lag):
    """Fraction of lag-matrix energy in the imaginary part."""
    frobenius = np.linalg.norm(lag)
    if frobenius == 0:
        return 0.0
    return float(np.linalg.norm(np.asarray(lag).imag) / frobenius)


def aligned_phase_error(estimated, true, weights):
    """Circular error after removing one global phase, in radians."""
    weights = np.asarray(weights, dtype=float)
    delta = np.exp(1j * (np.asarray(estimated) - np.asarray(true)))
    offset = np.angle(np.sum(weights * delta))
    return np.angle(np.exp(1j * (np.asarray(estimated) - np.asarray(true) - offset)))


def permute_observed_times(times, rng):
    """Permute spike times among the cells that fired, independently per event."""
    permuted = np.array(times, dtype=float, copy=True)
    for event in permuted:
        observed = np.isfinite(event)
        values = event[observed].copy()
        rng.shuffle(values)
        event[observed] = values
    return permuted


def randomize_phases(encoded, rng):
    """Multiply every nonzero phasor by an independent unit phase.

    This preserves each |s_ik| (hence the diagonal of the lag matrix) and
    destroys timing. It is the encoded form of the time-shuffle null, including
    for cells that fired more than once.
    """
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=encoded.shape))
    randomized = np.array(encoded, copy=True)
    nonzero = encoded != 0
    randomized[nonzero] *= phases[nonzero]
    return randomized
