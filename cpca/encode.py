"""Spike times to a complex event-by-cell matrix.

Silence stays 0. Column means are not removed.
"""

import numpy as np

_HOW = ("mean", "sum", "first")


def encode(times, T, how="mean", t0=0.0, rescale=False):
    """Encode spike times as phasors at angular frequency pi/T.

    times has shape (events, cells) or (events, cells, spikes). Non-finite
    entries are missing spikes. how is "mean" (default), "sum", or "first".
    t0 is a scalar or one reference time per event. rescale divides each cell
    by the root-mean-square of its arrow lengths.
    """
    if how not in _HOW:
        raise ValueError(f"how must be one of {_HOW}")
    times = np.asarray(times, dtype=float)
    if times.ndim not in (2, 3):
        raise ValueError("times must have shape (events, cells) or (events, cells, spikes)")
    phasors = _phasors(times, T, t0)
    finite = np.isfinite(times)
    if how == "first":
        encoded = _first(times, phasors, finite)
    elif times.ndim == 2:
        encoded = phasors
    else:
        encoded = phasors.sum(axis=-1)
        if how == "mean":
            count = finite.sum(axis=-1)
            np.divide(encoded, count, out=encoded, where=count > 0)
    if rescale:
        encoded = rescale_cells(encoded)
    return encoded


def rescale_cells(encoded):
    """Divide each cell by sqrt(mean |s|^2). Zero columns stay zero."""
    scale = np.sqrt(np.mean(np.abs(encoded) ** 2, axis=0))
    out = np.array(encoded, copy=True)
    np.divide(out, scale, out=out, where=scale > 0)
    return out


def _phasors(times, T, t0):
    t0 = np.asarray(t0, dtype=float)
    if t0.ndim == 0:
        dt = times - t0
    elif t0.shape == (times.shape[0],):
        dt = times - t0.reshape((times.shape[0],) + (1,) * (times.ndim - 1))
    else:
        raise ValueError("t0 must be a scalar or one value per event")
    with np.errstate(invalid="ignore"):
        phasors = np.exp(1j * np.pi * dt / T)
    phasors[~np.isfinite(times)] = 0
    return phasors


def _first(times, phasors, finite):
    if times.ndim == 2:
        return phasors
    filled = np.where(finite, times, np.inf)
    position = np.argmin(filled, axis=-1)
    picked = np.take_along_axis(phasors, position[..., None], axis=-1)[..., 0]
    picked[~finite.any(axis=-1)] = 0
    return picked
