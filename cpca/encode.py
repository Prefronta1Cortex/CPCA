"""Spike times to a complex event-by-cell matrix.

Each spike at time t becomes the unit arrow exp(j * pi * (t - t0) / T). A cell's
spikes in one event are combined into one arrow (mean by default). Silence is
the number 0.

T sets the time-to-angle scale. The half-circle rule from Haessig et al.
(2020) is that T equals the largest spread a pattern can have, so that a full
pattern spans the half-circle [0, pi]. Two different orders are then close to
orthogonal and PCA can give each its own component. If T is much longer than
the pattern, every pattern sits in a small arc near the same angle and PCA
merges them. T may therefore be one number for all events, or one number per
event (its own duration), see `event_scale`.
"""

import numpy as np

_HOW = ("mean", "sum", "first")


def encode(times, T, how="mean", t0=0.0, rescale=False, arc=np.pi):
    """Encode spike times (ms) as arrows at angular frequency arc/T (default pi: the half-circle).

    times has shape (events, cells) or (events, cells, spikes); non-finite
    entries are missing spikes. T is a scalar or one value per event. t0 is a
    scalar or one reference time per event. how is "mean" (default; arrow
    length between 0 and 1), "sum" (length counts spikes) or "first" (the
    paper's single first spike). rescale divides each cell by the
    root-mean-square of its arrow lengths across events.
    """
    if how not in _HOW:
        raise ValueError(f"how must be one of {_HOW}")
    times = np.asarray(times, dtype=float)
    if times.ndim not in (2, 3):
        raise ValueError("times must have shape (events, cells) or (events, cells, spikes)")
    phasors = _phasors(times, T, t0, arc)
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


def event_scale(times):
    """Per-event T: the spread of each event's spikes (max minus min, in ms).

    With t0 at each event's first spike, this maps each event's own extent
    onto the half-circle. Events with fewer than two spikes get T = 1 (their
    single arrow has angle 0 whatever T is).
    """
    times = np.asarray(times, dtype=float)
    flat = times.reshape(times.shape[0], -1)
    lo = np.nanmin(np.where(np.isfinite(flat), flat, np.inf), axis=1)
    hi = np.nanmax(np.where(np.isfinite(flat), flat, -np.inf), axis=1)
    span = hi - lo
    span[~np.isfinite(span) | (span <= 0)] = 1.0
    return span, lo


def rescale_cells(encoded):
    """Divide each cell by sqrt(mean |s|^2) over events. Zero columns stay zero."""
    scale = np.sqrt(np.mean(np.abs(encoded) ** 2, axis=0))
    out = np.array(encoded, copy=True)
    np.divide(out, scale, out=out, where=scale > 0)
    return out


def _per_event(value, times, name):
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        return value
    if value.shape == (times.shape[0],):
        return value.reshape((times.shape[0],) + (1,) * (times.ndim - 1))
    raise ValueError(f"{name} must be a scalar or one value per event")


def _phasors(times, T, t0, arc=np.pi):
    T = _per_event(T, times, "T")
    t0 = _per_event(t0, times, "t0")
    with np.errstate(invalid="ignore", divide="ignore"):
        phasors = np.exp(1j * arc * (times - t0) / T)
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
