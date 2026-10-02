"""Read a component as weights, relative times, and delays.

Short arcs are anchored on the strongest cell. Arcs longer than half a turn
are unwrapped by cutting the widest empty gap on the circle.
"""

import numpy as np

from cpca.theory import anchor_phase, relative_times


def readout(component, T, min_weight=0.1):
    """Weights, times in ms, and non-negative delays for one component.

    Cells below min_weight times the largest weight are dropped: their weight
    is set to 0 and their time and delay are NaN. Delays are D - time, with D
    the latest included time, so every included delay is >= 0.
    """
    component = np.asarray(component)
    raw_weight = np.abs(component)
    included = np.zeros(component.shape, dtype=bool)
    if raw_weight.max() > 0:
        included = raw_weight >= min_weight * raw_weight.max()
    times = np.full(component.shape, np.nan)
    n_included = int(included.sum())
    if n_included == 1:
        times[included] = 0.0
        span = 0.0
    elif n_included >= 2:
        angles = np.angle(component[included])
        span = _circular_span(angles)
        if span > np.pi:
            times[included] = _unwrap_largest_gap(angles, T)
        else:
            anchored = relative_times(anchor_phase(component), T)
            times[included] = anchored[included]
            # Anchor uses every cell. Re-express included times from the
            # strongest included cell so a dropped cell cannot set the zero.
            strongest = np.flatnonzero(included)[np.argmax(raw_weight[included])]
            times[included] = times[included] - times[strongest]
    else:
        span = 0.0
    weights = np.where(included, raw_weight, 0.0)
    delays = np.full(component.shape, np.nan)
    if n_included:
        delays[included] = np.nanmax(times) - times[included]
    return weights, times, delays, included, float(span)


def _circular_span(angles):
    gaps = _gaps(np.sort(angles))
    return float(2 * np.pi - gaps.max())


def _gaps(angles_sorted):
    return np.diff(angles_sorted, append=angles_sorted[0] + 2 * np.pi)


def _unwrap_largest_gap(angles, T):
    order = np.argsort(angles)
    ordered = angles[order]
    gaps = _gaps(ordered)
    cut = int(np.argmax(gaps))
    start = (cut + 1) % len(ordered)
    sequence = np.concatenate(
        [np.arange(start, len(ordered)), np.arange(0, start)]
    )
    unwrapped = ordered[sequence].astype(float)
    if start != 0:
        unwrapped[len(ordered) - start :] += 2 * np.pi
    unwrapped -= unwrapped[0]
    times = np.empty(len(angles))
    times[order[sequence]] = (T / np.pi) * unwrapped
    return times
