"""Population-burst events from pooled spike times.

The summed firing of all cells is binned, smoothed, and compared with its
mean and standard deviation inside the allowed intervals (e.g. NREM sleep).
An event starts and ends where the smoothed rate crosses the mean, and is
kept only if its peak exceeds mean + peak_sd standard deviations. Each event
is described by its start, end and peak time.
"""

import numpy as np


def in_intervals(times, intervals):
    keep = np.zeros(np.shape(times), dtype=bool)
    for start, end in intervals:
        keep |= (times >= start) & (times < end)
    return keep


def population_bursts(spike_times, intervals, bin_s=0.01, smooth_bins=1.0, peak_sd=3.0,
                      merge_gap_s=0.03, min_duration_s=0.05, max_duration_s=0.5):
    """Return (events, peaks): events is (n, 2) start/end in seconds, peaks is (n,) peak time."""
    spike_times = np.asarray(spike_times, dtype=float)
    if spike_times.size == 0:
        return np.zeros((0, 2)), np.zeros(0)
    start = max(0.0, spike_times.min() - 1)
    edges = np.arange(start, spike_times.max() + 1, bin_s)
    counts, _ = np.histogram(spike_times, bins=edges)
    centers = edges[:-1] + bin_s / 2
    half = int(np.ceil(3 * smooth_bins))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / smooth_bins) ** 2)
    smooth = np.convolve(counts, kernel / kernel.sum(), mode="same")
    allowed = in_intervals(centers, intervals)
    baseline = smooth[allowed]
    if baseline.size == 0:
        return np.zeros((0, 2)), np.zeros(0)
    edge_line = baseline.mean()
    peak_line = baseline.mean() + peak_sd * baseline.std()
    above = (smooth >= edge_line) & allowed
    if not above.any():
        return np.zeros((0, 2)), np.zeros(0)
    changes = np.diff(above.astype(int))
    starts = np.flatnonzero(changes == 1) + 1
    ends = np.flatnonzero(changes == -1) + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends, len(above)]
    events, peaks = [], []
    for begin, finish in zip(starts, ends):
        if smooth[begin:finish].max() < peak_line:
            continue
        t0 = centers[begin] - bin_s / 2
        t1 = centers[finish - 1] + bin_s / 2
        peak = centers[begin + int(np.argmax(smooth[begin:finish]))]
        if events and t0 - events[-1][1] <= merge_gap_s:
            events[-1][1] = t1
            if smooth[begin:finish].max() > smooth[int((peaks[-1] - start) / bin_s)]:
                peaks[-1] = peak
        else:
            events.append([t0, t1])
            peaks.append(peak)
    events = np.array(events)
    peaks = np.array(peaks)
    duration = np.diff(events, axis=1).ravel()
    keep = (duration >= min_duration_s) & (duration <= max_duration_s)
    return events[keep], peaks[keep]


def event_spike_times(by_cell, cell_ids, starts, ends, max_spikes=24):
    """Spike times (ms, relative to each window start) as an array (events, cells, max_spikes).

    by_cell maps cell id to a sorted array of spike times in seconds. Missing
    entries are NaN. Returns the array and the number of truncated (cell, event) pairs.
    """
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    n_events, n_cells = len(starts), len(cell_ids)
    times = np.full((n_events, n_cells, max_spikes), np.nan)
    truncated = 0
    for k, cell in enumerate(cell_ids):
        spikes = np.asarray(by_cell[cell], dtype=float)
        if spikes.size == 0:
            continue
        left = np.searchsorted(spikes, starts, side="left")
        right = np.searchsorted(spikes, ends, side="left")
        for i in range(n_events):
            count = int(right[i] - left[i])
            if count <= 0:
                continue
            take = min(count, max_spikes)
            truncated += int(count > max_spikes)
            times[i, k, :take] = (spikes[left[i]:left[i] + take] - starts[i]) * 1000
    return times, truncated
