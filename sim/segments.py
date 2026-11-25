"""Planted reactivation events with constant-speed partial replays (docs/segments.md).

Differences from sim/routes.py: a replay plays a segment of the route at a constant speed, so a longer segment lasts
longer; the event window follows the activity (the activity span plus a margin on each side) instead of a fixed window
over which every segment is stretched; non-replay events draw the same durations, so duration does not mark replays.
Realism switches, all on by default: per-cell background participation (uniform 0.05-0.30), a fixed per-cell latency
offset within the burst, random assemblies (sim/routes.py's rule), 1-3 spikes per active cell.

The answer key is each route cell's position (NaN off the route); the planted direction of each replay is +1 (forward)
or -1 (backward), 0 for other events.
"""

import numpy as np

ROUTE_MS = 250.0          # time to play the whole route at the nominal speed
SPEED_RANGE = (0.8, 1.25)  # per-event speed factor
SEGMENT = (0.3, 0.5)       # segment length as a fraction of the route
MARGIN = 0.1               # window margin on each side, as a fraction of the window


def generate(rng, n_cells=120, n_route=37, n_events=2500, replay_frac=0.05, reverse_frac=0.5, seq_participation=0.6,
             participation=(0.05, 0.30), jitter_ms=8.0, latency_frac=0.08, assemblies=6, assembly_membership=None,
             segment=SEGMENT, route_ms=ROUTE_MS, speed_range=SPEED_RANGE, max_spikes=3):
    """Spike times (events, cells, max_spikes) in ms from each window's start (NaN = no spike), window durations T (ms),
    the answer key (position per cell, NaN off route), the label (1 replay, 0 other) and direction (+1, -1, 0)."""
    pos = np.full(n_cells, np.nan)
    route = rng.permutation(n_cells)[:n_route]
    pos[route] = rng.uniform(0, 1, n_route)
    p_bg = rng.uniform(*participation, size=n_cells) if np.ndim(participation) else np.full(n_cells, float(participation))
    latency = rng.standard_normal(n_cells) * latency_frac if latency_frac > 0 else np.zeros(n_cells)
    if assembly_membership is not None:
        membership = np.asarray(assembly_membership, int)
        assemblies = int(membership.max()) + 1
    else:
        membership = rng.integers(0, assemblies, size=n_cells) if assemblies else None
    times = np.full((n_events, n_cells, max_spikes), np.nan)
    label = np.zeros(n_events, int)
    direction = np.zeros(n_events, int)
    n_ordered = np.zeros(n_events, int)
    seg_start = np.full(n_events, np.nan)
    seg_length = np.zeros(n_events)
    T = np.zeros(n_events)
    for i in range(n_events):
        L = rng.uniform(*segment)
        speed = route_ms * rng.uniform(*speed_range)              # ms per unit of route
        span = L * speed
        D = span / (1 - 2 * MARGIN)
        T[i] = D
        seg_length[i] = L
        prob = p_bg.copy()
        if membership is not None:
            g = rng.integers(assemblies)
            prob = np.where(membership == g, np.minimum(3 * p_bg, 0.9), p_bg / 2)
        fire = rng.random(n_cells) < prob
        n_sp = 1 + rng.poisson(0.7, n_cells).clip(0, max_spikes - 1)
        if rng.random() < replay_frac:
            label[i] = 1
            backward = rng.random() < reverse_frac
            direction[i] = -1 if backward else 1
            start = rng.uniform(0, 1 - L)
            seg_start[i] = start
            frac = (pos - start) / L
            in_seg = np.isfinite(frac) & (frac >= 0) & (frac <= 1) & (rng.random(n_cells) < seq_participation)
            frac = 1 - frac if backward else frac
            t_field = MARGIN * D + frac * span
            for c in np.flatnonzero(in_seg):
                times[i, c, :n_sp[c]] = t_field[c] + rng.normal(0, jitter_ms, n_sp[c])
            fire[in_seg] = False
            n_ordered[i] = int(in_seg.sum())
        for c in np.flatnonzero(fire):
            times[i, c, :n_sp[c]] = rng.normal(D / 2 + latency[c] * D, span / 4, n_sp[c])
    times = np.clip(times, 0, T[:, None, None] - 1e-6)
    return {"times": times, "T": T, "centres": pos, "label": label, "direction": direction, "n_ordered": n_ordered,
            "seg_start": seg_start, "seg_length": seg_length, "route": route, "n_cells": n_cells, "membership": membership}


def spatial_assemblies(rng, n_cells, route_cells, positions, n_groups=6):
    """Assembly membership that follows the map: route cells grouped by contiguous stretches, other cells at random."""
    membership = rng.integers(0, n_groups, size=n_cells)
    membership[route_cells] = np.minimum((positions[route_cells] * n_groups).astype(int), n_groups - 1)
    return membership


def n_route_for(cells_per_replay, seq_participation=0.6, segment=SEGMENT):
    """Route size that gives about the requested number of ordered route cells per replay."""
    return int(round(cells_per_replay / (seq_participation * np.mean(segment))))
