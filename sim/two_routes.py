"""Planted reactivation events with several routes, for testing whether routes separate onto components.

Same event model as sim/segments.py (constant-speed partial replays, windows that follow the activity, per-cell
background participation and latency, random assemblies, 1-3 spikes per active cell), with several maps:
  shared=False  each map has its own cells (n_route each, disjoint);
  shared=True   all maps use the same n_route cells, each with its own independent positions (as two running
                directions or two environments on one population).
rates gives each map's replay share (their sum is the share of replay events). The answer key is one position
array per map (NaN off the map).
"""

import numpy as np

from sim.segments import MARGIN, ROUTE_MS, SPEED_RANGE


def generate(rng, n_cells=120, n_route=40, rates=(0.1, 0.1), shared=False, n_events=2500, reverse_frac=0.5,
             seq_participation=0.6, participation=(0.05, 0.30), jitter_ms=8.0, latency_frac=0.08, assemblies=6,
             assembly_membership=None, segment=(0.5, 1.0), route_ms=ROUTE_MS, speed_range=SPEED_RANGE, max_spikes=3):
    n_maps = len(rates)
    perm = rng.permutation(n_cells)
    maps = []
    for m in range(n_maps):
        cells = perm[:n_route] if shared else perm[m * n_route:(m + 1) * n_route]
        pos = np.full(n_cells, np.nan)
        pos[cells] = rng.uniform(0, 1, n_route)
        maps.append(pos)
    p_bg = rng.uniform(*participation, size=n_cells)
    latency = rng.standard_normal(n_cells) * latency_frac if latency_frac > 0 else np.zeros(n_cells)
    if assembly_membership is not None:
        membership = np.asarray(assembly_membership, int)
        assemblies = int(membership.max()) + 1
    else:
        membership = rng.integers(0, assemblies, size=n_cells) if assemblies else None
    cum = np.cumsum(rates)
    times = np.full((n_events, n_cells, max_spikes), np.nan)
    which = np.full(n_events, -1)
    direction = np.zeros(n_events, int)
    T = np.zeros(n_events)
    for i in range(n_events):
        L = rng.uniform(*segment)
        speed = route_ms * rng.uniform(*speed_range)
        span = L * speed
        D = span / (1 - 2 * MARGIN)
        T[i] = D
        prob = p_bg.copy()
        if membership is not None:
            g = rng.integers(assemblies)
            prob = np.where(membership == g, np.minimum(3 * p_bg, 0.9), p_bg / 2)
        fire = rng.random(n_cells) < prob
        n_sp = 1 + rng.poisson(0.7, n_cells).clip(0, max_spikes - 1)
        u = rng.random()
        m = int(np.searchsorted(cum, u, side="right")) if u < cum[-1] else -1
        if m >= 0:
            which[i] = m
            backward = rng.random() < reverse_frac
            direction[i] = -1 if backward else 1
            start = rng.uniform(0, 1 - L)
            frac = (maps[m] - start) / L
            in_seg = np.isfinite(frac) & (frac >= 0) & (frac <= 1) & (rng.random(n_cells) < seq_participation)
            frac = 1 - frac if backward else frac
            t_field = MARGIN * D + frac * span
            for c in np.flatnonzero(in_seg):
                times[i, c, :n_sp[c]] = t_field[c] + rng.normal(0, jitter_ms, n_sp[c])
            fire[in_seg] = False
        for c in np.flatnonzero(fire):
            times[i, c, :n_sp[c]] = rng.normal(D / 2 + latency[c] * D, span / 4, n_sp[c])
    times = np.clip(times, 0, T[:, None, None] - 1e-6)
    return {"times": times, "T": T, "maps": maps, "which": which, "direction": direction, "membership": membership,
            "shared": shared, "n_cells": n_cells}
