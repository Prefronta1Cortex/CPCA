"""Synthetic reactivation events with planted routes.

A route is an ordered set of cells (each cell has a position in [0, 1] along
the route). A replay of a route plays a segment of it, forward or backward,
compressed into the event, with timing jitter. Events that are not replays
are synchronous volleys at the event centre. Optional properties of real
recordings can be switched on: direction-specific maps (a second route over
the same cells with independent positions), assembly structure in which cells
fire, and cells with no place on any route.

The answer key for scoring is, per route and direction, each cell's position
along that route (NaN for cells not on it).
"""

import numpy as np

EVENT_MS = 100.0


def make_routes(rng, n_cells, n_routes=1, shared=0.0, cells_per_route=None, direction_maps=False):
    """Return a list of routes; each route is a dict with 'cells' (indices) and 'pos' (positions in [0, 1])."""
    cells_per_route = cells_per_route or n_cells // max(n_routes, 1)
    routes = []
    pool = rng.permutation(n_cells)
    used = []
    for r in range(n_routes):
        if r == 0 or shared == 0:
            cells = pool[r * cells_per_route:(r + 1) * cells_per_route]
        else:
            n_shared = int(round(shared * cells_per_route))
            prev = routes[-1]["cells"]
            fresh = [c for c in pool if c not in np.concatenate(used)]
            cells = np.r_[rng.choice(prev, n_shared, replace=False), fresh[:cells_per_route - n_shared]]
        used.append(cells)
        routes.append({"cells": np.asarray(cells), "pos": rng.uniform(0, 1, size=len(cells))})
        if direction_maps:
            routes.append({"cells": np.asarray(cells), "pos": rng.uniform(0, 1, size=len(cells))})
    return routes


def route_centres(routes, n_cells):
    """Answer key: list of per-cell position vectors (NaN off-route), one per route."""
    out = []
    for route in routes:
        cen = np.full(n_cells, np.nan)
        cen[route["cells"]] = route["pos"]
        out.append(cen)
    return out


def generate(rng, n_cells=80, n_events=3000, routes=None, replay_frac=0.4, reverse_frac=0.5, participation=0.15,
             seq_participation=0.6, segment=1.0, span_ms=80.0, jitter_ms=6.0, assemblies=0, extra_rate=0.0,
             event_ms=EVENT_MS):
    """Spike times (events, cells, 3) in ms within [0, event_ms]; labels: route index (+1) or 0 for a volley."""
    routes = routes if routes is not None else make_routes(rng, n_cells)
    membership = rng.integers(0, assemblies, size=n_cells) if assemblies else None
    times = np.full((n_events, n_cells, 3), np.nan)
    label = np.zeros(n_events, dtype=int)
    direction = np.zeros(n_events, dtype=int)
    for i in range(n_events):
        if membership is not None:
            group = rng.integers(assemblies)
            prob = np.where(membership == group, min(3 * participation, 0.9), participation / 2)
        else:
            prob = np.full(n_cells, participation)
        fire = rng.random(n_cells) < prob
        if rng.random() < replay_frac:
            r = rng.integers(len(routes))
            route = routes[r]
            reverse = rng.random() < reverse_frac
            label[i], direction[i] = r + 1, -1 if reverse else 1
            start = rng.uniform(0, 1 - segment)
            frac = (route["pos"] - start) / segment
            in_seg = (frac >= 0) & (frac <= 1) & (rng.random(len(route["cells"])) < seq_participation)
            frac = 1 - frac if reverse else frac
            t = (event_ms - span_ms) / 2 + span_ms * frac
            cells = route["cells"][in_seg]
            times[i, cells, 0] = t[in_seg] + rng.normal(0, jitter_ms, size=cells.size)
            fire[cells] = False
            times[i, fire, 0] = rng.normal(event_ms / 2, 12.0, size=int(fire.sum()))
        else:
            times[i, fire, 0] = rng.normal(event_ms / 2, jitter_ms, size=int(fire.sum()))
        if extra_rate > 0:
            extra = rng.random(n_cells) < extra_rate
            times[i, extra, 1] = rng.uniform(0, event_ms, size=int(extra.sum()))
    times = np.clip(times, 0, event_ms - 1e-6)
    return {"times": times, "label": label, "direction": direction, "routes": routes,
            "centres": route_centres(routes, n_cells), "event_ms": event_ms, "n_cells": n_cells}


# Named conditions used by scripts/run_synthetic.py. Each value is (generator kwargs, route kwargs).
CONDITIONS = {
    "A one route, every event a replay":
        (dict(replay_frac=1.0, reverse_frac=0.0, participation=0.3), dict(n_routes=1)),
    "B one route, forward and backward 60/40, span 25 ms":
        (dict(replay_frac=1.0, reverse_frac=0.4, participation=0.3, span_ms=25.0), dict(n_routes=1)),
    "C one route, forward and backward 60/40, span 80 ms":
        (dict(replay_frac=1.0, reverse_frac=0.4, participation=0.3, span_ms=80.0), dict(n_routes=1)),
    "D two routes on different cells, 40% replay":
        (dict(replay_frac=0.4, reverse_frac=0.0, participation=0.15), dict(n_routes=2)),
    "E two routes sharing half their cells, 40% replay":
        (dict(replay_frac=0.4, reverse_frac=0.0, participation=0.15), dict(n_routes=2, shared=0.5, cells_per_route=40)),
    "F two routes, both directions, 15% replay, 15% cells":
        (dict(replay_frac=0.15, reverse_frac=0.5, participation=0.15, jitter_ms=8.0, extra_rate=0.03), dict(n_routes=2)),
    "G realistic: direction maps, segments, assemblies, non-place cells, 15% replay":
        (dict(replay_frac=0.15, reverse_frac=0.5, participation=0.15, segment=0.4, jitter_ms=8.0, assemblies=6),
         dict(n_routes=1, cells_per_route=60, direction_maps=True)),
    "H realistic at 40% replay":
        (dict(replay_frac=0.4, reverse_frac=0.5, participation=0.15, segment=0.4, jitter_ms=8.0, assemblies=6),
         dict(n_routes=1, cells_per_route=60, direction_maps=True)),
}
