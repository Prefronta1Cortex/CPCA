"""Checks of the constant-speed partial-replay generator (sim/segments.py) and of the direction oracle it supports."""

import numpy as np
from scipy.stats import spearmanr

from sim.segments import MARGIN, generate, n_route_for, spatial_assemblies


def _data(seed=0, **kw):
    rng = np.random.default_rng(seed)
    return generate(rng, n_route=n_route_for(9), n_events=600, replay_frac=0.3, **kw)


def test_replays_play_at_constant_speed_inside_the_window():
    d = _data(jitter_ms=0.0)
    pos, T = d["centres"], d["T"]
    checked = 0
    for i in np.flatnonzero(d["label"]):
        start, L, D = d["seg_start"][i], d["seg_length"][i], T[i]
        in_seg = np.isfinite(pos) & (pos >= start) & (pos <= start + L)
        mt = np.nanmean(d["times"][i], axis=-1)
        ordered = in_seg & np.isfinite(mt)
        if ordered.sum() < 4:
            continue
        frac = (pos[ordered] - start) / L
        frac = 1 - frac if d["direction"][i] < 0 else frac
        expected = MARGIN * D + frac * (1 - 2 * MARGIN) * D          # constant speed: time is affine in position
        on_line = np.abs(mt[ordered] - expected) < 1e-6
        # cells inside the segment that did not take part fire as background and are not on the line; those that did are
        assert on_line.sum() == d["n_ordered"][i]
        t_on = mt[ordered][on_line]
        assert t_on.min() >= MARGIN * D - 1e-6 and t_on.max() <= (1 - MARGIN) * D + 1e-6
        checked += 1
    assert checked > 20


def test_durations_do_not_mark_replays():
    d = _data()
    T, lab = d["T"], d["label"]
    assert abs(np.median(T[lab == 1]) - np.median(T[lab == 0])) < 0.15 * np.median(T)


def test_route_size_gives_requested_ordered_cells():
    d = _data()
    assert abs(np.mean(d["n_ordered"][d["label"] == 1]) - 9) < 2


def test_spatial_assemblies_follow_the_map():
    d = _data()
    m = spatial_assemblies(np.random.default_rng(1), d["n_cells"], d["route"], d["centres"])
    pos = d["centres"][d["route"]]
    assert spearmanr(m[d["route"]], pos).statistic > 0.9


def test_direction_oracle_recovers_the_order():
    from cpca import encode
    from cpca.direction import order_of
    from cpca.theory import eigh_desc, lag_matrix
    rng = np.random.default_rng(3)
    d = generate(rng, n_route=n_route_for(9), n_events=2000, replay_frac=0.1, reverse_frac=0.5)
    S = encode(d["times"], d["T"])
    signs = np.where(d["direction"] != 0, d["direction"], rng.choice([-1, 1], size=d["direction"].size))
    q = eigh_desc(lag_matrix(np.where((signs > 0)[:, None], S, np.conj(S))))[1][:, 0]
    o = order_of(q, member_frac=0.2)
    use = np.isfinite(o) & np.isfinite(d["centres"])
    assert use.sum() >= 10
    assert abs(spearmanr(o[use], d["centres"][use]).statistic) > 0.7
