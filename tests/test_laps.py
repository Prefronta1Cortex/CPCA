import numpy as np

from cpca.hc11 import laps


def _run(t, x, length, circular):
    dx = np.diff(x)
    if circular:
        dx = (dx + length / 2) % length - length / 2
    d = np.zeros(t.size)
    d[1:] = np.sign(dx)
    return {"t": t, "x": x, "direction": d, "length": length, "circular": circular}


def test_linear_track_laps_alternate_and_skip_short_runs():
    t = np.arange(0, 30, 0.025)
    # three full laps (0 -> 1.6, 1.6 -> 0, 0 -> 1.6) with 1 s stops at the ends, then a short run that does not count
    x = np.interp(t, [0, 3, 4, 7, 8, 11, 12, 13, 30], [0, 1.6, 1.6, 0, 0, 1.6, 1.6, 1.2, 1.2])
    out = laps(_run(t, x, 1.6, False))
    assert [d for *_, d in out] == [1.0, -1.0, 1.0]
    assert all(abs((b - a) - 3) < 0.1 for a, b, _ in out)


def test_circular_track_long_runs_are_cut_into_laps():
    t = np.arange(0, 12, 0.025)
    x = (t / 4 * 2.8) % 2.8                      # three loops of a 2.8 m circle in one direction, 4 s each
    out = laps(_run(t, x, 2.8, True))
    assert len(out) == 3 and all(d == 1.0 for *_, d in out)
    assert all(abs((b - a) - 4) < 0.1 for a, b, _ in out)
