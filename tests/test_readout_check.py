import numpy as np

from cpca.readout_check import circularity, redraw_latency, shuffle_timing
from sim.two_routes import generate


def test_circularity_of_real_and_spread_vectors():
    assert circularity(np.array([1.0, -2.0, 0.5, 3.0])) > 0.99
    assert circularity(np.exp(1j * np.linspace(0, np.pi, 50, endpoint=False))) < 0.05


def test_surrogates_keep_who_fires_and_spike_counts():
    rng = np.random.default_rng(0)
    d = generate(rng, n_cells=30, n_route=10, rates=(0.2,), n_events=50)
    for s in (shuffle_timing(d["times"], rng), redraw_latency(d["times"], d["T"], rng)):
        assert np.array_equal(np.isfinite(s).any(axis=-1), np.isfinite(d["times"]).any(axis=-1))
    r = redraw_latency(d["times"], d["T"], rng)
    assert np.array_equal(np.isfinite(r).sum(axis=-1), np.isfinite(d["times"]).sum(axis=-1))


def test_two_route_generator_maps():
    rng = np.random.default_rng(1)
    a = generate(rng, n_cells=60, n_route=20, rates=(0.1, 0.1), shared=False, n_events=20)
    assert not np.any(np.isfinite(a["maps"][0]) & np.isfinite(a["maps"][1]))
    b = generate(rng, n_cells=60, n_route=20, rates=(0.1, 0.1), shared=True, n_events=20)
    assert np.array_equal(np.isfinite(b["maps"][0]), np.isfinite(b["maps"][1]))
