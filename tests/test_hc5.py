import numpy as np

from cpca import fit, hc5
from cpca.decode import readout
from cpca.evaluate import best_order


def fake_session(n_runs=10, n_cells=30, W=12.0, seed=0):
    """Cells with time fields spread over the run, Poisson spikes around each field, plus background."""
    rng = np.random.default_rng(seed)
    fields = np.sort(rng.uniform(0.5, W - 0.5, n_cells))
    spikes = []
    for _ in range(n_runs):
        run = []
        for c in range(n_cells):
            s = np.r_[rng.normal(fields[c], 0.4, rng.poisson(8)), rng.uniform(0, W, rng.poisson(1))]
            run.append(np.sort(s[(s >= 0) & (s < W)]))
        spikes.append(run)
    return {"name": "fake", "W_s": W, "onsets_s": np.arange(n_runs) * 30.0, "cell_ids": np.arange(n_cells), "spikes": spikes}, fields


def test_best_order_signed_and_calibrated():
    rng = np.random.default_rng(1)
    key = np.r_[np.linspace(0, 10, 20), np.full(5, np.nan)]
    perfect = best_order([key.copy()], key, rng, n_null=500)
    assert perfect["rho"] > 0.99 and perfect["p"] < 0.01 and perfect["coverage"] == 1.0
    backwards = best_order([-key], key, rng, n_null=500)
    assert backwards["rho"] < -0.99 and backwards["p"] > 0.9
    ps = [best_order([rng.permutation(key)], key, rng, n_null=200)["p"] for _ in range(60)]
    assert 0.25 < np.mean(ps) < 0.75


def test_best_order_null_grows_with_outputs():
    rng = np.random.default_rng(2)
    key = np.linspace(0, 10, 15)
    few = best_order([rng.permutation(key) for _ in range(1)], key, rng, n_null=400)
    many = best_order([rng.permutation(key) for _ in range(12)], key, rng, n_null=400)
    assert many["null95"] > few["null95"]


def test_best_order_skips_small_outputs():
    rng = np.random.default_rng(3)
    key = np.linspace(0, 10, 20)
    small = np.full(20, np.nan)
    small[:4] = key[:4]
    out = best_order([small], key, rng, min_members=6)
    assert out["n_testable"] == 0 and not np.isfinite(out["rho"])


def test_window_times_offsets_and_clipping():
    session, _ = fake_session()
    runs = np.array([0, 3])
    off = np.array([0.0, 2.0])
    times, T = hc5.window_times(session, runs, off, 8.0)
    assert T == 8000.0
    assert np.nanmax(times) < 8000.0 and np.nanmin(times) >= 0
    raw = session["spikes"][3][5]
    expect = (raw[(raw >= 2.0) & (raw < 10.0)] - 2.0) * 1000
    got = times[1, 5][np.isfinite(times[1, 5])]
    np.testing.assert_allclose(np.sort(got), np.sort(expect))


def test_time_fields_and_cpca_recover_fake_sequence():
    session, fields = fake_session(n_runs=12)
    key = hc5.time_fields(session, np.arange(6, 12))
    assert np.isfinite(key).mean() > 0.8
    assert np.corrcoef(key[np.isfinite(key)], fields[np.isfinite(key)])[0, 1] > 0.95
    times, T = hc5.window_times(session, np.arange(6))
    r = fit(times, T, ica=False, center=False, scale=False)
    _, t, _, inc, _ = readout(r["patterns"][:, 0], r["T_read_ms"], min_weight=0.2)
    score = best_order([np.where(inc, t, np.nan)], key, np.random.default_rng(0), n_null=300)
    assert score["rho"] > 0.8 and score["p"] < 0.01
