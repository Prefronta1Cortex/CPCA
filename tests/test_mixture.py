import numpy as np
from scipy.stats import spearmanr

from cpca import fit
from cpca.direction import order_of
from cpca.mixture import alternate, sequences, similarity
from sim.routes import generate, make_routes


def _two_orders(rng, n_events=600, n_cells=30):
    """Each event plays one of two independent orders of the same cells, forward or backward."""
    pos = rng.uniform(0.05, 0.95, (2, n_cells))
    which = rng.integers(0, 2, n_events)
    backward = rng.random(n_events) < 0.5
    S = np.zeros((n_events, n_cells), complex)
    for i in range(n_events):
        t = pos[which[i]] + rng.normal(0, 0.02, n_cells)
        t = 1 - t if backward[i] else t
        fire = rng.random(n_cells) < 0.5
        S[i, fire] = np.exp(1j * np.pi * t[fire])
    return S, pos, which, backward


def test_similarity_ignores_phase_and_time_reversal():
    rng = np.random.default_rng(0)
    q = np.exp(1j * rng.uniform(0, np.pi, 20))
    assert np.isclose(similarity(q, q * np.exp(1j * 1.3)), 1.0)
    assert np.isclose(similarity(q, -np.conj(q)), 1.0)
    assert similarity(q, np.exp(1j * rng.uniform(0, np.pi, 20))) < 0.8


def test_alternation_separates_two_orders_on_the_same_cells():
    rng = np.random.default_rng(1)
    S, pos, which, _ = _two_orders(rng)
    # a poor start: the blend of the two orders and a slightly perturbed copy of it
    blend = np.exp(1j * np.pi * pos[0]) + np.exp(1j * np.pi * pos[1])
    Q0 = np.stack([blend, blend * np.exp(1j * rng.normal(0, 0.6, blend.size))], axis=1)
    r = alternate(S, Q0, rng)
    assert r["templates"].shape[1] == 2
    for k in range(2):
        rho = []
        for j in range(2):
            o = order_of(r["templates"][:, j])
            m = np.isfinite(o)
            rho.append(abs(spearmanr(o[m], pos[k][m]).statistic))
        assert max(rho) > 0.9
    agree = np.mean(r["label"][r["label"] >= 0] == which[r["label"] >= 0])
    assert max(agree, 1 - agree) > 0.95


def test_duplicate_starts_merge_into_one_template():
    rng = np.random.default_rng(2)
    S, pos, which, _ = _two_orders(rng)
    S = S[which == 0]
    q = np.exp(1j * np.pi * pos[0])
    r = alternate(S, np.stack([q, q * np.exp(1j * 0.4)], axis=1), rng)
    assert r["templates"].shape[1] == 1


def test_no_validated_template_without_order():
    rng = np.random.default_rng(3)
    routes = make_routes(rng, n_cells=60, n_routes=1, cells_per_route=40)
    data = generate(rng, n_cells=60, n_events=1500, routes=routes, replay_frac=0.4, reverse_frac=0.5, participation=0.15, jitter_ms=8.0)
    times = data["times"].copy()
    for e in range(times.shape[0]):
        active = np.flatnonzero(np.isfinite(times[e]).any(axis=1))
        times[e, active] = times[e, rng.permutation(active)]
    res = fit(times, data["event_ms"], rng=rng, ica=False, n_shuffles=30)
    sq = sequences(res, rng)
    assert sq is None or not sq["check"]["keep"].any()
