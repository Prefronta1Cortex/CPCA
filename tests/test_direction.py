import numpy as np

from cpca import encode, fit
from cpca.direction import expression_p, fit_template, heldout_test, permute_within_events, split_component
from cpca.evaluate import order_vs_place
from sim.routes import generate, make_routes


def test_time_reversal_is_conjugation_and_separates_when_the_half_circle_is_filled():
    T = 100.0
    t = np.linspace(5, 95, 20)
    forward = encode(t[None, :], T)[0]
    backward = encode((T - t)[None, :], T)[0]
    np.testing.assert_allclose(backward, -np.conj(forward), atol=1e-12)
    q = forward / np.linalg.norm(forward)
    shifted = forward * np.exp(1j * 0.7)                   # a different clock: same match
    assert np.isclose(abs(np.vdot(q, shifted)), abs(np.vdot(q, forward)))
    assert abs(np.vdot(q, backward)) < 0.25 * abs(np.vdot(q, forward))
    assert np.isclose(abs(np.vdot(np.conj(q), backward)), abs(np.vdot(q, forward)))


def test_within_event_shuffle_keeps_who_fired_and_the_set_of_arrows():
    rng = np.random.default_rng(0)
    S = (rng.random((50, 12)) < 0.4) * np.exp(1j * rng.uniform(0, np.pi, (50, 12)))
    P = permute_within_events(S, rng)
    assert np.array_equal(S != 0, P != 0)
    for a, b in zip(S, P):
        np.testing.assert_allclose(np.sort_complex(a[a != 0]), np.sort_complex(b[b != 0]))


def test_split_recovers_order_and_direction_of_a_route_played_both_ways():
    rng = np.random.default_rng(1)
    routes = make_routes(rng, n_cells=40, n_routes=1)
    data = generate(rng, n_cells=40, n_events=1500, routes=routes, replay_frac=1.0, reverse_frac=0.4, participation=0.3, span_ms=80.0)
    res = fit(data["times"], data["event_ms"], rng=rng, ica=False, n_shuffles=30)
    assert res["n_kept"] >= 1
    sp = split_component(res["encoded"], res["pca_components"][:, 0], rng)
    assert sp["accepted"]
    rho = order_vs_place(sp["order"], data["centres"][0], False, 1.0, rng, n_null=20)[0]
    assert abs(rho) > 0.95
    labels = sp["forward"][sp["expresses"]]
    truth = data["direction"][sp["events"][sp["expresses"]]] > 0
    accuracy = np.mean(labels == truth)
    assert max(accuracy, 1 - accuracy) > 0.95
    assert abs(sp["minority_fraction"] - 0.4) < 0.08


def test_no_backward_group_in_one_direction_data():
    rng = np.random.default_rng(2)
    routes = make_routes(rng, n_cells=40, n_routes=1)
    data = generate(rng, n_cells=40, n_events=1500, routes=routes, replay_frac=1.0, reverse_frac=0.0, participation=0.3, span_ms=80.0)
    res = fit(data["times"], data["event_ms"], rng=rng, ica=False, center=False, scale=False, n_shuffles=30)
    sp = split_component(res["encoded"], res["pca_components"][:, 0], rng)
    assert sp["minority_fraction"] < 0.05


def _orderless(rng, n_events=400, n_cells=20, p=0.4):
    """Events in which a random subset of cells fires at random times: who fires carries nothing about order."""
    return (rng.random((n_events, n_cells)) < p) * np.exp(1j * rng.uniform(0, np.pi, (n_events, n_cells)))


def test_expression_p_is_uniform_without_order():
    rng = np.random.default_rng(4)
    S = _orderless(rng, n_events=2000)
    q = np.exp(1j * rng.uniform(0, np.pi, 20)) / np.sqrt(20)
    p = expression_p(S, q, rng)
    assert np.mean(p <= 0.05) <= 0.06
    assert abs(np.mean(p <= 0.5) - 0.5) < 0.05


def test_heldout_test_holds_its_level_without_order_and_finds_a_route():
    rng = np.random.default_rng(5)
    ps = [heldout_test(_orderless(rng), rng)[1] for _ in range(40)]
    assert np.mean(np.array(ps) <= 0.05) <= 0.1
    q = np.exp(1j * np.linspace(0.1, 3.0, 20))
    played = np.array([(rng.random(20) < 0.4) * (q if k % 2 else -np.conj(q)) * np.exp(1j * rng.normal(0, 0.15, 20)) for k in range(400)])
    share, p = heldout_test(played, rng)
    assert p < 1e-6 and share > 0.3


def test_fit_template_labels_flip_with_conjugation():
    rng = np.random.default_rng(3)
    q = np.exp(1j * np.linspace(0, np.pi, 15))
    S = np.array([q * np.exp(1j * rng.uniform(0, 6)) if k % 3 else np.conj(q) * np.exp(1j * rng.uniform(0, 6)) for k in range(90)])
    _, forward, _, _, _ = fit_template(S)
    truth = np.array([k % 3 != 0 for k in range(90)])
    assert np.all(forward == truth) or np.all(forward == ~truth)
