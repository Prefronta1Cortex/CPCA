"""The conjugate-pair read-out (cpca/pairs.py)."""

import numpy as np
from scipy.stats import spearmanr

from cpca.pairs import circular_combination, pair_orders


def test_real_orthonormal_pair_gives_the_phases():
    theta = np.linspace(0.05, np.pi - 0.05, 40)
    c, s = np.cos(theta), np.sin(theta)
    u = circular_combination(c / np.linalg.norm(c), s / np.linalg.norm(s))
    assert abs(np.sum(u * u)) < 1e-9
    phase = np.angle(u * np.exp(-1j * np.angle(u[0])))
    assert abs(spearmanr(np.unwrap(phase), theta).statistic) > 0.999


def test_invariant_to_global_phases_and_mixing():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, np.pi, 50)
    q = np.exp(1j * theta)
    c, s = np.real(q), np.imag(q)
    # an arbitrary rotation of the pair within its span and arbitrary global phases, as eigh may return them
    a = rng.uniform(0, 2 * np.pi)
    v1 = (np.cos(a) * c + np.sin(a) * s) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    v2 = (-np.sin(a) * c + np.cos(a) * s) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    u = circular_combination(v1, v2)
    # u is q or its conjugate up to a global phase (exactly so when q's doubled phases are balanced; a finite random
    # sample of phases is not quite, so the zero-resultant vector differs from q slightly)
    overlap = max(abs(np.vdot(u, q)), abs(np.vdot(u, np.conj(q)))) / np.linalg.norm(q)
    assert overlap > 0.98


def test_recovers_a_balanced_whole_route_from_centred_components():
    from cpca import fit
    from sim.segments import generate
    rng = np.random.default_rng(1)
    d = generate(rng, n_route=40, n_events=2000, replay_frac=0.3, reverse_frac=0.5, segment=(1.0, 1.0), assemblies=0, latency_frac=0.0)
    res = fit(d["times"], d["T"], rng=rng, ica=False)
    assert res["n_kept"] >= 2
    orders = pair_orders(res["patterns"][:, :2], how="adjacent")
    o = orders[0]
    use = np.isfinite(o) & np.isfinite(d["centres"])
    assert use.sum() >= 20
    assert abs(spearmanr(o[use], d["centres"][use]).statistic) > 0.9
