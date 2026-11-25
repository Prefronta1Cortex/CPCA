import numpy as np

from cpca.replay import _smooth, decoder, null_templates, score_event


def _toy(seed=0, P=80, n=40):
    rng = np.random.default_rng(seed)
    x = (np.arange(P) + 0.5) * 0.02
    centres = rng.uniform(0.1, 1.5, n)
    raw = 15 * np.exp(-0.5 * ((x[None] - centres[:, None]) / 0.08) ** 2) + 0.2
    return rng, x, centres, decoder(_smooth(raw, axis=1), null_templates(raw, rng, 300))


def _event(rng, centres, n, M=8, ordered=True):
    act = rng.choice(n, 16, replace=False)
    order = act[np.argsort(centres[act])]
    c = np.zeros((M, n))
    for i, cell in enumerate(order):
        c[min(int(i / len(order) * M), M - 1) if ordered else rng.integers(M), cell] += 1
    return c


def test_ordered_events_score_high_in_both_directions():
    rng, x, centres, dec = _toy()
    for _ in range(10):
        c = _event(rng, centres, 40)
        fwd, rev = score_event(c, dec, x, rng, n_time=300), score_event(c[::-1], dec, x, rng, n_time=300)
        assert fwd["r"] > 0.8 and rev["r"] < -0.8
        assert fwd["p"] < 0.05 and fwd["p_time"] < 0.05 and rev["p"] < 0.05


def test_unordered_events_are_rarely_significant():
    rng, x, centres, dec = _toy(1)
    sig = [score_event(_event(rng, centres, 40, ordered=False), dec, x, rng, n_time=300) for _ in range(80)]
    assert np.mean([s["p"] < 0.05 and s["p_time"] < 0.05 for s in sig]) <= 0.1
