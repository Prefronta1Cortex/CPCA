import numpy as np

from cpca import fit
from cpca.encode import encode
from experimental.rotation import quartimax
from experimental.separate import otsu_mask, separate
from sim.two_routes import generate


def test_quartimax_recovers_sparse_components_from_a_mixture():
    rng = np.random.default_rng(0)
    V = np.zeros((90, 3), complex)
    for j in range(3):
        V[30 * j:30 * (j + 1), j] = np.exp(1j * np.pi * rng.random(30))
    V /= np.linalg.norm(V, axis=0)
    U, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    W, history = quartimax(V @ U)
    assert np.all(np.diff(history) >= -1e-12)
    assert np.all(np.abs(V.conj().T @ W).max(axis=1) > 0.99)


def test_otsu_mask_splits_two_groups():
    a = np.r_[np.full(10, 1.0), np.full(30, 0.1)]
    assert otsu_mask(a).sum() == 10


def test_separate_gives_one_route_per_template():
    rng = np.random.default_rng(3)
    d = generate(rng, rates=(0.15, 0.15), shared=False, n_events=2000)
    r = fit(d["times"], d["T"], rng=rng, ica=False, center=True, scale=True, n_shuffles=30)
    temps = separate(r, encode(d["times"], d["T"]), rng, explain=True)
    routes = []
    for t in temps:
        on = [int((t["members"] & np.isfinite(m)).sum()) for m in d["maps"]]
        if max(on) >= 10:
            assert max(on) / sum(on) >= 0.8
            routes.append(int(np.argmax(on)))
    assert sorted(set(routes)) == [0, 1]


def test_forward_only_separates_two_running_directions():
    from experimental.scripts.dev_waking import waking_outputs
    from sim.laps import generate
    rng = np.random.default_rng(11)
    d = generate(rng, both=0.7, rho=0.2, laps_per_direction=(30, 30))
    dirs = d["direction"]
    half = np.zeros(dirs.size, bool)
    for s in (1, -1):
        idx = rng.permutation(np.flatnonzero(dirs == s))
        half[idx[: idx.size // 2]] = True
    tr, te = np.flatnonzero(half), np.flatnonzero(~half)
    outs = waking_outputs(d["times"][tr], d["T"][tr], d["times"][te], d["T"][te], rng, center=False, explain=True,
                          conjugate=False, min_events=4, event_starts=True, trim=True, adaptive=True)
    outs = [o for o in outs if o[1] < 0.05]
    assert len(outs) == 2
    from scipy.stats import spearmanr
    matched = set()
    for order, _, _ in outs:
        r = [abs(spearmanr(order[np.isfinite(order) & np.isfinite(m)], m[np.isfinite(order) & np.isfinite(m)])[0]) for m in d["maps"]]
        assert max(r) > 0.9
        matched.add(int(np.argmax(r)))
    assert matched == {0, 1}
