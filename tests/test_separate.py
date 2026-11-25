import numpy as np

from cpca import fit
from cpca.rotation import quartimax
from cpca.separate import otsu_mask, separate
from sim.laps import generate as laps
from sim.two_routes import generate as routes


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
    assert otsu_mask(np.r_[np.full(10, 1.0), np.full(30, 0.1)]).sum() == 10


def _route_of(t, maps):
    on = [int((t["members"] & np.isfinite(m)).sum()) for m in maps]
    return int(np.argmax(on)), max(on) / max(sum(on), 1)


def test_reactivations_give_one_output_per_route():
    rng = np.random.default_rng(3)
    d = routes(rng, rates=(0.15, 0.15), shared=False, n_events=2000)
    perm = rng.permutation(d["times"].shape[0])
    tr, te = perm[:1000], perm[1000:]
    r = fit(d["times"][tr], d["T"][tr], rng=rng, ica=False, center=True, scale=True, n_shuffles=30)
    outs, _ = separate(r, d["times"][tr], d["T"][tr], d["times"][te], d["T"][te], rng, reversal=True)
    found = [_route_of(t, d["maps"]) for t in outs]
    assert all(purity >= 0.8 for _, purity in found)
    assert sorted({k for k, _ in found}) == [0, 1]


def test_waking_traversals_give_one_output_per_direction():
    rng = np.random.default_rng(11)
    d = laps(rng, both=0.7, rho=0.2, laps_per_direction=(30, 30))
    half = np.zeros(d["direction"].size, bool)
    for s in (1, -1):
        idx = rng.permutation(np.flatnonzero(d["direction"] == s))
        half[idx[: idx.size // 2]] = True
    tr, te = np.flatnonzero(half), np.flatnonzero(~half)
    r = fit(d["times"][tr], d["T"][tr], rng=rng, ica=False, center=False, scale=False, n_shuffles=30)
    outs, _ = separate(r, d["times"][tr], d["T"][tr], d["times"][te], d["T"][te], rng, reversal=False, min_events=4)
    assert len(outs) == 2
