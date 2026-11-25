import numpy as np

from cpca.active import centre_active, shuffle_null_active
from cpca.decompose import n_above_null, prepare, shuffle_null
from cpca.theory import eigh_desc, lag_matrix


def _assemblies(rng, M=3000, N=40, sequence_frac=0.0):
    """Events of one of four co-firing assemblies; each cell has its own fixed latency plus jitter.

    In a share sequence_frac of events, cells 0-19 fire instead in an order shared across cells,
    started at a random time.
    """
    latency = rng.uniform(0, np.pi, N)
    groups = np.arange(N) % 4
    S = np.zeros((M, N), complex)
    for i in range(M):
        if rng.random() < sequence_frac:
            on = rng.random(20) < 0.7
            start = rng.uniform(-0.5, 0.5)
            S[i, :20][on] = np.exp(1j * (start + np.linspace(0, np.pi, 20)[on] + rng.normal(0, 0.1, on.sum())))
        else:
            on = (groups == rng.integers(4)) & (rng.random(N) < 0.8) | (rng.random(N) < 0.05)
            S[i, on] = np.exp(1j * (latency[on] + rng.normal(0, 0.6, on.sum())))
    return S


def _kept(X, null):
    return n_above_null(eigh_desc(lag_matrix(X))[0], null, centred=True)[0]


def test_active_centring_removes_co_firing_only_on_a_shared_clock():
    rng = np.random.default_rng(0)
    S = _assemblies(rng)
    Xc = prepare(S, center=True, scale=True)
    Xa = centre_active(S)
    assert _kept(Xc, shuffle_null(Xc, rng, 30)) >= 3
    assert _kept(Xa, shuffle_null_active(Xa, rng, 30)) == 0
    # a shift of each event's clock is shared by the cells firing together, and the shuffle does not keep it
    Sd = S * np.exp(1j * rng.uniform(-0.3, 0.3, S.shape[0]))[:, None]
    Xd = centre_active(Sd)
    assert _kept(Xd, shuffle_null_active(Xd, rng, 30)) >= 1


def test_active_centring_keeps_a_sequence():
    rng = np.random.default_rng(1)
    S = _assemblies(rng, sequence_frac=0.1)
    Xa = prepare(S, center="active", scale=True)
    k = _kept(Xa, shuffle_null_active(Xa, rng, 30))
    assert k >= 1
    lead = eigh_desc(lag_matrix(Xa))[1][:, 0]
    phase = np.unwrap(np.angle(lead[:20] * np.conj(lead[0])))
    assert abs(np.corrcoef(phase, np.arange(20))[0, 1]) > 0.9


def test_active_shuffle_keeps_which_cells_fire():
    rng = np.random.default_rng(2)
    X = prepare(_assemblies(rng, M=500), center="active", scale=True)
    active = [np.flatnonzero(X[:, c] != 0) for c in range(X.shape[1])]
    P = np.zeros_like(X)
    for c, idx in enumerate(active):
        P[idx, c] = X[rng.permutation(idx), c]
    assert np.array_equal(P != 0, X != 0)
    assert shuffle_null_active(X, rng, 3).shape == (3, X.shape[1])
