"""Event-sparse complex templates: find a sequence that is present in only a few events.

Complex PCA averages over all events, so a sequence played in a few percent of them is diluted below the
noise bound. Here each template is fitted to the events that match it best instead:

  1. events are encoded and centred/scaled exactly as for complex PCA (cpca.encode, cpca.decompose.prepare);
  2. a template starts from one event's own arrows (many seed events);
  3. every event is matched to the template forward (<z, q>) or backward (<conj z, q>, a reversed play), whichever
     is larger; the share `frac` of events with the largest match is kept;
  4. the kept events, each reversed if it matched backward and rotated to the template's phase (a shift of the
     event's clock), are averaged into the new template; 3-4 repeat until the kept events stop changing;
  5. templates are ranked by the summed match of their kept events, and near-duplicates (|<q_i, q_j>| or
     |<q_i, conj q_j>| above 0.8) are dropped.

normalize=True matches by cosine (each event's arrows scaled to unit length, events with fewer than min_active
active cells never kept). With raw inner products (normalize=False, the first version) the match grows with
the number of active cells, and on real sleep events the kept events were the largest population bursts rather
than the replays (docs/power.md).

Each template is read out like a complex PCA component (cpca.decode.readout). The method returns several
templates; scoring with a best-of-outputs null (as for every method here) makes that fair.
"""

import numpy as np

from cpca.decode import readout
from cpca.decompose import prepare
from cpca.encode import encode


def _iterate(Z, Q, k, n_iter, eligible=None):
    Zc = np.conj(Z)
    prev = None
    for _ in range(n_iter):
        Gf, Gb = Zc @ Q, Z @ Q                                            # (events, seeds)
        back = np.abs(Gb) > np.abs(Gf)
        G = np.where(back, Gb, Gf)
        mag = np.abs(G)
        if eligible is not None:
            mag = np.where(eligible[:, None], mag, 0.0)
        cut = -np.partition(-mag, k - 1, axis=0)[k - 1]
        keep = mag >= cut[None, :]
        U = np.where(keep, G / np.where(mag > 0, mag, 1), 0)
        Q = Z.T @ np.where(back, 0, U) + Zc.T @ np.where(back, U, 0)
        Q /= np.maximum(np.linalg.norm(Q, axis=0, keepdims=True), 1e-12)
        if prev is not None and np.array_equal(keep, prev):
            break
        prev = keep
    return Q, np.where(keep, mag, 0).sum(axis=0)


def fit(times_ms, T, rng, fracs=(0.03, 0.10), n_seeds=400, n_iter=25, n_out=5, center=True, scale=True,
        member_frac=0.2, min_active=5, distinct=0.8, normalize=True):
    Z = prepare(encode(times_ms, T), center=center, scale=scale)
    active = (np.abs(encode(times_ms, T)) > 0).sum(axis=1)
    eligible = active >= min_active
    if normalize:
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    pool = np.flatnonzero(eligible)
    seeds = rng.choice(pool, min(n_seeds, pool.size), replace=False)
    T_read = float(np.median(T)) if np.ndim(T) else float(T)
    out = []
    for frac in fracs:
        k = max(10, int(round(frac * Z.shape[0])))
        Q = Z[seeds].T.copy()
        Q /= np.maximum(np.linalg.norm(Q, axis=0, keepdims=True), 1e-12)
        Q, obj = _iterate(Z, Q, k, n_iter, eligible if normalize else None)
        picked = []
        for s in np.argsort(-obj):
            q = Q[:, s]
            if any(max(abs(np.vdot(p, q)), abs(np.vdot(np.conj(p), q))) > distinct for p in picked):
                continue
            picked.append(q)
            if len(picked) == n_out:
                break
        for q in picked:
            _, t, _, inc, _ = readout(q, T_read, min_weight=member_frac)
            out.append({"frac": frac, "template": q, "order": np.where(inc, t, np.nan)})
    return {"templates": out, "orders": [o["order"] for o in out]}
