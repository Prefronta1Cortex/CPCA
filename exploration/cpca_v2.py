"""Unified, principled Complex PCA + Riemannian Complex ICA (exploration/cpca_v2.py).

Does NOT modify any existing code in cpca/.

Five first-principles foundations:
1. Phase arc alpha = 1.5 * pi: reduces unrelated-map overlap 4.5x (0.41 -> 0.09) and
   background burst mean energy >2x, while keeping a 90 deg empty gap for linear unwrapping.
2. Reactivation preprocessing (regime='reactivation'):
   - Per-event circular phase alignment: cancels shared burst-peak timing skew.
   - Active-centring: subtract each cell's mean active phasor ONLY on active entries
     (silent entries remain 0) -> algebraically annihilates arbitrary co-firing covariance
     AND per-cell latency in expectation.
   - Global RMS scaling: gives low-rate place cells equal footing with high-rate cells.
   - Active-phase shuffle null (upper noise edge null[:, 0], q=99%): permutes each cell's
     active phasors only among its active events, preserving exact co-firing & latency.
3. Waking preprocessing (regime='waking'):
   - Uncentred, unscaled phasors (every event is a traversal).
   - Circular time-shift shuffle null (q=99%): component 0 tested against null[:, 0] (rejects
     no-sequence controls); components j >= 1 tested against the noise edge null[:, 1].
4. Exact Riemannian Complex ICA in whitened event space (no circularity assumption, fixes
   both conjugation bugs in cpca/ica.py, uses robust super-linear contrast |y|^(2*p) with
   p=1.5 so high-expression sequence events drive unmixing rather than background noise or
   single extreme bursts). Exact mixing pattern reconstruction: P = V Lambda^(1/2) W*.
5. Pairwise route-symmetry quotient (reactivation regime only):
   - Merges conjugate pairs (q, conj(q)) representing forward & backward plays of one route.
   - Combines co-occurring real pairs (c, s) via Takagi circular combination (cpca.pairs).
"""

import numpy as np


def encode_arc(times, T, alpha=1.5 * np.pi, how="mean", t0=0.0):
    """Encode spike times as unit phasors exp(j * alpha * (t - t0) / T), averaged per cell."""
    times = np.asarray(times, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    if T_arr.ndim == 0:
        T_arr = np.full(times.shape[0], float(T_arr))
    T_b = T_arr.reshape((times.shape[0],) + (1,) * (times.ndim - 1))
    t0_arr = np.asarray(t0, dtype=float)
    if t0_arr.ndim == 1:
        t0_arr = t0_arr.reshape((times.shape[0],) + (1,) * (times.ndim - 1))
    finite = np.isfinite(times)
    with np.errstate(invalid="ignore", divide="ignore"):
        phasors = np.exp(1j * alpha * (times - t0_arr) / T_b)
    phasors[~finite] = 0.0
    if times.ndim == 2:
        return phasors
    encoded = phasors.sum(axis=-1)
    if how == "mean":
        count = finite.sum(axis=-1)
        np.divide(encoded, count, out=encoded, where=count > 0)
    return encoded


def align_event_phase(S):
    """Rotate each event row i so its circular mean phase arg(sum_k S_ik) is 0."""
    phi = np.angle(S.sum(axis=1))
    return S * np.exp(-1j * phi)[:, None]


def active_centre_scale(S, scale="global_rms"):
    """Subtract each cell's mean active phasor ONLY on active entries (zeros stay 0), then scale."""
    X = np.array(S, dtype=complex, copy=True)
    active = X != 0
    counts = active.sum(axis=0)
    mean_act = np.zeros(X.shape[1], dtype=complex)
    np.divide(X.sum(axis=0), counts, out=mean_act, where=counts > 0)
    X = np.where(active, X - mean_act[None, :], 0.0)
    if scale == "global_rms":
        rms = np.sqrt(np.mean(np.abs(X) ** 2, axis=0))
        np.divide(X, rms, out=X, where=rms > 0)
    elif scale == "sqrt_rms":
        rms = np.sqrt(np.mean(np.abs(X) ** 2, axis=0))
        s = np.sqrt(rms)
        np.divide(X, s, out=X, where=s > 0)
    elif scale == "active_rms":
        ms = np.zeros(X.shape[1], dtype=float)
        np.divide(np.sum(np.abs(X) ** 2, axis=0), counts, out=ms, where=counts > 0)
        rms = np.sqrt(ms)
        np.divide(X, rms, out=X, where=rms > 0)
    elif scale != "none":
        raise ValueError(f"Unknown scale: {scale}")
    return X


def lag_matrix(X):
    return (X.T @ X.conj()) / X.shape[0]


def eigh_desc(C):
    H = 0.5 * (C + C.conj().T)
    vals, vecs = np.linalg.eigh(H)
    order = np.argsort(vals)[::-1]
    return vals[order].real, vecs[:, order]


def shuffle_null_active(X, rng, n_shuffles=100):
    """Permute each cell's nonzero entries ONLY among its active events."""
    M, N = X.shape
    active_idx = [np.flatnonzero(X[:, c] != 0) for c in range(N)]
    null = np.empty((n_shuffles, N), dtype=float)
    P = np.zeros_like(X)
    for s in range(n_shuffles):
        P.fill(0.0)
        for c, idx in enumerate(active_idx):
            if idx.size > 0:
                P[idx, c] = X[rng.permutation(idx), c]
        null[s] = eigh_desc(lag_matrix(P))[0]
    return null


def shuffle_null_circ_time(times, T, rng, alpha=1.5 * np.pi, n_shuffles=100):
    """Circularly shift each cell's spike times in each event by U(0, T_i) mod T_i."""
    M, N = times.shape[:2]
    T_arr = np.asarray(T, dtype=float)
    if T_arr.ndim == 0:
        T_arr = np.full(M, float(T_arr))
    T3 = T_arr[:, None, None]
    null = np.empty((n_shuffles, N), dtype=float)
    for s in range(n_shuffles):
        shift = rng.uniform(0.0, 1.0, size=(M, N, 1)) * T3
        sh = np.mod(times + shift, T3)
        X = encode_arc(sh, T_arr, alpha=alpha)
        null[s] = eigh_desc(lag_matrix(X))[0]
    return null


def count_above_noise(vals, null, has_mean=False, quantile=99.0, max_components=20):
    """Components above the noise edge.

    - Zero-mean matrices (has_mean=False, active-centred): all components tested against
      the upper noise edge percentile(null[:, 0], quantile).
    - Uncentred waking matrices (has_mean=True): component 0 tested against percentile(null[:, 0], quantile)
      (so no-sequence controls yield k=0), and components j >= 1 tested against the noise bulk
      edge percentile(null[:, 1], quantile).
    """
    lim = min(vals.size, max_components)
    if lim == 0:
        return 0
    edge0 = float(np.percentile(null[:, 0], quantile))
    if vals[0] <= edge0:
        return 0
    edge = float(np.percentile(null[:, 1], quantile)) if (has_mean and null.shape[1] > 1) else edge0
    k = 1
    while k < lim and vals[k] > edge:
        k += 1
    return k


def _sym_orth(W):
    G = W.conj().T @ W
    vals, vecs = np.linalg.eigh(G)
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.conj().T
    return W @ inv_sqrt


def complex_ica_riemannian(Z, rng, power=1.5, n_init=8, max_iter=200, tol=1e-6):
    """Riemannian unitary Complex ICA maximizing sum_j mean_i (|y_ij|^2)^power on whitened Z."""
    M, K = Z.shape
    if K <= 1:
        return np.eye(K, dtype=complex)

    def obj_and_grad(W):
        Y = Z @ W
        U = np.abs(Y) ** 2
        if power == 2.0:
            loss = -float(np.sum(np.mean(U * U, axis=0)))
            g = -2.0 * U
        else:
            Up = np.power(U + 1e-12, power - 1.0)
            loss = -float(np.sum(np.mean(U * Up, axis=0)))
            g = -power * Up
        Egrad = (Z.conj().T @ (Y * g)) / M
        return loss, Egrad

    best_loss, best_W = np.inf, np.eye(K, dtype=complex)
    inits = [np.eye(K, dtype=complex)]
    for _ in range(n_init - 1):
        R = rng.normal(size=(K, K)) + 1j * rng.normal(size=(K, K))
        inits.append(_sym_orth(R))

    for W0 in inits:
        W = W0.copy()
        loss, Egrad = obj_and_grad(W)
        step = 0.5
        for _ in range(max_iter):
            G = W.conj().T @ Egrad
            A = -(G - G.conj().T)
            gnorm2 = float(np.real(np.sum(A.conj() * A)))
            if gnorm2 < tol ** 2:
                break
            accepted = False
            for _ls in range(20):
                W_try = _sym_orth(W + step * (W @ A))
                loss_try, Egrad_try = obj_and_grad(W_try)
                if loss_try <= loss - 1e-4 * step * gnorm2:
                    W, loss, Egrad = W_try, loss_try, Egrad_try
                    step = min(step * 1.5, 2.0)
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
        if loss < best_loss:
            best_loss, best_W = loss, W
    return best_W


def circular_combination(v1, v2):
    """Unit combination in span(v1, v2) minimizing |sum_k u_k^2| (Takagi / cpca.pairs)."""
    v1 = np.asarray(v1, complex)
    v2 = np.asarray(v2, complex)
    v1 = v1 / max(np.linalg.norm(v1), 1e-300)
    v2 = v2 - np.vdot(v1, v2) * v1
    n2 = np.linalg.norm(v2)
    if n2 < 1e-12:
        return v1
    v2 = v2 / n2
    p11, p22, p12 = np.sum(v1 * v1), np.sum(v2 * v2), np.sum(v1 * v2)
    if abs(p22) > 1e-12:
        disc = np.sqrt(p12 ** 2 - p11 * p22 + 0j)
        roots = [(-p12 + disc) / p22, (-p12 - disc) / p22]
    elif abs(p12) > 1e-12:
        roots = [-p11 / (2 * p12)]
    else:
        roots = [1j]
    best = None
    for r in roots:
        u = v1 + r * v2
        u = u / max(np.linalg.norm(u), 1e-300)
        res = abs(np.sum(u * u))
        if best is None or res < best[0]:
            best = (res, u)
    return best[1]


def quotient_symmetry(P, Y, C, sim_thresh=0.5, circ_thresh=0.75, corr_thresh=0.20, fold_harmonics=True):
    """Pairwise (non-transitive) quotient of reactivation ICs by route symmetries:

    1. Conjugate / near-duplicate pairs (|<p_a, conj(p_b)>| or |<p_a, p_b>| >= sim_thresh):
       combine forward & backward copies of one route into p_a + e^{j phi} conj(p_b).
    2. Co-occurring real pairs (both circularity >= circ_thresh & corr(|y_a|^2, |y_b|^2) >= corr_thresh):
       combine (c, s) via Takagi circular_combination into q = c + j s.
    3. Optional harmonic folding (fold_harmonics=True): if a weaker component j co-occurs
       strongly ONLY with one stronger component i (corr >= corr_thresh and exclusively paired
       with i), replace i by the leading eigenvector of C inside span(p_i, p_j).
    """
    K = P.shape[1]
    if K <= 1:
        return P
    Pn = P / np.maximum(np.linalg.norm(P, axis=0, keepdims=True), 1e-12)
    U = np.abs(Y) ** 2
    Uc = U - U.mean(axis=0, keepdims=True)
    Us = np.maximum(np.linalg.norm(Uc, axis=0, keepdims=True), 1e-12)
    Cev = (Uc.T @ Uc) / (Us.T @ Us)
    circs = np.array([abs(np.sum(Pn[:, j] ** 2)) for j in range(K)])

    used = np.zeros(K, dtype=bool)
    out = []
    out_idx = []
    # Pass 1: conjugate pairs & co-occurring real pairs (highest energy first)
    for i in range(K):
        if used[i]:
            continue
        used[i] = True
        acc = Pn[:, i].copy()
        paired = False
        for j in range(i + 1, K):
            if used[j]:
                continue
            s_dir = np.vdot(Pn[:, j], acc / max(np.linalg.norm(acc), 1e-12))
            s_con = np.vdot(np.conj(Pn[:, j]), acc / max(np.linalg.norm(acc), 1e-12))
            if max(abs(s_dir), abs(s_con)) >= sim_thresh:
                used[j] = True
                paired = True
                if abs(s_con) >= abs(s_dir):
                    acc = acc + np.conj(Pn[:, j]) * np.exp(1j * np.angle(s_con))
                else:
                    acc = acc + Pn[:, j] * np.exp(1j * np.angle(s_dir))
        if not paired and circs[i] >= circ_thresh:
            best_j, best_c = -1, corr_thresh
            for j in range(i + 1, K):
                if not used[j] and circs[j] >= circ_thresh and Cev[i, j] > best_c:
                    best_j, best_c = j, Cev[i, j]
            if best_j >= 0:
                used[best_j] = True
                acc = circular_combination(Pn[:, i], Pn[:, best_j])
        acc /= max(np.linalg.norm(acc), 1e-12)
        out.append(acc)
        out_idx.append(i)

    # Pass 2: if a weaker unpaired component j is exclusively co-occurring with one stronger
    # component a (partial-replay 2nd harmonic), fold it back into a's fundamental mode in C.
    if fold_harmonics and len(out) >= 2:
        keep = np.ones(len(out), dtype=bool)
        for b in range(len(out) - 1, 0, -1):
            ib = out_idx[b]
            # Check correlation with all stronger kept components a < b
            corrs = [(a, Cev[out_idx[a], ib]) for a in range(b) if keep[a]]
            strong = [(a, c) for a, c in corrs if c >= corr_thresh]
            if len(strong) == 1:
                a = strong[0][0]
                Q, _ = np.linalg.qr(np.column_stack([out[a], out[b]]))
                _, ev = eigh_desc(Q.conj().T @ C @ Q)
                q = Q @ ev[:, 0]
                q /= max(np.linalg.norm(q), 1e-12)
                out[a] = q
                keep[b] = False
        out = [v for v, k_flag in zip(out, keep) if k_flag]
    return np.column_stack(out)


def readout_gap(component, T_read=1.0, alpha=1.5 * np.pi, member_frac=0.2):
    """Read out member mask and unwrapped relative times by cutting the widest empty gap on the circle."""
    q = np.asarray(component, dtype=complex)
    w = np.abs(q)
    if w.max() <= 0:
        return np.full(q.shape, np.nan), np.zeros(q.shape, dtype=bool), 0.0
    inc = w >= member_frac * w.max()
    times = np.full(q.shape, np.nan)
    n_inc = int(inc.sum())
    if n_inc == 1:
        times[inc] = 0.0
        return times, inc, 0.0
    if n_inc < 2:
        return times, inc, 0.0
    angles = np.angle(q[inc])
    order = np.argsort(angles)
    ordered = angles[order]
    gaps = np.diff(ordered, append=ordered[0] + 2 * np.pi)
    cut = int(np.argmax(gaps))
    span = float(2 * np.pi - gaps[cut])
    start = (cut + 1) % len(ordered)
    seq = np.concatenate([np.arange(start, len(ordered)), np.arange(0, start)])
    unwrapped = ordered[seq].astype(float)
    if start != 0:
        unwrapped[len(ordered) - start:] += 2 * np.pi
    unwrapped -= unwrapped[0]
    out_t = np.empty(len(angles))
    out_t[order[seq]] = (T_read / alpha) * unwrapped
    times[inc] = out_t
    return times, inc, span


def fit_cpca_v2(times_ms, T, rng=None, regime="reactivation", alpha=1.5 * np.pi,
                power=1.5, scale="global_rms", n_shuffles=100, quantile=99.0,
                member_frac=0.2, max_components=20, ica=True, reconcile=True,
                fold_harmonics=True):
    """Unified, principled Complex PCA (+ Riemannian Complex ICA).

    regime:
      - 'reactivation': event-aligned + active-centred + global_rms + active-phase shuffle null
                        + Riemannian Complex ICA + pairwise route-symmetry quotient.
      - 'waking':       uncentred + circular time-shift null + Riemannian Complex ICA.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    T_arr = np.asarray(T, dtype=float)
    if T_arr.ndim == 0:
        T_arr = np.full(times_ms.shape[0], float(T_arr))

    S = encode_arc(times_ms, T_arr, alpha=alpha)
    if regime == "reactivation":
        S = align_event_phase(S)
        X = active_centre_scale(S, scale=scale)
        C = lag_matrix(X)
        vals, vecs = eigh_desc(C)
        null = shuffle_null_active(X, rng, n_shuffles=n_shuffles)
        k = count_above_noise(vals, null, has_mean=False, quantile=quantile, max_components=max_components)
    elif regime == "waking":
        X = S
        C = lag_matrix(X)
        vals, vecs = eigh_desc(C)
        null = shuffle_null_circ_time(times_ms, T_arr, rng, alpha=alpha, n_shuffles=n_shuffles)
        k = count_above_noise(vals, null, has_mean=True, quantile=quantile, max_components=max_components)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    if k == 0:
        return {
            "k": 0, "eigenvalues": vals, "pca_components": vecs[:, :0],
            "patterns": np.zeros((X.shape[1], 0), complex),
            "scores": np.zeros((X.shape[0], 0), complex),
            "orders": [], "masks": [], "spans": [],
        }

    if ica and k > 1:
        Z = (X @ np.conj(vecs[:, :k])) / np.sqrt(np.maximum(vals[:k], 1e-12))[None, :]
        W = complex_ica_riemannian(Z, rng, power=power)
        P = vecs[:, :k] @ (np.sqrt(vals[:k])[:, None] * np.conj(W))
        norms = np.linalg.norm(P, axis=0)
        P = P / np.maximum(norms, 1e-12)
        order = np.argsort(-norms)
        P, Y = P[:, order], (Z @ W)[:, order]
        if regime == "reactivation" and reconcile:
            P = quotient_symmetry(P, Y, C, fold_harmonics=fold_harmonics)
    else:
        P = vecs[:, :k].copy()
        P /= np.maximum(np.linalg.norm(P, axis=0, keepdims=True), 1e-12)
        Y = X @ np.conj(P)

    T_read = float(np.median(T_arr))
    orders, masks, spans = [], [], []
    for j in range(P.shape[1]):
        t, inc, sp = readout_gap(P[:, j], T_read=T_read, alpha=alpha, member_frac=member_frac)
        orders.append(t)
        masks.append(inc)
        spans.append(sp)

    return {
        "k": k, "eigenvalues": vals, "pca_components": vecs[:, :k],
        "patterns": P, "scores": Y, "orders": orders, "masks": masks, "spans": spans,
    }
