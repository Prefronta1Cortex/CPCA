"""Exploratory clean-slate Complex PCA + Complex ICA experiments.

Lives entirely in exploration/ without altering any existing code in cpca/.
Tests fundamental design choices from first principles:
1. Phase arc alpha in (0, 2*pi] (half-circle pi vs wider arcs like 1.5*pi, 1.6*pi, 2*pi)
   and fixed T vs per-event T.
2. Active-centring: subtracting each cell's mean active phasor ONLY on active entries
   (leaving silent entries at 0), which algebraically annihilates arbitrary co-firing
   and per-cell latency in expectation.
3. Active-phase shuffle null: permuting each cell's active phasors only across the
   events where it fired (exact co-firing + latency null for eigenvalue cutoff).
4. Correct Complex ICA in whitened event space:
   - Fixed Bingham-Hyvarinen Complex FastICA (fixing both conjugation bugs in cpca/ica.py)
   - Exact Riemannian unitary optimization (no circularity assumption; works for both
     random-clock reactivations and aligned waking laps).
5. Conjugate-pair quotienting (merging (q, conj(q)) when reversal=True).
"""

import numpy as np
from scipy.linalg import expm


def encode_arc(times, T, alpha=np.pi, how="mean", t0=0.0):
    """Encode spike times as phasors exp(j * alpha * (t - t0) / T).

    times: (events, cells) or (events, cells, spikes), NaN where no spike.
    T: scalar or (events,) array.
    alpha: phase arc in radians (np.pi = half-circle, 2*np.pi = full circle).
    """
    times = np.asarray(times, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    if T_arr.ndim == 1:
        T_arr = T_arr.reshape((times.shape[0],) + (1,) * (times.ndim - 1))
    t0_arr = np.asarray(t0, dtype=float)
    if t0_arr.ndim == 1:
        t0_arr = t0_arr.reshape((times.shape[0],) + (1,) * (times.ndim - 1))
    finite = np.isfinite(times)
    with np.errstate(invalid="ignore", divide="ignore"):
        phasors = np.exp(1j * alpha * (times - t0_arr) / T_arr)
    phasors[~finite] = 0.0
    if times.ndim == 2:
        return phasors
    encoded = phasors.sum(axis=-1)
    if how == "mean":
        count = finite.sum(axis=-1)
        np.divide(encoded, count, out=encoded, where=count > 0)
    return encoded


def prepare_matrix(encoded, mode="uncentred", scale="none"):
    """Prepare event-by-cell complex matrix X.

    mode:
      - 'uncentred': raw encoded arrows
      - 'centred': subtract global column mean (including zeros) -- existing cpca convention
      - 'active_centred': subtract per-cell mean ONLY over active entries (zeros stay 0),
        annihilating arbitrary co-firing + per-cell latency in expectation.
    scale:
      - 'none': no column scaling
      - 'global_rms': divide each column by sqrt(mean(|x|^2)) over all events
      - 'active_rms': divide each column by sqrt(mean(|x|^2 | active))
      - 'sqrt_rms': divide each column by (global_rms)**0.5 (tempered scaling)
    """
    X = np.array(encoded, dtype=complex, copy=True)
    active = X != 0
    if mode == "centred":
        X = X - X.mean(axis=0, keepdims=True)
    elif mode == "active_centred":
        counts = active.sum(axis=0)
        mean_act = np.zeros(X.shape[1], dtype=complex)
        np.divide(X.sum(axis=0), counts, out=mean_act, where=counts > 0)
        X = np.where(active, X - mean_act[None, :], 0.0)
    elif mode != "uncentred":
        raise ValueError(f"Unknown mode: {mode}")

    if scale == "global_rms":
        rms = np.sqrt(np.mean(np.abs(X) ** 2, axis=0))
        np.divide(X, rms, out=X, where=rms > 0)
    elif scale == "active_rms":
        counts = active.sum(axis=0)
        ms = np.zeros(X.shape[1], dtype=float)
        np.divide(np.sum(np.abs(X) ** 2, axis=0), counts, out=ms, where=counts > 0)
        rms = np.sqrt(ms)
        np.divide(X, rms, out=X, where=rms > 0)
    elif scale == "sqrt_rms":
        rms = np.sqrt(np.mean(np.abs(X) ** 2, axis=0))
        s = np.sqrt(rms)
        np.divide(X, s, out=X, where=s > 0)
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
    """Active-phase shuffle null: permute each cell's nonzero entries ONLY among its active events.

    Preserves the exact binary co-firing matrix (who fires with whom in every event) AND each
    cell's marginal phase/latency distribution; destroys only within-event timing coordination.
    """
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


def shuffle_null_column(X, rng, n_shuffles=100):
    """Standard column-wise shuffle null (same as cpca.decompose.shuffle_null)."""
    M, N = X.shape
    rows = np.broadcast_to(np.arange(M)[:, None], (M, N))
    null = np.empty((n_shuffles, N), dtype=float)
    for s in range(n_shuffles):
        P = np.take_along_axis(X, rng.permuted(rows, axis=0), axis=0)
        null[s] = eigh_desc(lag_matrix(P))[0]
    return null


def shuffle_null_circ_phase(X, rng, n_shuffles=100):
    """Multiply every nonzero entry by a random unit phase exp(j * U(0, 2*pi)).

    Useful for waking data where every event has the same sequence: preserves co-firing
    and spike-count magnitudes, destroys sequence phases.
    """
    M, N = X.shape
    nz = X != 0
    null = np.empty((n_shuffles, N), dtype=float)
    for s in range(n_shuffles):
        P = np.where(nz, X * np.exp(1j * rng.uniform(0, 2 * np.pi, size=(M, N))), 0.0)
        null[s] = eigh_desc(lag_matrix(P))[0]
    return null


def count_above_null(eigenvalues, null, structural=0, quantile=95.0):
    ceiling = float(np.percentile(null[:, structural], quantile))
    k = structural
    while k < eigenvalues.size and eigenvalues[k] > ceiling:
        k += 1
    return min(k, eigenvalues.size), ceiling


def _sym_orth(W):
    G = W.conj().T @ W
    vals, vecs = np.linalg.eigh(G)
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.conj().T
    return W @ inv_sqrt


def complex_ica_riemannian(Z, rng, contrast="log", a=0.1, n_init=8, max_iter=300, tol=1e-7):
    """Exact Riemannian unitary optimization for Complex ICA on whitened scores Z (M x K).

    Makes ZERO circularity assumption: works identically for aligned waking events
    (non-circular sources) and random-clock reactivations (circular sources).
    Finds unitary W minimizing:
      contrast='log':  sum_j mean_i log(a + |y_ij|^2)       (robust sparsity/bimodality)
      contrast='sqrt': sum_j mean_i sqrt(a + |y_ij|^2)      (smoothed L1)
      contrast='kurt': -sum_j mean_i |y_ij|^4               (4th moment / event quartimax)
    """
    M, K = Z.shape
    if K <= 1:
        return np.eye(K, dtype=complex)

    def obj_and_grad(W):
        Y = Z @ W
        U = np.abs(Y) ** 2
        if contrast == "log":
            loss = float(np.sum(np.mean(np.log(a + U), axis=0)))
            g = 1.0 / (a + U)
        elif contrast == "sqrt":
            s = np.sqrt(a + U)
            loss = float(np.sum(np.mean(s, axis=0)))
            g = 0.5 / s
        elif contrast == "kurt":
            loss = -float(np.sum(np.mean(U ** 2, axis=0)))
            g = -2.0 * U
        else:
            raise ValueError(f"Unknown contrast: {contrast}")
        # Exact Euclidean gradient w.r.t. W*: (1/M) Z^H (Y * g)
        Egrad = (Z.conj().T @ (Y * g)) / M
        return loss, Egrad

    best_loss, best_W = np.inf, np.eye(K, dtype=complex)
    # Include identity as one deterministic init, plus random unitary inits
    inits = [np.eye(K, dtype=complex)]
    for _ in range(n_init - 1):
        R = rng.normal(size=(K, K)) + 1j * rng.normal(size=(K, K))
        inits.append(_sym_orth(R))

    for W0 in inits:
        W = W0.copy()
        loss, Egrad = obj_and_grad(W)
        step = 1.0
        for _ in range(max_iter):
            # Skew-Hermitian Riemannian gradient in Lie algebra u(K):
            # directional derivative of L along W*A (A skew-Hermitian) is 2*Re Tr(Egrad^H W A)
            # = Re Tr((W^H Egrad - Egrad^H W)^H A) -> steepest descent A = -(W^H Egrad - Egrad^H W)
            G = W.conj().T @ Egrad
            A = -(G - G.conj().T)
            grad_norm_sq = float(np.real(np.sum(A.conj() * A)))  # ||A||_F^2
            if grad_norm_sq < tol ** 2:
                break
            # Armijo backtracking line search along geodesic W exp(step * A)
            accepted = False
            for _ls in range(25):
                W_try = W @ expm(step * A)
                loss_try, Egrad_try = obj_and_grad(W_try)
                if loss_try <= loss - 1e-4 * step * grad_norm_sq:
                    W, loss, Egrad = W_try, loss_try, Egrad_try
                    step = min(step * 1.5, 4.0)
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
        if loss < best_loss:
            best_loss, best_W = loss, W
    return best_W


def unmix_patterns(X, eigenvalues, eigenvectors, rng, **ica_kwargs):
    """Whiten X in the kept PCA subspace, run Riemannian Complex ICA, reconstruct mixing patterns.

    Because lag_matrix(X) = X^T X* / M = V Lambda V^H:
      X ~ Z Lambda^(1/2) V^T  with  Z = X V* Lambda^(-1/2),  Z^H Z / M = I.
    If Y = Z W (W unitary), then:
      X ~ Y (V Lambda^(1/2) W*)^T  ->  true mixing patterns are P = V Lambda^(1/2) W*.
    """
    K = eigenvectors.shape[1]
    if K == 0:
        return eigenvectors.copy(), np.zeros((X.shape[0], 0), dtype=complex), np.zeros((0, 0), dtype=complex)
    if K == 1:
        p = eigenvectors.copy()
        p /= max(np.linalg.norm(p), 1e-12)
        return p, X @ np.conj(p), np.ones((1, 1), dtype=complex)
    Z = (X @ np.conj(eigenvectors)) / np.sqrt(np.maximum(eigenvalues, 1e-12))[None, :]
    W = complex_ica_riemannian(Z, rng, **ica_kwargs)
    # EXACT mixing patterns (note np.conj(W)!):
    P = eigenvectors @ (np.sqrt(eigenvalues)[:, None] * np.conj(W))
    norms = np.linalg.norm(P, axis=0)
    P = P / np.maximum(norms, 1e-12)
    Y = Z @ W
    # Order by variance explained in X (i.e. norm^2 of raw mixing column before unit normalization)
    order = np.argsort(-norms)
    return P[:, order], Y[:, order], W[:, order]


def merge_conjugate_pairs(P, Y=None, sim_thresh=0.5):
    """Quotient patterns by time-reversal symmetry (q ~ conj(q)).

    When a route plays both forward and backward, Complex ICA recovers q and conj(q) as
    two separate sparse components. Merging pairs with |<p_a, conj(p_b)>| >= sim_thresh
    (or near-duplicates |<p_a, p_b>| >= sim_thresh) combines them into one sequence template
    with higher SNR: q = w_a p_a + w_b exp(j phi) conj(p_b).
    """
    K = P.shape[1]
    if K == 0:
        return P, []
    P_norm = P / np.maximum(np.linalg.norm(P, axis=0, keepdims=True), 1e-12)
    used = np.zeros(K, dtype=bool)
    merged = []
    groups = []
    for i in range(K):
        if used[i]:
            continue
        used[i] = True
        acc = P_norm[:, i].copy()
        grp = [(i, "fwd")]
        for j in range(i + 1, K):
            if used[j]:
                continue
            s_dir = np.vdot(P_norm[:, j], acc / max(np.linalg.norm(acc), 1e-12))
            s_con = np.vdot(np.conj(P_norm[:, j]), acc / max(np.linalg.norm(acc), 1e-12))
            if max(abs(s_dir), abs(s_con)) >= sim_thresh:
                used[j] = True
                if abs(s_con) >= abs(s_dir):
                    acc = acc + np.conj(P_norm[:, j]) * np.exp(1j * np.angle(s_con))
                    grp.append((j, "rev"))
                else:
                    acc = acc + P_norm[:, j] * np.exp(1j * np.angle(s_dir))
                    grp.append((j, "fwd"))
        acc /= max(np.linalg.norm(acc), 1e-12)
        merged.append(acc)
        groups.append(grp)
    return np.column_stack(merged), groups


def readout_gap(component, T_read=1.0, alpha=np.pi, member_rule="frac", member_frac=0.2):
    """Read out cell order from a complex pattern by always cutting the widest empty gap on the circle.

    member_rule:
      - 'frac': |w| >= member_frac * max(|w|)
      - 'otsu': two-group Otsu split of |w|
      - 'z2': |w| >= mean(|w|) + 2 * std(|w|)
    """
    q = np.asarray(component, dtype=complex)
    w = np.abs(q)
    if w.max() <= 0:
        return np.full(q.shape, np.nan), np.zeros(q.shape, dtype=bool), 0.0
    if member_rule == "frac":
        inc = w >= member_frac * w.max()
    elif member_rule == "otsu":
        s = np.sort(w)
        c1, c2 = np.cumsum(s), np.cumsum(s ** 2)
        best, cut = np.inf, s[-1]
        for i in range(1, s.size):
            lo = c2[i - 1] - c1[i - 1] ** 2 / i
            hi = (c2[-1] - c2[i - 1]) - (c1[-1] - c1[i - 1]) ** 2 / (s.size - i)
            if lo + hi < best:
                best, cut = lo + hi, s[i]
        inc = w >= cut
    elif member_rule == "z2":
        inc = w >= w.mean() + 2.0 * w.std()
    else:
        raise ValueError(f"Unknown member_rule: {member_rule}")

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


def fit_clean(times_ms, T, rng, alpha=np.pi, mode="uncentred", scale="none",
              null_kind="column", n_shuffles=50, ica=True, contrast="log", a=0.1,
              merge_conj=False, merge_thresh=0.5, max_components=20,
              member_rule="frac", member_frac=0.2, fixed_T=None):
    """Unified, clean cPCA (+ Riemannian Complex ICA) fit."""
    T_use = float(fixed_T) if fixed_T is not None else T
    encoded = encode_arc(times_ms, T_use, alpha=alpha)
    X = prepare_matrix(encoded, mode=mode, scale=scale)
    C = lag_matrix(X)
    vals, vecs = eigh_desc(C)

    if null_kind == "active":
        null = shuffle_null_active(X, rng, n_shuffles=n_shuffles)
        structural = 0
    elif null_kind == "column":
        null = shuffle_null_column(X, rng, n_shuffles=n_shuffles)
        structural = 0 if mode != "uncentred" else 1
    elif null_kind == "circ_phase":
        null = shuffle_null_circ_phase(X, rng, n_shuffles=n_shuffles)
        structural = 0
    else:
        raise ValueError(f"Unknown null_kind: {null_kind}")

    k, bound = count_above_null(vals, null, structural=structural)
    k = int(min(max(k, 0), max_components))
    kept_vals, kept_vecs = vals[:k], vecs[:, :k]

    if ica and k > 1:
        patterns, scores, W = unmix_patterns(X, kept_vals, kept_vecs, rng, contrast=contrast, a=a)
    else:
        patterns = kept_vecs.copy()
        scores = X @ np.conj(patterns) if k else np.zeros((X.shape[0], 0), dtype=complex)
        W = np.eye(k, dtype=complex)

    groups = [[(j, "fwd")] for j in range(patterns.shape[1])]
    if merge_conj and patterns.shape[1] > 1:
        patterns, groups = merge_conjugate_pairs(patterns, scores, sim_thresh=merge_thresh)

    T_read = float(np.median(T)) if np.ndim(T) else float(T)
    orders, masks, spans = [], [], []
    for j in range(patterns.shape[1]):
        t, inc, sp = readout_gap(patterns[:, j], T_read=T_read, alpha=alpha,
                                 member_rule=member_rule, member_frac=member_frac)
        orders.append(t)
        masks.append(inc)
        spans.append(sp)

    return {
        "encoded": encoded,
        "X": X,
        "eigenvalues": vals,
        "bound": bound,
        "n_kept": k,
        "pca_components": kept_vecs,
        "patterns": patterns,
        "scores": scores,
        "W": W,
        "groups": groups,
        "orders": orders,
        "masks": masks,
        "spans": spans,
        "T_read": T_read,
    }
