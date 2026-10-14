"""seqNMF (Mackevicius et al., 2019), run the way its authors describe.

Algorithm: the `seqnmf` Python port of the FeeLab toolbox (convolutive NMF
with the cross-orthogonality penalty). Procedure, following the paper:

1. Lambda is chosen by the crossover rule: fit over a range of lambda, normalise
   the reconstruction cost and the cross-orthogonality cost to [0, 1], find the
   lambda at which they cross (lambda_0), and use 2 * lambda_0 (the value the
   paper used for its hippocampal example).
2. K is set larger than the expected number of sequences; the penalty drives
   unneeded factors to zero. Best of several restarts by final reconstruction cost.
3. Significance (the toolbox's test_significance): events are split into a
   training and a held-out set; for each factor the overlap of W with the
   held-out data is computed by convolution and its skewness compared with
   skewness after circularly shifting each neuron's held-out data independently;
   a factor is significant if its p-value is below 0.05 / K (Bonferroni).
4. A factor's cell order is the lag of each cell's peak loading in W (the
   quantity the toolbox sorts neurons by when it plots a factor); members are
   cells whose peak loading is at least 20% of the factor's maximum.

Deviation from the paper, stated: each neuron's row is divided by its standard
deviation before fitting, so that a few high-rate cells do not own every factor.
"""

import time
import warnings

import numpy as np
from scipy.stats import skew

warnings.filterwarnings("ignore")


def _concatenate(blocks, max_events, rng):
    idx = np.arange(len(blocks))
    if len(idx) > max_events:
        idx = np.sort(rng.choice(idx, max_events, replace=False))
    X = np.concatenate([blocks[i].T for i in idx], axis=1)          # (cells, time)
    return X, idx


def _normalise(X):
    scale = X.std(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return X / scale


def _reconstruct(W, H):
    N, K, L = W.shape
    T = H.shape[1]
    Xhat = np.zeros((N, T))
    for l in range(L):
        Xhat[:, l:] += W[:, :, l] @ H[:, :T - l]
    return Xhat


def _xortho_cost(W, H, X):
    """Cross-orthogonality cost of the paper: || (W^T (*) X) S H^T ||_1 off the diagonal."""
    N, K, L = W.shape
    T = X.shape[1]
    WTX = np.zeros((K, T))
    for l in range(L):
        WTX[:, :T - l] += W[:, :, l].T @ X[:, l:]
    S = np.ones(2 * L - 1)
    smoothed = np.array([np.convolve(row, S, mode="same") for row in WTX])
    C = smoothed @ H.T
    return float(np.abs(C - np.diag(np.diag(C))).sum())


def choose_lambda(X, K, L, rng, lambdas=np.logspace(-5, -1, 9), iters=30):
    """Crossover of normalised reconstruction and cross-orthogonality costs; returns (2*lambda_0, sweep table)."""
    import seqnmf
    recon, xortho = [], []
    for lam in lambdas:
        np.random.seed(int(rng.integers(1 << 31)))
        W, H, cost, loadings, power = seqnmf.seqnmf(X, K=K, L=L, Lambda=lam, max_iter=iters, plot_it=False)
        recon.append(float(np.sum((X - _reconstruct(W, H)) ** 2)))
        xortho.append(_xortho_cost(W, H, X))
    recon, xortho = np.array(recon), np.array(xortho)
    r = (recon - recon.min()) / max(recon.max() - recon.min(), 1e-12)
    x = (xortho - xortho.min()) / max(xortho.max() - xortho.min(), 1e-12)
    diff = r - x
    cross = np.flatnonzero(np.diff(np.sign(diff)) != 0)
    if cross.size:
        i = cross[0]
        # linear interpolation of the crossing in log-lambda
        t = diff[i] / (diff[i] - diff[i + 1]) if diff[i] != diff[i + 1] else 0.5
        lam0 = 10 ** (np.log10(lambdas[i]) + t * (np.log10(lambdas[i + 1]) - np.log10(lambdas[i])))
    else:
        lam0 = lambdas[int(np.argmin(np.abs(diff)))]
    return 2 * lam0, {"lambdas": lambdas.tolist(), "recon": r.tolist(), "xortho": x.tolist(), "lambda_0": float(lam0)}


def significance(W, X_test, rng, n_shuffles=100):
    """p-value per factor: skewness of the factor's overlap with held-out data vs circularly shifted data."""
    N, K, L = W.shape
    T = X_test.shape[1]

    def overlaps(Xt):
        out = np.zeros((K, T))
        for l in range(L):
            out[:, :T - l] += W[:, :, l].T @ Xt[:, l:]
        return out

    real = skew(overlaps(X_test), axis=1)
    null = np.empty((n_shuffles, K))
    for s in range(n_shuffles):
        shifted = np.array([np.roll(row, int(rng.integers(T))) for row in X_test])
        null[s] = skew(overlaps(shifted), axis=1)
    p = (np.sum(null >= real[None, :], axis=0) + 1) / (n_shuffles + 1)
    return p, real


def run(blocks, bin_s, rng, K=5, L=12, iters=100, restarts=3, max_events=1500, test_frac=0.25):
    """Fit seqNMF to zero-padded event blocks; return factors with orders, significance, and settings."""
    import seqnmf
    t0 = time.perf_counter()
    idx_all = np.arange(len(blocks))
    rng.shuffle(idx_all)
    n_test = int(round(test_frac * len(idx_all)))
    test_idx, train_idx = idx_all[:n_test], idx_all[n_test:]
    X_train, used = _concatenate([blocks[i] for i in train_idx], max_events, rng)
    X_test, _ = _concatenate([blocks[i] for i in test_idx], max_events // 3, rng)
    scale = X_train.std(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    X_train, X_test = X_train / scale, X_test / scale
    lam, sweep = choose_lambda(X_train, K, L, rng)
    best = None
    for _ in range(restarts):
        np.random.seed(int(rng.integers(1 << 31)))
        W, H, cost, loadings, power = seqnmf.seqnmf(X_train, K=K, L=L, Lambda=lam, max_iter=iters, plot_it=False)
        recon = float(np.sum((X_train - _reconstruct(W, H)) ** 2))
        if best is None or recon < best["recon"]:
            best = {"W": W, "H": H, "recon": recon, "loadings": np.asarray(loadings).ravel(), "power": float(power)}
    W = best["W"]
    p_values, skewness = significance(W, X_test, rng)
    lags_ms = np.arange(L) * bin_s * 1000
    factors = []
    for k in range(K):
        Wk = W[:, k, :]
        peak = Wk.max(axis=1)
        if peak.max() <= 0:
            continue
        members = peak >= 0.2 * peak.max()
        lag = np.where(members, lags_ms[np.argmax(Wk, axis=1)], np.nan)
        factors.append({"k": k, "members": members, "lag_ms": lag, "n_members": int(members.sum()),
                        "p": float(p_values[k]), "significant": bool(p_values[k] < 0.05 / K),
                        "loading": float(best["loadings"][k]) if k < best["loadings"].size else np.nan,
                        "lag_span_ms": float(np.nanmax(lag) - np.nanmin(lag)) if members.sum() > 1 else np.nan})
    return {"factors": factors, "lambda": float(lam), "sweep": sweep, "K": K, "L_ms": L * bin_s * 1000,
            "n_train_events": int(len(used)), "n_test_events": int(min(n_test, max_events // 3)),
            "runtime_s": time.perf_counter() - t0}
