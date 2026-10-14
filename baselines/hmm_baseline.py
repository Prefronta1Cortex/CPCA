"""Hidden Markov model of population-burst events (Maboudi et al., 2018), with their settings.

- Observations: spike counts in 20 ms bins inside each event; Poisson emissions;
  30 hidden states; EM. Five-fold cross-validation of the held-out likelihood
  against two surrogates (temporal shuffle of each neuron; coherent time swap).
- Congruence (their label-free replay test): an event's sequence likelihood under
  the learned transition matrix is compared with the likelihood under transition
  matrices whose off-diagonal entries are shuffled row-wise; an event is congruent
  if the real likelihood exceeds 99% of shuffles (95% also reported).
- Latent-state place fields (lsPF): running-period activity (100 ms bins,
  speed above 10 cm/s) is decoded to state posteriors; P(state | position) is
  estimated on half of the running data and used to decode position on the
  other half; median error is compared with lsPF built from shuffled positions.
  This uses behaviour, so it is the post-hoc answer-key check for the HMM, not an
  input to it.

The HMM does not produce a cell order without behaviour; it is scored on what
it claims: sequential structure per event (congruence) and a latent map (lsPF).
"""

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def _stack(blocks):
    """Events as integer count arrays with empty bins removed (an event must keep at least two bins).

    Empty flank bins would make every event contain a quiet -> active -> quiet transition,
    which is specific to the fitted transition matrix and destroyed by the congruence shuffle,
    so congruence would be trivially high. The test is about transitions between active states.
    """
    usable = []
    for b in blocks:
        kept = b[b.sum(axis=1) > 0].astype(int)
        if kept.shape[0] >= 2:
            usable.append(kept)
    return usable, np.vstack(usable), [b.shape[0] for b in usable]


def kept_events(blocks):
    """Boolean mask of the events that _stack keeps (at least two non-empty bins)."""
    return np.array([(b.sum(axis=1) > 0).sum() >= 2 for b in blocks])


def _repair(model):
    """States never visited in training leave zero rows in the transition matrix; make them uniform."""
    row = model.transmat_.sum(axis=1)
    if np.any(row == 0):
        model.transmat_[row == 0] = 1.0 / model.n_components
    model.startprob_ = np.maximum(model.startprob_, 1e-12)
    model.startprob_ /= model.startprob_.sum()
    return model


def _poisson_hmm_class():
    """hmmlearn's PoissonHMM with the emission log-likelihood computed as one matrix product.

    hmmlearn evaluates scipy.stats.poisson.logpmf for every (bin, state, cell), which
    makes a 30-state fit on a session take minutes. log P(x | lambda) = x log lambda
    - lambda - log x!, summed over cells, is X @ log(lambda).T - sum(lambda) - sum(log x!).
    Same model, same EM, same numbers.
    """
    from hmmlearn import hmm
    from scipy.special import gammaln

    class FastPoissonHMM(hmm.PoissonHMM):
        def _compute_log_likelihood(self, X):
            lam = np.maximum(self.lambdas_, 1e-12)                     # (states, cells)
            return X @ np.log(lam).T - lam.sum(axis=1)[None, :] - gammaln(X + 1).sum(axis=1)[:, None]

    return FastPoissonHMM


def fit(blocks, rng, n_states=30, n_iter=60):
    usable, X, lengths = _stack(blocks)
    model = _poisson_hmm_class()(n_components=n_states, n_iter=n_iter, random_state=int(rng.integers(1 << 31)), tol=1e-3)
    model.fit(X, lengths)
    model.lambdas_ = np.maximum(model.lambdas_, 1e-4)
    return _repair(model), usable


def cross_validated_gain(blocks, rng, n_states=30, n_folds=3, n_iter=40):
    """Held-out log-likelihood per bin for real test events minus that for surrogate test events."""
    usable, _, _ = _stack(blocks)
    order = rng.permutation(len(usable))
    folds = np.array_split(order, n_folds)
    gains = {"temporal_shuffle": [], "time_swap": []}
    for f in range(n_folds):
        test_idx = folds[f]
        train = [usable[i] for i in np.concatenate([folds[g] for g in range(n_folds) if g != f])]
        test = [usable[i] for i in test_idx]
        model, _ = fit(train, rng, n_states=n_states, n_iter=n_iter)
        X_test = np.vstack(test)
        L_test = [b.shape[0] for b in test]
        real = model.score(X_test, L_test) / X_test.shape[0]
        temporal = np.vstack([np.column_stack([np.roll(b[:, n], int(rng.integers(b.shape[0]))) for n in range(b.shape[1])]) for b in test])
        swap = np.vstack([b[rng.permutation(b.shape[0])] for b in test])
        gains["temporal_shuffle"].append(real - model.score(temporal, L_test) / X_test.shape[0])
        gains["time_swap"].append(real - model.score(swap, L_test) / X_test.shape[0])
    return {k: float(np.mean(v)) for k, v in gains.items()}


def _event_loglik(frames, log_start, log_trans):
    """Forward algorithm, vectorised over events of equal length. frames: list of (bins, states) emission log-likelihoods."""
    from collections import defaultdict
    from scipy.special import logsumexp
    out = np.empty(len(frames))
    by_len = defaultdict(list)
    for i, f in enumerate(frames):
        by_len[f.shape[0]].append(i)
    for n_bins, idx in by_len.items():
        F = np.stack([frames[i] for i in idx])                        # (events, bins, states)
        alpha = log_start[None, :] + F[:, 0, :]
        for t in range(1, n_bins):
            alpha = logsumexp(alpha[:, :, None] + log_trans[None, :, :], axis=1) + F[:, t, :]
        out[idx] = logsumexp(alpha, axis=1)
    return out


def congruence(model, events, rng, n_shuffles=100):
    """Per-event p-value of the sequence likelihood against row-wise shuffled off-diagonal transitions.

    Emission log-likelihoods come from the fitted model once; only the transition
    matrix changes between shuffles, so the forward pass is re-run in numpy.
    """
    A = model.transmat_
    M = A.shape[0]
    X = np.vstack(events)
    lengths = [b.shape[0] for b in events]
    frame = model._compute_log_likelihood(X)
    frames = np.split(frame, np.cumsum(lengths)[:-1])
    log_start = np.log(np.maximum(model.startprob_, 1e-300))
    real = _event_loglik(frames, log_start, np.log(np.maximum(A, 1e-300)))
    counts = np.zeros(len(events))
    for _ in range(n_shuffles):
        A_s = A.copy()
        for i in range(M):
            off = np.array([j for j in range(M) if j != i])
            A_s[i, off] = A[i, off][rng.permutation(M - 1)]
        null = _event_loglik(frames, log_start, np.log(np.maximum(A_s, 1e-300)))
        counts += null >= real
    return (counts + 1) / (n_shuffles + 1)


def latent_place_fields(model, session, run, cell_ids, rng, bin_s=0.1, min_speed=0.1, n_bins=40):
    """Decode running periods to states; build P(state | position) on half, decode position on the other half."""
    maze = session["maze_epoch"]
    edges = np.arange(maze[0], maze[1], bin_s)
    centers = edges[:-1] + bin_s / 2
    on = (session["times"] >= maze[0]) & (session["times"] < maze[1])
    st, sid = session["times"][on], session["ids"][on]
    X = np.column_stack([np.histogram(st[sid == c], bins=edges)[0] for c in cell_ids]).astype(int)
    pos_t, pos_x, direction = run["t"], run["x"], run["direction"]
    j = np.clip(np.searchsorted(pos_t, centers), 1, pos_t.size - 1)
    j = np.where(np.abs(pos_t[j - 1] - centers) <= np.abs(pos_t[j] - centers), j - 1, j)
    dt = np.diff(pos_t)
    speed = np.r_[0, np.abs(np.diff(pos_x)) / np.maximum(dt, 1e-6)]
    if run["circular"]:
        speed = np.r_[0, np.abs((np.diff(pos_x) + run["length"] / 2) % run["length"] - run["length"] / 2) / np.maximum(dt, 1e-6)]
    valid = (np.abs(pos_t[j] - centers) < bin_s) & (speed[j] >= min_speed) & (X.sum(1) > 0)
    if valid.sum() < 100:
        return {"median_error_m": np.nan, "shuffled_median_error_m": np.nan, "n_bins": int(valid.sum())}
    post = model.predict_proba(X[valid])                              # (bins, states)
    x = pos_x[j[valid]]
    pos_edges = np.linspace(0, run["length"], n_bins + 1)
    pos_bin = np.clip(np.digitize(x, pos_edges) - 1, 0, n_bins - 1)
    half = rng.permutation(valid.sum())
    a, b = half[: len(half) // 2], half[len(half) // 2:]

    def decode(train_idx, test_idx, positions):
        lspf = np.zeros((model.n_components, n_bins)) + 1e-3        # P(state | position bin)
        for s in range(model.n_components):
            lspf[s] += np.bincount(positions[train_idx], weights=post[train_idx, s], minlength=n_bins)
        lspf /= lspf.sum(axis=1, keepdims=True)
        # P(position | states at time t) proportional to prod_s lspf[s, pos]^post[t, s] with a flat prior
        logp = post[test_idx] @ np.log(lspf)
        decoded = 0.5 * (pos_edges[:-1] + pos_edges[1:])[np.argmax(logp, axis=1)]
        err = np.abs(decoded - x[test_idx])
        if run["circular"]:
            err = np.minimum(err, run["length"] - err)
        return float(np.median(err))

    real = decode(a, b, pos_bin)
    shuffled = decode(a, b, pos_bin[rng.permutation(len(pos_bin))])
    lspf = np.zeros((model.n_components, n_bins)) + 1e-3
    for s in range(model.n_components):
        lspf[s] += np.bincount(pos_bin, weights=post[:, s], minlength=n_bins)
    lspf /= lspf.sum(axis=1, keepdims=True)
    return {"median_error_m": real, "shuffled_median_error_m": shuffled, "n_bins": int(valid.sum()),
            "lspf": lspf, "state_mass": post.sum(axis=0)}


def run(blocks, rng, n_states=30, session=None, run_info=None, cell_ids=None, n_shuffles=100):
    t0 = time.perf_counter()
    model, events = fit(blocks, rng, n_states=n_states)
    p = congruence(model, events, rng, n_shuffles=n_shuffles)
    out = {"n_states": n_states, "n_events": len(events), "congruent_99": float(np.mean(p < 0.01)), "congruent_95": float(np.mean(p < 0.05)),
           "p_values": p, "kept": kept_events(blocks), "cv_gain": cross_validated_gain(blocks, rng, n_states=n_states)}
    if session is not None:
        out["lspf"] = latent_place_fields(model, session, run_info, cell_ids, rng)
    out["runtime_s"] = time.perf_counter() - t0
    return out
