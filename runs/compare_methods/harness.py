"""Label-free comparison harness.

Every method here sees spikes only. The waking place map is used once, at
the end, to score the cell order each method produced. Nothing is selected,
tuned, or filtered with the map.

Methods
- cPCA single template: the current pipeline (C, C - C0, C on events with
  positive timing excess), 100 ms window on the population peak.
- cPCA mixture: cpca.cluster.fit_mixture with K templates; K chosen by
  split-half stability.
- seqNMF (Mackevicius et al. 2019): convolutive NMF on binned event blocks,
  per-neuron normalised; a factor's cell order is the time centre of mass of
  its W column.
- Poisson HMM (Maboudi et al. 2018): fit on 20 ms event bins; number of
  states chosen by held-out log-likelihood; states seriated label-free by the
  Fiedler vector of the symmetrised transition graph; a cell's position is
  its rate-weighted mean state rank.
- Count PCA/ICA (Lopes-dos-Santos et al. 2013): assemblies from binned
  counts; gives membership, not order, and is reported as such.

Scoring: best |Spearman| between a method's cell order and place-field
centre, over the two running directions and over the method's templates or
factors; the null is the 95th percentile of the same maximum after permuting
cell identities, so methods with more templates are not favoured.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts", ROOT / "runs" / "hc11_post", ROOT / "runs" / "hc11_t100"):
    sys.path.insert(0, str(p))

from analyze_hc11_window import MAT_DIR, load_session  # noqa: E402
from post_timing import WINDOW_MS, load_position, post_spikes, running_track, templates, wide_bursts  # noqa: E402
from run_cpca import build_times  # noqa: E402
from cpca import encode, readout  # noqa: E402
from cpca.cluster import fit_mixture, split_half_stability  # noqa: E402

NBINS = 40
MIN_RANK_CELLS = 8
MIN_WEIGHT = 0.1
RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = Path(__file__).resolve().parent / "figures"


# ----------------------------------------------------------------------------- real data
def session_data(path):
    sess = load_session(path)
    by_cell, post_nrem, spike_times = post_spikes(sess)
    cell_ids = np.array(sorted(by_cell))
    bursts, peaks = wide_bursts(spike_times, post_nrem)
    pos_t, pos_x = load_position(path)
    circular = "ircular" in sess["maze"]
    pos_t, pos_x, direction, length = running_track(pos_t, pos_x, circular)
    centres = place_centres(sess, cell_ids, pos_t, pos_x, direction, length)
    return {
        "name": path.name.split("_sess")[0], "maze": sess["maze"].strip(), "sess": sess,
        "by_cell": by_cell, "cell_ids": cell_ids, "bursts": bursts, "peaks": peaks,
        "length": length, "circular": circular, "centres": centres,
    }


def place_centres(sess, cell_ids, pos_t, pos_x, direction, length, min_spikes=40, min_peak=3.0):
    """Place-field centre per cell and running direction. This is the answer key."""
    maze = sess["maze_epoch"]
    on = (sess["times"] >= maze[0]) & (sess["times"] < maze[1])
    st, sid = sess["times"][on], sess["ids"][on]
    j = np.clip(np.searchsorted(pos_t, st), 1, pos_t.size - 1)
    j = np.where(np.abs(pos_t[j - 1] - st) <= np.abs(pos_t[j] - st), j - 1, j)
    keep = (np.abs(pos_t[j] - st) < 0.04) & (direction[j] != 0)
    sid, sx, sdir = sid[keep], pos_x[j[keep]], direction[j[keep]]
    dt = float(np.median(np.diff(pos_t)))
    edges = np.linspace(0, length, NBINS + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    out = {}
    for sign in (1.0, -1.0):
        occ = np.histogram(pos_x[direction == sign], bins=edges)[0] * dt
        valid = occ > 0.2
        centres = np.full(len(cell_ids), np.nan)
        for k, c in enumerate(cell_ids):
            m = (sid == c) & (sdir == sign)
            if m.sum() < min_spikes:
                continue
            rate = np.zeros(NBINS)
            rate[valid] = np.histogram(sx[m], bins=edges)[0][valid] / occ[valid]
            rate = gaussian_filter1d(rate, 1.0, mode="nearest")
            if rate.max() >= min_peak:
                centres[k] = mid[np.argmax(rate)]
        out[sign] = centres
    return out


def event_bins(d, bin_s, pad=0):
    """Binned counts per event (list of (bins, cells) arrays), full event window, optional zero padding."""
    cells, by_cell = d["cell_ids"], d["by_cell"]
    blocks = []
    for a, b in d["bursts"]:
        nb = max(int(np.ceil((b - a) / bin_s)), 1)
        edges = a + np.arange(nb + 1) * bin_s
        X = np.zeros((nb + pad, len(cells)))
        for k, c in enumerate(cells):
            s = by_cell[c]
            lo, hi = np.searchsorted(s, a), np.searchsorted(s, b)
            if hi > lo:
                X[:nb, k] = np.histogram(s[lo:hi], bins=edges)[0]
        blocks.append(X)
    return blocks


def peak_window_times(d):
    peaks = d["peaks"]
    bursts = np.column_stack([peaks - WINDOW_MS / 2000, peaks + WINDOW_MS / 2000])
    times, _, _ = build_times(d["by_cell"], bursts, peaks, d["cell_ids"])
    return times


# ----------------------------------------------------------------------------- scoring
def order_score(times, centres, circular, length, nshift=12):
    m = np.isfinite(times) & np.isfinite(centres)
    if m.sum() < MIN_RANK_CELLS:
        return np.nan
    if not circular:
        return float(spearmanr(times[m], centres[m]).statistic)
    best = 0.0
    for s in range(nshift):
        moved = (centres[m] + s * length / nshift) % length
        r = spearmanr(times[m], moved).statistic
        if np.isfinite(r) and abs(r) > abs(best):
            best = float(r)
    return best


def evaluate_orders(orders, d, rng, n_null=200):
    """orders: list of cell-time vectors (one per template/factor).

    Returns best |rho| over templates and directions, the matched null's 95th
    percentile, and the per-template per-direction values.
    """
    orders = [o for o in orders if np.isfinite(o).sum() >= MIN_RANK_CELLS]
    if not orders:
        return {"best": np.nan, "null95": np.nan, "table": [], "n_templates": 0}
    directions = list(d["centres"].keys())

    def best_of(perm=None):
        vals = []
        for o in orders:
            oo = o if perm is None else o[perm]
            vals.append([abs(order_score(oo, d["centres"][k], d["circular"], d["length"])) for k in directions])
        vals = np.array(vals, dtype=float)
        return (np.nanmax(vals) if np.isfinite(vals).any() else np.nan), vals

    best, table = best_of()
    null = np.array([best_of(rng.permutation(len(orders[0])))[0] for _ in range(n_null)])
    return {"best": float(best), "null95": float(np.nanpercentile(null, 95)), "null_mean": float(np.nanmean(null)),
            "table": table.tolist(), "n_templates": len(orders)}


def template_to_times(component, T):
    if not np.isfinite(component).all():
        return np.full(component.shape, np.nan)
    _, t, _, inc, _ = readout(component, T, min_weight=MIN_WEIGHT)
    return np.where(inc, t, np.nan)


# ----------------------------------------------------------------------------- methods on encoded arrows
def method_cpca_single(S, active, T):
    t0 = time.perf_counter()
    from_c, from_delta, from_sel, n_kept, n_ev = templates(S, active)
    out = {"C": [template_to_times(from_c, T)], "C-C0": [template_to_times(from_delta, T)],
           "selected": [template_to_times(from_sel, T)], "kept": n_kept, "runtime_s": time.perf_counter() - t0}
    return out


def method_cpca_mixture(S, T, rng, Ks=(2, 3, 4, 6), n_init=8):
    t0 = time.perf_counter()
    out = {"by_K": {}, "runtime_s": None}
    for K in Ks:
        try:
            fit = fit_mixture(S, K, rng, n_init=n_init)
            stab = split_half_stability(S, K, rng, n_init=max(2, n_init // 2))
        except ValueError:
            continue
        out["by_K"][K] = {
            "orders_delta": [template_to_times(fit["templates_delta"][:, k], T) for k in range(K)],
            "orders_c": [template_to_times(fit["templates_c"][:, k], T) for k in range(K)],
            "sizes": fit["sizes"].tolist(), "objective": fit["objective"], "stability": stab,
        }
    if out["by_K"]:
        out["K_stability"] = max(out["by_K"], key=lambda K: out["by_K"][K]["stability"])
    out["runtime_s"] = time.perf_counter() - t0
    return out


# ----------------------------------------------------------------------------- baselines on binned counts
def method_seqnmf(blocks, bin_s, rng, K=3, L=12, lam=0.001, iters=80, restarts=3, max_events=1500):
    """blocks: list of (bins + L, cells) zero-padded count arrays."""
    import seqnmf
    t0 = time.perf_counter()
    idx = np.arange(len(blocks))
    if len(idx) > max_events:
        idx = np.sort(rng.choice(idx, max_events, replace=False))
    X = np.concatenate([blocks[i].T for i in idx], axis=1)          # (cells, time)
    scale = X.std(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    X = X / scale
    best = None
    for r in range(restarts):
        np.random.seed(int(rng.integers(1 << 31)))
        W, H, cost, loadings, power = seqnmf.seqnmf(X, K=K, L=L, Lambda=lam, max_iter=iters, plot_it=False)
        final = float(np.asarray(cost).ravel()[-1])
        if best is None or final < best["cost"]:
            best = {"W": W, "cost": final, "loadings": np.asarray(loadings).ravel(), "power": float(power)}
    W = best["W"]
    lags = np.arange(W.shape[2]) * bin_s * 1000
    orders, spans, n_cells = [], [], []
    for k in range(W.shape[1]):
        Wk = W[:, k, :]
        peak = Wk.max(axis=1)
        if peak.max() <= 0:
            continue
        inc = peak >= 0.2 * peak.max()
        com = (Wk * lags[None, :]).sum(axis=1) / np.maximum(Wk.sum(axis=1), 1e-12)
        t = np.where(inc, com, np.nan)
        orders.append(t)
        spans.append(float(np.nanmax(t) - np.nanmin(t)) if inc.sum() > 1 else np.nan)
        n_cells.append(int(inc.sum()))
    return {"orders": orders, "spans_ms": spans, "n_cells": n_cells, "loadings": best["loadings"].tolist(),
            "power": best["power"], "runtime_s": time.perf_counter() - t0}


def method_hmm(blocks, rng, n_states_options=(20, 35), n_iter=50):
    """blocks: list of (bins, cells) count arrays (unpadded, 20 ms)."""
    from hmmlearn import hmm
    t0 = time.perf_counter()
    usable = [b for b in blocks if b.shape[0] >= 2 and b.sum() > 0]
    perm = rng.permutation(len(usable))
    n_test = max(len(usable) // 5, 1)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    train = np.vstack([usable[i] for i in train_idx]).astype(int)
    train_len = [usable[i].shape[0] for i in train_idx]
    test = np.vstack([usable[i] for i in test_idx]).astype(int)
    test_len = [usable[i].shape[0] for i in test_idx]
    best = None
    for M in n_states_options:
        model = hmm.PoissonHMM(n_components=M, n_iter=n_iter, random_state=0, tol=1e-3)
        model.fit(train, train_len)
        model.lambdas_ = np.maximum(model.lambdas_, 1e-4)
        ll = model.score(test, test_len) / test.shape[0]
        if best is None or ll > best["ll"]:
            best = {"M": M, "ll": ll, "model": model}
    model = best["model"]
    A, lam = model.transmat_, model.lambdas_                        # (M, M), (M, N)
    Wg = (A + A.T) / 2
    np.fill_diagonal(Wg, 0)
    Lap = np.diag(Wg.sum(1)) - Wg
    _, evecs = np.linalg.eigh(Lap)
    rank = np.argsort(np.argsort(evecs[:, 1])).astype(float)
    mass = lam.sum(0)
    cell_pos = (lam * rank[:, None]).sum(0) / np.maximum(mass, 1e-12)
    cell_pos[mass < np.percentile(mass, 10)] = np.nan            # cells the model barely uses
    # how chain-like is the transition graph?  (ratio of 2nd to 3rd Laplacian eigenvalue, small = chain)
    evals = np.linalg.eigvalsh(Lap)
    return {"orders": [cell_pos], "n_states": best["M"], "heldout_ll_per_bin": float(best["ll"]),
            "chain_ratio": float(evals[1] / max(evals[2], 1e-12)), "runtime_s": time.perf_counter() - t0}


def method_count_ica(blocks, d, rng):
    from sklearn.decomposition import FastICA
    t0 = time.perf_counter()
    counts = np.array([b.sum(0) for b in blocks])
    keep = counts.std(0) > 0
    Z = (counts[:, keep] - counts[:, keep].mean(0)) / counts[:, keep].std(0)
    M, N = Z.shape
    ev, V = np.linalg.eigh(Z.T @ Z / M)
    ev, V = ev[::-1], V[:, ::-1]
    mp = (1 + np.sqrt(N / M)) ** 2
    n_sig = int((ev > mp).sum())
    out = {"n_assemblies": n_sig, "runtime_s": None}
    if n_sig >= 1:
        ica = FastICA(n_components=n_sig, random_state=0, max_iter=2000, tol=1e-4).fit(Z @ V[:, :n_sig])
        W = (V[:, :n_sig] @ ica.components_.T).T
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        W *= np.sign(W[np.arange(n_sig), np.argmax(np.abs(W), 1)])[:, None]
        full = np.full((n_sig, len(keep)), np.nan)
        full[:, keep] = W
        cen = np.nanmean(np.column_stack([d["centres"][k] for k in d["centres"]]), axis=1) if d is not None else None
        if cen is not None:
            spread, null = [], []
            cand = cen[np.isfinite(cen)]
            for a in range(n_sig):
                top = np.argsort(np.nan_to_num(full[a], nan=-1))[-6:]
                c = cen[top]
                c = c[np.isfinite(c)]
                if c.size >= 3:
                    pair = np.triu_indices(c.size, 1)
                    spread.append(np.mean(np.abs(c[:, None] - c[None, :])[pair]))
                    null.append(np.mean([np.mean(np.abs(x[:, None] - x[None, :])[pair]) for x in (rng.choice(cand, c.size, replace=False) for _ in range(200))]))
            out["member_field_spread"] = float(np.mean(spread)) if spread else np.nan
            out["member_field_spread_null"] = float(np.mean(null)) if null else np.nan
        out["weights"] = full
    out["runtime_s"] = time.perf_counter() - t0
    return out


# ----------------------------------------------------------------------------- synthetic data with a planted answer
def synthetic_session(rng, n_cells=80, n_events=3000, participation=0.15, replay_frac=0.4, reverse_frac=0.5,
                      jitter_ms=8.0, span_ms=80.0, background_rate=0.0, T=100.0):
    """Events of length T ms. Replay events play the planted order forward or backward, compressed to span_ms.
    Other events are synchronous volleys at the centre. Returns a dict shaped like session_data."""
    order = np.linspace((T - span_ms) / 2, (T + span_ms) / 2, n_cells)
    times = np.full((n_events, n_cells, 4), np.nan)
    label = np.zeros(n_events, dtype=int)
    for i in range(n_events):
        fire = rng.random(n_cells) < participation
        if rng.random() < replay_frac:
            rev = rng.random() < reverse_frac
            label[i] = 1 + int(rev)
            base = order[::-1] if rev else order
            times[i, fire, 0] = base[fire] + rng.normal(0, jitter_ms, size=int(fire.sum()))
        else:
            times[i, fire, 0] = rng.normal(T / 2, jitter_ms, size=int(fire.sum()))
        if background_rate > 0:
            extra = rng.random(n_cells) < background_rate
            times[i, extra, 1] = rng.uniform(0, T, size=int(extra.sum()))
    times = np.clip(times, 0, T - 1e-6)
    # a fake "place map": the planted order is the answer key for both directions
    centres = {1.0: order.copy(), -1.0: order.copy()}
    cell_ids = np.arange(n_cells)
    return {"name": "synthetic", "times": times, "label": label, "T": T, "centres": centres, "circular": False,
            "length": float(T), "cell_ids": cell_ids}


def synthetic_blocks(times, T, bin_s, pad=0):
    n_events, n_cells, _ = times.shape
    nb = int(np.ceil(T / (bin_s * 1000)))
    blocks = []
    for i in range(n_events):
        X = np.zeros((nb + pad, n_cells))
        for k in range(n_cells):
            t = times[i, k]
            t = t[np.isfinite(t)]
            if t.size:
                X[:nb, k] = np.histogram(t, bins=np.linspace(0, T, nb + 1))[0]
        blocks.append(X)
    return blocks
