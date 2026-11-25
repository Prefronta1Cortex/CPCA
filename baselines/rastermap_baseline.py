"""Rastermap (Stringer et al., 2024) on reactivation events, with the authors' package.

Settings of the authors' notebook for fewer than 200 neurons (rastermap_singleneurons.ipynb, which uses hc-11
itself): activity z-scored per neuron; n_clusters=None (single neurons are sorted, no clustering),
n_PCs=64 (fewer PCs than neurons: min(64, neurons - 1)), locality=0.1, time_lag_window=15,
grid_upsample=0. The notebook bins running at 200 ms; replay is about 20 times faster, so the events are
binned at 10 ms, which keeps the lag window (15 bins = 150 ms) at the time scale of an event. The events'
activity is concatenated, so Rastermap sees exactly the spikes the other methods see.

Output: one order of the cells (their position in the sorting); cells without activity are left out.
"""

import numpy as np

BIN_MS = 10.0


def fit(times_ms, T_ms, rng=None, bin_ms=BIN_MS):
    from baselines.count_assemblies import binned_events
    from rastermap import Rastermap
    from scipy.stats import zscore
    X = binned_events(times_ms, np.asarray(T_ms, float), bin_ms).T        # cells x time
    active = X.std(axis=1) > 0
    Z = np.zeros_like(X)
    Z[active] = zscore(X[active], axis=1)
    n_pcs = int(min(64, active.sum() - 1))
    model = Rastermap(n_clusters=None, n_PCs=n_pcs, locality=0.1, time_lag_window=15, grid_upsample=0, verbose=False).fit(Z[active])
    order = np.full(X.shape[0], np.nan)
    position = np.empty(active.sum())
    position[model.isort] = np.arange(active.sum())
    order[active] = position
    return {"orders": [order], "n_active": int(active.sum()), "n_pcs": n_pcs}
