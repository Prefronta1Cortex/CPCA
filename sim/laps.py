"""Planted running laps on a linear track: two running directions as two sequences (docs/unmixing.md, waking design).

Each lap is one full traversal in one direction. A place cell fires a burst at its field with probability
`reliability` per pass (Poisson number of spikes around the time the animal passes its field), plus sparse spikes
anywhere in the lap. Direction +1 runs from position 0 to 1, direction -1 from 1 to 0, so a cell at position x fires
at x * D in a +1 lap and at (1 - x) * D in a -1 lap. Cells may have a field in one direction only, or in both at
positions whose rank correlation is about `rho` (Gaussian copula). Cells without a field fire only sparsely.

The answer key is one position array per direction (NaN where the cell has no field in that direction), in track
coordinates. With `shuffle=True` each cell's spikes in each lap are circularly shifted by an independent random
amount (the no-sequence control).
"""

import numpy as np
from scipy.stats import norm


def generate(rng, n_place=60, n_other=20, both=0.7, rho=0.2, laps_per_direction=(30, 30), lap_s=(2.0, 3.5),
             spikes=10.0, field_sd=0.06, reliability=0.85, background=0.5, max_spikes=40, shuffle=False):
    n_cells = n_place + n_other
    n_both = int(round(both * n_place))
    n_one = (n_place - n_both) // 2
    maps = [np.full(n_cells, np.nan), np.full(n_cells, np.nan)]
    z1, z2 = rng.standard_normal(n_both), rng.standard_normal(n_both)
    r = 2 * np.sin(np.pi * rho / 6)                  # Pearson correlation of the copula for Spearman rho
    maps[0][:n_both] = norm.cdf(z1)
    maps[1][:n_both] = norm.cdf(r * z1 + np.sqrt(1 - r ** 2) * z2)
    maps[0][n_both:n_both + n_one] = rng.uniform(0, 1, n_one)
    maps[1][n_both + n_one:n_place] = rng.uniform(0, 1, n_place - n_both - n_one)
    perm = rng.permutation(n_cells)
    maps = [m[perm] for m in maps]
    dirs = np.r_[np.ones(laps_per_direction[0], int), -np.ones(laps_per_direction[1], int)]
    dirs = dirs[rng.permutation(dirs.size)]
    n = dirs.size
    T = rng.uniform(*lap_s, size=n) * 1000
    times = np.full((n, n_cells, max_spikes), np.nan)
    for i in range(n):
        m = maps[0] if dirs[i] > 0 else maps[1]
        for c in range(n_cells):
            spk = []
            if np.isfinite(m[c]) and rng.random() < reliability:
                x = m[c] if dirs[i] > 0 else 1 - m[c]
                k = rng.poisson(spikes)
                spk.append(rng.normal(x, field_sd, k) * T[i])
            k = rng.poisson(background)
            spk.append(rng.uniform(0, T[i], k))
            s = np.concatenate(spk) if spk else np.zeros(0)
            s = s[(s >= 0) & (s < T[i])]
            if shuffle and s.size:
                s = np.mod(s + rng.uniform(0, T[i]), T[i])
            s = np.sort(s)[:max_spikes]
            times[i, c, :s.size] = s
    return {"times": times, "T": T, "maps": maps, "direction": dirs, "n_cells": n_cells}
