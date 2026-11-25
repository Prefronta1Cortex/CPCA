# hc-11 waking laps: complex PCA against count PCA

Produced by `scripts/run_laps_hc11.py`. Design: `docs/laps_hc11.md`. Per session: fits (of 6, 3 splits x 2 folds) whose best output matches the held-out place fields (p < 0.05), and the median |ρ|.

| session | track | laps (train/test) | place cells, held-out key | complex PCA, uncentred | complex PCA, centred | count PCA | averaging reference (uses lap direction) |
|---|---|---|---|---|---|---|---|
| Achilles_10252013 | linear | 40/41 | 55/59 | 6/6 (0.65) | 6/6 (0.61) | 1/6 (0.34) | 6/6 (0.81) |
| Achilles_11012013 | circular | 19/20 | 0/62 | 6/6 (0.68) | 6/6 (0.62) | 4/6 (0.46) | 6/6 (0.64) |
| Buddy_06272013 | linear | 26/27 | 12/13 | 0/6 (0.56) | 0/6 (0.22) | 1/6 (0.64) | 2/6 (0.60) |
| Cicero_09012014 | linear | 33/34 | 20/19 | 5/6 (0.67) | 3/6 (0.56) | 2/6 (0.48) | 6/6 (0.80) |
| Cicero_09102014 | circular | 6/7 | 0/35 | 6/6 (0.68) | 2/6 (0.44) | 5/6 (0.63) | 6/6 (0.68) |
| Cicero_09172014 | linear | 9/11 | 22/24 | 4/6 (0.67) | 2/6 (0.45) | 0/6 (0.34) | 5/6 (0.74) |
| Gatsby_08022013 | linear | 35/36 | 20/22 | 4/6 (0.66) | 4/6 (0.61) | 1/6 (0.43) | 6/6 (0.77) |
| Gatsby_08282013 | circular | 25/26 | 0/25 | 6/6 (0.82) | 3/6 (0.55) | 1/6 (0.55) | 4/6 (0.56) |
| **all fits** | | | | 37/48 | 26/48 | 15/48 | 41/48 |
