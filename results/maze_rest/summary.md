# hc-11: awake rest on the maze

Produced by `scripts/run_maze_rest.py` and `scripts/summarize_maze_rest.py`. Design: `docs/maze_rest.md`.

## Events and replay content

| session | rest (min) | events | place cells active per event (median) | clear replays (tested) | PRE control: clear replays |
|---|---|---|---|---|---|
| Achilles_10252013 | 12 | 373 | 18 | 18% (244) | 4% |
| Achilles_11012013 | 9 | 321 | 12 | 11% (182) | 1% |
| Buddy_06272013 | 14 | 504 | 3 | 9% (111) | 3% |
| Cicero_09012014 | 49 | 1546 | 7 | 5% (889) | 1% |
| Cicero_09102014 | 6 | 184 | 8 | 4% (105) | 1% |
| Cicero_09172014 | 15 | 432 | 10 | 6% (279) | 4% |
| Gatsby_08022013 | 15 | 474 | 10 | 10% (359) | 7% |
| Gatsby_08282013 | 15 | 531 | 7 | 2% (297) | 2% |

## Order and topology (p; planned analyses)

| session | epoch | complex PCA centred, best component | uncentred | seriation, complex PCA | seriation, count PCA | topology, complex PCA | topology, count PCA |
|---|---|---|---|---|---|---|---|
| Achilles_10252013 | MAZE rest | 0.001 | 0.010 | 0.009 | 0.015 | 0.000 | 0.000 |
| Achilles_10252013 | PRE control | 0.577 | 0.632 | 0.488 | 0.615 | 0.828 | 0.949 |
| Achilles_11012013 | MAZE rest | 0.002 | 0.006 | 0.001 | 0.005 | 0.000 | 0.000 |
| Achilles_11012013 | PRE control | 0.267 | 0.995 | 0.850 | 0.902 | 0.058 | 0.329 |
| Buddy_06272013 | MAZE rest | 0.334 | 0.594 | 0.004 | 0.010 | 0.244 | 0.367 |
| Buddy_06272013 | PRE control | 0.247 | 0.468 | 0.346 | 0.139 | 0.008 | 0.193 |
| Cicero_09012014 | MAZE rest | 0.814 | 0.305 | 0.520 | 0.404 | 0.000 | 0.008 |
| Cicero_09012014 | PRE control | 0.990 | 0.467 | 0.563 | 0.953 | 0.577 | 0.521 |
| Cicero_09102014 | MAZE rest | 0.034 | 0.888 | 0.308 | 0.010 | 0.000 | 0.001 |
| Cicero_09102014 | PRE control | 0.479 | 0.416 | 0.702 | 0.166 | 0.541 | 0.431 |
| Cicero_09172014 | MAZE rest | 0.028 | 0.021 | 0.364 | 0.244 | 0.000 | 0.001 |
| Cicero_09172014 | PRE control | 0.512 | 0.228 | 0.765 | 0.400 | 0.105 | 0.132 |
| Gatsby_08022013 | MAZE rest | 0.002 | 0.073 | 0.021 | 0.269 | 0.000 | 0.000 |
| Gatsby_08022013 | PRE control | 0.519 | 0.007 | 0.302 | 0.294 | 0.378 | 0.708 |
| Gatsby_08282013 | MAZE rest | 0.264 | 0.400 | 0.407 | 0.123 | 0.003 | 0.006 |
| Gatsby_08282013 | PRE control | 0.525 | 0.287 | 0.819 | 0.738 | 0.072 | 0.401 |
| **sessions with p < 0.05** | MAZE rest | 5/8 | 3/8 | 4/8 | 4/8 | 7/8 | 7/8 |
| **sessions with p < 0.05** | PRE control | 0/8 | 1/8 | 0/8 | 0/8 | 1/8 | 0/8 |

## Timing control (added after the first results): the same events with spike trains shuffled among each event's active cells

Share of 100 timing-shuffled fits whose best output matches the map (p < 0.05), and the share whose |ρ| is at least the real fit's (the timing p: small = the real order is more than co-firing explains). \|ρ\|: best output's absolute Spearman correlation.

| session | MAZE centred: real \|ρ\| (p) | shuffled pass | timing p | MAZE uncentred: real \|ρ\| (p) | shuffled pass | timing p | POST sleep centred: real \|ρ\| (p) | shuffled pass | timing p |
|---|---|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 0.75 (0.001) | 98/100 | 0.149 | 0.54 (0.019) | 97/100 | 0.663 | 0.54 (0.005) | 35/100 | 0.109 |
| Achilles_11012013 | 0.60 (0.005) | 87/100 | 0.406 | 0.68 (0.007) | 94/100 | 0.545 | 0.53 (0.260) | 9/100 | 0.426 |
| Buddy_06272013 | 0.65 (0.338) | 21/100 | 0.624 | 0.56 (0.557) | 10/100 | 0.713 | 0.57 (0.491) | 15/100 | 0.901 |
| Cicero_09012014 | 0.40 (0.792) | 4/100 | 0.713 | 0.63 (0.288) | 0/100 | 0.040 | 0.29 (0.958) | 2/100 | 0.851 |
| Cicero_09102014 | 0.70 (0.031) | 6/100 | 0.030 | 0.42 (0.875) | 7/100 | 0.604 | 0.64 (0.311) | 3/100 | 0.277 |
| Cicero_09172014 | 0.65 (0.038) | 16/100 | 0.079 | 0.71 (0.029) | 7/100 | 0.010 | 0.50 (0.413) | 2/100 | 0.277 |
| Gatsby_08022013 | 0.80 (0.004) | 21/100 | 0.020 | 0.61 (0.060) | 0/100 | 0.010 | 0.61 (0.159) | 13/100 | 0.228 |
| Gatsby_08282013 | 0.63 (0.228) | 2/100 | 0.158 | 0.56 (0.434) | 4/100 | 0.475 | 0.60 (0.773) | 4/100 | 0.465 |
