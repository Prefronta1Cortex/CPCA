# Are the waking components found within the sleep components?

Produced by `scripts/run_containment.py` and `scripts/summarize_containment.py`. Design: `docs/containment.md`. Waking inside sleep: mean share of the kept waking components (uncentred, laps) inside the span of the kept sleep components (centred, bursts) and their conjugates, against the same with the sleep fit's cell identities permuted (its dimension kept).

## Planted data (5 draws per condition)

| condition | draws passing (p < 0.05) | share inside sleep (median) vs null | waking components contained individually |
|---|---|---|---|
| both maps replayed | 5/5 | 0.95 vs 0.34 | 10/10 |
| one map only (subset) | 5/5 | 0.64 vs 0.31 | 10/10 |
| 40% segments of the maps | 5/5 | 0.87 vs 0.44 | 10/10 |
| other maps on the same cells | 1/5 | 0.37 vs 0.34 | 4/10 |
| assemblies along the map, no order | 5/5 | 0.42 vs 0.16 | 10/10 |
| no replays | 0/5 | 0.01 vs 0.02 | 0/2 |

## hc-11, sleep after the maze (all NREM bursts)

| session | laps | sleep events | cells | components, waking / sleep | share inside sleep vs null (p) | waking components contained individually |
|---|---|---|---|---|---|---|
| Achilles_10252013 | 81 | 3260 | 115 | 3 / 11 | 0.53 vs 0.35 (0.001) | 2/3 |
| Achilles_11012013 | 39 | 2678 | 89 | 3 / 11 | 0.58 vs 0.41 (0.001) | 2/3 |
| Buddy_06272013 | 53 | 1483 | 32 | 2 / 3 | 0.45 vs 0.40 (0.133) | 1/2 |
| Cicero_09012014 | 67 | 3702 | 50 | 2 / 5 | 0.50 vs 0.44 (0.042) | 1/2 |
| Cicero_09102014 | 13 | 4892 | 59 | 2 / 5 | 0.64 vs 0.44 (0.001) | 2/2 |
| Cicero_09172014 | 20 | 3462 | 46 | 2 / 4 | 0.47 vs 0.46 (0.361) | 0/2 |
| Gatsby_08022013 | 71 | 2782 | 53 | 2 / 6 | 0.65 vs 0.43 (0.001) | 2/2 |
| Gatsby_08282013 | 51 | 4940 | 41 | 2 / 8 | 0.69 vs 0.60 (0.015) | 2/2 |

Sessions passing: 6/8. Waking components contained individually: 12/18.
