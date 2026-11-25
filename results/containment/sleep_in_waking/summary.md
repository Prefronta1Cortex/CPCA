# Are the sleep components found within the waking components?

Produced by `scripts/run_containment.py` and `scripts/summarize_containment.py`. Design: `docs/containment.md`. Span containment: mean share of the sleep components inside the span of the waking components and their conjugates, against the same with the waking fit's cell identities permuted. Single match: each sleep component's best squared overlap with one waking component (or its conjugate).

## Planted data (5 draws per condition)

| condition | sessions passing (span, p < 0.05) | span containment (median) vs null | sleep components contained individually | single match (median) |
|---|---|---|---|---|
| both maps replayed | 5/5 | 0.99 vs 0.29 | 16/16 | 0.64 |
| one map only (subset) | 5/5 | 0.99 vs 0.42 | 10/10 | 0.48 |
| 40% segments of the maps | 5/5 | 0.40 vs 0.14 | 37/39 | 0.15 |
| other maps on the same cells | 1/5 | 0.32 vs 0.29 | 3/17 | 0.07 |
| assemblies along the map, no order | 5/5 | 0.30 vs 0.06 | 20/25 | 0.07 |
| no replays | 0/5 | 0.02 vs 0.05 | 0/1 | 0.01 |

## hc-11, sleep after the maze (all NREM bursts)

| session | laps | sleep events | cells | span containment vs null | sleep components contained individually | single match: median; components beating the null | components, waking / sleep |
|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 81 | 3260 | 115 | 0.18 vs 0.11 (p 0.001) | 9/11 | 0.06; 6/11 beat the null | 3 / 11 |
| Achilles_11012013 | 39 | 2678 | 89 | 0.21 vs 0.13 (p 0.001) | 6/11 | 0.06; 6/11 beat the null | 3 / 11 |
| Buddy_06272013 | 53 | 1483 | 32 | 0.33 vs 0.27 (p 0.046) | 1/3 | 0.12; 1/3 beat the null | 2 / 3 |
| Cicero_09012014 | 67 | 3702 | 50 | 0.23 vs 0.19 (p 0.046) | 1/5 | 0.04; 1/5 beat the null | 2 / 5 |
| Cicero_09102014 | 13 | 4892 | 59 | 0.32 vs 0.20 (p 0.001) | 3/5 | 0.15; 4/5 beat the null | 2 / 5 |
| Cicero_09172014 | 20 | 3462 | 46 | 0.26 vs 0.24 (p 0.245) | 0/4 | 0.08; 1/4 beat the null | 2 / 4 |
| Gatsby_08022013 | 71 | 2782 | 53 | 0.25 vs 0.16 (p 0.001) | 3/6 | 0.12; 4/6 beat the null | 2 / 6 |
| Gatsby_08282013 | 51 | 4940 | 41 | 0.23 vs 0.17 (p 0.001) | 4/8 | 0.11; 2/8 beat the null | 2 / 8 |

Sessions passing: 7/8. Sleep components contained individually: 27/53; matched by a single waking component beyond the null: 25/53.
