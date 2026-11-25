# Per-component re-scoring

Produced by `scripts/rescore_components.py`. Design: `docs/rescore.md`. Every entry counts output components (not the best one).

## hc-5 wheel runs (6 sessions x 2 window conditions x 3 repeats x 2 folds)

| windows | convention | components per fit (median) | components that are the sequence | fits where every component is |
|---|---|---|---|---|
| aligned | uncentred | 1 | 34/46 (74%) | 27/36 |
| aligned | centred | 1 | 3/25 (12%) | 3/36 |
| unaligned | uncentred | 1 | 36/42 (86%) | 30/36 |
| unaligned | centred | 1 | 11/34 (32%) | 11/36 |

## hc-11 waking laps (8 sessions x 3 splits x 2 folds)

| session | convention | components per fit (median) | components matching exactly one direction | fits with every direction represented | fits where every component passes |
|---|---|---|---|---|---|
| Achilles_10252013 | uncentred | 2 | 0/12 (0%) | 0/6 | 0/6 |
| Achilles_10252013 | centred | 1 | 0/8 (0%) | 0/6 | 0/6 |
| Achilles_11012013 | uncentred | 2 | 12/12 (100%) | 6/6 | 6/6 |
| Achilles_11012013 | centred | 2 | 7/11 (64%) | 6/6 | 2/6 |
| Buddy_06272013 | uncentred | 2 | 3/12 (25%) | 0/6 | 0/6 |
| Buddy_06272013 | centred | 1 | 1/7 (14%) | 0/6 | 0/6 |
| Cicero_09012014 | uncentred | 2 | 9/12 (75%) | 2/6 | 3/6 |
| Cicero_09012014 | centred | 1 | 3/6 (50%) | 0/6 | 3/6 |
| Cicero_09102014 | uncentred | 2 | 6/9 (67%) | 6/6 | 3/6 |
| Cicero_09102014 | centred | 1 | 2/6 (33%) | 2/6 | 2/6 |
| Cicero_09172014 | uncentred | 2 | 6/11 (55%) | 1/6 | 2/6 |
| Cicero_09172014 | centred | 1 | 3/4 (75%) | 0/6 | 3/6 |
| Gatsby_08022013 | uncentred | 2 | 7/12 (58%) | 2/6 | 2/6 |
| Gatsby_08022013 | centred | 2 | 4/9 (44%) | 0/6 | 2/6 |
| Gatsby_08282013 | uncentred | 2 | 10/12 (83%) | 6/6 | 4/6 |
| Gatsby_08282013 | centred | 2 | 3/9 (33%) | 3/6 | 1/6 |

## hc-11 sleep (all NREM bursts, 2 folds; outputs = components passing the latency check)

| session | epoch | method | outputs (fold 1 + fold 2) | outputs matching exactly one direction |
|---|---|---|---|---|
| Achilles_10252013 | POST | centred PCA | 18 | 6 |
| Achilles_10252013 | POST | templates | 10 | 0 |
| Achilles_10252013 | PRE | centred PCA | 19 | 0 |
| Achilles_10252013 | PRE | templates | 13 | 0 |
| Achilles_11012013 | POST | centred PCA | 17 | 3 |
| Achilles_11012013 | POST | templates | 11 | 1 |
| Achilles_11012013 | PRE | centred PCA | 8 | 1 |
| Achilles_11012013 | PRE | templates | 10 | 1 |
| Buddy_06272013 | POST | centred PCA | 5 | 2 |
| Buddy_06272013 | POST | templates | 6 | 1 |
| Buddy_06272013 | PRE | centred PCA | 5 | 2 |
| Buddy_06272013 | PRE | templates | 5 | 0 |
| Cicero_09012014 | POST | centred PCA | 5 | 1 |
| Cicero_09012014 | POST | templates | 8 | 0 |
| Cicero_09012014 | PRE | centred PCA | 5 | 0 |
| Cicero_09012014 | PRE | templates | 7 | 1 |
| Cicero_09102014 | POST | centred PCA | 11 | 1 |
| Cicero_09102014 | POST | templates | 7 | 1 |
| Cicero_09102014 | PRE | centred PCA | 9 | 2 |
| Cicero_09102014 | PRE | templates | 7 | 1 |
| Cicero_09172014 | POST | centred PCA | 9 | 1 |
| Cicero_09172014 | POST | templates | 4 | 1 |
| Cicero_09172014 | PRE | centred PCA | 9 | 1 |
| Cicero_09172014 | PRE | templates | 5 | 0 |
| Gatsby_08022013 | POST | centred PCA | 9 | 3 |
| Gatsby_08022013 | POST | templates | 8 | 1 |
| Gatsby_08022013 | PRE | centred PCA | 10 | 0 |
| Gatsby_08022013 | PRE | templates | 8 | 2 |
| Gatsby_08282013 | POST | centred PCA | 13 | 1 |
| Gatsby_08282013 | POST | templates | 13 | 1 |
| Gatsby_08282013 | PRE | centred PCA | 8 | 1 |
| Gatsby_08282013 | PRE | templates | 8 | 1 |

| epoch | method | outputs | outputs matching exactly one direction | sessions with at least one |
|---|---|---|---|---|
| POST | centred PCA | 87 | 18 (21%) | 8/8 |
| POST | templates | 67 | 6 (9%) | 6/8 |
| PRE | centred PCA | 73 | 7 (10%) | 5/8 |
| PRE | templates | 63 | 6 (10%) | 5/8 |
