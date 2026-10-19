# Matched rerun: cPCA, seqNMF (K x L grid), PP-Seq (corrected)

Every method saw the same fixed subset of at most 1500 peak-centred event windows per dataset (each window is the event's own duration centred on its population peak). Order-producing methods are scored identically (`cpca.evaluate.order_vs_place`): Spearman between a component/factor/type's member order and the answer key, against the 95th percentile of a cell-permutation null; * marks above chance. Counts carry 95% Clopper–Pearson intervals. Corrections relative to the first comparison are listed in docs/baselines.md.

## Synthetic planted routes

| condition | replay | cPCA uncentred | cPCA centred | seqNMF K=15 L=120 (paper's guidance) | seqNMF best of 9 (K, L) cells | PP-Seq type membership (R, warp) | PP-Seq package membership | PP-Seq detection AUC (sequence types) |
|---|---|---|---|---|---|---|---|---|
| A one route, every event a replay | 100% | 1.00 [0.22]* (k=1) | 0.00 [nan] (k=0) | 0.98 [0.22]* | 0.98 [0.34]* (K=5, L=200) | 1.00 [0.21]* (R=2, no warp) | 1.00 [0.34]* | — |
| B one route, forward and backward 60/40, span 25 ms | 100% | 0.99 [0.21]* (k=2) | 0.81 [0.24]* (k=1) | 0.77 [0.21]* | 0.87 [0.25]* (K=5, L=60) | 1.00 [0.22]* (R=4, warp) | 1.00 [0.69]* | — |
| C one route, forward and backward 60/40, span 80 ms | 100% | 1.00 [0.21]* (k=2) | 0.80 [0.23]* (k=1) | 0.98 [0.21]* | 0.99 [0.21]* (K=10, L=120) | 1.00 [0.22]* (R=7, no warp) | 0.36 [0.52] | — |
| D two routes on different cells, 40% replay | 41% | 1.00 [0.29]*; 1.00 [0.28]* (k=3) | 1.00 [0.34]*; 1.00 [0.31]* (k=2) | 0.98 [0.33]*; 0.98 [0.30]* | 0.99 [0.32]*; 0.98 [0.30]* (K=15, L=200) | 1.00 [0.32]*; 1.00 [0.30]* (R=8, no warp) | 1.00 [0.83]*; 0.95 [0.67]* | 1.00 |
| E two routes sharing half their cells, 40% replay | 41% | 0.90 [0.32]*; 0.85 [0.32]* (k=3) | 0.92 [0.30]*; 0.94 [0.32]* (k=2) | 0.98 [0.33]*; 0.98 [0.31]* | 0.98 [0.31]*; 0.97 [0.33]* (K=15, L=200) | 1.00 [0.34]*; 1.00 [0.29]* (R=3, no warp) | 1.00 [0.38]*; 0.99 [0.37]* | 1.00 |
| F two routes, both directions, 15% replay, 15% cells | 16% | 0.82 [0.30]*; 0.94 [0.32]* (k=4) | 0.95 [0.32]*; 0.93 [0.33]* (k=4) | 0.35 [0.32]*; 0.28 [0.27]* | 0.35 [0.32]*; 0.28 [0.27]* (K=15, L=120) | 0.56 [0.34]*; 0.99 [0.28]* (R=4, warp) | 0.52 [0.37]*; 0.99 [0.43]* | 1.00 |
| G realistic: direction maps, segments, assemblies, non-place cells, 15% replay | 16% | 0.31 [0.30]*; 0.26 [0.30] (k=8) | 0.33 [0.43]; 0.28 [0.27]* (k=8) | 0.21 [0.71]; 0.22 [0.29] | 0.48 [0.70]; 0.67 [0.67] (K=5, L=60) | 0.31 [0.45]; 0.54 [0.47]* (R=6, no warp) | 0.60 [0.83]; 1.00 [0.83]* | 1.00 |
| H realistic at 40% replay | 37% | 0.47 [0.29]*; 0.45 [0.28]* (k=9) | 0.54 [0.29]*; 0.46 [0.24]* (k=10) | 0.05 [0.26]; 0.02 [0.25] | 0.32 [0.67]; 0.14 [0.56] (K=15, L=60) | 0.89 [0.32]*; 0.81 [0.34]* (R=8, warp) | 0.95 [0.72]*; 1.00 [0.71]* | 0.80 |

## hc-11, POST vs PRE sleep (same 1500 events per epoch for cPCA, seqNMF, PP-Seq)

| session | cells / place (+,−) | events (all → used) | cPCA centred, all events POST / PRE | cPCA centred, subset POST / PRE | PP-Seq type rule POST / PRE (R) | PP-Seq package rule POST / PRE | seqNMF K=15 L=120 POST / PRE | seqNMF best cell POST / PRE |
|---|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 120 / 57,63 | 3260 → 1500; 6009 → 1500 | 6/22 / 0/24 | 4/18 / 1/16 | 1/12 / 0/8 (6/4) | 1/7 / 0/8 | 0/0 / 0/2 | 0/0 / 1/2 |
| Achilles_11012013 | 92 / 51,62 | 2678 → 1500; 3928 → 1500 | 8/22 / 1/14 | 4/16 / 0/10 | 0/4 / 1/6 (2/3) | 0/4 / 0/6 | 1/2 / 0/0 | 1/2 / 1/2 |
| Buddy_06272013 | 48 / 13,13 | 1483 → 1483; 3391 → 1500 | 1/10 / 1/10 | 0/10 / 1/6 | 0/3 / 0/4 (6/3) | 0/0 / 0/2 | 0/0 / 0/0 | 1/2 / 0/0 |
| Cicero_09012014 | 55 / 17,15 | 3702 → 1500; 4153 → 1500 | 1/10 / 1/12 | 0/6 / 1/8 | 1/8 / 0/8 (5/8) | 0/1 / 0/0 | 0/0 / 0/0 | 1/2 / 0/0 |
| Cicero_09102014 | 81 / 22,36 | 4892 → 1500; 3289 → 1500 | 1/14 / 0/12 | 2/10 / 0/8 | 0/4 / 1/6 (2/4) | 0/3 / 0/4 | 0/2 / 0/2 | 0/0 / 1/2 |
| Cicero_09172014 | 59 / 23,22 | 3462 → 1500; 4059 → 1500 | 1/10 / 0/12 | 0/10 / 0/8 | 0/4 / 0/4 (3/3) | 0/3 / 0/2 | 0/2 / 0/0 | 1/2 / 0/0 |
| Gatsby_08022013 | 66 / 19,21 | 2782 → 1500; 3775 → 1500 | 3/14 / 1/12 | 4/10 / 1/10 | 0/7 / 0/2 (6/2) | 0/2 / 0/2 | 0/0 / 0/2 | 0/0 / 0/0 |
| Gatsby_08282013 | 41 / 20,25 | 4940 → 1500; 4364 → 1500 | 0/16 / 0/8 | 0/10 / 1/8 | 0/5 / 0/6 (3/7) | 0/4 / 0/2 | 0/0 / 0/0 | 0/0 / 0/0 |
| **pooled (hits/tests, % and 95% CI; chance 5%)** | | | **POST** 21/118 (18%, CI 11–26%)<br>**PRE** 4/104 (4%, CI 1–10%) | **POST** 14/90 (16%, CI 9–25%)<br>**PRE** 5/74 (7%, CI 2–15%) | **POST** 2/47 (4%, CI 1–15%)<br>**PRE** 2/44 (5%, CI 1–15%) | **POST** 1/24 (4%, CI 0–21%)<br>**PRE** 0/26 (0%, CI 0–13%) | **POST** 1/6 (17%, CI 0–64%)<br>**PRE** 0/6 (0%, CI 0–46%) | **POST** 4/8 (50%, CI 16–84%)<br>**PRE** 3/6 (50%, CI 12–88%) |

### Reading the pooled row

A method shows maze-specific order when its POST interval excludes 5% and its PRE interval does not. Note that components of one decomposition share cells, so the binomial intervals are somewhat optimistic for cPCA; the session-level pattern (which sessions carry the POST hits) is given in the per-session rows.

## PP-Seq sequence types on hc-11 (members under the type rule / the package rule, offset span, latent events per posterior sample)

- Achilles_10252013 POST (R=6, no warp, 57% of events carry a type spanning ≥ 20 ms): 36/21 cells, span 68 ms, 191 events; 19/17 cells, span 24 ms, 160 events; 85/10 cells, span 18 ms, 1242 events; 29/18 cells, span 20 ms, 283 events; 79/11 cells, span 69 ms, 378 events; 47/19 cells, span 37 ms, 303 events
- Achilles_10252013 PRE (R=4, no warp, 43% of events carry a type spanning ≥ 20 ms): 35/22 cells, span 37 ms, 292 events; 38/25 cells, span 20 ms, 294 events; 32/24 cells, span 33 ms, 339 events; 86/25 cells, span 15 ms, 1664 events
- Achilles_11012013 POST (R=2, warp, 15% of events carry a type spanning ≥ 20 ms): 76/52 cells, span 13 ms, 1985 events; 17/21 cells, span 30 ms, 242 events
- Achilles_11012013 PRE (R=3, warp, 34% of events carry a type spanning ≥ 20 ms): 15/25 cells, span 21 ms, 259 events; 55/25 cells, span 46 ms, 368 events; 77/23 cells, span 12 ms, 1562 events
- Buddy_06272013 POST (R=6, warp, 0% of events carry a type spanning ≥ 20 ms): 16/8 cells, span 14 ms, 398 events; 6/6 cells, span 9 ms, 450 events; 7/5 cells, span 16 ms, 248 events; 8/7 cells, span 19 ms, 403 events; 27/7 cells, span 14 ms, 761 events; 8/5 cells, span 13 ms, 418 events
- Buddy_06272013 PRE (R=3, warp, 0% of events carry a type spanning ≥ 20 ms): 24/16 cells, span 15 ms, 629 events; 8/10 cells, span 16 ms, 340 events; 34/12 cells, span 14 ms, 1496 events
- Cicero_09012014 POST (R=5, warp, 7% of events carry a type spanning ≥ 20 ms): 13/11 cells, span 14 ms, 151 events; 8/1 cells, span 21 ms, 119 events; 16/12 cells, span 9 ms, 683 events; 15/9 cells, span 12 ms, 866 events; 7/11 cells, span 15 ms, 271 events
- Cicero_09012014 PRE (R=8, no warp, 19% of events carry a type spanning ≥ 20 ms): 21/6 cells, span 42 ms, 232 events; 1/2 cells, span nan ms, 93 events; 26/5 cells, span 15 ms, 563 events; 4/3 cells, span 11 ms, 88 events; 2/4 cells, span 7 ms, 171 events; 17/8 cells, span 29 ms, 82 events; 5/6 cells, span 23 ms, 67 events; 15/10 cells, span 9 ms, 593 events
- Cicero_09102014 POST (R=2, warp, 0% of events carry a type spanning ≥ 20 ms): 45/37 cells, span 15 ms, 1622 events; 19/27 cells, span 18 ms, 379 events
- Cicero_09102014 PRE (R=4, no warp, 47% of events carry a type spanning ≥ 20 ms): 46/14 cells, span 11 ms, 1003 events; 8/14 cells, span 16 ms, 318 events; 12/13 cells, span 24 ms, 369 events; 37/23 cells, span 22 ms, 603 events
- Cicero_09172014 POST (R=3, warp, 0% of events carry a type spanning ≥ 20 ms): 28/16 cells, span 19 ms, 497 events; 24/22 cells, span 7 ms, 1844 events; 6/9 cells, span 10 ms, 212 events
- Cicero_09172014 PRE (R=3, warp, 0% of events carry a type spanning ≥ 20 ms): 7/11 cells, span 14 ms, 199 events; 32/25 cells, span 11 ms, 1676 events; 15/11 cells, span 17 ms, 469 events
- Gatsby_08022013 POST (R=6, no warp, 21% of events carry a type spanning ≥ 20 ms): 8/11 cells, span 8 ms, 586 events; 19/12 cells, span 12 ms, 1122 events; 5/4 cells, span 8 ms, 102 events; 5/3 cells, span 6 ms, 292 events; 20/9 cells, span 17 ms, 605 events; 12/13 cells, span 25 ms, 428 events
- Gatsby_08022013 PRE (R=2, no warp, 0% of events carry a type spanning ≥ 20 ms): 3/9 cells, span 2 ms, 408 events; 15/43 cells, span 10 ms, 1886 events
- Gatsby_08282013 POST (R=3, no warp, 0% of events carry a type spanning ≥ 20 ms): 18/13 cells, span 13 ms, 714 events; 6/11 cells, span 13 ms, 540 events; 8/8 cells, span 18 ms, 212 events
- Gatsby_08282013 PRE (R=7, no warp, 23% of events carry a type spanning ≥ 20 ms): 12/6 cells, span 17 ms, 490 events; 10/8 cells, span 11 ms, 452 events; 7/6 cells, span 201 ms, 284 events; 5/4 cells, span 5 ms, 427 events; 4/2 cells, span 29 ms, 135 events; 5/2 cells, span 6 ms, 403 events; 3/4 cells, span 2 ms, 382 events
