# Method comparison: results

All methods see the same events and cells; the waking map is used only to score. Order-producing methods (cPCA, seqNMF, PP-Seq) are scored per component/factor/sequence type by the Spearman correlation between member order and the answer key against a cell-permutation null; the HMM is scored on per-event congruence and on latent-state place-field decoding; Bayesian decoding takes the map as input and is the reference ceiling. convSeq was not run; see docs/baselines.md.

## Synthetic planted routes (one draw per condition, 80 cells, 3000 events)

| condition | replay | cPCA uncentred | cPCA centred | seqNMF (significant / K) | PP-Seq (R chosen) | detection AUC, replays vs volleys: HMM congruence / PP-Seq sequence types | count PCA/ICA |
|---|---|---|---|---|---|---|---|
| A one route, every event a replay | 100% | 1.00 [0.20]* (k=1) | 0.00 [nan] (k=0) | 0.99 [0.21]* (2/4) | 0.96 [0.21]* (R=2) | nan / — | 0 assemblies, no order |
| B one route, forward and backward 60/40, span 25 ms | 100% | 0.99 [0.20]* (k=2) | 0.79 [0.21]* (k=1) | 0.81 [0.23]* (5/5) | 1.00 [0.23]* (R=2) | nan / — | 0 assemblies, no order |
| C one route, forward and backward 60/40, span 80 ms | 100% | 1.00 [0.21]* (k=2) | 0.76 [0.24]* (k=1) | 0.98 [0.38]* (5/5) | 1.00 [0.22]* (R=2) | nan / — | 0 assemblies, no order |
| D two routes on different cells, 40% replay | 41% | 1.00 [0.33]*; 1.00 [0.29]* (k=3) | 1.00 [0.27]*; 1.00 [0.31]* (k=2) | 0.99 [0.61]*; 0.98 [0.33]* (5/5) | 1.00 [0.33]*; 1.00 [0.31]* (R=2) | 0.86 / 0.51 | 2 assemblies, no order |
| E two routes sharing half their cells, 40% replay | 40% | 0.90 [0.29]*; 0.85 [0.33]* (k=3) | 0.94 [0.30]*; 0.93 [0.30]* (k=2) | 0.97 [0.33]*; 0.97 [0.35]* (3/5) | 1.00 [0.32]*; 0.99 [0.30]* (R=3) | 0.81 / 1.00 | 2 assemblies, no order |
| F two routes, both directions, 15% replay, 15% cells | 16% | 0.95 [0.46]*; 0.99 [0.33]* (k=4) | 0.99 [0.37]*; 0.98 [0.32]* (k=4) | 0.52 [0.50]*; 0.47 [0.41]* (5/5) | 0.82 [0.33]*; 0.10 [0.60] (R=3) | 0.79 / 0.69 | 2 assemblies, no order |
| G realistic: direction maps, segments, assemblies, non-place cells, 15% replay | 16% | 0.50 [0.27]*; 0.57 [0.54]* (k=9) | 0.37 [0.32]*; 0.38 [0.32]* (k=9) | 0.23 [0.51]; 0.35 [0.53] (5/5) | 0.15 [0.25]; 0.76 [0.45]* (R=4) | 0.49 / 0.76 | 9 assemblies, no order |
| H realistic at 40% replay | 38% | 0.37 [0.32]*; 0.64 [0.29]* (k=10) | 0.46 [0.29]*; 0.36 [0.33]* (k=12) | 0.01 [0.26]; 0.11 [0.25] (1/3) | 0.15 [0.24]; 0.19 [0.27] (R=2) | 0.65 / 0.50 | 10 assemblies, no order |

## hc-11: POST vs PRE sleep

| session | cells / place (+,−) | cPCA (centred) map hits POST / PRE [tests] | PP-Seq map hits POST / PRE [tests] (R) | seqNMF map hits POST / PRE [tests] (significant) | HMM congruent 99% POST / PRE | HMM lsPF error POST (shuffled) / PRE (shuffled), m | Bayes significant POST / PRE |
|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 120 / 57,63 | 6 / 0 [22/24] | 0 / 0 [4/4] (2/2) | 0 / 0 [2/2] (3/3) | 29.6% / 31.8% | 0.21 (0.67) / 0.50 (0.63) | 21.6% / 18.8% |
| Achilles_11012013 | 92 / 51,62 | 8 / 1 [22/14] | 0 / 0 [4/4] (2/2) | 0 / 0 [2/2] (2/2) | 24.4% / 14.1% | 0.10 (0.62) / 0.57 (0.61) | 11.2% / 8.2% |
| Buddy_06272013 | 48 / 13,13 | 1 / 1 [10/10] | 0 / 0 [4/2] (3/3) | 0 / 0 [2/2] (1/1) | 14.0% / 10.1% | 0.23 (0.59) / 0.20 (0.53) | 20.5% / 21.6% |
| Cicero_09012014 | 55 / 17,15 | 1 / 1 [10/12] | 0 / 1 [3/2] (2/3) | 0 / 0 [2/2] (1/2) | 22.4% / 17.5% | 0.42 (0.45) / 0.39 (0.89) | 16.6% / 16.9% |
| Cicero_09102014 | 81 / 22,36 | 1 / 0 [14/12] | 0 / 0 [4/3] (2/2) | 1 / 0 [2/2] (1/1) | 22.3% / 23.3% | 0.12 (0.68) / 0.35 (0.63) | 13.1% / 12.3% |
| Cicero_09172014 | 59 / 23,22 | 1 / 0 [10/12] | 0 / 0 [3/2] (2/2) | 0 / 0 [2/2] (2/2) | 21.6% / 23.7% | 0.22 (0.75) / 0.40 (1.00) | 13.0% / 11.8% |
| Gatsby_08022013 | 66 / 19,21 | 3 / 1 [14/12] | 1 / 0 [4/4] (2/2) | 0 / 1 [2/2] (1/2) | 30.1% / 29.0% | 0.30 (0.47) / 0.34 (0.69) | 18.9% / 17.6% |
| Gatsby_08282013 | 41 / 20,25 | 0 / 0 [16/8] | 0 / 0 [4/4] (2/2) | 0 / 0 [1/0] (4/2) | 17.6% / 16.4% | 0.21 (0.61) / 0.22 (0.59) | 10.0% / 8.6% |
| **total** | | **21 / 4** [118/104] (chance ≈ 5.9 / 5.2) | **1 / 1** [30/25] (chance ≈ 1.5 / 1.2) | **1 / 1** [15/14] (chance ≈ 0.8 / 0.7) | | | |

### PP-Seq sequence types on hc-11 (POST): members, offset span, latent events per posterior sample

- Achilles_10252013 POST (R=2, 1500 events fitted, 100% of events carry a sequence type that spans ≥ 20 ms): type 1: 99 cells, span 25 ms, width 18 ms, 2508 events; type 2: 32 cells, span 37 ms, width 27 ms, 745 events
- Achilles_10252013 PRE (R=2, 1500 events fitted, 100% of events carry a sequence type that spans ≥ 20 ms): type 1: 87 cells, span 25 ms, width 20 ms, 2458 events; type 2: 52 cells, span 41 ms, width 23 ms, 861 events
- Achilles_11012013 POST (R=2, 1500 events fitted, 0% of events carry a sequence type that spans ≥ 20 ms): type 1: 52 cells, span 14 ms, width 19 ms, 1497 events; type 2: 61 cells, span 18 ms, width 18 ms, 1647 events
- Achilles_11012013 PRE (R=2, 1500 events fitted, 0% of events carry a sequence type that spans ≥ 20 ms): type 1: 77 cells, span 15 ms, width 18 ms, 2749 events; type 2: 11 cells, span 18 ms, width 22 ms, 413 events
- Buddy_06272013 POST (R=3, 1483 events fitted, 79% of events carry a sequence type that spans ≥ 20 ms): type 1: 22 cells, span 200 ms, width 24 ms, 667 events; type 2: 10 cells, span 18 ms, width 29 ms, 315 events; type 3: 32 cells, span 32 ms, width 28 ms, 863 events
- Buddy_06272013 PRE (R=3, 1500 events fitted, 74% of events carry a sequence type that spans ≥ 20 ms): type 1: 35 cells, span 22 ms, width 21 ms, 823 events; type 2: 9 cells, span 21 ms, width 19 ms, 282 events; type 3: 14 cells, span 24 ms, width 22 ms, 365 events
- Cicero_09012014 POST (R=2, 1500 events fitted, 0% of events carry a sequence type that spans ≥ 20 ms): type 1: 17 cells, span 8 ms, width 19 ms, 2145 events; type 2: 8 cells, span 14 ms, width 27 ms, 361 events
- Cicero_09012014 PRE (R=3, 1500 events fitted, 7% of events carry a sequence type that spans ≥ 20 ms): type 1: 4 cells, span 226 ms, width 49 ms, 114 events; type 2: 5 cells, span 18 ms, width 31 ms, 125 events; type 3: 29 cells, span 13 ms, width 19 ms, 1510 events
- Cicero_09102014 POST (R=2, 1500 events fitted, 23% of events carry a sequence type that spans ≥ 20 ms): type 1: 14 cells, span 23 ms, width 24 ms, 468 events; type 2: 49 cells, span 11 ms, width 18 ms, 2852 events
- Cicero_09102014 PRE (R=2, 1500 events fitted, 0% of events carry a sequence type that spans ≥ 20 ms): type 1: 9 cells, span 20 ms, width 27 ms, 506 events; type 2: 47 cells, span 13 ms, width 20 ms, 2733 events
- Cicero_09172014 POST (R=2, 1500 events fitted, 17% of events carry a sequence type that spans ≥ 20 ms): type 1: 10 cells, span 23 ms, width 23 ms, 328 events; type 2: 27 cells, span 13 ms, width 18 ms, 2840 events
- Cicero_09172014 PRE (R=2, 1500 events fitted, 0% of events carry a sequence type that spans ≥ 20 ms): type 1: 28 cells, span 12 ms, width 19 ms, 2867 events; type 2: 9 cells, span 10 ms, width 23 ms, 343 events
- Gatsby_08022013 POST (R=2, 1500 events fitted, 38% of events carry a sequence type that spans ≥ 20 ms): type 1: 22 cells, span 22 ms, width 21 ms, 721 events; type 2: 14 cells, span 8 ms, width 18 ms, 1869 events
- Gatsby_08022013 PRE (R=2, 1500 events fitted, 0% of events carry a sequence type that spans ≥ 20 ms): type 1: 14 cells, span 15 ms, width 19 ms, 584 events; type 2: 14 cells, span 10 ms, width 19 ms, 1938 events
- Gatsby_08282013 POST (R=2, 1500 events fitted, 0% of events carry a sequence type that spans ≥ 20 ms): type 1: 9 cells, span 4 ms, width 16 ms, 1100 events; type 2: 19 cells, span 13 ms, width 21 ms, 1404 events
- Gatsby_08282013 PRE (R=2, 1500 events fitted, 0% of events carry a sequence type that spans ≥ 20 ms): type 1: 6 cells, span 8 ms, width 20 ms, 951 events; type 2: 19 cells, span 9 ms, width 16 ms, 1750 events
