# Un-mixing: results

Produced by `scripts/run_unmixing.py`. Design: `docs/unmixing.md`.

## Part 1: are the routes inside the kept span?

| condition | space | components kept (median) | target | routes above the null's 99th percentile | share (median) | null (median) |
|---|---|---|---|---|---|---|
| A different cells, 10% + 10%, both ways | complex PCA, centred | 9 | membership | 10/10 | 0.77 | 0.28 |
| A different cells, 10% + 10%, both ways | complex PCA, centred | 9 | order | 10/10 | 0.92 | 0.20 |
| A different cells, 10% + 10%, both ways | complex PCA, uncentred | 10 | membership | 10/10 | 0.82 | 0.40 |
| A different cells, 10% + 10%, both ways | complex PCA, uncentred | 10 | order | 10/10 | 0.92 | 0.26 |
| A different cells, 10% + 10%, both ways | count PCA | 17 | membership | 10/10 | 0.81 | 0.41 |
| B different cells, 13% + 7%, both ways | complex PCA, centred | 9 | membership | 10/10 | 0.77 | 0.27 |
| B different cells, 13% + 7%, both ways | complex PCA, centred | 9 | order | 10/10 | 0.90 | 0.20 |
| B different cells, 13% + 7%, both ways | complex PCA, uncentred | 10 | membership | 10/10 | 0.80 | 0.40 |
| B different cells, 13% + 7%, both ways | complex PCA, uncentred | 10 | order | 10/10 | 0.89 | 0.26 |
| B different cells, 13% + 7%, both ways | count PCA | 17 | membership | 10/10 | 0.84 | 0.41 |
| C same cells, 10% + 10%, forward | complex PCA, centred | 7 | membership | 10/10 | 0.86 | 0.19 |
| C same cells, 10% + 10%, forward | complex PCA, centred | 7 | order | 10/10 | 0.91 | 0.15 |
| C same cells, 10% + 10%, forward | complex PCA, uncentred | 8 | membership | 10/10 | 0.88 | 0.37 |
| C same cells, 10% + 10%, forward | complex PCA, uncentred | 8 | order | 10/10 | 0.93 | 0.23 |
| C same cells, 10% + 10%, forward | count PCA | 14 | membership | 10/10 | 0.91 | 0.39 |
| D same cells, 10% + 10%, both ways | complex PCA, centred | 9 | membership | 10/10 | 0.87 | 0.21 |
| D same cells, 10% + 10%, both ways | complex PCA, centred | 9 | order | 10/10 | 0.89 | 0.17 |
| D same cells, 10% + 10%, both ways | complex PCA, uncentred | 9 | membership | 10/10 | 0.87 | 0.39 |
| D same cells, 10% + 10%, both ways | complex PCA, uncentred | 9 | order | 10/10 | 0.87 | 0.25 |
| D same cells, 10% + 10%, both ways | count PCA | 15 | membership | 10/10 | 0.89 | 0.39 |
| E one map, 20%, both ways | complex PCA, centred | 7 | membership | 5/5 | 0.78 | 0.18 |
| E one map, 20%, both ways | complex PCA, centred | 7 | order | 5/5 | 0.93 | 0.14 |
| E one map, 20%, both ways | complex PCA, uncentred | 8 | membership | 5/5 | 0.79 | 0.38 |
| E one map, 20%, both ways | complex PCA, uncentred | 8 | order | 5/5 | 0.92 | 0.23 |
| E one map, 20%, both ways | count PCA | 14 | membership | 5/5 | 0.90 | 0.39 |

## Part 2: candidates (screen)

Outputs pass the latency check on held-out events; clean = every map represented and every output passes; coverage = members on the route / route cells, median over passing outputs.

| condition | candidate | rule | outputs per fit (median) | outputs passing | clean fits | coverage | solves |
|---|---|---|---|---|---|---|---|
| A different cells, 10% + 10%, both ways | PCA | 20% of max | 8 | 2/82 (2%) | 0/10 | 0.84 | no |
| A different cells, 10% + 10%, both ways | PCA | mean + 2 SD | 5 | 7/51 (14%) | 0/10 | 0.17 | no |
| A different cells, 10% + 10%, both ways | PCA | two-group split | 8 | 7/76 (9%) | 0/10 | 0.55 | no |
| A different cells, 10% + 10%, both ways | ICA | 20% of max | 9 | 0/90 (0%) | 0/10 | – | no |
| A different cells, 10% + 10%, both ways | ICA | mean + 2 SD | 7 | 0/69 (0%) | 0/10 | – | no |
| A different cells, 10% + 10%, both ways | ICA | two-group split | 9 | 0/89 (0%) | 0/10 | – | no |
| A different cells, 10% + 10%, both ways | Quartimax | 20% of max | 5 | 26/50 (52%) | 0/10 | 0.78 | no |
| A different cells, 10% + 10%, both ways | Quartimax | mean + 2 SD | 4 | 9/43 (21%) | 0/10 | 0.23 | no |
| A different cells, 10% + 10%, both ways | Quartimax | two-group split | 4 | 16/43 (37%) | 0/10 | 0.61 | no |
| A different cells, 10% + 10%, both ways | Count ICA + phase | 20% of max | 15 | 64/147 (44%) | 0/10 | 0.23 | no |
| A different cells, 10% + 10%, both ways | Count ICA + phase | mean + 2 SD | 12 | 44/122 (36%) | 0/10 | 0.17 | no |
| A different cells, 10% + 10%, both ways | Count ICA + phase | two-group split | 13 | 46/129 (36%) | 0/10 | 0.17 | no |
| B different cells, 13% + 7%, both ways | PCA | 20% of max | 9 | 9/87 (10%) | 0/10 | 0.82 | no |
| B different cells, 13% + 7%, both ways | PCA | mean + 2 SD | 6 | 11/57 (19%) | 0/10 | 0.20 | no |
| B different cells, 13% + 7%, both ways | PCA | two-group split | 7 | 16/69 (23%) | 0/10 | 0.54 | no |
| B different cells, 13% + 7%, both ways | ICA | 20% of max | 9 | 0/87 (0%) | 0/10 | – | no |
| B different cells, 13% + 7%, both ways | ICA | mean + 2 SD | 8 | 1/74 (1%) | 0/10 | 0.15 | no |
| B different cells, 13% + 7%, both ways | ICA | two-group split | 9 | 0/89 (0%) | 0/10 | – | no |
| B different cells, 13% + 7%, both ways | Quartimax | 20% of max | 5 | 32/52 (62%) | 0/10 | 0.75 | no |
| B different cells, 13% + 7%, both ways | Quartimax | mean + 2 SD | 4 | 9/45 (20%) | 0/10 | 0.20 | no |
| B different cells, 13% + 7%, both ways | Quartimax | two-group split | 4 | 16/42 (38%) | 0/10 | 0.54 | no |
| B different cells, 13% + 7%, both ways | Count ICA + phase | 20% of max | 14 | 61/146 (42%) | 0/10 | 0.20 | no |
| B different cells, 13% + 7%, both ways | Count ICA + phase | mean + 2 SD | 12 | 35/125 (28%) | 0/10 | 0.17 | no |
| B different cells, 13% + 7%, both ways | Count ICA + phase | two-group split | 14 | 47/143 (33%) | 0/10 | 0.17 | no |
| C same cells, 10% + 10%, forward | PCA | 20% of max | 6 | 19/57 (33%) | 0/10 | 0.75 | no |
| C same cells, 10% + 10%, forward | PCA | mean + 2 SD | 2 | 8/23 (35%) | 0/10 | 0.17 | no |
| C same cells, 10% + 10%, forward | PCA | two-group split | 4 | 16/46 (35%) | 0/10 | 0.50 | no |
| C same cells, 10% + 10%, forward | ICA | 20% of max | 7 | 31/69 (45%) | 0/10 | 0.90 | no |
| C same cells, 10% + 10%, forward | ICA | mean + 2 SD | 6 | 14/60 (23%) | 0/10 | 0.17 | no |
| C same cells, 10% + 10%, forward | ICA | two-group split | 7 | 44/66 (67%) | 0/10 | 0.50 | no |
| C same cells, 10% + 10%, forward | Quartimax | 20% of max | 2 | 17/28 (61%) | 2/10 | 0.82 | no |
| C same cells, 10% + 10%, forward | Quartimax | mean + 2 SD | 2 | 9/20 (45%) | 1/10 | 0.20 | no |
| C same cells, 10% + 10%, forward | Quartimax | two-group split | 2 | 8/21 (38%) | 0/10 | 0.61 | no |
| C same cells, 10% + 10%, forward | Count ICA + phase | 20% of max | 10 | 58/100 (58%) | 0/10 | 0.24 | no |
| C same cells, 10% + 10%, forward | Count ICA + phase | mean + 2 SD | 8 | 34/78 (44%) | 0/10 | 0.20 | no |
| C same cells, 10% + 10%, forward | Count ICA + phase | two-group split | 8 | 43/85 (51%) | 0/10 | 0.20 | no |
| D same cells, 10% + 10%, both ways | PCA | 20% of max | 6 | 38/62 (61%) | 1/10 | 0.78 | no |
| D same cells, 10% + 10%, both ways | PCA | mean + 2 SD | 4 | 4/36 (11%) | 0/10 | 0.16 | no |
| D same cells, 10% + 10%, both ways | PCA | two-group split | 5 | 28/54 (52%) | 0/10 | 0.59 | no |
| D same cells, 10% + 10%, both ways | ICA | 20% of max | 8 | 52/77 (68%) | 0/10 | 0.82 | no |
| D same cells, 10% + 10%, both ways | ICA | mean + 2 SD | 7 | 3/73 (4%) | 0/10 | 0.17 | no |
| D same cells, 10% + 10%, both ways | ICA | two-group split | 8 | 40/78 (51%) | 0/10 | 0.44 | no |
| D same cells, 10% + 10%, both ways | Quartimax | 20% of max | 4 | 13/35 (37%) | 0/10 | 0.65 | no |
| D same cells, 10% + 10%, both ways | Quartimax | mean + 2 SD | 4 | 8/36 (22%) | 0/10 | 0.20 | no |
| D same cells, 10% + 10%, both ways | Quartimax | two-group split | 4 | 8/36 (22%) | 0/10 | 0.46 | no |
| D same cells, 10% + 10%, both ways | Count ICA + phase | 20% of max | 12 | 63/116 (54%) | 0/10 | 0.20 | no |
| D same cells, 10% + 10%, both ways | Count ICA + phase | mean + 2 SD | 8 | 28/84 (33%) | 0/10 | 0.17 | no |
| D same cells, 10% + 10%, both ways | Count ICA + phase | two-group split | 10 | 33/103 (32%) | 0/10 | 0.20 | no |
| E one map, 20%, both ways | PCA | 20% of max | 5 | 33/56 (59%) | 1/10 | 0.78 | no |
| E one map, 20%, both ways | PCA | mean + 2 SD | 2 | 2/28 (7%) | 0/10 | 0.26 | no |
| E one map, 20%, both ways | PCA | two-group split | 5 | 24/47 (51%) | 0/10 | 0.53 | no |
| E one map, 20%, both ways | ICA | 20% of max | 6 | 41/63 (65%) | 1/10 | 0.85 | no |
| E one map, 20%, both ways | ICA | mean + 2 SD | 6 | 5/59 (8%) | 0/10 | 0.20 | no |
| E one map, 20%, both ways | ICA | two-group split | 7 | 37/63 (59%) | 1/10 | 0.47 | no |
| E one map, 20%, both ways | Quartimax | 20% of max | 2 | 13/23 (57%) | 3/10 | 0.75 | no |
| E one map, 20%, both ways | Quartimax | mean + 2 SD | 2 | 7/21 (33%) | 1/10 | 0.28 | no |
| E one map, 20%, both ways | Quartimax | two-group split | 2 | 9/24 (38%) | 2/10 | 0.55 | no |
| E one map, 20%, both ways | Count ICA + phase | 20% of max | 11 | 56/111 (50%) | 0/10 | 0.20 | no |
| E one map, 20%, both ways | Count ICA + phase | mean + 2 SD | 8 | 24/81 (30%) | 0/10 | 0.17 | no |
| E one map, 20%, both ways | Count ICA + phase | two-group split | 10 | 30/95 (32%) | 0/10 | 0.17 | no |
| F no replays | PCA | 20% of max | 0 | 0/3 (0%) | 8/10 | – | – |
| F no replays | PCA | mean + 2 SD | 0 | 0/1 (0%) | 9/10 | – | – |
| F no replays | PCA | two-group split | 0 | 0/2 (0%) | 8/10 | – | – |
| F no replays | ICA | 20% of max | 0 | 0/3 (0%) | 8/10 | – | – |
| F no replays | ICA | mean + 2 SD | 0 | 0/1 (0%) | 9/10 | – | – |
| F no replays | ICA | two-group split | 0 | 0/3 (0%) | 8/10 | – | – |
| F no replays | Quartimax | 20% of max | 0 | 0/4 (0%) | 7/10 | – | – |
| F no replays | Quartimax | mean + 2 SD | 0 | 0/4 (0%) | 6/10 | – | – |
| F no replays | Quartimax | two-group split | 0 | 0/1 (0%) | 9/10 | – | – |
| F no replays | Count ICA + phase | 20% of max | 0 | 0/0 | 10/10 | – | – |
| F no replays | Count ICA + phase | mean + 2 SD | 0 | 0/0 | 10/10 | – | – |
| F no replays | Count ICA + phase | two-group split | 0 | 0/1 (0%) | 9/10 | – | – |

No replays (F): clean = no outputs; a candidate must also have outputs in at most 2 of 10 F fits.

## Part 3: ladder outputs per output (stored fits, no latency check)

| rung | method | outputs per fit (median) | outputs matching exactly one map | clean fits |
|---|---|---|---|---|
| L0 | complex PCA, centred | 2 | 5/6 (83%) | 2/3 |
| L0 | complex PCA, uncentred | 2 | 5/6 (83%) | 2/3 |
| L0 | PP-Seq | 4 | 6/13 (46%) | 0/3 |
| L0 | seqNMF (K 15, L 120 ms) | 8 | 2/26 (8%) | 0/3 |
| L1 | complex PCA, centred | 2 | 1/6 (17%) | 0/3 |
| L1 | complex PCA, uncentred | 3 | 1/9 (11%) | 0/3 |
| L1 | PP-Seq | 3 | 6/10 (60%) | 0/3 |
| L1 | seqNMF (K 15, L 120 ms) | 8 | 4/25 (16%) | 0/3 |
| L2 | complex PCA, centred | 3 | 4/9 (44%) | 0/3 |
| L2 | complex PCA, uncentred | 3 | 6/9 (67%) | 1/3 |
| L2 | PP-Seq | 7 | 10/22 (45%) | 0/3 |
| L2 | seqNMF (K 15, L 120 ms) | 10 | 2/30 (7%) | 0/3 |
| L3 | complex PCA, centred | 7 | 13/21 (62%) | 1/3 |
| L3 | complex PCA, uncentred | 7 | 14/20 (70%) | 1/3 |
| L3 | PP-Seq | 5 | 3/15 (20%) | 0/3 |
| L3 | seqNMF (K 15, L 120 ms) | 9 | 0/28 (0%) | 0/3 |
| L4 | complex PCA, centred | 7 | 5/21 (24%) | 0/3 |
| L4 | complex PCA, uncentred | 5 | 6/16 (38%) | 0/3 |
| L4 | PP-Seq | 6 | 13/19 (68%) | 0/3 |
| L4 | seqNMF (K 15, L 120 ms) | 11 | 2/34 (6%) | 0/3 |
| L5 | complex PCA, centred | 3 | 6/9 (67%) | 1/3 |
| L5 | complex PCA, uncentred | 3 | 7/9 (78%) | 2/3 |
| L5 | PP-Seq | 7 | 12/20 (60%) | 0/3 |
| L5 | seqNMF (K 15, L 120 ms) | 9 | 0/24 (0%) | 0/3 |
| L6 | complex PCA, centred | 10 | 10/31 (32%) | 0/3 |
| L6 | complex PCA, uncentred | 10 | 11/29 (38%) | 0/3 |
| L6 | PP-Seq | 4 | 7/12 (58%) | 1/3 |
| L6 | seqNMF (K 15, L 120 ms) | 10 | 0/29 (0%) | 0/3 |
