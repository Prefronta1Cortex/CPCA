# Per-component separation benchmark

Produced by `scripts/run_separation.py`. Design: `docs/separation.md`. Outputs: components (or templates) that pass the latency check on held-out events. 10 fits per cell (5 draws x 2 folds).

| condition | method | outputs per fit (median; before the check) | outputs passing the criterion | maps represented (fits with all maps) | clean fits |
|---|---|---|---|---|---|
| A different cells, 10% + 10%, both ways | centred PCA | 9 (9) | 2/88 (2%) | 0/10 | 0/10 |
| A different cells, 10% + 10%, both ways | centred ICA | 9 (9) | 0/90 (0%) | 0/10 | 0/10 |
| A different cells, 10% + 10%, both ways | uncentred ICA | 8 (10) | 0/84 (0%) | 0/10 | 0/10 |
| A different cells, 10% + 10%, both ways | templates | 3 (7) | 1/30 (3%) | 0/10 | 0/10 |
| B different cells, 13% + 7%, both ways | centred PCA | 9 (9) | 10/86 (12%) | 1/10 | 0/10 |
| B different cells, 13% + 7%, both ways | centred ICA | 9 (9) | 0/89 (0%) | 0/10 | 0/10 |
| B different cells, 13% + 7%, both ways | uncentred ICA | 8 (9) | 0/77 (0%) | 0/10 | 0/10 |
| B different cells, 13% + 7%, both ways | templates | 3 (7) | 4/30 (13%) | 0/10 | 0/10 |
| C same cells, 10% + 10%, forward | centred PCA | 6 (7) | 22/64 (34%) | 7/10 | 0/10 |
| C same cells, 10% + 10%, forward | centred ICA | 7 (7) | 25/70 (36%) | 6/10 | 0/10 |
| C same cells, 10% + 10%, forward | uncentred ICA | 8 (8) | 41/75 (55%) | 7/10 | 0/10 |
| C same cells, 10% + 10%, forward | templates | 2 (6) | 19/25 (76%) | 8/10 | 7/10 |
| D same cells, 10% + 10%, both ways | centred PCA | 6 (8) | 34/59 (58%) | 9/10 | 1/10 |
| D same cells, 10% + 10%, both ways | centred ICA | 8 (8) | 38/74 (51%) | 8/10 | 0/10 |
| D same cells, 10% + 10%, both ways | uncentred ICA | 8 (9) | 40/81 (49%) | 9/10 | 0/10 |
| D same cells, 10% + 10%, both ways | templates | 3 (6) | 26/31 (84%) | 10/10 | 6/10 |
| E one map, 20%, both ways | centred PCA | 6 (7) | 32/65 (49%) | 10/10 | 0/10 |
| E one map, 20%, both ways | centred ICA | 7 (7) | 44/67 (66%) | 10/10 | 2/10 |
| E one map, 20%, both ways | uncentred ICA | 7 (8) | 27/72 (38%) | 10/10 | 0/10 |
| E one map, 20%, both ways | templates | 2 (6) | 11/18 (61%) | 10/10 | 4/10 |
| F no replays | centred PCA | 0 (5) | 0/4 (0%) | 0/10 | 6/10 |
| F no replays | centred ICA | 0 (5) | 0/3 (0%) | 0/10 | 8/10 |
| F no replays | uncentred ICA | 0 (6) | 0/2 (0%) | 0/10 | 8/10 |
| F no replays | templates | 0 (5) | 0/0 | 0/10 | 10/10 |

No replays (F): a clean fit is one with no outputs.
