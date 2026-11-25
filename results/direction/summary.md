# Forward and backward plays: results

Produced by `scripts/summarize_direction.py` from `results/direction/` (`scripts/run_direction.py`).

## Planted data (10 fresh draws per condition)

Order recovered per planted route (|Spearman ρ| of the best component, median over draws and routes), before and after the split; direction accuracy and reverse share for the best component after the split.

| condition | centred before | centred after | uncentred before | uncentred after | accuracy (centred) | reverse share planted / estimated | held-out events expressing | components passing held-out test | splits accepted |
|---|---|---|---|---|---|---|---|---|---|
| B  one route, forward and backward 60/40, span 25 ms | 0.74 | 1.00 | 0.99 | 1.00 | 1.00 | 0.40 / 0.40 | 99.5% | 100% | 100% |
| C  one route, forward and backward 60/40, span 80 ms | 0.74 | 1.00 | 1.00 | 1.00 | 1.00 | 0.40 / 0.40 | 100.0% | 100% | 100% |
| F  two routes, both directions, 15% replay, 15% cells | 0.97 | 1.00 | 0.98 | 1.00 | 1.00 | 0.47 / 0.48 | 14.9% | 100% | 55% |
| G  realistic: direction maps, segments, assemblies, non-place cells, 15% replay | 0.45 | 0.46 | 0.48 | 0.46 | 0.54 | 0.46 / 0.47 | 7.9% | 93% | 4% |
| H  realistic at 40% replay | 0.55 | 0.57 | 0.48 | 0.47 | 0.60 | 0.48 / 0.49 | 15.7% | 100% | 8% |
| L0  one map on 60 cells, both directions | 0.81 | 1.00 | 0.81 | 1.00 | 1.00 | 0.49 / 0.49 | 42.6% | 100% | 70% |
| L2  two maps on the same cells, both directions | 0.80 | 0.80 | 0.76 | 0.78 | 1.00 | 0.49 / 0.49 | 41.4% | 100% | 57% |
| L3  L2 + partial replays (40% segments) | 0.60 | 0.61 | 0.54 | 0.54 | 0.60 | 0.47 / 0.48 | 16.4% | 100% | 22% |
| L4  L2 + assemblies | 0.72 | 0.81 | 0.73 | 0.80 | 1.00 | 0.49 / 0.49 | 36.7% | 100% | 23% |
| L5  L2 + variable pacing (30-90 ms) | 0.75 | 0.81 | 0.71 | 0.81 | 1.00 | 0.49 / 0.49 | 41.0% | 100% | 70% |
| A  one route, every event a replay | – | – | 1.00 | 1.00 | – | – / – | no component | – | – |
| D  two routes on different cells, 40% replay | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 / 0.04 | 42.1% | 100% | 50% |
| E  two routes sharing half their cells, 40% replay | 0.91 | 0.88 | 0.87 | 0.89 | 1.00 | 0.00 / 0.05 | 41.5% | 100% | 65% |
| L1  two maps on the same 60 cells, forward only | 0.87 | 0.86 | 0.82 | 0.85 | 1.00 | 0.00 / 0.05 | 40.7% | 100% | 55% |
| S0  L0, cells shuffled in each event | 0.11 | 0.11 | 0.10 | 0.10 | 0.52 | 0.47 / 0.44 | 3.7% | 0% | 0% |

## hc-11: one template on the whole population

Map ρ is the template's order against each running direction's place map (null 95th percentile in brackets). Labels: expressing events that are significant replays of the template's map, labelled in the map's direction. Groups: templates of the forward and the backward events against that map.

| session | epoch | events | expressing | reverse share | map ρ (+) | map ρ (−) | labels correct | groups ρ fwd / bwd | held-out expressing (p) |
|---|---|---|---|---|---|---|---|---|---|
| Achilles 10/25 | POST | 3260 | 378 | 0.29 | +0.46 (0.26) | +0.31 (0.24) | 43/46 | +0.58 / -0.45 | 6.2% (1e-02) |
| Achilles 10/25 | PRE | 6009 | 757 | 0.32 | -0.16 (0.27) | +0.05 (0.26) | no map match | – | 8.1% (1e-14) |
| Achilles 11/01 | POST | 2678 | 319 | 0.37 | -0.38 (0.39) | -0.20 (0.36) | no map match | – | 5.1% (5e-01) |
| Achilles 11/01 | PRE | 3928 | 337 | 0.39 | +0.29 (0.45) | +0.27 (0.41) | no map match | – | 3.8% (1e+00) |
| Buddy 06/27 | POST | 1470 | 124 | 0.46 | -0.19 (0.72) | +0.17 (0.68) | no map match | – | 3.9% (1e+00) |
| Buddy 06/27 | PRE | 3326 | 282 | 0.46 | -0.37 (0.62) | +0.03 (0.67) | no map match | – | 3.8% (1e+00) |
| Cicero 09/01 | POST | 3702 | 240 | 0.33 | +0.15 (0.51) | -0.05 (0.56) | no map match | – | 3.4% (1e+00) |
| Cicero 09/01 | PRE | 4144 | 272 | 0.39 | -0.21 (0.57) | -0.41 (0.61) | no map match | – | 3.3% (1e+00) |
| Cicero 09/10 | POST | 4892 | 557 | 0.31 | -0.36 (0.60) | +0.41 (0.50) | no map match | – | 6.7% (3e-06) |
| Cicero 09/10 | PRE | 3289 | 428 | 0.21 | +0.32 (0.58) | -0.32 (0.47) | no map match | – | 7.5% (5e-09) |
| Cicero 09/17 | POST | 3462 | 243 | 0.36 | -0.23 (0.50) | +0.02 (0.49) | no map match | – | 4.0% (1e+00) |
| Cicero 09/17 | PRE | 4059 | 296 | 0.41 | +0.18 (0.46) | -0.11 (0.49) | no map match | – | 3.8% (1e+00) |
| Gatsby 08/02 | POST | 2781 | 278 | 0.38 | -0.38 (0.51) | -0.47 (0.53) | no map match | – | 4.2% (9e-01) |
| Gatsby 08/02 | PRE | 3772 | 338 | 0.38 | -0.07 (0.61) | -0.39 (0.53) | no map match | – | 5.0% (2e-01) |
| Gatsby 08/28 | POST | 4774 | 500 | 0.50 | -0.40 (0.67) | -0.41 (0.58) | no map match | – | 6.4% (1e-03) |
| Gatsby 08/28 | PRE | 4280 | 549 | 0.43 | +0.59 (0.65) | +0.55 (0.62) | no map match | – | 7.1% (2e-06) |

- POST: 43/46 labels correct (93.5%; binomial p against 50%: 2.3e-10).

## hc-11: the split of each component

| session | epoch | kept | pass held-out test | split accepted | map hits before | map hits after | template hits (unguarded) | template labels correct |
|---|---|---|---|---|---|---|---|---|
| Achilles 10/25 | POST | 11 | 5 | 0 | 5 | 5 | 8 | 281/306 |
| Achilles 10/25 | PRE | 12 | 12 | 1 | 0 | 0 | 3 | 97/104 |
| Achilles 11/01 | POST | 11 | 1 | 0 | 5 | 5 | 7 | 78/136 |
| Achilles 11/01 | PRE | 7 | 0 | 0 | 1 | 1 | 0 | 0/0 |
| Buddy 06/27 | POST | 5 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| Buddy 06/27 | PRE | 5 | 0 | 1 | 1 | 1 | 0 | 0/0 |
| Cicero 09/01 | POST | 5 | 0 | 0 | 1 | 1 | 0 | 0/0 |
| Cicero 09/01 | PRE | 5 | 0 | 0 | 0 | 0 | 1 | 2/2 |
| Cicero 09/10 | POST | 7 | 5 | 0 | 1 | 1 | 2 | 29/51 |
| Cicero 09/10 | PRE | 6 | 5 | 1 | 0 | 0 | 0 | 0/0 |
| Cicero 09/17 | POST | 6 | 0 | 0 | 2 | 1 | 0 | 0/0 |
| Cicero 09/17 | PRE | 6 | 0 | 1 | 0 | 0 | 0 | 0/0 |
| Gatsby 08/02 | POST | 7 | 0 | 1 | 3 | 3 | 3 | 20/22 |
| Gatsby 08/02 | PRE | 6 | 0 | 0 | 0 | 1 | 0 | 0/0 |
| Gatsby 08/28 | POST | 8 | 5 | 0 | 0 | 0 | 1 | 7/9 |
| Gatsby 08/28 | PRE | 5 | 2 | 1 | 0 | 0 | 0 | 0/0 |
