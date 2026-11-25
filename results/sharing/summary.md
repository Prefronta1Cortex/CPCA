# Sequences that share cells: results

Produced by `scripts/summarize_sharing.py` from `results/sharing/` (`scripts/run_sharing.py`). Acceptance criteria: `docs/sharing.md`.

## Planted data (10 fresh draws per condition, 3000 events)

Per planted map, median over maps and draws: |Spearman ρ| of the best Step 2 order (direction split of each component) and of the best validated Step 3 template, with the template's coverage of the map; templates per fit (mean) against planted maps; for the best template per map, the share of the map's replays assigned to it (recall), the share of its events that are the map's replays (precision), and the direction accuracy of those events.

| condition | convention | Step 2 | Step 3 | coverage | validated / planted | matching no map (total) | recall | precision | direction | Step 3 time |
|---|---|---|---|---|---|---|---|---|---|---|
| E two routes sharing half their cells | centred | 0.89 | 1.00 | 1.00 | 2.0 / 2 | 0 | 1.00 | 0.88 | 1.00 | 3.6 s |
| E two routes sharing half their cells | uncentred | 0.91 | 1.00 | 1.00 | 2.1 / 2 | 1 | 1.00 | 0.97 | 1.00 | 12.7 s |
| L1 two maps, same cells, forward only | centred | 0.80 | 1.00 | 1.00 | 2.0 / 2 | 0 | 1.00 | 0.88 | 1.00 | 4.7 s |
| L1 two maps, same cells, forward only | uncentred | 0.83 | 1.00 | 1.00 | 2.0 / 2 | 0 | 1.00 | 0.88 | 1.00 | 12.9 s |
| L2 two maps, same cells | centred | 0.81 | 1.00 | 1.00 | 2.0 / 2 | 0 | 1.00 | 0.90 | 1.00 | 9.8 s |
| L2 two maps, same cells | uncentred | 0.81 | 0.74 | 1.00 | 1.0 / 2 | 0 | 1.00 | 0.46 | 1.00 | 8.6 s |
| L4 L2 + assemblies | centred | 0.78 | 1.00 | 1.00 | 2.1 / 2 | 1 | 1.00 | 0.98 | 1.00 | 19.1 s |
| L4 L2 + assemblies | uncentred | 0.82 | 1.00 | 1.00 | 2.0 / 2 | 0 | 1.00 | 0.94 | 1.00 | 15.3 s |
| L5 L2 + variable pacing | centred | 0.80 | 0.90 | 1.00 | 1.5 / 2 | 0 | 1.00 | 0.65 | 1.00 | 6.7 s |
| L5 L2 + variable pacing | uncentred | 0.80 | 0.74 | 1.00 | 1.0 / 2 | 0 | 0.99 | 0.44 | 1.00 | 7.4 s |
| L5 (own scale) L2 + variable pacing, per-event scale | centred | 0.80 | 1.00 | 1.00 | 2.0 / 2 | 0 | 1.00 | 0.90 | 1.00 | 11.1 s |
| A one route in every event | centred | – | – | – | 0.0 / 1 | 0 | – | – | – | 0.0 s |
| A one route in every event | uncentred | 1.00 | 1.00 | 1.00 | 1.0 / 1 | 0 | 1.00 | 1.00 | 1.00 | 1.7 s |
| B one route, both ways, 25 ms | centred | 1.00 | 1.00 | 0.81 | 1.0 / 1 | 0 | 1.00 | 1.00 | 1.00 | 1.4 s |
| B one route, both ways, 25 ms | uncentred | 1.00 | 1.00 | 1.00 | 1.0 / 1 | 0 | 1.00 | 1.00 | 1.00 | 7.9 s |
| C one route, both ways, 80 ms | centred | 1.00 | 1.00 | 0.91 | 1.0 / 1 | 0 | 1.00 | 1.00 | 1.00 | 1.6 s |
| C one route, both ways, 80 ms | uncentred | 1.00 | 1.00 | 1.00 | 1.0 / 1 | 0 | 1.00 | 1.00 | 1.00 | 8.3 s |
| D two routes, different cells | centred | 1.00 | 1.00 | 1.00 | 2.0 / 2 | 0 | 1.00 | 0.88 | 1.00 | 4.7 s |
| D two routes, different cells | uncentred | 1.00 | 1.00 | 1.00 | 1.6 / 2 | 0 | 1.00 | 0.97 | 1.00 | 9.7 s |
| F two routes, 15% replay | centred | 1.00 | 1.00 | 1.00 | 2.0 / 2 | 0 | 1.00 | 0.92 | 1.00 | 19.8 s |
| F two routes, 15% replay | uncentred | 1.00 | 1.00 | 1.00 | 1.8 / 2 | 0 | 1.00 | 0.67 | 1.00 | 8.0 s |
| L0 one map on 60 cells | centred | 1.00 | 1.00 | 1.00 | 1.0 / 1 | 0 | 1.00 | 0.88 | 1.00 | 7.3 s |
| L0 one map on 60 cells | uncentred | 1.00 | 1.00 | 1.00 | 1.0 / 1 | 0 | 1.00 | 0.93 | 1.00 | 10.0 s |
| G realistic, 15% replay | centred | 0.45 | 0.73 | 0.64 | 2.7 / 2 | 1 | 0.28 | 0.77 | 1.00 | 26.2 s |
| G realistic, 15% replay | uncentred | 0.46 | 0.70 | 0.68 | 2.4 / 2 | 1 | 0.28 | 0.80 | 1.00 | 26.9 s |
| H realistic, 40% replay | centred | 0.51 | 0.80 | 0.62 | 7.4 / 2 | 9 | 0.25 | 0.91 | 1.00 | 36.6 s |
| H realistic, 40% replay | uncentred | 0.46 | 0.77 | 0.64 | 6.5 / 2 | 6 | 0.29 | 0.90 | 1.00 | 33.4 s |
| L3 L2 + partial replays | centred | 0.58 | 0.64 | 0.76 | 6.2 / 2 | 20 | 0.29 | 0.88 | 1.00 | 24.7 s |
| L3 L2 + partial replays | uncentred | 0.53 | 0.22 | 0.92 | 1.3 / 2 | 8 | 0.41 | 0.56 | 0.93 | 18.9 s |
| S0 L0, cells shuffled per event | centred | 0.10 | – | – | 0.0 / 1 | 0 | – | – | – | 1.9 s |
| S0 L0, cells shuffled per event | uncentred | 0.12 | – | – | 0.0 / 1 | 0 | – | – | – | 2.1 s |

## Difficulty ladder (the ladder's own events; 1500 per draw, 3 draws)

Median |ρ| per planted map and maps recovered (p < 0.05 against the best-of-outputs null).

| rung | components (centred) | + Step 2 | + Step 3 (centred) | + Step 3 (uncentred) | PP-Seq | seqNMF K = 15, L = 120 ms |
|---|---|---|---|---|---|---|
| L0 one map, both ways | 0.90 (3/3) | 1.00 (3/3) | 1.00 (3/3) | 1.00 (3/3) | 1.00 (3/3) | 0.97 (2/3) |
| L1 two maps, same cells, forward | 0.88 (6/6) | 0.80 (6/6) | 1.00 (6/6) | 1.00 (6/6) | 1.00 (6/6) | 0.99 (4/6) |
| L2 two maps, same cells, both ways | 0.83 (6/6) | 0.82 (6/6) | 1.00 (6/6) | 0.90 (5/6) | 1.00 (6/6) | 0.47 (4/6) |
| L3 L2 + partial replays | 0.59 (4/6) | 0.59 (4/6) | 0.70 (6/6) | 0.13 (2/6) | 0.35 (3/6) | 0.08 (0/6) |
| L4 L2 + assemblies | 0.85 (6/6) | 0.80 (6/6) | 1.00 (6/6) | 1.00 (6/6) | 1.00 (6/6) | 0.36 (4/6) |
| L5 L2 + variable pacing | 0.81 (6/6) | 0.78 (6/6) | 1.00 (5/6) | 0.88 (5/6) | 0.99 (6/6) | 0.26 (3/6) |
| L6 L2 + partial + assemblies (= H) | 0.48 (5/6) | 0.48 (5/6) | 0.81 (6/6) | 0.82 (6/6) | 0.39 (3/6) | 0.15 (0/6) |

Run time per fit (median): step3 centred 3.5 s, step3 uncentred 4.4 s, cpca 1.4 s, ppseq 6971.9 s, seqnmf 3006.1 s.

## hc-11 (centred, all population bursts)

| session | epoch | components | templates | validated | validated and matching a map | map ρ (+ / −) of validated templates | labels correct |
|---|---|---|---|---|---|---|---|
| Achilles 10/25 | POST | 11 | 8 | 1 | 1 | +0.66 (0.30) / +0.50 (0.25) | 34/34 |
| Achilles 10/25 | PRE | 12 | 8 | 1 | 0 | +0.00 (0.45) / -0.03 (0.43) | – |
| Achilles 11/01 | POST | 11 | 9 | 0 | 0 | – | – |
| Achilles 11/01 | PRE | 7 | 7 | 0 | 0 | – | – |
| Buddy 06/27 | POST | 5 | 5 | 0 | 0 | – | – |
| Buddy 06/27 | PRE | 5 | 5 | 0 | 0 | – | – |
| Cicero 09/01 | POST | 5 | 5 | 0 | 0 | – | – |
| Cicero 09/01 | PRE | 5 | 5 | 0 | 0 | – | – |
| Cicero 09/10 | POST | 7 | 4 | 0 | 0 | – | – |
| Cicero 09/10 | PRE | 6 | 3 | 1 | 0 | +0.50 (0.65) / +0.30 (0.54) | – |
| Cicero 09/17 | POST | 6 | 2 | 0 | 0 | – | – |
| Cicero 09/17 | PRE | 6 | 3 | 0 | 0 | – | – |
| Gatsby 08/02 | POST | 7 | 6 | 0 | 0 | – | – |
| Gatsby 08/02 | PRE | 6 | 5 | 0 | 0 | – | – |
| Gatsby 08/28 | POST | 8 | 8 | 0 | 0 | – | – |
| Gatsby 08/28 | PRE | 5 | 5 | 0 | 0 | – | – |

Components matching a map (either direction), for reference: POST 18/60, PRE 2/52.
