# Synthetic routes: results

3 independent draws per condition, 80 cells, 3000 events. Per planted route and draw, the best pattern is the one whose members' relative times correlate most with the route positions (|Spearman|). Reported: median |rho| over routes and draws; fraction of (route, draw) pairs above the 95th percentile of the cell-permutation null; median number of components kept (Marchenko-Pastur); median label-free rank of the route's pattern (1 = would be picked first without the answer key; patterns whose members span under 60 degrees rank last).

| condition | replay | uncentred, fixed T | centred, PCA only | centred, T per event | centred, PCA + ICA |
|---|---|---|---|---|---|
| A one route, every event a replay | 100% | ρ 1.00, 100% above chance, k 1, rank 1 | ρ 0.00, 0% above chance, k 0, rank — | ρ 0.96, 100% above chance, k 1, rank 1 | ρ 0.00, 0% above chance, k 0, rank — |
| B one route, forward and backward 60/40, span 25 ms | 100% | ρ 0.99, 100% above chance, k 2, rank 2 | ρ 0.78, 100% above chance, k 1, rank 1 | ρ 0.79, 100% above chance, k 2, rank 1 | ρ 0.78, 100% above chance, k 1, rank 1 |
| C one route, forward and backward 60/40, span 80 ms | 100% | ρ 1.00, 100% above chance, k 2, rank 1 | ρ 0.74, 100% above chance, k 1, rank 1 | ρ 0.72, 100% above chance, k 2, rank 2 | ρ 0.74, 100% above chance, k 1, rank 1 |
| D two routes on different cells, 40% replay | 40% | ρ 1.00, 100% above chance, k 3, rank 2 | ρ 1.00, 100% above chance, k 2, rank 2 | ρ 1.00, 100% above chance, k 2, rank 2 | ρ 1.00, 100% above chance, k 2, rank 2 |
| E two routes sharing half their cells, 40% replay | 40% | ρ 0.86, 100% above chance, k 3, rank 1 | ρ 0.91, 100% above chance, k 2, rank 2 | ρ 0.94, 100% above chance, k 2, rank 2 | ρ 0.90, 100% above chance, k 2, rank 2 |
| F two routes, both directions, 15% replay, 15% cells | 15% | ρ 0.97, 100% above chance, k 4, rank 2 | ρ 0.97, 100% above chance, k 4, rank 3 | ρ 0.97, 100% above chance, k 4, rank 2 | ρ 0.97, 100% above chance, k 4, rank 2 |
| G realistic: direction maps, segments, assemblies, non-place cells, 15% replay | 16% | ρ 0.53, 83% above chance, k 9, rank 8 | ρ 0.38, 83% above chance, k 9, rank 6 | ρ 0.39, 67% above chance, k 10, rank 6 | ρ 0.45, 83% above chance, k 9, rank 5 |
| H realistic at 40% replay | 40% | ρ 0.54, 100% above chance, k 10, rank 4 | ρ 0.60, 100% above chance, k 12, rank 10 | ρ 0.50, 100% above chance, k 11, rank 6 | ρ 0.45, 100% above chance, k 12, rank 3 |
