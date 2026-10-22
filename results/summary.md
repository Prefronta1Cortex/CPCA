# Results

## Synthetic routes

See `results/synthetic/summary.md` and `figures/synthetic_routes.png`.


## hc-11: components whose internal order matches the place map

For every kept component and each running direction: Spearman between members' relative times and their place-field centres, against the 95th percentile of a cell-permutation null. 'hits / tests' counts the (component, direction) pairs above chance; about 5% of tests are expected by chance. POST is sleep after the novel maze; PRE is sleep before it (control).

### uncentred, T = event duration

| session | maze | cells | place cells (+/−) | POST events | POST k | POST hits / tests | PRE events | PRE k | PRE hits / tests | split-half match POST / PRE |
|---|---|---|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 1.6m Linear Maze | 120 | 57/63 | 3260 | 6 | 2 / 12 | 6009 | 6 | 0 / 12 | 0.67 / 0.85 |
| Achilles_11012013 | Circular Maze | 92 | 51/62 | 2678 | 4 | 0 / 8 | 3928 | 2 | 0 / 4 | 0.90 / 0.91 |
| Buddy_06272013 | 1.6m Linear Maze | 48 | 13/13 | 1483 | 2 | 0 / 4 | 3391 | 2 | 0 / 2 | 0.73 / 0.83 |
| Cicero_09012014 | 1.6m Linear Maze | 55 | 17/15 | 3702 | 2 | 0 / 4 | 4153 | 2 | 0 / 4 | 0.99 / 0.93 |
| Cicero_09102014 | Circular Maze | 81 | 22/36 | 4892 | 3 | 0 / 6 | 3289 | 3 | 1 / 6 | 0.92 / 0.91 |
| Cicero_09172014 | 2m Linear Maze | 59 | 23/22 | 3462 | 3 | 0 / 6 | 4059 | 2 | 0 / 4 | 0.88 / 0.96 |
| Gatsby_08022013 | 1.6m Linear Maze | 66 | 19/21 | 2782 | 3 | 1 / 6 | 3775 | 2 | 0 / 4 | 0.92 / 0.99 |
| Gatsby_08282013 | Circular Maze | 41 | 20/25 | 4940 | 4 | 0 / 8 | 4364 | 4 | 0 / 8 | 0.91 / 0.95 |
| **total** | | | | | | **3 / 54** (chance ≈ 2.7) | | | **1 / 44** (chance ≈ 2.2) | |

### centred, T = event duration

| session | maze | cells | place cells (+/−) | POST events | POST k | POST hits / tests | PRE events | PRE k | PRE hits / tests | split-half match POST / PRE |
|---|---|---|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 1.6m Linear Maze | 120 | 57/63 | 3260 | 11 | 6 / 22 | 6009 | 12 | 0 / 24 | 0.56 / 0.65 |
| Achilles_11012013 | Circular Maze | 92 | 51/62 | 2678 | 11 | 8 / 22 | 3928 | 7 | 1 / 14 | 0.50 / 0.57 |
| Buddy_06272013 | 1.6m Linear Maze | 48 | 13/13 | 1483 | 5 | 0 / 10 | 3391 | 5 | 1 / 10 | 0.55 / 0.60 |
| Cicero_09012014 | 1.6m Linear Maze | 55 | 17/15 | 3702 | 5 | 1 / 10 | 4153 | 5 | 0 / 10 | 0.66 / 0.56 |
| Cicero_09102014 | Circular Maze | 81 | 22/36 | 4892 | 7 | 1 / 14 | 3289 | 6 | 0 / 12 | 0.59 / 0.66 |
| Cicero_09172014 | 2m Linear Maze | 59 | 23/22 | 3462 | 6 | 1 / 12 | 4059 | 6 | 0 / 12 | 0.68 / 0.68 |
| Gatsby_08022013 | 1.6m Linear Maze | 66 | 19/21 | 2782 | 7 | 3 / 14 | 3775 | 6 | 0 / 12 | 0.62 / 0.64 |
| Gatsby_08282013 | Circular Maze | 41 | 20/25 | 4940 | 8 | 0 / 16 | 4364 | 6 | 0 / 12 | 0.61 / 0.63 |
| **total** | | | | | | **20 / 120** (chance ≈ 6.0) | | | **2 / 106** (chance ≈ 5.3) | |

### uncentred, T = spike spread

| session | maze | cells | place cells (+/−) | POST events | POST k | POST hits / tests | PRE events | PRE k | PRE hits / tests | split-half match POST / PRE |
|---|---|---|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 1.6m Linear Maze | 120 | 57/63 | 3260 | 5 | 1 / 10 | 6009 | 6 | 0 / 12 | 0.66 / 0.88 |
| Achilles_11012013 | Circular Maze | 92 | 51/62 | 2678 | 3 | 1 / 6 | 3928 | 1 | 0 / 2 | 0.80 / 0.99 |
| Buddy_06272013 | 1.6m Linear Maze | 48 | 13/13 | 1483 | 1 | 0 / 2 | 3391 | 1 | 0 / 2 | 0.99 / 0.99 |
| Cicero_09012014 | 1.6m Linear Maze | 55 | 17/15 | 3702 | 2 | 0 / 4 | 4153 | 1 | 0 / 2 | 0.92 / 1.00 |
| Cicero_09102014 | Circular Maze | 81 | 22/36 | 4892 | 2 | 0 / 4 | 3289 | 3 | 0 / 6 | 0.96 / 0.88 |
| Cicero_09172014 | 2m Linear Maze | 59 | 23/22 | 3462 | 2 | 0 / 4 | 4059 | 2 | 0 / 4 | 0.88 / 0.91 |
| Gatsby_08022013 | 1.6m Linear Maze | 66 | 19/21 | 2782 | 2 | 1 / 4 | 3775 | 2 | 0 / 4 | 0.96 / 0.98 |
| Gatsby_08282013 | Circular Maze | 41 | 20/25 | 4940 | 4 | 0 / 8 | 4364 | 3 | 0 / 6 | 0.88 / 0.97 |
| **total** | | | | | | **3 / 42** (chance ≈ 2.1) | | | **0 / 38** (chance ≈ 1.9) | |

### centred, T = spike spread

| session | maze | cells | place cells (+/−) | POST events | POST k | POST hits / tests | PRE events | PRE k | PRE hits / tests | split-half match POST / PRE |
|---|---|---|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 1.6m Linear Maze | 120 | 57/63 | 3260 | 11 | 6 / 22 | 6009 | 12 | 0 / 24 | 0.49 / 0.63 |
| Achilles_11012013 | Circular Maze | 92 | 51/62 | 2678 | 11 | 10 / 22 | 3928 | 6 | 1 / 12 | 0.49 / 0.63 |
| Buddy_06272013 | 1.6m Linear Maze | 48 | 13/13 | 1483 | 4 | 0 / 8 | 3391 | 4 | 0 / 8 | 0.44 / 0.56 |
| Cicero_09012014 | 1.6m Linear Maze | 55 | 17/15 | 3702 | 4 | 1 / 8 | 4153 | 4 | 0 / 8 | 0.63 / 0.57 |
| Cicero_09102014 | Circular Maze | 81 | 22/36 | 4892 | 7 | 2 / 14 | 3289 | 6 | 1 / 12 | 0.63 / 0.59 |
| Cicero_09172014 | 2m Linear Maze | 59 | 23/22 | 3462 | 5 | 1 / 10 | 4059 | 5 | 0 / 10 | 0.65 / 0.71 |
| Gatsby_08022013 | 1.6m Linear Maze | 66 | 19/21 | 2782 | 6 | 2 / 12 | 3775 | 6 | 1 / 12 | 0.59 / 0.62 |
| Gatsby_08282013 | Circular Maze | 41 | 20/25 | 4940 | 8 | 0 / 16 | 4364 | 5 | 1 / 10 | 0.65 / 0.60 |
| **total** | | | | | | **22 / 112** (chance ≈ 5.6) | | | **4 / 96** (chance ≈ 4.8) | |

### Components matching the map in POST (uncentred, T = event duration)

| session | component | direction | rho | null 95% | members with a field | timing z | span (deg) | eigenvalue share |
|---|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 2 | + | +0.37 | 0.33 | 37 | 43.0 | 327 | 1.8% |
| Achilles_10252013 | 6 | − | -0.51 | 0.42 | 23 | 5.7 | 316 | 1.4% |
| Gatsby_08022013 | 2 | + | -0.77 | 0.64 | 11 | 32.2 | 245 | 3.8% |

### Count PCA/ICA baseline (same events, POST)

| session | assemblies | assemblies with compact fields (spread below null 5%) | best Jaccard of each complex component's members with a count assembly (median) |
|---|---|---|---|
| Achilles_10252013 | 14 | 1 | 0.27 |
| Achilles_11012013 | 13 | 0 | 0.30 |
| Buddy_06272013 | 6 | 0 | 0.41 |
| Cicero_09012014 | 7 | 0 | 0.46 |
| Cicero_09102014 | 10 | 0 | 0.32 |
| Cicero_09172014 | 8 | 1 | 0.39 |
| Gatsby_08022013 | 9 | 2 | 0.38 |
| Gatsby_08282013 | 10 | 0 | 0.25 |
