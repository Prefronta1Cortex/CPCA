# Results

## Synthetic routes

See `results/synthetic/summary.md` and `figures/synthetic_routes.png`.


## hc-11: components whose internal order matches the place map

For every kept component and each running direction: Spearman between members' relative times and their place-field centres, against the 95th percentile of a cell-permutation null. 'hits / tests' counts the (component, direction) pairs above chance; about 5% of tests are expected by chance. POST is sleep after the novel maze; PRE is sleep before it (control).

### uncentred, T = event duration

| session | maze | cells | place cells (+/−) | POST events | POST k | POST hits / tests | PRE events | PRE k | PRE hits / tests | split-half match POST / PRE |
|---|---|---|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 1.6m Linear Maze | 120 | 57/63 | 3260 | 11 | 4 / 22 | 6009 | 14 | 1 / 28 | 0.48 / 0.59 |
| Achilles_11012013 | Circular Maze | 92 | 51/62 | 2678 | 10 | 3 / 20 | 3928 | 12 | 0 / 20 | 0.54 / 0.47 |
| Buddy_06272013 | 1.6m Linear Maze | 48 | 13/13 | 1483 | 8 | 0 / 7 | 3391 | 8 | 0 / 5 | 0.61 / 0.58 |
| Cicero_09012014 | 1.6m Linear Maze | 55 | 17/15 | 3702 | 9 | 1 / 18 | 4153 | 10 | 1 / 18 | 0.65 / 0.62 |
| Cicero_09102014 | Circular Maze | 81 | 22/36 | 4892 | 14 | 3 / 27 | 3289 | 10 | 2 / 20 | 0.52 / 0.55 |
| Cicero_09172014 | 2m Linear Maze | 59 | 23/22 | 3462 | 3 | 0 / 6 | 4059 | 7 | 0 / 14 | 0.91 / 0.68 |
| Gatsby_08022013 | 1.6m Linear Maze | 66 | 19/21 | 2782 | 9 | 2 / 16 | 3775 | 9 | 0 / 16 | 0.62 / 0.61 |
| Gatsby_08282013 | Circular Maze | 41 | 20/25 | 4940 | 9 | 1 / 18 | 4364 | 9 | 0 / 18 | 0.72 / 0.73 |
| **total** | | | | | | **14 / 134** (chance ≈ 6.7) | | | **4 / 139** (chance ≈ 7.0) | |

### centred, T = event duration

| session | maze | cells | place cells (+/−) | POST events | POST k | POST hits / tests | PRE events | PRE k | PRE hits / tests | split-half match POST / PRE |
|---|---|---|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 1.6m Linear Maze | 120 | 57/63 | 3260 | 11 | 6 / 22 | 6009 | 12 | 0 / 24 | 0.53 / 0.71 |
| Achilles_11012013 | Circular Maze | 92 | 51/62 | 2678 | 11 | 8 / 22 | 3928 | 7 | 1 / 14 | 0.57 / 0.60 |
| Buddy_06272013 | 1.6m Linear Maze | 48 | 13/13 | 1483 | 5 | 1 / 10 | 3391 | 5 | 1 / 10 | 0.52 / 0.68 |
| Cicero_09012014 | 1.6m Linear Maze | 55 | 17/15 | 3702 | 5 | 1 / 10 | 4153 | 6 | 1 / 12 | 0.59 / 0.58 |
| Cicero_09102014 | Circular Maze | 81 | 22/36 | 4892 | 7 | 1 / 14 | 3289 | 6 | 0 / 12 | 0.65 / 0.65 |
| Cicero_09172014 | 2m Linear Maze | 59 | 23/22 | 3462 | 5 | 1 / 10 | 4059 | 6 | 0 / 12 | 0.76 / 0.74 |
| Gatsby_08022013 | 1.6m Linear Maze | 66 | 19/21 | 2782 | 7 | 3 / 14 | 3775 | 6 | 1 / 12 | 0.61 / 0.67 |
| Gatsby_08282013 | Circular Maze | 41 | 20/25 | 4940 | 8 | 0 / 16 | 4364 | 4 | 0 / 8 | 0.62 / 0.85 |
| **total** | | | | | | **21 / 118** (chance ≈ 5.9) | | | **4 / 104** (chance ≈ 5.2) | |

### uncentred, T = spike spread

| session | maze | cells | place cells (+/−) | POST events | POST k | POST hits / tests | PRE events | PRE k | PRE hits / tests | split-half match POST / PRE |
|---|---|---|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 1.6m Linear Maze | 120 | 57/63 | 3260 | 14 | 1 / 28 | 6009 | 17 | 1 / 34 | 0.43 / 0.52 |
| Achilles_11012013 | Circular Maze | 92 | 51/62 | 2678 | 12 | 6 / 24 | 3928 | 14 | 1 / 24 | 0.47 / 0.49 |
| Buddy_06272013 | 1.6m Linear Maze | 48 | 13/13 | 1483 | 9 | 0 / 5 | 3391 | 11 | 0 / 8 | 0.54 / 0.53 |
| Cicero_09012014 | 1.6m Linear Maze | 55 | 17/15 | 3702 | 11 | 3 / 18 | 4153 | 12 | 2 / 12 | 0.70 / 0.59 |
| Cicero_09102014 | Circular Maze | 81 | 22/36 | 4892 | 18 | 1 / 31 | 3289 | 12 | 1 / 23 | 0.49 / 0.51 |
| Cicero_09172014 | 2m Linear Maze | 59 | 23/22 | 3462 | 7 | 0 / 14 | 4059 | 10 | 1 / 20 | 0.58 / 0.55 |
| Gatsby_08022013 | 1.6m Linear Maze | 66 | 19/21 | 2782 | 10 | 1 / 20 | 3775 | 11 | 1 / 20 | 0.59 / 0.59 |
| Gatsby_08282013 | Circular Maze | 41 | 20/25 | 4940 | 10 | 0 / 20 | 4364 | 12 | 0 / 21 | 0.72 / 0.69 |
| **total** | | | | | | **12 / 160** (chance ≈ 8.0) | | | **7 / 162** (chance ≈ 8.1) | |

### centred, T = spike spread

| session | maze | cells | place cells (+/−) | POST events | POST k | POST hits / tests | PRE events | PRE k | PRE hits / tests | split-half match POST / PRE |
|---|---|---|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 1.6m Linear Maze | 120 | 57/63 | 3260 | 11 | 6 / 22 | 6009 | 12 | 0 / 24 | 0.50 / 0.64 |
| Achilles_11012013 | Circular Maze | 92 | 51/62 | 2678 | 10 | 10 / 20 | 3928 | 6 | 1 / 12 | 0.53 / 0.60 |
| Buddy_06272013 | 1.6m Linear Maze | 48 | 13/13 | 1483 | 5 | 0 / 10 | 3391 | 4 | 0 / 8 | 0.47 / 0.56 |
| Cicero_09012014 | 1.6m Linear Maze | 55 | 17/15 | 3702 | 4 | 1 / 8 | 4153 | 4 | 0 / 8 | 0.63 / 0.53 |
| Cicero_09102014 | Circular Maze | 81 | 22/36 | 4892 | 7 | 1 / 14 | 3289 | 6 | 1 / 12 | 0.58 / 0.64 |
| Cicero_09172014 | 2m Linear Maze | 59 | 23/22 | 3462 | 5 | 1 / 10 | 4059 | 5 | 0 / 10 | 0.62 / 0.68 |
| Gatsby_08022013 | 1.6m Linear Maze | 66 | 19/21 | 2782 | 6 | 2 / 12 | 3775 | 6 | 1 / 12 | 0.56 / 0.68 |
| Gatsby_08282013 | Circular Maze | 41 | 20/25 | 4940 | 7 | 0 / 14 | 4364 | 5 | 1 / 10 | 0.58 / 0.65 |
| **total** | | | | | | **21 / 110** (chance ≈ 5.5) | | | **4 / 96** (chance ≈ 4.8) | |

### Components matching the map in POST (uncentred, T = event duration)

| session | component | direction | rho | null 95% | members with a field | timing z | span (deg) | eigenvalue share |
|---|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 2 | + | +0.37 | 0.33 | 37 | 41.7 | 327 | 1.8% |
| Achilles_10252013 | 5 | − | -0.32 | 0.32 | 40 | 15.3 | 334 | 1.5% |
| Achilles_10252013 | 6 | − | -0.51 | 0.42 | 23 | 7.8 | 316 | 1.4% |
| Achilles_10252013 | 10 | + | -0.50 | 0.36 | 28 | -0.8 | 323 | 1.2% |
| Achilles_11012013 | 7 | + | -0.53 | 0.45 | 34 | 11.6 | 329 | 1.7% |
| Achilles_11012013 | 7 | − | -0.67 | 0.42 | 41 | 11.6 | 329 | 1.7% |
| Achilles_11012013 | 10 | + | +0.72 | 0.66 | 16 | -0.7 | 285 | 1.6% |
| Cicero_09012014 | 4 | + | -0.78 | 0.65 | 10 | -22.1 | 271 | 2.8% |
| Cicero_09102014 | 7 | + | +1.00 | 0.94 | 6 | -11.5 | 299 | 1.8% |
| Cicero_09102014 | 7 | − | +0.95 | 0.86 | 8 | -11.5 | 299 | 1.8% |
| Cicero_09102014 | 9 | + | +0.81 | 0.77 | 10 | -14.2 | 312 | 1.8% |
| Gatsby_08022013 | 2 | + | -0.77 | 0.62 | 11 | 29.2 | 245 | 3.8% |
| Gatsby_08022013 | 5 | − | +0.81 | 0.63 | 11 | -6.5 | 300 | 2.3% |
| Gatsby_08282013 | 8 | − | -0.73 | 0.71 | 13 | -14.9 | 272 | 3.2% |

### Count PCA/ICA baseline (same events, POST)

| session | assemblies | assemblies with compact fields (spread below null 5%) | best Jaccard of each complex component's members with a count assembly (median) |
|---|---|---|---|
| Achilles_10252013 | 14 | 1 | 0.26 |
| Achilles_11012013 | 13 | 0 | 0.28 |
| Buddy_06272013 | 6 | 0 | 0.35 |
| Cicero_09012014 | 7 | 0 | 0.31 |
| Cicero_09102014 | 10 | 0 | 0.32 |
| Cicero_09172014 | 8 | 0 | 0.39 |
| Gatsby_08022013 | 9 | 3 | 0.35 |
| Gatsby_08282013 | 10 | 0 | 0.25 |
