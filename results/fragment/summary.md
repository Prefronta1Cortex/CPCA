# The recurring-fragment criterion

Produced by `scripts/run_fragment.py` and `scripts/summarize_fragment.py`. Design: `docs/fragment.md`.

## Planted data (80 cells, 60 on the route, 2000 events, 5 draws)

| condition | complex PCA bar 1 | complex PCA bar 2 (timing) | complex PCA both | count PCA bar 1 | median \|ρ\| complex / count |
|---|---|---|---|---|---|
| whole-route replays, 20% | 5/5 | 5/5 | 5/5 | 0/5 | 0.96 / 0.45 |
| 40% segment replays, 20% | 4/5 | 4/5 | 4/5 | 5/5 | 0.99 / 0.98 |
| 40% segment replays, 10%, busy events | 5/5 | 4/5 | 4/5 | 4/5 | 0.91 / 0.81 |
| assemblies along the map, no order | 1/5 | 0/5 | 0/5 | 4/5 | 0.83 / 0.90 |
| no replays | 0/5 | 0/5 | 0/5 | 0/5 | nan / nan |

## hc-11

| session | epoch | events | complex PCA \|ρ\| (bar 1 p) | timing p | both | count PCA \|ρ\| (bar 1 p) |
|---|---|---|---|---|---|---|
| Achilles_10252013 | POST | 3260 | 0.90 (0.128) | 0.159 | – | 0.91 (0.170) |
| Achilles_11012013 | POST | 2678 | 0.90 (0.042) | 0.204 | – | 0.92 (0.072) |
| Buddy_06272013 | POST | 1483 | 0.29 (0.924) | 0.975 | – | 0.39 (0.495) |
| Cicero_09012014 | POST | 3702 | 0.81 (0.152) | 0.035 | – | 0.54 (0.829) |
| Cicero_09102014 | POST | 3289 | 0.84 (0.257) | 0.328 | – | 0.92 (0.073) |
| Cicero_09172014 | POST | 3462 | 0.86 (0.213) | 0.194 | – | 0.75 (0.559) |
| Gatsby_08022013 | POST | 2782 | 0.77 (0.442) | 0.244 | – | 0.63 (0.710) |
| Gatsby_08282013 | POST | 4364 | 0.84 (0.304) | 0.199 | – | 0.86 (0.190) |
| Achilles_10252013 | PRE | 3260 | 0.81 (0.517) | 0.423 | – | 0.78 (0.809) |
| Achilles_11012013 | PRE | 2678 | 0.75 (0.497) | 0.517 | – | 0.78 (0.467) |
| Buddy_06272013 | PRE | 1483 | 0.57 (0.323) | 0.239 | – | 0.34 (0.755) |
| Cicero_09012014 | PRE | 3702 | 0.40 (0.628) | 0.841 | – | 0.52 (0.831) |
| Cicero_09102014 | PRE | 3289 | 0.76 (0.585) | 0.498 | – | 0.77 (0.683) |
| Cicero_09172014 | PRE | 3462 | 0.92 (0.051) | 0.055 | – | 0.71 (0.636) |
| Gatsby_08022013 | PRE | 2782 | 0.68 (0.686) | 0.662 | – | 0.70 (0.410) |
| Gatsby_08282013 | PRE | 4364 | 0.71 (0.729) | 0.627 | – | 0.75 (0.668) |
| Achilles_10252013 | MAZE | 373 | 0.87 (0.163) | 0.189 | – | 0.82 (0.443) |
| Achilles_11012013 | MAZE | 321 | 0.81 (0.137) | 0.532 | – | 0.91 (0.040) |
| Buddy_06272013 | MAZE | 504 | 0.24 (0.867) | 0.970 | – | 0.78 (0.085) |
| Cicero_09012014 | MAZE | 1546 | 0.57 (0.481) | 0.851 | – | 0.68 (0.495) |
| Cicero_09102014 | MAZE | 184 | 0.72 (0.416) | 0.502 | – | 0.86 (0.117) |
| Cicero_09172014 | MAZE | 432 | 0.78 (0.321) | 0.458 | – | 0.97 (0.007) |
| Gatsby_08022013 | MAZE | 474 | 0.84 (0.131) | 0.114 | – | 0.48 (0.961) |
| Gatsby_08282013 | MAZE | 531 | 0.72 (0.677) | 0.602 | – | 0.83 (0.201) |

| epoch | complex PCA bar 1 | bar 2 | both | count PCA bar 1 |
|---|---|---|---|---|
| POST | 1/8 | 1/8 | 0/8 | 0/8 |
| PRE | 0/8 | 0/8 | 0/8 | 0/8 |
| MAZE | 0/8 | 0/8 | 0/8 | 2/8 |
