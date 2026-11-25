# hc-11: six methods on the same sleep events

Produced by `scripts/run_methods_hc11.py` and `scripts/summarize_methods_hc11.py` (figure: `figures/methods_hc11.png`). Matched 1500-event subsets of the rerun. Complex PCA, seqNMF and PP-Seq scores are those of `results/hc11_sequences` (unchanged).

## Order test: |ρ| of the best output against the waking map (p, best-of-outputs null)

| session | epoch | Complex PCA | seqNMF | PP-Seq | SPOTDis | Rastermap | random order |
|---|---|---|---|---|---|---|---|
| Achilles_10252013 | POST | 0.54 (p 0.004, 47/57 cells) | no output | 0.43 (p 0.542, 13/57 cells) | no output | 0.20 (p 0.222, 57/57 cells) | 0.19 (p 0.262, 63/63 cells) |
| Achilles_10252013 | PRE | 0.45 (p 0.053, 41/63 cells) | 0.17 (p 0.389, 52/57 cells) | 0.28 (p 0.796, 21/57 cells) | no output | 0.27 (p 0.056, 63/63 cells) | 0.19 (p 0.242, 63/63 cells) |
| Achilles_11012013 | POST | 0.53 (p 0.216, 33/51 cells) | 0.94 (p 0.078, 6/62 cells) | 0.71 (p 0.246, 10/51 cells) | no output | 0.36 (p 0.119, 62/62 cells) | 0.15 (p 0.987, 62/62 cells) |
| Achilles_11012013 | PRE | 0.42 (p 0.528, 40/62 cells) | no output | 0.80 (p 0.078, 10/51 cells) | no output | 0.31 (p 0.312, 62/62 cells) | 0.15 (p 0.993, 62/62 cells) |
| Buddy_06272013 | POST | 0.57 (p 0.496, 13/13 cells) | no output | 0.55 (p 0.390, 8/13 cells) | no output | 0.70 (p 0.024, 13/13 cells) | 0.28 (p 0.602, 13/13 cells) |
| Buddy_06272013 | PRE | 0.77 (p 0.101, 11/13 cells) | no output | 0.43 (p 0.625, 10/13 cells) | no output | 0.43 (p 0.283, 13/13 cells) | 0.28 (p 0.595, 13/13 cells) |
| Cicero_09012014 | POST | 0.29 (p 0.960, 14/17 cells) | no output | 0.84 (p 0.128, 9/15 cells) | no output | 0.21 (p 0.692, 15/15 cells) | 0.54 (p 0.063, 15/15 cells) |
| Cicero_09012014 | PRE | 0.70 (p 0.075, 12/15 cells) | no output | 0.29 (p 0.990, 7/17 cells) | no output | 0.20 (p 0.688, 17/17 cells) | 0.54 (p 0.062, 15/15 cells) |
| Cicero_09102014 | POST | 0.64 (p 0.280, 11/22 cells) | 0.58 (p 0.410, 11/22 cells) | 0.54 (p 0.785, 13/36 cells) | no output | 0.41 (p 0.397, 22/22 cells) | 0.31 (p 0.813, 22/22 cells) |
| Cicero_09102014 | PRE | 0.48 (p 0.921, 9/22 cells) | 0.41 (p 0.549, 19/22 cells) | 0.78 (p 0.358, 9/36 cells) | no output | 0.39 (p 0.484, 22/22 cells) | 0.31 (p 0.810, 22/22 cells) |
| Cicero_09172014 | POST | 0.50 (p 0.399, 14/23 cells) | 0.39 (p 0.193, 20/23 cells) | 0.53 (p 0.323, 11/23 cells) | no output | 0.34 (p 0.210, 22/22 cells) | 0.08 (p 0.921, 22/22 cells) |
| Cicero_09172014 | PRE | 0.21 (p 0.985, 18/23 cells) | no output | 0.54 (p 0.267, 9/23 cells) | no output | 0.22 (p 0.551, 22/22 cells) | 0.08 (p 0.920, 22/22 cells) |
| Gatsby_08022013 | POST | 0.61 (p 0.181, 20/21 cells) | no output | 0.71 (p 0.395, 8/21 cells) | no output | 0.09 (p 0.917, 19/19 cells) | 0.28 (p 0.409, 19/19 cells) |
| Gatsby_08022013 | PRE | 0.61 (p 0.144, 16/21 cells) | 0.27 (p 0.475, 16/19 cells) | 0.17 (p 0.921, 6/21 cells) | no output | 0.23 (p 0.550, 21/21 cells) | 0.28 (p 0.423, 19/19 cells) |
| Gatsby_08282013 | POST | 0.60 (p 0.788, 10/20 cells) | no output | 0.81 (p 0.580, 6/25 cells) | no output | 0.40 (p 0.629, 25/25 cells) | 0.46 (p 0.359, 25/25 cells) |
| Gatsby_08282013 | PRE | 0.68 (p 0.338, 18/25 cells) | no output | 0.83 (p 0.623, 6/20 cells) | no output | 0.37 (p 0.720, 25/25 cells) | 0.46 (p 0.341, 25/25 cells) |
| **sessions with p < 0.05** | POST | 1/8 | 0/8 | 0/8 | 0/8 | 1/8 | 0/8 |
| **sessions with p < 0.05** | PRE | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |

## Assembly locality (descriptive): mean field distance of the most local output over its null mean (p)

A route-wide sequence covers the whole track and is not expected to pass; this describes grouping, it does not rank sequence methods.

| session | epoch | Complex PCA | seqNMF | PP-Seq | SPOTDis | Count ICA |
|---|---|---|---|---|---|---|
| Achilles_10252013 | POST | 0.97 (p 0.986, 47 cells) | no output | 0.89 (p 0.060, 37 cells) | no output | 0.35 (p 0.018, 7 cells) |
| Achilles_10252013 | PRE | 0.94 (p 0.048, 49 cells) | 0.97 (p 0.076, 52 cells) | 0.86 (p 0.375, 17 cells) | no output | 0.43 (p 0.461, 4 cells) |
| Achilles_11012013 | POST | 0.98 (p 0.359, 49 cells) | 1.12 (p 0.940, 4 cells) | 0.99 (p 0.196, 46 cells) | no output | 0.13 (p 0.002, 6 cells) |
| Achilles_11012013 | PRE | 0.99 (p 0.521, 41 cells) | no output | 1.01 (p 0.974, 10 cells) | no output | 0.56 (p 0.070, 4 cells) |
| Buddy_06272013 | POST | 0.77 (p 0.475, 11 cells) | no output | 0.18 (p 0.220, 5 cells) | no output | no output |
| Buddy_06272013 | PRE | 0.86 (p 0.836, 8 cells) | no output | 0.51 (p 0.140, 8 cells) | no output | no output |
| Cicero_09012014 | POST | 0.80 (p 0.038, 13 cells) | 1.40 (p 0.882, 4 cells) | 1.00 (p 0.998, 7 cells) | no output | no output |
| Cicero_09012014 | PRE | 0.88 (p 0.385, 13 cells) | no output | 0.72 (p 0.407, 7 cells) | no output | no output |
| Cicero_09102014 | POST | 0.96 (p 0.214, 22 cells) | 1.01 (p 0.848, 17 cells) | 0.92 (p 0.238, 13 cells) | no output | 0.86 (p 0.267, 4 cells) |
| Cicero_09102014 | PRE | 0.95 (p 0.012, 25 cells) | 0.94 (p 0.098, 19 cells) | 0.36 (p 0.074, 4 cells) | no output | 0.36 (p 0.050, 4 cells) |
| Cicero_09172014 | POST | 0.93 (p 0.423, 20 cells) | 0.95 (p 0.353, 19 cells) | 0.95 (p 0.892, 12 cells) | no output | no output |
| Cicero_09172014 | PRE | 0.93 (p 0.423, 20 cells) | no output | 0.84 (p 0.259, 15 cells) | no output | no output |
| Gatsby_08022013 | POST | 0.98 (p 0.970, 20 cells) | 0.88 (p 0.555, 4 cells) | 0.70 (p 0.467, 8 cells) | no output | 0.78 (p 0.371, 5 cells) |
| Gatsby_08022013 | PRE | 0.98 (p 0.906, 16 cells) | 0.98 (p 0.445, 16 cells) | 0.86 (p 0.313, 7 cells) | no output | no output |
| Gatsby_08282013 | POST | 0.92 (p 0.192, 18 cells) | no output | 0.71 (p 0.263, 6 cells) | no output | no output |
| Gatsby_08282013 | PRE | 0.94 (p 0.194, 19 cells) | no output | 0.41 (p 0.355, 4 cells) | no output | 1.17 (p 0.754, 4 cells) |
| **sessions with p < 0.05** | POST | 1/8 | 0/8 | 0/8 | 0/8 | 2/8 |
| **sessions with p < 0.05** | PRE | 2/8 | 0/8 | 0/8 | 0/8 | 1/8 |

## Outputs and runtime

| session | epoch | complex PCA components | seqNMF factors | PP-Seq types | SPOTDis clusters (noise share) | count ICA assemblies | Rastermap cells | Complex PCA s | seqNMF s | PP-Seq s | SPOTDis s | Rastermap s | Count ICA s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Achilles_10252013 | POST | 9 | 14 | 6 | 0 (1.00) | 16 | 120 | 0.8 | 228.7 | 3823.5 | 27.5 | 69.3 | 0.3 |
| Achilles_10252013 | PRE | 8 | 11 | 4 | 2 (0.97) | 15 | 120 | 1.4 | 161.2 | 3066.8 | 39.8 | 79.3 | 0.5 |
| Achilles_11012013 | POST | 9 | 14 | 2 | 0 (1.00) | 14 | 92 | 1.0 | 198.4 | 1926.0 | 20.7 | 36.1 | 0.4 |
| Achilles_11012013 | PRE | 5 | 15 | 3 | 0 (1.00) | 8 | 92 | 0.9 | 199.1 | 1728.9 | 17.8 | 40.8 | 0.4 |
| Buddy_06272013 | POST | 5 | 12 | 6 | 0 (1.00) | 8 | 47 | 0.4 | 91.2 | 964.0 | 9.4 | 1.9 | 0.2 |
| Buddy_06272013 | PRE | 4 | 11 | 3 | 2 (0.97) | 6 | 48 | 0.4 | 104.4 | 747.2 | 8.8 | 1.9 | 0.2 |
| Cicero_09012014 | POST | 4 | 13 | 5 | 2 (0.98) | 5 | 55 | 0.5 | 110.1 | 1087.8 | 15.6 | 3.6 | 0.3 |
| Cicero_09012014 | PRE | 3 | 10 | 8 | 0 (1.00) | 5 | 54 | 0.5 | 109.0 | 637.1 | 12.5 | 3.7 | 0.3 |
| Cicero_09102014 | POST | 5 | 11 | 2 | 0 (1.00) | 8 | 81 | 0.8 | 105.5 | 1057.7 | 20.4 | 20.6 | 0.3 |
| Cicero_09102014 | PRE | 4 | 14 | 4 | 0 (1.00) | 8 | 81 | 0.7 | 168.3 | 979.9 | 19.7 | 22.3 | 0.4 |
| Cicero_09172014 | POST | 5 | 12 | 3 | 0 (1.00) | 7 | 59 | 0.4 | 119.6 | 1889.7 | 34.2 | 6.7 | 0.3 |
| Cicero_09172014 | PRE | 4 | 15 | 3 | 0 (1.00) | 6 | 59 | 0.4 | 124.0 | 1334.7 | 27.1 | 4.2 | 0.3 |
| Gatsby_08022013 | POST | 6 | 11 | 6 | 0 (1.00) | 8 | 66 | 0.5 | 114.8 | 1252.1 | 18.5 | 6.3 | 0.3 |
| Gatsby_08022013 | PRE | 5 | 12 | 2 | 0 (1.00) | 6 | 66 | 0.5 | 137.0 | 937.4 | 16.0 | 4.8 | 0.2 |
| Gatsby_08282013 | POST | 5 | 11 | 3 | 0 (1.00) | 9 | 41 | 0.2 | 85.0 | 538.9 | 6.5 | 1.0 | 0.2 |
| Gatsby_08282013 | PRE | 4 | 14 | 7 | 2 (0.98) | 7 | 41 | 0.2 | 88.3 | 639.8 | 6.0 | 0.8 | 0.2 |
