# hc-11: each method's best output against the waking map

Produced by `scripts/plot_hc11_sequences.py` (figure: `figures/hc11_sequences.png`). Matched 1500-event subsets of the rerun; |ρ| of the best output (over outputs, running directions, senses and, on circular tracks, rotations), p against the same maximum under permutations of the field centres, and the cells it covers.

| session | epoch | Complex PCA | seqNMF | PP-Seq | random order |
|---|---|---|---|---|---|
| Achilles_10252013 | POST | 0.54 (p 0.004, 47/57 cells) | – | 0.43 (p 0.542, 13/57 cells) | 0.19 (p 0.262, 63/63 cells) |
| Achilles_10252013 | PRE | 0.45 (p 0.053, 41/63 cells) | 0.17 (p 0.389, 52/57 cells) | 0.28 (p 0.796, 21/57 cells) | 0.19 (p 0.242, 63/63 cells) |
| Achilles_11012013 | POST | 0.53 (p 0.216, 33/51 cells) | 0.94 (p 0.078, 6/62 cells) | 0.71 (p 0.246, 10/51 cells) | 0.15 (p 0.987, 62/62 cells) |
| Achilles_11012013 | PRE | 0.42 (p 0.528, 40/62 cells) | – | 0.80 (p 0.078, 10/51 cells) | 0.15 (p 0.993, 62/62 cells) |
| Buddy_06272013 | POST | 0.57 (p 0.496, 13/13 cells) | – | 0.55 (p 0.390, 8/13 cells) | 0.28 (p 0.602, 13/13 cells) |
| Buddy_06272013 | PRE | 0.77 (p 0.101, 11/13 cells) | – | 0.43 (p 0.625, 10/13 cells) | 0.28 (p 0.595, 13/13 cells) |
| Cicero_09012014 | POST | 0.29 (p 0.960, 14/17 cells) | – | 0.84 (p 0.128, 9/15 cells) | 0.54 (p 0.063, 15/15 cells) |
| Cicero_09012014 | PRE | 0.70 (p 0.075, 12/15 cells) | – | 0.29 (p 0.990, 7/17 cells) | 0.54 (p 0.062, 15/15 cells) |
| Cicero_09102014 | POST | 0.64 (p 0.280, 11/22 cells) | 0.58 (p 0.410, 11/22 cells) | 0.54 (p 0.785, 13/36 cells) | 0.31 (p 0.813, 22/22 cells) |
| Cicero_09102014 | PRE | 0.48 (p 0.921, 9/22 cells) | 0.41 (p 0.549, 19/22 cells) | 0.78 (p 0.358, 9/36 cells) | 0.31 (p 0.810, 22/22 cells) |
| Cicero_09172014 | POST | 0.50 (p 0.399, 14/23 cells) | 0.39 (p 0.193, 20/23 cells) | 0.53 (p 0.323, 11/23 cells) | 0.08 (p 0.921, 22/22 cells) |
| Cicero_09172014 | PRE | 0.21 (p 0.985, 18/23 cells) | – | 0.54 (p 0.267, 9/23 cells) | 0.08 (p 0.920, 22/22 cells) |
| Gatsby_08022013 | POST | 0.61 (p 0.181, 20/21 cells) | – | 0.71 (p 0.395, 8/21 cells) | 0.28 (p 0.409, 19/19 cells) |
| Gatsby_08022013 | PRE | 0.61 (p 0.144, 16/21 cells) | 0.27 (p 0.475, 16/19 cells) | 0.17 (p 0.921, 6/21 cells) | 0.28 (p 0.423, 19/19 cells) |
| Gatsby_08282013 | POST | 0.60 (p 0.788, 10/20 cells) | – | 0.81 (p 0.580, 6/25 cells) | 0.46 (p 0.359, 25/25 cells) |
| Gatsby_08282013 | PRE | 0.68 (p 0.338, 18/25 cells) | – | 0.83 (p 0.623, 6/20 cells) | 0.46 (p 0.341, 25/25 cells) |
| **sessions with p < 0.05** | POST | 1/8 | 0/8 | 0/8 | 0/8 |
| **sessions with p < 0.05** | PRE | 0/8 | 0/8 | 0/8 | 0/8 |
