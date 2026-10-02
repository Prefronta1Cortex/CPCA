# Label-free comparison: results

Score: best |Spearman| between the method's cell order and the answer key (planted order, or place-field centre per running direction), maximised over the method's templates/factors and the two directions. In brackets: 95th percentile of the same maximum after shuffling cell identities. A star marks a value above that chance level. The answer key is used for scoring only.

## Synthetic events with a planted order

| condition | replay events | cPCA one template (C) | cPCA one template (C − C⁰) | cPCA mixture (K by stability) | seqNMF | HMM | count PCA/ICA |
|---|---|---|---|---|---|---|---|
| S1 one route, all events replay | 100% | 1.00 [0.21]* | 1.00 [0.22]* | 1.00 [0.21]* (K=2) | 1.00 [0.39]* | 0.96 [0.24]* (M=20) | 0 assemblies, no order |
| S2 two directions 50/50, all events replay | 100% | 0.18 [0.21] | 0.76 [0.22]* | 1.00 [0.23]* (K=2) | 1.00 [0.22]* | 0.95 [0.23]* (M=20) | 0 assemblies, no order |
| S3 two directions, 40% replay, 20% cells | 40% | 0.40 [0.24]* | 0.88 [0.22]* | 1.00 [0.29]* (K=3) | 0.33 [0.44] | 0.57 [0.22]* (M=20) | 0 assemblies, no order |
| S4 two directions, 15% replay, 15% cells | 15% | 0.03 [0.25] | 0.72 [0.20]* | 0.98 [0.23]* (K=2) | 0.15 [0.35] | 0.09 [0.24] (M=20) | 0 assemblies, no order |

## Realistic synthetic sleep (3000 events, 15% replay unless stated): which property removes the order?

Each replay covers a 40% segment of the track in one of two direction-specific maps, forward or backward; 'assemblies' means which cells fire is structured (6 groups); 'non-place' means half the recorded cells have no field and fire anyway.

| condition | replay events | cPCA one template (C − C⁰) | cPCA mixture | cPCA mixture, unit event arrows | seqNMF | HMM |
|---|---|---|---|---|---|---|
| R0 all three properties | 15% | 0.13 [0.40] | 0.19 [0.39] (K=6) | 0.19 [0.42] (K=6) | 0.31 [0.71] | 0.24 [0.31] |
| R1 no assembly structure | 15% | 0.16 [0.40] | 0.32 [0.44] (K=2) | 0.34 [0.43] (K=2) | 0.49 [0.65] | 0.22 [0.32] |
| R2 one map for both directions | 16% | 0.00 [0.41] | 0.45 [0.38]* (K=6) | 0.02 [0.37] (K=2) | 0.47 [0.70] | 0.24 [0.26] |
| R3 only place cells recorded | 15% | 0.17 [0.28] | 0.26 [0.39] (K=6) | 0.24 [0.37] (K=6) | 0.55 [0.68] | 0.20 [0.29] |
| R4 none of the three | 15% | 0.37 [0.27]* | 0.32 [0.31]* (K=2) | 0.32 [0.32]* (K=2) | 0.88 [0.50]* | 0.51 [0.26]* |
| R5 all three, 40% replay | 39% | 0.28 [0.42] | 0.24 [0.39] (K=6) | 0.24 [0.42] (K=6) | 0.38 [0.68] | 0.32 [0.31]* |

Membership-blind mixture (event and template arrows both unit length), same conditions, K = 2 / 4: R4 0.25 [0.28] / 0.51 [0.28]*; R2 0.21 [0.27] / 0.39 [0.31]*; R0 0.22 [0.34] / 0.24 [0.34]; R5 0.39 [0.34]* / 0.34 [0.32]*.

## hc-11 POST NREM, wide events (edges at mean rate, peak above mean + 3 SD), pyramidal cells

| session | maze | cells | place cells (+/−) | events | cPCA one template (C) | cPCA one template (C − C⁰) | cPCA mixture (K by stability) | cPCA mixture, membership-blind | seqNMF | HMM | count PCA/ICA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 1.6m Linear Maze | 120 | 57/63 | 3260 | 0.11 [0.29] | 0.13 [0.29] | 0.16 [0.34] (K=2) | 0.13 [0.31] (K=2) | 0.27 [0.66] | 0.12 [0.29] (M=35) | 14 assemblies; member field spread 0.63 m vs random 0.60 m |
| Achilles_11012013 | Circular Maze | 92 | 51/62 | 2678 | 0.32 [0.39] | 0.34 [0.43] | 0.35 [0.47] (K=2) | 0.39 [0.46] (K=2) | 0.65 [0.81] | 0.21 [0.42] (M=35) | 13 assemblies; member field spread 0.92 m vs random 0.92 m |
| Buddy_06272013 | 1.6m Linear Maze | 48 | 13/13 | 1483 | 0.38 [0.73] | 0.50 [0.68] | 0.41 [0.73] (K=2) | 0.22 [0.73] (K=2) | 0.47 [0.69] | 0.69 [0.67]* (M=20) | 6 assemblies; member field spread 0.25 m vs random 0.59 m |
| Cicero_09012014 | 1.6m Linear Maze | 55 | 17/15 | 3702 | 0.39 [0.67] | 0.32 [0.58] | 0.32 [0.77] (K=3) | 0.38 [0.69] (K=2) | 0.46 [0.65] | 0.24 [0.59] (M=20) | 7 assemblies; member field spread 0.48 m vs random 0.51 m |
| Cicero_09102014 | Circular Maze | 81 | 22/36 | 4892 | 0.28 [0.58] | 0.35 [0.66] | 0.35 [0.74] (K=2) | 0.37 [0.66] (K=2) | 0.66 [0.80] | 0.51 [0.61] (M=35) | 10 assemblies; member field spread 1.04 m vs random 0.91 m |
| Cicero_09172014 | 2m Linear Maze | 59 | 23/22 | 3462 | 0.18 [0.51] | 0.37 [0.44] | 0.25 [0.59] (K=2) | 0.09 [0.53] (K=2) | 0.31 [0.54] | 0.41 [0.49] (M=35) | 8 assemblies; member field spread 0.60 m vs random 0.62 m |
| Gatsby_08022013 | 1.6m Linear Maze | 66 | 19/21 | 2782 | 0.38 [0.56] | 0.60 [0.51]* | 0.56 [0.65] (K=2) | 0.54 [0.65] (K=2) | 0.77 [0.70]* | 0.24 [0.51] (M=35) | 9 assemblies; member field spread 0.44 m vs random 0.54 m |
| Gatsby_08282013 | Circular Maze | 41 | 20/25 | 4940 | 0.51 [0.69] | 0.51 [0.68] | 0.57 [0.86] (K=3) | 0.53 [0.67] (K=2) | — | 0.44 [0.64] (M=35) | 9 assemblies; member field spread 0.72 m vs random 0.74 m |
| **sessions above chance** | | | | | 0/8 | 1/8 | 0/8 | 0/8 | 1/8 | 1/8 | — |

## Mixture cPCA per number of templates (real sessions)

| session | K=2 | K=3 | K=4 | K=6 | chosen K |
|---|---|---|---|---|---|
| Achilles_10252013 | 0.16 [0.34] stab 0.96 | 0.24 [0.35] stab 0.94 | 0.20 [0.39] stab 0.90 | 0.33 [0.45] stab 0.89 | 2 |
| Achilles_11012013 | 0.35 [0.47] stab 0.99 | 0.39 [0.61] stab 0.98 | 0.62 [0.61]* stab 0.96 | 0.64 [0.63]* stab 0.90 | 2 |
| Buddy_06272013 | 0.41 [0.73] stab 0.98 | 0.46 [0.77] stab 0.89 | 0.40 [0.83] stab 0.80 | 0.78 [0.85] stab 0.81 | 2 |
| Cicero_09012014 | 0.38 [0.78] stab 0.99 | 0.32 [0.77] stab 0.99 | 0.51 [0.82] stab 0.95 | 0.59 [0.78] stab 0.91 | 3 |
| Cicero_09102014 | 0.35 [0.74] stab 0.99 | 0.36 [0.78] stab 0.97 | 0.43 [0.83] stab 0.94 | 0.45 [0.82] stab 0.91 | 2 |
| Cicero_09172014 | 0.25 [0.59] stab 0.99 | 0.35 [0.61] stab 0.97 | 0.39 [0.69] stab 0.93 | 0.42 [0.72] stab 0.97 | 2 |
| Gatsby_08022013 | 0.56 [0.65] stab 0.99 | 0.54 [0.68] stab 0.99 | 0.45 [0.72] stab 0.98 | 0.57 [0.77] stab 0.93 | 2 |
| Gatsby_08282013 | 0.66 [0.78] stab 0.97 | 0.57 [0.86] stab 0.99 | 0.74 [0.86] stab 0.99 | 0.72 [0.88] stab 0.93 | 3 |

## Run time per session (seconds)

| session | cPCA one template | cPCA mixture (4 values of K, with stability fits) | seqNMF (3 restarts) | HMM (2 state counts) | count PCA/ICA |
|---|---|---|---|---|---|
| Achilles_10252013 | 0.03 | 27 | 86 | 188 | 0.04 |
| Achilles_11012013 | 0.01 | 12 | 100 | 201 | 0.03 |
| Buddy_06272013 | 0.00 | 3 | 44 | 98 | 0.01 |
| Cicero_09012014 | 0.01 | 7 | 60 | 248 | 0.02 |
| Cicero_09102014 | 0.03 | 24 | 110 | 295 | 0.03 |
| Cicero_09172014 | 0.01 | 8 | 55 | 196 | 0.02 |
| Gatsby_08022013 | 0.01 | 7 | 64 | 152 | 0.02 |
| Gatsby_08282013 | 0.01 | 4 | 39 | 243 | 0.02 |

seqNMF factor lag spans (ms) and cells per factor, real sessions: Achilles 1025: spans [28, 20, 16] cells [83, 16, 10]; Achilles 1101: spans [24, 33, 20] cells [41, 22, 13]; Buddy 0627: spans [30, 35, 78] cells [23, 7, 8]; Cicero 0901: spans [17, 21, 58] cells [41, 8, 3]; Cicero 0910: spans [22, 28, 62] cells [43, 12, 6]; Cicero 0917: spans [17, 12, 24] cells [42, 6, 6]; Gatsby 0802: spans [23, 9, 33] cells [37, 6, 20]; Gatsby 0828: spans [3, 8, 0] cells [6, 4, 2]
