# Method comparison: results

All methods see the same events and cells; the waking map is used only to score. Order-producing methods (cPCA, seqNMF) are scored per component/factor by the Spearman correlation between member order and the answer key against a cell-permutation null; the HMM is scored on per-event congruence and on latent-state place-field decoding; Bayesian decoding takes the map as input and is the reference ceiling. PP-Seq (Julia) and convSeq (research code) were not run; see docs/baselines.md.

## Synthetic planted routes (one draw per condition, 80 cells, 3000 events)

| condition | replay | cPCA uncentred | cPCA centred | seqNMF (significant factors / K) | HMM congruence: AUC replay vs volley; % replays / % volleys flagged | count PCA/ICA |
|---|---|---|---|---|---|---|
| A one route, every event a replay | 100% | 1.00 [0.20]* (k=1) | 0.00 [nan] (k=0) | 0.99 [0.21]* (2/4, λ=7.7e-03) | AUC nan; 99% / nan% | 0 assemblies, no order |
| B one route, forward and backward 60/40, span 25 ms | 100% | 0.99 [0.20]* (k=2) | 0.79 [0.21]* (k=1) | 0.81 [0.23]* (5/5, λ=1.7e-04) | AUC nan; 60% / nan% | 0 assemblies, no order |
| C one route, forward and backward 60/40, span 80 ms | 100% | 1.00 [0.21]* (k=2) | 0.76 [0.24]* (k=1) | 0.98 [0.38]* (5/5, λ=1.7e-03) | AUC nan; 100% / nan% | 0 assemblies, no order |
| D two routes on different cells, 40% replay | 41% | 1.00 [0.33]*; 1.00 [0.29]* (k=3) | 1.00 [0.27]*; 1.00 [0.31]* (k=2) | 0.99 [0.61]*; 0.98 [0.33]* (5/5, λ=1.7e-03) | AUC 0.86; 100% / 29% | 2 assemblies, no order |
| E two routes sharing half their cells, 40% replay | 40% | 0.90 [0.29]*; 0.85 [0.33]* (k=3) | 0.94 [0.30]*; 0.93 [0.30]* (k=2) | 0.97 [0.33]*; 0.97 [0.35]* (3/5, λ=8.3e-03) | AUC 0.81; 100% / 37% | 2 assemblies, no order |
| F two routes, both directions, 15% replay, 15% cells | 16% | 0.95 [0.46]*; 0.99 [0.33]* (k=4) | 0.99 [0.37]*; 0.98 [0.32]* (k=4) | 0.52 [0.50]*; 0.47 [0.41]* (5/5, λ=9.4e-05) | AUC 0.79; 99% / 40% | 2 assemblies, no order |
| G realistic: direction maps, segments, assemblies, non-place cells, 15% replay | 16% | 0.50 [0.27]*; 0.57 [0.54]* (k=9) | 0.37 [0.32]*; 0.38 [0.32]* (k=9) | 0.23 [0.51]; 0.35 [0.53] (5/5, λ=4.0e-03) | AUC 0.49; 30% / 31% | 9 assemblies, no order |
| H realistic at 40% replay | 38% | 0.37 [0.32]*; 0.64 [0.29]* (k=10) | 0.46 [0.29]*; 0.36 [0.33]* (k=12) | 0.01 [0.26]; 0.11 [0.25] (1/3, λ=5.4e-02) | AUC 0.65; 48% / 26% | 10 assemblies, no order |

## hc-11: POST vs PRE sleep

| session | cells / place (+,−) | cPCA (centred) map hits POST / PRE [tests] | seqNMF map hits POST / PRE [tests] (significant factors) | HMM congruent 99% POST / PRE | HMM lsPF error POST (shuffled) / PRE (shuffled), m | Bayes significant POST / PRE (strict) |
|---|---|---|---|---|---|---|
| Achilles_10252013 | 120 / 57,63 | 6 / 0 [22/24] | 0 / 0 [2/2] (3/3) | 29.6% / 31.8% | 0.21 (0.67) / 0.50 (0.63) | 21.6% / 18.8% (7.2% / 5.7%) |
| Achilles_11012013 | 92 / 51,62 | 8 / 1 [22/14] | 0 / 0 [2/2] (2/2) | 24.4% / 14.1% | 0.10 (0.62) / 0.57 (0.61) | 11.2% / 8.2% (3.2% / 1.2%) |
| Buddy_06272013 | 48 / 13,13 | 1 / 1 [10/10] | 0 / 0 [2/2] (1/1) | 14.0% / 10.1% | 0.23 (0.59) / 0.20 (0.53) | 20.5% / 21.6% (1.2% / 0.4%) |
| Cicero_09012014 | 55 / 17,15 | 1 / 1 [10/12] | 0 / 0 [2/2] (1/2) | 22.4% / 17.5% | 0.42 (0.45) / 0.39 (0.89) | 16.6% / 16.9% (0.1% / 0.0%) |
| Cicero_09102014 | 81 / 22,36 | 1 / 0 [14/12] | 1 / 0 [2/2] (1/1) | 22.3% / 23.3% | 0.12 (0.68) / 0.35 (0.63) | 13.1% / 12.3% (1.8% / 1.6%) |
| Cicero_09172014 | 59 / 23,22 | 1 / 0 [10/12] | 0 / 0 [2/2] (2/2) | 21.6% / 23.7% | 0.22 (0.75) / 0.40 (1.00) | 13.0% / 11.8% (1.1% / 0.7%) |
| Gatsby_08022013 | 66 / 19,21 | 3 / 1 [14/12] | 0 / 1 [2/2] (1/2) | 30.1% / 29.0% | 0.30 (0.47) / 0.34 (0.69) | 18.9% / 17.6% (1.1% / 1.4%) |
| Gatsby_08282013 | 41 / 20,25 | 0 / 0 [16/8] | 0 / 0 [1/0] (4/2) | 17.6% / 16.4% | 0.21 (0.61) / 0.22 (0.59) | 10.0% / 8.6% (0.8% / 0.2%) |
| **total** | | **21 / 4** [118/104] (chance ≈ 5.9 / 5.2) | **1 / 1** [15/14] (chance ≈ 0.8 / 0.7) | | | |

seqNMF λ chosen by the crossover rule (2 λ₀), per session POST/PRE: Achilles: 9.3e-03/1.7e-02; Achilles: 2.7e-02/2.7e-02; Buddy: 2.9e-02/2.4e-02; Cicero: 2.7e-02/2.1e-02; Cicero: 3.5e-02/2.2e-02; Cicero: 1.2e-02/1.9e-02; Gatsby: 4.1e-02/2.9e-02; Gatsby: 4.5e-03/3.3e-02

HMM cross-validated likelihood gain per bin over surrogates (POST), temporal shuffle / time swap: Achilles: 1.08/0.46; Achilles: 0.76/0.40; Buddy: 0.18/0.18; Cicero: 0.26/0.18; Cicero: 0.43/0.24; Cicero: 0.49/0.30; Gatsby: 0.43/0.27; Gatsby: 0.37/0.22
