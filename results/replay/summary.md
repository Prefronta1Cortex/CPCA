# hc-11: replay in sleep (the original paper's analysis) and how much complex PCA needs

Produced by `scripts/run_replay_hc11.py`, `scripts/run_spikein_hc11.py`, `scripts/summarize_replay.py`. Design: `docs/replay.md`.

## Bayesian replay analysis (paper's events)

| session | place cells (paper's criterion, per direction) | running decode error (m; shuffled) | events PRE / MAZE / POST | mean rZ PRE / MAZE / POST | clear replays PRE / MAZE / POST | POST − PRE (points) |
|---|---|---|---|---|---|---|
| Achilles_10252013 | 78, 84 | 0.04 (0.59); 0.04 (0.56) | 3943 / 277 / 2243 | 0.26 / 0.91 / 0.42 | 4.8% / 18.8% / 8.2% | +3.4 |
| Achilles_11012013 | 84 | 0.04 (1.04) | 1849 / 156 / 1773 | 0.01 / 0.34 / 0.15 | 2.6% / 7.7% / 5.1% | +2.5 |
| Buddy_06272013 | 19, 23 | 0.07 (0.62); 0.08 (0.45) | 1056 / 90 / 640 | 0.52 / 0.86 / 0.54 | 6.3% / 7.8% / 3.9% | -2.4 |
| Cicero_09012014 | 26, 26 | 0.09 (0.38); 0.08 (0.60) | 2096 / 902 / 2668 | 0.20 / 0.60 / 0.36 | 1.9% / 7.8% / 4.5% | +2.7 |
| Cicero_09102014 | 54 | 0.04 (0.55) | 2226 / 59 / 2957 | 0.14 / 0.57 / 0.15 | 3.2% / 5.1% / 3.2% | -0.1 |
| Cicero_09172014 | 36, 36 | 0.07 (0.48); 0.08 (0.64) | 2138 / 189 / 2706 | 0.30 / 0.81 / 0.27 | 4.5% / 15.9% / 4.7% | +0.2 |
| Gatsby_08022013 | 30, 35 | 0.06 (0.57); 0.05 (0.32) | 1669 / 272 / 1235 | 0.39 / 1.30 / 0.48 | 5.0% / 18.0% / 6.1% | +1.1 |
| Gatsby_08282013 | 35 | 0.05 (1.00) | 2407 / 514 / 2832 | -0.04 / 0.09 / -0.03 | 1.5% / 2.7% / 1.9% | +0.4 |

Our population-burst windows (the events every method was given), scored the same way:

| session | mean rZ PRE / POST | clear replays PRE / POST |
|---|---|---|
| Achilles_10252013 | 0.31 / 0.37 | 4.3% / 5.6% |
| Achilles_11012013 | 0.00 / 0.20 | 1.8% / 4.8% |
| Buddy_06272013 | 0.68 / 0.62 | 5.2% / 3.4% |
| Cicero_09012014 | 0.21 / 0.35 | 1.5% / 3.4% |
| Cicero_09102014 | 0.18 / 0.17 | 2.7% / 2.7% |
| Cicero_09172014 | 0.34 / 0.45 | 4.0% / 5.3% |
| Gatsby_08022013 | 0.58 / 0.53 | 7.7% / 4.6% |
| Gatsby_08282013 | 0.03 / -0.03 | 1.2% / 1.3% |

## Complex PCA on PRE events with replays added (share of 5 draws with p < 0.05)

| session | ideal 0% | ideal 1% | ideal 2% | ideal 3% | ideal 5% | ideal 10% | ideal 20% | awake 1% | awake 2% | awake 3% |
|---|---|---|---|---|---|---|---|---|---|---|
| Achilles_10252013 | 2/5 | 0/5 | 0/5 | 0/5 | 1/5 | 5/5 | 5/5 | 0/5 | 0/5 | 0/5 |
| Achilles_11012013 | 0/5 | 1/5 | 0/5 | 0/5 | 0/5 | 4/5 | 5/5 | – | – | – |
| Buddy_06272013 | 0/5 | 0/5 | 0/5 | 1/5 | 0/5 | 1/5 | 1/5 | – | – | – |
| Cicero_09012014 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 3/5 | 3/5 | 0/5 | 0/5 | 0/5 |
| Cicero_09102014 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | – | – | – |
| Cicero_09172014 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 5/5 | 0/5 | 0/5 | – |
| Gatsby_08022013 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 1/5 | 0/5 | 0/5 | 0/5 |
| Gatsby_08282013 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 1/5 | 0/5 | – | – | – |
