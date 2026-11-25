# Waking un-mixing on real data

Design and bars: `docs/unmixing.md` (waking design). Outputs = templates passing the waking check on held-out events. Clean = every scored map represented and every output passing.

| data | track | criterion | fits | outputs passing | clean fits | fits with any output |
|---|---|---|---|---|---|---|
| hc-11 laps | linear | partial correlation (fixed) | 10 | 13/17 | 5/10 | 10/10 |
| hc-11 laps | linear | variance share | 10 | 12/17 | 4/10 | 10/10 |
| hc-11 laps | linear | original | 10 | 13/17 | 5/10 | 10/10 |
| hc-11 laps | circular | partial correlation (fixed) | 6 | 6/6 | 6/6 | 6/6 |
| hc-11 laps | circular | variance share | 6 | 6/6 | 6/6 | 6/6 |
| hc-11 laps | circular | original | 6 | 6/6 | 6/6 | 6/6 |
| hc-5 wheel | wheel | partial correlation (fixed) | 12 | 9/11 | 9/12 | 11/12 |
| hc-5 wheel | wheel | variance share | 12 | 9/11 | 9/12 | 11/12 |
| hc-5 wheel | wheel | original | 12 | 9/11 | 9/12 | 11/12 |
| hc-5 wheel (control) | wheel | partial correlation (fixed) | 12 | 0/0 | 12/12 | 0/12 |
| hc-11 laps (control) | linear | partial correlation (fixed) | 10 | 0/0 | 10/10 | 0/10 |
| hc-11 laps (control) | circular | partial correlation (fixed) | 6 | 0/0 | 6/6 | 0/6 |

## Per fit (partial-correlation criterion)

| data | session | fold | maps scored | outputs (map or fail) |
|---|---|---|---|---|
| hc-11 laps | Achilles_10252013 | 0 | 2 | 1, 0 |
| hc-11 laps | Achilles_10252013 | 1 | 2 | 0, 1 |
| hc-11 laps | Achilles_11012013 | 0 | 1 | 0 |
| hc-11 laps | Achilles_11012013 | 1 | 1 | 0 |
| hc-11 laps | Buddy_06272013 | 0 | 2 | fail, 0 |
| hc-11 laps | Buddy_06272013 | 1 | 2 | fail, fail |
| hc-11 laps | Cicero_09012014 | 0 | 2 | 1, 0 |
| hc-11 laps | Cicero_09012014 | 1 | 2 | 1 |
| hc-11 laps | Cicero_09102014 | 0 | 1 | 0 |
| hc-11 laps | Cicero_09102014 | 1 | 1 | 0 |
| hc-11 laps | Cicero_09172014 | 0 | 2 | fail |
| hc-11 laps | Cicero_09172014 | 1 | 2 | 1 |
| hc-11 laps | Gatsby_08022013 | 0 | 2 | 1, 0 |
| hc-11 laps | Gatsby_08022013 | 1 | 2 | 1, 0 |
| hc-11 laps | Gatsby_08282013 | 0 | 1 | 0 |
| hc-11 laps | Gatsby_08282013 | 1 | 1 | 0 |
| hc-11 laps (control) | Achilles_10252013 | 0 | 0 | none |
| hc-11 laps (control) | Achilles_10252013 | 1 | 0 | none |
| hc-11 laps (control) | Achilles_11012013 | 0 | 0 | none |
| hc-11 laps (control) | Achilles_11012013 | 1 | 0 | none |
| hc-11 laps (control) | Buddy_06272013 | 0 | 0 | none |
| hc-11 laps (control) | Buddy_06272013 | 1 | 0 | none |
| hc-11 laps (control) | Cicero_09012014 | 0 | 0 | none |
| hc-11 laps (control) | Cicero_09012014 | 1 | 0 | none |
| hc-11 laps (control) | Cicero_09102014 | 0 | 0 | none |
| hc-11 laps (control) | Cicero_09102014 | 1 | 0 | none |
| hc-11 laps (control) | Cicero_09172014 | 0 | 0 | none |
| hc-11 laps (control) | Cicero_09172014 | 1 | 0 | none |
| hc-11 laps (control) | Gatsby_08022013 | 0 | 0 | none |
| hc-11 laps (control) | Gatsby_08022013 | 1 | 0 | none |
| hc-11 laps (control) | Gatsby_08282013 | 0 | 0 | none |
| hc-11 laps (control) | Gatsby_08282013 | 1 | 0 | none |
| hc-5 wheel | i01_maze05_MS.005 | 0 | 1 | 0 |
| hc-5 wheel | i01_maze05_MS.005 | 1 | 1 | 0 |
| hc-5 wheel | i01_maze06_MS.002 | 0 | 1 | 0 |
| hc-5 wheel | i01_maze06_MS.002 | 1 | 1 | 0 |
| hc-5 wheel | i01_maze06_MS.005 | 0 | 1 | 0 |
| hc-5 wheel | i01_maze06_MS.005 | 1 | 1 | 0 |
| hc-5 wheel | i01_maze08_MS.001 | 0 | 1 | fail |
| hc-5 wheel | i01_maze08_MS.001 | 1 | 1 | fail |
| hc-5 wheel | i01_maze08_MS.004 | 0 | 1 | none |
| hc-5 wheel | i01_maze08_MS.004 | 1 | 1 | 0 |
| hc-5 wheel | i01_maze15_MS.002 | 0 | 1 | 0 |
| hc-5 wheel | i01_maze15_MS.002 | 1 | 1 | 0 |
| hc-5 wheel (control) | i01_maze05_MS.005 | 0 | 1 | none |
| hc-5 wheel (control) | i01_maze05_MS.005 | 1 | 1 | none |
| hc-5 wheel (control) | i01_maze06_MS.002 | 0 | 1 | none |
| hc-5 wheel (control) | i01_maze06_MS.002 | 1 | 1 | none |
| hc-5 wheel (control) | i01_maze06_MS.005 | 0 | 1 | none |
| hc-5 wheel (control) | i01_maze06_MS.005 | 1 | 1 | none |
| hc-5 wheel (control) | i01_maze08_MS.001 | 0 | 1 | none |
| hc-5 wheel (control) | i01_maze08_MS.001 | 1 | 1 | none |
| hc-5 wheel (control) | i01_maze08_MS.004 | 0 | 1 | none |
| hc-5 wheel (control) | i01_maze08_MS.004 | 1 | 1 | none |
| hc-5 wheel (control) | i01_maze15_MS.002 | 0 | 1 | none |
| hc-5 wheel (control) | i01_maze15_MS.002 | 1 | 1 | none |
