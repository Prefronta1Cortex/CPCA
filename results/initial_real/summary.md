# Initial real-data tests of the un-mixing step

Design: `docs/unmixing.md` (initial real-data tests).

## Waking, figure-eight (DANDI 000552)

- fits: 130; outputs: 133 (left arm 0, right arm 0, stem 0, fail 133)
- fits with both arms as separate outputs: 0/130; clean (both arms, no failing output): 0/130
- fits with any output: 127/130; median training trials 11

## Waking, 001695 laps

| criterion | outputs passing | clean fits | fits with any output |
|---|---|---|---|
| partial correlation | 42/54 | 12/30 | 30/30 |
| variance share | 43/54 | 14/30 | 30/30 |
| original | 36/54 | 12/30 | 30/30 |

## Reactivations, hc-11

Outputs passing the latency check; matching exactly one running direction (partial / variance share / original); chance per output about 9.5% with two maps, 5% with one.

| epoch | fits | outputs | partial | share | original | expected by chance | sessions with a match (partial) |
|---|---|---|---|---|---|---|---|
| maze | 16 | 6 | 1 | 1 | 1 | 0.6 | 1/8 |
| post | 16 | 16 | 3 | 3 | 2 | 1.2 | 2/8 |
| pre | 16 | 15 | 1 | 1 | 0 | 1.2 | 1/8 |

## Reactivations, 001695

Outputs passing the latency check; matching exactly one running direction (partial / variance share / original); chance per output about 9.5% with two maps, 5% with one.

| epoch | fits | outputs | partial | share | original | expected by chance | sessions with a match (partial) |
|---|---|---|---|---|---|---|---|
| maze | 30 | 25 | 4 | 5 | 4 | 2.4 | 3/15 |
| post | 30 | 25 | 0 | 0 | 0 | 2.4 | 0/15 |
| pre | 30 | 17 | 1 | 0 | 1 | 1.6 | 1/15 |

