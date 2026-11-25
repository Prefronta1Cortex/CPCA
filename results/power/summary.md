# hc-11: the bound from perfect event choice (Step 0) and how much replay each method needs (Step 1)

Produced by `scripts/run_oracle_hc11.py`, `scripts/run_power_hc11.py`, `scripts/summarize_power.py`. Design: `docs/power.md`.

## Step 0: splits (of 10) whose order matches the map on the held-out half of the place cells

| session | epoch | clear replays chosen (median) | decoder, clear replays | decoder, loose | random events | complex PCA (no map) |
|---|---|---|---|---|---|---|
| Achilles_10252013 | POST | 38 of 916 testable | 3/10 | 6/10 | 2/10 | 5/10 |
| Achilles_10252013 | PRE | 30 of 962 testable | 0/10 | 0/10 | 1/10 | 2/10 |
| Achilles_11012013 | POST | 17 of 817 testable | 0/10 | 0/10 | 1/10 | 2/10 |
| Achilles_11012013 | PRE | 11 of 818 testable | 2/10 | 1/10 | 0/10 | 2/10 |
| Buddy_06272013 | POST | 3 of 102 testable | 0/10 | 1/10 | 1/10 | 0/10 |
| Buddy_06272013 | PRE | 6 of 172 testable | 0/10 | 2/10 | 0/10 | 0/10 |
| Cicero_09012014 | POST | 10 of 408 testable | 0/10 | 0/10 | 0/10 | 1/10 |
| Cicero_09012014 | PRE | 4 of 284 testable | 0/10 | 0/10 | 1/10 | 2/10 |
| Cicero_09102014 | POST | 14 of 791 testable | 1/10 | 1/10 | 0/10 | 1/10 |
| Cicero_09102014 | PRE | 14 of 801 testable | 0/10 | 1/10 | 0/10 | 0/10 |
| Cicero_09172014 | POST | 31 of 808 testable | 0/10 | 0/10 | 0/10 | 0/10 |
| Cicero_09172014 | PRE | 31 of 748 testable | 2/10 | 0/10 | 1/10 | 0/10 |
| Gatsby_08022013 | POST | 18 of 442 testable | 1/10 | 2/10 | 0/10 | 3/10 |
| Gatsby_08022013 | PRE | 20 of 472 testable | 2/10 | 0/10 | 0/10 | 0/10 |
| Gatsby_08282013 | POST | 4 of 369 testable | 0/10 | 2/10 | 0/10 | 1/10 |
| Gatsby_08282013 | PRE | 2 of 224 testable | 0/10 | 1/10 | 0/10 | 0/10 |

## Step 1: share of draws whose best output matches the map

| method | sessions | 0% | 1% | 2% | 3% | 5% | 10% | 20% |
|---|---|---|---|---|---|---|---|---|
| Complex PCA (1500 events) | two largest | 2/10 | 1/10 | 0/10 | 0/10 | 1/10 | 9/10 | 10/10 |
| Complex PCA (1500 events) | other six | 0/30 | 0/30 | 0/30 | 1/30 | 0/30 | 5/30 | 10/30 |
| Complex PCA (all events) | two largest | 0/10 | 0/10 | 0/10 | 0/10 | 2/10 | 8/10 | 10/10 |
| Complex PCA (all events) | other six | 0/30 | 0/30 | 0/30 | 3/30 | 3/30 | 5/30 | 14/30 |
| Event-sparse templates (raw match) | two largest | 0/10 | 0/10 | 0/10 | 0/10 | 3/10 | 2/10 | 8/10 |
| Event-sparse templates (raw match) | other six | 0/30 | 1/30 | 0/30 | 1/30 | 2/30 | 1/30 | 11/30 |
| Event-sparse templates (cosine match) | two largest | 0/10 | 0/10 | 0/10 | 0/10 | 1/10 | 0/10 | 1/10 |
| Event-sparse templates (cosine match) | other six | 0/30 | 1/30 | 1/30 | 0/30 | 0/30 | 0/30 | 2/30 |
| Rastermap | two largest | 0/10 | 1/10 | 0/10 | 0/10 | 0/10 | 1/10 | 3/10 |
| Rastermap | other six | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 1/30 | 1/30 |
| PP-Seq (fixed setting) | two largest | 0/2 | – | – | – | 0/2 | 0/2 | 0/2 |

## Runtime per fit (median, s)

| method | seconds |
|---|---|
| Complex PCA (all events) | 4.8 |
| Event-sparse templates (raw match) | 5.7 |
| Event-sparse templates (cosine match) | 4.5 |
| Rastermap | 7.2 |
| PP-Seq (fixed setting) | 760.8 |
