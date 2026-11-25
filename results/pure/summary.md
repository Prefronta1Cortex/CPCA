# Pure capabilities of complex PCA

Design: `docs/unmixing.md` (pure capabilities). One fit per row of the source; components read at 20% of the largest weight; no rotation, templates or split.

| data | case | fits | components kept (median) | routes with order present | routes inside the span | components that are one route | clean fits | latency-passing components that are one route |
|---|---|---|---|---|---|---|---|---|
| planted reactivations | A different cells, 10% + 10%, both | 5 | 9 | 10/10 | 10/10 | 1/45 | 0/5 | 1/41 |
| planted reactivations | B different cells, 13% + 7%, both  | 5 | 9 | 10/10 | 10/10 | 3/45 | 0/5 | 3/42 |
| planted reactivations | C same cells, 10% + 10%, forward | 5 | 7 | 6/10 | 10/10 | 8/35 | 0/5 | 7/30 |
| planted reactivations | D same cells, 10% + 10%, both ways | 5 | 8 | 10/10 | 10/10 | 22/42 | 0/5 | 19/35 |
| planted reactivations | E one map, 20%, both ways | 5 | 7 | 5/5 | 5/5 | 17/35 | 0/5 | 17/33 |
| planted reactivations | F no replays | 5 | 5 | – | – | 0/25 | 0/5 | 0/1 |
| planted laps | W1 both ways rho 0.2 | 5 | 2 | 10/10 | 10/10 | 5/10 | 0/5 | – |
| planted laps | W2 both ways rho 0.7 | 5 | 2 | 10/10 | 10/10 | 5/10 | 0/5 | – |
| planted laps | W3 10 laps each | 5 | 2 | 10/10 | 10/10 | 5/10 | 0/5 | – |
| planted laps | W4 one direction | 5 | 1 | 5/5 | 5/5 | 5/5 | 5/5 | – |
| planted laps | W5 control, no sequence | 5 | 1 | – | – | 0/6 | 0/5 | – |
| hc-5 wheel | all | 12 | 1 | 10/12 | 9/12 | 11/16 | 7/12 | – |
| hc-11 laps (circular) | all | 6 | 2 | 6/6 | 6/6 | 9/13 | 2/6 | – |
| hc-11 laps (linear) | all | 10 | 2 | 10/20 | 20/20 | 7/19 | 1/10 | – |
| 001695 laps (linear) | all | 30 | 2 | 50/60 | 54/60 | 35/67 | 6/30 | – |
| 000552 figure-eight | all | 130 | 2 | 175/260 | 152/260 | 0/253 | 0/130 | – |
| hc-11 maze | all | 16 | 3 | 10/26 | 18/26 | 20/55 | 1/16 | 20/48 |
| hc-11 post | all | 16 | 5 | 2/26 | 10/26 | 14/89 | 0/16 | 14/83 |
| hc-11 pre | all | 16 | 4 | 2/26 | 2/26 | 5/74 | 0/16 | 5/64 |
| 001695 maze | all | 30 | 5 | 26/60 | 39/60 | 51/154 | 0/30 | 48/141 |
| 001695 post | all | 30 | 5 | 9/60 | 13/60 | 25/170 | 0/30 | 23/165 |
| 001695 pre | all | 30 | 5 | 4/60 | 6/60 | 22/165 | 0/30 | 22/158 |

For data without a sequence (planted no-replay and no-sequence controls), clean = no component kept.
Components that are one route: variance-share criterion where maps share cells (the original criterion is in rows.pkl).
