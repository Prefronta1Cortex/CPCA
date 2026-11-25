# Simplified separation step: stage check and confirmation

Entries: outputs that are one route, and clean fits (every route its own output and every output one route; with no routes, clean = no output). Criteria: original / variance share / partial correlation for maps on shared cells.

## planted reactivations, development draws: one stage switched off at a time (original criterion)

| condition | full | without rotation | without refine | without explain | without split | without select |
|---|---|---|---|---|---|---|
| A different cells, 10% + 10%, both | 20/20 (100%), 10/10 | 10/20 (50%), 5/10 | 25/36 (69%), 5/10 | 20/32 (62%), 2/10 | 20/20 (100%), 10/10 | 20/27 (74%), 3/10 |
| B different cells, 13% + 7%, both  | 20/21 (95%), 9/10 | 20/20 (100%), 10/10 | 23/34 (68%), 3/10 | 20/30 (67%), 1/10 | 20/20 (100%), 10/10 | 20/28 (71%), 3/10 |
| C same cells, 10% + 10%, forward | 13/20 (65%), 6/10 | 16/20 (80%), 6/10 | 14/25 (56%), 4/10 | 15/29 (52%), 1/10 | 0/10 (0%), 0/10 | 13/31 (42%), 1/10 |
| D same cells, 10% + 10%, both ways | 12/21 (57%), 4/10 | 14/20 (70%), 6/10 | 19/36 (53%), 0/10 | 12/23 (52%), 4/10 | 1/11 (9%), 0/10 | 12/28 (43%), 0/10 |
| E one map, 20%, both ways | 10/10 (100%), 10/10 | 10/10 (100%), 10/10 | 21/24 (88%), 8/10 | 10/11 (91%), 9/10 | 10/10 (100%), 10/10 | 10/12 (83%), 8/10 |
| F no replays | 0/0 (0%), 10/10 | 0/0 (0%), 10/10 | 0/0 (0%), 10/10 | 0/0 (0%), 10/10 | 0/0 (0%), 10/10 | 0/43 (0%), 0/10 |

## planted laps, development draws: one stage switched off at a time (original criterion)

| condition | full | without rotation | without refine | without explain | without split | without select |
|---|---|---|---|---|---|---|
| W1 70% cells both ways, positions  | 8/17 (47%), 3/10 | 8/18 (44%), 3/10 | 9/26 (35%), 0/10 | 8/17 (47%), 3/10 | 0/10 (0%), 0/10 | 8/17 (47%), 3/10 |
| W2 70% both ways, rho 0.7 | 11/19 (58%), 5/10 | 11/19 (58%), 5/10 | 19/28 (68%), 4/10 | 11/19 (58%), 5/10 | 1/10 (10%), 0/10 | 11/19 (58%), 5/10 |
| W3 70% both ways, rho 0.2, 10 laps | 0/10 (0%), 0/10 | 0/10 (0%), 0/10 | 1/20 (5%), 0/10 | 0/10 (0%), 0/10 | 0/10 (0%), 0/10 | 0/10 (0%), 0/10 |
| W4 one direction, 40 laps | 10/10 (100%), 10/10 | 10/10 (100%), 10/10 | 10/10 (100%), 10/10 | 10/10 (100%), 10/10 | 10/10 (100%), 10/10 | 10/10 (100%), 10/10 |
| W5 control: spikes circularly shif | 0/0 (0%), 10/10 | 0/0 (0%), 10/10 | 0/0 (0%), 10/10 | 0/0 (0%), 10/10 | 0/0 (0%), 10/10 | 0/10 (0%), 0/10 |

## planted reactivations, confirmation draws, full step

| condition | original | variance share | partial |
|---|---|---|---|
| A different cells, 10% + 10%, both | 20/20 (100%), 10/10 | 20/20 (100%), 10/10 | 20/20 (100%), 10/10 |
| B different cells, 13% + 7%, both  | 20/20 (100%), 10/10 | 20/20 (100%), 10/10 | 20/20 (100%), 10/10 |
| C same cells, 10% + 10%, forward | 18/19 (95%), 9/10 | 18/19 (95%), 9/10 | 18/19 (95%), 9/10 |
| D same cells, 10% + 10%, both ways | 16/20 (80%), 8/10 | 20/20 (100%), 10/10 | 20/20 (100%), 10/10 |
| E one map, 20%, both ways | 10/10 (100%), 10/10 | 10/10 (100%), 10/10 | 10/10 (100%), 10/10 |
| F no replays | 0/0 (0%), 10/10 | 0/0 (0%), 10/10 | 0/0 (0%), 10/10 |

## planted laps, confirmation draws, full step

| condition | original | variance share | partial |
|---|---|---|---|
| W1 70% cells both ways, positions  | 14/17 (82%), 7/10 | 14/17 (82%), 7/10 | 13/17 (76%), 6/10 |
| W2 70% both ways, rho 0.7 | 12/19 (63%), 6/10 | 18/19 (95%), 9/10 | 18/19 (95%), 9/10 |
| W3 70% both ways, rho 0.2, 10 laps | 0/10 (0%), 0/10 | 0/10 (0%), 0/10 | 0/10 (0%), 0/10 |
| W4 one direction, 40 laps | 10/10 (100%), 10/10 | 10/10 (100%), 10/10 | 10/10 (100%), 10/10 |
| W5 control: spikes circularly shif | 0/0 (0%), 10/10 | 0/0 (0%), 10/10 | 0/0 (0%), 10/10 |

## Real waking data, full step (waking mode)

| data | original | variance share | partial |
|---|---|---|---|
| hc-11 laps (linear) | 8/11 (73%), 1/10 | 7/11 (64%), 1/10 | 8/11 (73%), 1/10 |
| hc-11 laps (circular) | 6/6 (100%), 6/6 | 6/6 (100%), 6/6 | 6/6 (100%), 6/6 |
| hc-5 wheel | 9/11 (82%), 9/12 | 9/11 (82%), 9/12 | 9/11 (82%), 9/12 |
