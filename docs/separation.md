# Do routes separate onto components? Per-component benchmark (repair step 2)

Code: `sim/two_routes.py`, `scripts/run_separation.py`. Results: `results/separation/`. Goal and criterion:
`docs/goals.md`. Read-out check: `docs/readout_check.md`.

## Design (fixed November 5 before running)

**Data.** `sim/two_routes.py`: 120 cells, 2500 events, constant-speed partial replays (segments of 50–100% of the
route), windows that follow the activity, per-cell background and latency, random assemblies, 5 draws. Two maps of
40 cells each, with these conditions:

| Condition | Maps | Replay shares | Directions |
|---|---|---|---|
| A | on different cells | 10% + 10% | both |
| B | on different cells | 13% + 7% | both |
| C | on the same cells | 10% + 10% | forward only |
| D | on the same cells | 10% + 10% | both |
| E | one map only | 20% | both |
| F | no replays | – | – |

**Methods.** Each is fitted on one half of the events, its outputs checked on the other half (both folds, so 10 fits
per condition):
1. complex PCA centred, no rotation;
2. complex PCA centred, ICA rotation;
3. complex PCA uncentred, ICA rotation;
4. Step 3 templates (`cpca.mixture.sequences` on the centred fit, without its internal held-out check).

**Outputs.** The components (or templates) that pass the latency check on the held-out half (p < 0.05). Raw
outputs, before the check, are also reported.

**Per-output criterion.**
- **Maps on different cells:**
  - at least 6 members on one map;
  - at least 80% of the members that lie on any map on that same map;
  - member order matches that map (|Spearman|, 1000 permutations, p < 0.05).
- **Maps on the same cells:** member order matches exactly one map (p < 0.05 for one map, p ≥ 0.05 for the other).
- **No replays:** every output counts as a failure.

**Measures per fit.**
- the number of outputs;
- the share of outputs that pass;
- the maps represented by at least one passing output;
- **clean** = every map of the condition is represented and every output passes (the goal).

## Results (`results/separation/summary.md`)

10 fits per cell. Outputs are those passing the latency check; "clean" = every map represented and every output
passes.

| Condition | Centred PCA | Centred ICA | Uncentred ICA | Templates |
|---|---|---|---|---|
| A: different cells, 10% + 10% | 2% pass, 0 clean | 0%, 0 | 0%, 0 | 3%, 0 |
| B: different cells, 13% + 7% | 12%, 0 | 0%, 0 | 0%, 0 | 13%, 0 |
| C: same cells, forward | 34%, 0 | 36%, 0 | 55%, 0 | **76%, 7 clean** |
| D: same cells, both ways | 58%, 1 | 51%, 0 | 49%, 0 | **84%, 6 clean** |
| E: one map | 49%, 0 | 66%, 2 | 38%, 0 | 61%, 4 |
| F: no replays (clean = no outputs) | 6 clean | 8 | 8 | 10 |

**Diagnostic (added after the results).** Weight on each route, for centred components in condition A: components
spread their weight over both routes (for example 0.43 and 0.32) and over off-route cells (0.15–0.48), with 50–100
members of 120 cells. The failures in A and B are real mixing, not a scoring artefact. Even with one map (E), only
the first component is dominated by the route (90–94% of its weight); the others are mostly background structure.

**Reading.** On data with realistic background structure, complex PCA's components (with or without ICA rotation)
are not one route each. The Step 3 templates come closest: clean in 6–7 of 10 fits when two maps share cells. They
fail when maps are on different cells, where their outputs are mixtures as well. **The goal is not met by any
method tested** in the conditions with two routes on different cells.

## Diagnostic 2: count PCA/ICA and the membership rule (added November 5, after the results)

**Question.** Do count PCA and count ICA also mix the routes, and how much of complex PCA's mixing comes from its
membership rule?

**Design.**
- **Data:** the same generator and conditions A and B (two routes on different cells), 5 fresh draws.
- **Fit:** one fit per method on all events, with no latency check.
- **Score:** membership only, since count methods give no order:
  - **one route:** at least 6 members on one route, and at least 80% of the members on any route on that one;
  - **mixed:** at least 6 route members, but less than 80% on one route;
  - **background:** fewer than 6 route members.
- **Membership rules:**
  - count methods: weight above the mean + 2 SD (the rule used with Lopes-dos-Santos et al. 2013);
  - complex PCA: its own rule (at least 20% of the largest weight) and the count methods' rule.

| Condition | Method | Outputs | One route | Mixed | Background | Median members | Fits with both routes on a one-route output |
|---|---|---|---|---|---|---|---|
| A | count PCA | 89 | 26 | 31 | 32 | 7 | 5/5 |
| A | count ICA | 89 | 36 | 4 | 49 | 7 | 5/5 |
| A | complex PCA, 20% rule | 45 | 0 | 45 | 0 | 74 | 0/5 |
| A | complex PCA, mean + 2 SD | 45 | 9 | 8 | 28 | 6 | 2/5 |
| A | complex ICA, mean + 2 SD | 45 | 8 | 6 | 31 | 5 | 3/5 |
| B | count PCA | 88 | 29 | 22 | 37 | 7 | 5/5 |
| B | count ICA | 88 | 34 | 5 | 49 | 6 | 5/5 |
| B | complex PCA, 20% rule | 45 | 3 | 42 | 0 | 78 | 0/5 |
| B | complex PCA, mean + 2 SD | 45 | 10 | 9 | 26 | 7 | 3/5 |
| B | complex ICA, mean + 2 SD | 45 | 8 | 7 | 30 | 5 | 2/5 |

**Reading.**
- **Count PCA mixes routes too:** 31 and 22 mixed outputs.
- **Count ICA unmixes them:** 4–5 mixed outputs, and both routes on a one-route output in every fit. Its background
  outputs are mostly the generator's random assemblies, which are real co-firing groups.
- **Complex PCA's 20% rule** makes every component a mixture by membership: 74–78 of 120 cells are members. With the
  sparser rule, part of the mixing goes away, but less well than for count ICA.
- **Caveats:**
  - This scores membership only, not order.
  - The sparse rule's 5–7 members cover only a small part of a 40-cell route.
  - One-map and no-replay conditions are not informative under this score: with one map, any output with 6 route cells
    counts as "one route".
