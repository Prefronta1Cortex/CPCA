# Project goal and success criteria (agreed November 5, 2026)

Every task and report is checked against this page. If a task does not test the goal directly, the report says so.

## Goal

From reactivation events, without the waking map, complex PCA (with any separation step it needs) outputs
components that each correspond to **one** route or sequence. The waking map is used only afterwards, to check and
interpret the components.

## Success criterion: per component

Every output component must be interpretable as **one** sequence. The measure is the share of output components
that pass; a component that mixes two sequences counts as a failure.

**Planted data**, where the routes are known. A component passes if:
- at least 80% of its members that belong to any route belong to the **same** route; and
- the members' order matches that route (|Spearman| with the route positions above the permutation null, p < 0.05).

The report also gives, per route, whether some passing component represents it.

**Real data**, where the routes are unknown. A component passes if:
- it is reproducible on held-out events, beyond the timing and latency nulls of `docs/sequence_discovery.md`; and
- if it matches the waking map, it does so for one map or running direction only, with the timing control.

These real-data criteria are provisional; they are fixed before any real-data run.

## What earlier results mean under this goal

Results scored with the best-output test (best component, best-of-outputs null) show that the order is present
among the components. They do not show that routes are separated onto components. Planted check, November 5
(centred complex PCA, no rotation; 4 draws): both routes on separate, well-matched components in 2/4 draws (two
routes on different cells), 2/4 (two maps on the same cells, forward) and 1/4 (the same, both directions). With
ICA rotation: 1/4, 4/4 and 2/4.

## Main problem: un-mixing (November 5)

Scored per component, complex PCA's kept components are mixtures of the routes, on planted data and on hc-11 laps
(`docs/separation.md`, `docs/rescore.md`). Choosing among them requires the waking map, which defeats the purpose of
a label-free method. Getting one route per output without the map, by a step after PCA or a change to it, is
therefore the main problem. Design: `docs/unmixing.md`.

**Status (November 5):** solved on the planted two-route generator by the bar fixed before the confirmation run (`cpca/separate.py`). Maps that share all their cells at low replay shares, and three maps on the same cells, remain below the bar.

**Waking test (November 19, `results/unmix_waking/`):** the latency check is left off. With the planted event floor (20), the wheel produces no template. With that floor lowered to 4, one wheel sequence stays one output and matches the held-out time fields in 9 of 12 fits. The two running directions of the linear-track laps do not separate: the split's best held-out similarity is 0.96–0.98 and is not kept at 0.97, so one direction is left without an output.

## Scope

- **Focus:** complex PCA itself, meaning the separation and interpretability of its components.
- **Keep:** the comparisons already done that matter for the paper.
- **Do not start:** new method comparisons, or new dataset searches, unless asked.

## Reporting

- Reports are brief.
- Every report states what the test checks.
- Results are reported per component and per route.
- "Order present" and "routes separated" are always kept apart.

**Waking separation (November 19, `docs/unmixing.md`):**
- **The decision:** the two running directions of a linear track count as two sequences, so waking data are analysed forward only.
- **Planted laps:** the bar is met.
- **Real data:** circular tracks 6/6 clean, the wheel 9/12, and both no-sequence controls produce no output.
- **Linear tracks:** 5/10 fits clean, against a bar of 7/10. The two directions separate in the sessions with enough laps and cells.
