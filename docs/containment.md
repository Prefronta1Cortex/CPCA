# Are the waking components found within the sleep components?

Code: `cpca/containment.py`, `scripts/run_containment.py`. Results: `results/containment/`. Replaces the
symmetric comparison of `docs/wake_sleep.md` as the sanity check of waking against sleep.

## Why

Sleep replays only part of what was experienced, and waking activity holds structure sleep never replays. The
expected relation is one-sided: what the sleep fit finds should lie inside what the waking fit finds, not the
reverse. The earlier check correlated the two fits' cell-pair closeness over all pairs. That is symmetric, so
pairs close in waking but never replayed count against agreement.

## Design (fixed on November 2 before running)

**Fits, on the same cells** (active in at least 5 waking events and at least 1% of the sleep events):
- the waking fit: uncentred, on the laps;
- the sleep fit: centred, on the bursts;
- the kept components of each (shuffle-null bound).

**Statistic.** For each kept sleep component, the share of it inside the span of the kept waking components and
their conjugates (a reversed replay is the conjugate): ||B^H v||², from 0 to 1. The score is the mean over
sleep components.

**Bar 1.** The same score with the waking fit's cell identities permuted, which keeps the subspace's dimension
(1000 permutations; 500 for the planted checks).

**Bar 2, timing.** The same score after refitting sleep on spike trains shuffled among each event's active
cells (100 shuffles; 50 planted). This keeps firing rates and co-firing, so it also controls for cells that fire
a lot loading heavily in both fits. The timing p is the share of shuffled fits at least as high.

**A data set passes if both p < 0.05.** The reverse direction (waking components inside the sleep fit) is
reported for description; the subset relation lets it stay low.

**Planted checks** (80 cells; 60 waking laps, each playing one of two maps on the same 60 cells forward; 3000
sleep events; 5 draws):

| Sleep condition | Expected |
|---|---|
| Both maps replayed | pass |
| One map only (the subset case) | pass |
| 40% segments of the maps | pass |
| Other maps on the same cells | fail |
| Assemblies along the map, no order | fail bar 2 |
| No replays | no sleep components |

**hc-11.**
- **POST:** all NREM bursts.
- **PRE:** the same number of NREM bursts, drawn at random; if PRE has fewer, POST is subsampled instead.

The sanity check counts as passed if POST passes in more sessions than PRE.

## Planted check: the design fails on segments (hc-11 not run)

| Sleep condition | Both bars | Containment, real / cell null / timing-shuffled |
|---|---|---|
| Both maps replayed | 5/5 | 0.99 / 0.29 / 0.94 |
| One map only (subset) | 5/5 | 0.99 / 0.42 / 0.94 |
| 40% segments of the maps | 0/5 | 0.40 / 0.14 / 0.62 |
| Other maps on the same cells | 0/5 | 0.32 / 0.29 / 0.93 |
| Assemblies along the map, no order | 0/5 | 0.30 / 0.06 / 0.30 |
| No replays | no sleep components | – |

Two flaws in the design:
1. **Span containment cannot represent fragments.** A sleep component on one stretch's cells is not a linear
   combination of whole-route waking components.
2. **Including the conjugates puts timing-free directions in the waking span.** A component plus its conjugate
   is a real vector, so timing-shuffled sleep components look contained, and bar 2 rejects true segment replays.

A revised statistic is proposed below: the lags within each sleep component's cells, compared with the waking
fit's lags.

## Revised design (November 2, after discussion, before the revised runs)

The check is about what complex PCA outputs, its components: whatever the method does to the data, its
components on sleep after the maze should be found within its components on the waking laps.
- **No timing bar.** Requiring timing beyond co-firing would recreate the main task, which a sanity check
  should not do.
- **No PRE.** Only POST against waking.
- **The span statistic and the cell-identity null are kept from the first design** (p < 0.05 passes). With
  timing dropped, the problems of the first design no longer bear on the check, because segment replays pass
  the cell-identity null.
- **Added, for description:**
  - each sleep component's own containment, against the same null;
  - its best match with a single waking component (or its conjugate), against the same null.

  The span is the primary measure, because PCA components can mix within near-equal eigenvalues.
- **Expected:** shared co-firing counts as "seen within waking" here, as it should for a sanity check.

## Results (`results/containment/summary.md`)

**Planted data (5 draws per condition):**

| Sleep condition | Draws passing | Span containment (median) vs null | Components contained individually |
|---|---|---|---|
| Both maps replayed | 5/5 | 0.99 vs 0.29 | 16/16 |
| One map only | 5/5 | 0.99 vs 0.42 | 10/10 |
| 40% segments of the maps | 5/5 | 0.40 vs 0.14 | 37/39 |
| Other maps on the same cells | 1/5 | 0.32 vs 0.29 | 3/17 |
| Assemblies along the map, no order | 5/5 | 0.30 vs 0.06 | 20/25 |
| No replays | 0/5 | no sleep components in 4 of 5 draws | – |

**hc-11, sleep after the maze:**
- **Sessions:** the sleep components lie in the waking fit's span more than chance in **7 of 8** sessions
  (p ≤ 0.046). The exception is Cicero 09/17, with only 20 laps.
- **Components:** 27 of 53 sleep components are contained individually beyond chance, and 25 of 53 are matched
  by a single waking component beyond chance. In Achilles 10/25, 9 of 11 are contained.
- **Size of the shared part:** containment is 0.18–0.33 against 0.11–0.27 by chance, and a single waking
  component matches a median of 0.04–0.15 of a sleep component.

**Reading.** The implementation passes the sanity check. Complex PCA's components of sleep after the maze are
found, in part, within its components of waking activity, in nearly every session. The shared part is modest:
the sleep components also carry much that the waking fit does not (the waking fit has only 2–3 components). As
designed, this does not say whether the shared part is timing or co-firing.

## Final design: waking inside sleep (November 2, after discussion, before running)

**The question.** Whatever complex PCA finds on the clean waking laps should leave traces in what it finds on
the messier sleep reactivations.

**Why this direction.** The sleep fit also holds content that is not about the maze, such as volleys,
assemblies and other environments. In the sleep-inside-waking direction, that content counted against the score.
In this direction it does not.

**Statistic.** Each kept waking component's share inside the span of the kept sleep components and their
conjugates. The session score is the mean over waking components.

**Null.** The same score with the sleep fit's cell identities permuted, 1000 times (500 planted). This keeps
the sleep span's dimension, which matters because the sleep fit has more components than the waking fit.

**Passing.** p < 0.05. There is no timing bar, no PRE, and no reverse direction.

**Expected.** Waking structure is found in sleep when sleep replays both maps, one map or segments. It is not
found when sleep replays other maps on the same cells. Because sleep may replay only part of waking, the share
need not be near 1.

## Results (`results/containment/summary.md`)

**Planted data (5 draws per condition):**

| Sleep condition | Draws passing | Waking share inside sleep (median) vs null | Waking components contained |
|---|---|---|---|
| Both maps replayed | 5/5 | 0.95 vs 0.34 | 10/10 |
| One map only | 5/5 | 0.64 vs 0.31 | 10/10 |
| 40% segments of the maps | 5/5 | 0.87 vs 0.44 | 10/10 |
| Other maps on the same cells | 1/5 | 0.37 vs 0.34 | 4/10 |
| Assemblies along the map, no order | 5/5 | 0.42 vs 0.16 | 10/10 |
| No replays | 0/5 | no sleep components in 4 of 5 draws | – |

**hc-11, sleep after the maze:**
- **6 of 8 sessions pass:** the waking components lie in the sleep fit's span more than chance (p ≤ 0.042).
  The shares are 0.45–0.69 against 0.35–0.60 by chance.
- **12 of 18 waking components are contained individually.**
- **The two sessions that fail** are Buddy 06/27, the fewest cells (32) and only 3 sleep components, and
  Cicero 09/17, with only 20 laps.

**Reading.** The implementation passes the sanity check. What complex PCA finds on the waking laps is found
again, beyond chance, in what it finds on sleep after the maze, in 6 of 8 sessions. As designed, this does not
say whether the shared structure is sequence order or co-firing (map-ordered co-firing passes on planted data).
