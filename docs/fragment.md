# The recurring-fragment criterion

Code: `cpca/fragment.py`, `scripts/run_fragment.py`. Results: `results/fragment/`. Proposed after
`docs/maze_rest.md`.

## Why

The old order test is too strict and also not specific enough:
- **Too strict:** it asks one output to put its members across the whole track in the map's order. Real
  replays are often partial, and even perfect event selection by the map-based decoder recovers the whole-track
  order rarely on hc-11 (`docs/power.md`, Step 0).
- **Not specific enough:** co-firing alone can pass it (`docs/maze_rest.md`, timing control).

## Criterion (fixed on October 29 before running)

**Bar 1, the fragment.** For each output and map, the statistic is |Spearman| between the members' order and
their field centres, restricted to the members whose centres lie in one contiguous stretch.
- **The stretch** is 40% of the track, the planted segment length. Starts are every 5% of the track, and
  tracks are treated as linear (hc-11's circular track is a one-way run).
- **At least 8 members** must lie in the stretch.
- **The best is taken** over outputs, maps, senses and stretch positions.
- **The null** permutes each map's field centres among the cells with a field (1000 permutations; 500 for the
  planted checks) and takes the same best.

**Bar 2, timing.** The real statistic must beat the same refit (complex PCA, full pipeline) after spike trains
are shuffled among each event's active cells, which keeps who fires and destroys who fires when.
- **Shuffles:** 200 on hc-11, 50 for the planted checks.
- **Timing p** is the share of shuffled fits at least as high as the real one.

**A pass** requires p < 0.05 on both bars.

**Methods.**
- **Complex PCA, centred** (the rerun's read-out) is the primary convention.
- **Count PCA is scored on bar 1 only.** Its input is spike counts, which the timing shuffle leaves unchanged,
  so bar 2 cannot apply to it. Its outputs are each component's members ordered by weight, and the seriation of
  its reduced space.

**Events.**
- **POST:** all NREM bursts.
- **PRE:** the same number of NREM bursts, drawn at random (if PRE has fewer, POST is subsampled to PRE's number
  instead).
- **MAZE rest:** all immobility bursts on the maze.

**Validation on planted data first** (80 cells, 60 on one route, 2000 events, 5 draws):
- whole-route replays at 20%, and 40% segment replays at 20%: should pass;
- 40% segment replays at 10% in busy events: the hard case of `docs/ceiling.md`;
- assemblies along stretches of the map, firing without order: may pass bar 1, must fail bar 2;
- no replays: must fail.

**What would show that timing contributes:** complex PCA passes both bars in POST, in sessions where PRE does
not pass. Across 8 sessions, 3 or more passing by chance has p = 0.006.

## Results (`results/fragment/summary.md`)

**Planted data: the criterion works as intended.**

| Condition | Complex PCA, both bars | Count PCA, bar 1 |
|---|---|---|
| Whole-route replays, 20% | 5/5 | 0/5 |
| 40% segment replays, 20% | 4/5 | 5/5 |
| 40% segment replays, 10%, busy events | 4/5 | 4/5 |
| Assemblies along the map, no order | 0/5 | 4/5 |
| No replays | 0/5 (no outputs) | 0/5 |

- **It recovers partial replays,** including the busy-event case where the ceiling test gave complex PCA 16–24%.
- **The timing bar rejects co-firing.** On the no-order assemblies, count PCA passes bar 1 in 4/5 draws, but
  complex PCA never clears bar 2.
- **Bar 1 alone is not specific:** count PCA passes it whenever cells of one stretch co-fire, with or without
  order.

**hc-11: nothing passes.**

| Epoch | Complex PCA, both bars | Complex PCA, bar 1 | Complex PCA, bar 2 | Count PCA, bar 1 |
|---|---|---|---|---|
| POST | 0/8 | 1/8 | 1/8 | 0/8 |
| PRE | 0/8 | 0/8 | 0/8 | 0/8 |
| MAZE rest | 0/8 | 0/8 | 0/8 | 2/8 |

Real fragment correlations are high (|ρ| 0.7–0.9), but so is the null's best fragment. A 40% stretch holds only
about 8–20 of hc-11's place cells per direction, and the search over positions, components and maps makes chance
fragments look good.

**Power at hc-11's size (added after the hc-11 results).** On planted data with 25 cells on the route instead of
60 (20% replays, 4 draws, 30 timing shuffles):

| Condition | New criterion, both bars | Old whole-track test |
|---|---|---|
| Whole-route replays | 3/4 | 4/4 |
| 40% segment replays | 1/4 (bar 1: 4/4) | 4/4 |
| Assemblies, no order | 0/4 | 1/4 |

With few cells, co-firing within a segment reproduces much of its order, so the timing bar rarely clears. hc-11
carries far less replay than these planted sets (`docs/replay.md`).

## Reading

**What the criterion adds is specificity.** It is the first order criterion here that co-firing cannot pass,
and it credits partial replays. A pass under it would be a claim about timing.

**On hc-11 its power is too low to say anything.** Even 20% planted segment replays clear both bars in 1 of 4
draws at hc-11's cell counts. hc-11 has 0–3.4% excess clear replays in sleep and 2–18% on maze rest. That no
session passes is what this power predicts, not evidence that timing carries nothing.

**On maze rest, the earlier timing-specific signals do not survive the fragment search.** Gatsby 08/02 and
Cicero 09/10 beat the timing shuffle under the whole-track statistic (`docs/maze_rest.md`). The search over
stretches costs power.

**The criterion needs data with more cells per stretch.** DANDI 000552 (median 57 place cells per map) or
larger recordings are candidates. On hc-11, a pre-specified pair of tests is the best available, with a
Bonferroni correction: the fragment criterion for partial replays, and the whole-track statistic with the same
timing bar for whole replays.
