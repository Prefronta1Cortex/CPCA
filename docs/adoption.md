# Adopting parts of the exploration (`exploration/cpca_v2.py`), November 25

The exploration proposed five changes. Three are taken up here, at the current half-circle arc (α = π):
the corrected complex ICA, centring on active entries with its matching shuffle, and the ICA by gradient
ascent on unitary rotations. The wider arc (1.5π) and the pairwise merge rules are not.

## What was checked before adopting

Independent re-runs, fresh seeds, the project's own scoring (`exploration/verify_adoption.py`,
`verify_ablation.py`, `verify_kept.py`).

- **No leakage in the exploration's waking results.** Each fit uses one half of the laps or runs; the answer key
  comes from the other half.
- **Its numbers reproduce at 1.5π:** hc-5 aligned 11/12 clean fits, unaligned 12/12; hc-11 linear 6/10, circular
  5/6; spike-in 9/10 at 5%; 1/160 timing-shuffled refits pass the order test on rest on the maze.
- **At π (what is adopted here), on reactivations:**
  - spike-in, two largest hc-11 sessions: 0/10 at 0%, 7/10 at 3%, 10/10 at 5% (column centring: 1/10 at 5%);
  - rest on the maze: 0/160 timing-shuffled refits pass the order test.
- **One part at a time (spike-in at 3% and 5%; timing-shuffled refits passing):**
  - without centring: 0/10 and 0/10; 0/160;
  - without the per-event phase alignment: 5/10 and 8/10; 4/160;
  - without ICA: 8/10 and 9/10; 0/160.

  The centring gives the gain. ICA does not change detection.
- **At π, on waking data,** the exploration's version is worse than the current pipeline (hc-5 6/12 against
  9/12, hc-11 linear 3/10 against 5/10). Its waking gains come from the wider arc.
- **At π, planted reactivations A–M,** the exploration's whole method gives 102/130 clean fits against about 103
  for the current pipeline. It is better on weak or shared maps (I: 8 against 1; K: 9 against 5) and worse on
  unequal rates, one map and many cells per route (B 6 against 10; E and M 8 against 10; J 6 against 10).

### A claim that does not hold: centring on active entries does not remove co-firing

The exploration states that centring on active entries removes co-firing and fixed latencies exactly. That is
true only when every event has the same clock. With the half-circle encoding, every cell has a preferred phase
within the event window (all its phases lie in [0, π]). When the event's start varies, cells that fire together
share that variation. The cross term of two cells is then

  E[(s_k − μ_k) conj(s_l − μ_l)] = μ'_k conj(μ'_l) (1 − |E e^{jδ}|²),

where μ' is each cell's mean arrow and δ the event's shift. This term is not zero. The shuffle within active events moves each cell on its own,
so it does not keep δ. The per-event alignment does not remove it either: it replaces δ by the error of the
event's own mean phase.

On four co-firing assemblies, each cell at its own phase plus jitter, independently of the others (3000
events; `tests/test_active.py`):

- events with the same clock, no alignment: 0 components kept;
- with alignment, or with a ±0.3 rad shift per event: 2–4 kept (column centring: 3).

On real hc-11 PRE events with nothing spiked in (1500 per session, π, no ICA), it keeps 20 components over
the 8 sessions, against 36 with column centring. No session's best output passes the order test.

**Reading.** Centring on active entries reduces what co-firing adds; it does not remove it. Outputs still need
the held-out timing check and the map score. The exploration's controls count only outputs that pass the order
test, so they did not show this.

### A finding on the ICA itself

The conjugation fix alone is not enough. The corrected fixed-point ICA (Bingham and Hyvärinen) does not
reliably split two routes on shared cells that each take part in about 30% of events
(`tests/test_pipeline.py`). In 6 data draws its weaker route's best match was 0.57–0.85 (the test's bar is
0.9), and in 2 draws one output carried both routes. Gradient ascent of Σ E|y|³ over unitary rotations split
them in all 6 (best match 0.94–0.98). It is now the default in `cpca/ica.py`; the
fixed-point version is kept as `method="fastica"`.

**Every earlier result with ICA on** (`cpca.fit` default `ica=True`, and the ICA starts in the split stage of
`cpca/separate.py`) was produced with the broken ICA.

## Code changes

- `cpca/ica.py`: both conjugation errors fixed; `riemannian_ica` added and made the default.
  Test: `tests/test_ica.py` (known sources recovered; fails on the old code).
- `cpca/active.py` (new): `align_events`, `centre_active`, `shuffle_null_active`. `prepare` and `fit` accept
  `center="active"`. Scaling is unchanged (each cell divided by its root-mean-square over all events). The noise
  bound stays the project's 95th percentile (the exploration used 99). Tests: `tests/test_active.py`.
- `cpca/separate.py`: `rotation="ica"` seeds the templates from the ICA patterns instead of quartimax. The default
  is unchanged.

## Step 2 design: detection (written before running)

Spike-in of `docs/power.md`: all 8 hc-11 sessions, the matched 1500 PRE events, ideal replays, 5 draws per
share. Shares: 0, 2, 3, 5, 10 and 20%. Scoring: the best output's order against the map, p < 0.05.
`cpca.fit` at π, ICA off, 95% bound, column centring and active centring on the same draws.

**Bar.** Active centring becomes the reactivation default if both hold:

- at 0%, no more hits than column centring in either group (two largest, other six);
- at 5%, in the two largest sessions, at least 5 more hits than column centring.

## Step 3 design: separation (written before running)

Planted reactivations A–F (confirmation draws) and G–M (stress confirmation draws): 5 draws × 2 folds, the
halves and scoring of `scripts/run_separation_check.py`. Variants, all at π:

- **V0, current:** column centring, then the separation step (quartimax seeds).
- **V1:** active centring, then the separation step (quartimax seeds).
- **V2:** active centring, then the separation step seeded by the ICA patterns.
- **V3:** active centring and ICA, patterns read out directly. A pattern is dropped if it is alike
  (`cpca.separate.similarity` ≥ 0.5, direction-blind) to a stronger one. No held-out selection.

Score: clean fits (every route its own output and every output one route; with no routes, no output). Maps on
shared cells use the variance-share criterion; the original criterion is reported alongside.

**Bar.**

- A variant replaces V0 if it has at least 5 more clean fits over the 130;
- and no condition is worse than V0 by more than 2 of 10;
- and F (no replays) has at least 9/10 clean.

If several pass, the one with most clean fits wins; ties go to the simpler one (V3, then V1, then V2).

**Waking re-check.** The fixed ICA changes the split stage in waking mode too. Planted laps W1–W5 (confirmation
draws), hc-11 laps and hc-5 are re-run with the current step. Bar: no set loses more than 1 clean fit against
`results/separation_check/summary_restored.md`.

Output: `results/adoption/`.

## Results (`scripts/run_adoption_check.py`, `results/adoption/summary.md`)

### Step 2: detection

Hits (best output's order p < 0.05), with the median number of components kept in brackets:

| sessions | centring | 0% | 2% | 3% | 5% | 10% | 20% |
|---|---|---|---|---|---|---|---|
| two largest | column | 0/10 (6) | 2/10 (6) | 0/10 (6) | 1/10 (6) | 10/10 (6) | 10/10 (7) |
| two largest | active | 0/10 (3) | 4/10 (3) | 8/10 (3) | 9/10 (4) | 10/10 (4) | 10/10 (4) |
| other six | column | 0/30 (4) | 1/30 (4) | 0/30 (4) | 0/30 (4) | 5/30 (4) | 10/30 (4) |
| other six | active | 1/30 (2) | 2/30 (2) | 9/30 (2) | 12/30 (2) | 15/30 (2) | 21/30 (3) |

- **Gain at 5%:** active centring found 9/10 replays in the two largest sessions, against 1/10 with column
  centring (8 more; the bar was 5). The threshold falls from about 10% to about 3–5%.
- **The six smaller sessions gain too:** 12/30 against 0/30 at 5%, and 21/30 against 10/30 at 20%.
- **Fewer components:** active centring keeps about half as many.
- **The false-positive bar fails by one hit:** at 0%, the six smaller sessions give 1/30 hits with active
  centring, against 0/30 with column centring. 1/30 is within the test's nominal 5% rate, but the bar was "no
  more than column centring".

**Decision (by the bar): active centring does not become the default.** It stays available as
`center="active"`. Loosening the bar now would be a choice made after seeing the result. It is left to the user,
and if taken it should be recorded as such.

### Step 3: separation

Clean fits of 10 (variance share for maps on shared cells; original criterion in brackets):

| condition | V0 | V1 | V2 | V3 |
|---|---|---|---|---|
| A different cells, 10% + 10%, both ways | 9 (9) | 10 (10) | 10 (10) | 7 (7) |
| B different cells, 13% + 7%, both ways | 10 (10) | 9 (9) | 9 (9) | 8 (8) |
| C same cells, 10% + 10%, forward | 10 (10) | 10 (9) | 10 (10) | 5 (5) |
| D same cells, 10% + 10%, both ways | 10 (8) | 10 (7) | 10 (7) | 8 (8) |
| E one map, 20%, both ways | 10 (10) | 10 (10) | 10 (10) | 7 (7) |
| F no replays | 10 (10) | 10 (10) | 10 (10) | 8 (8) |
| G three routes, different cells, 7% each | 9 (9) | 10 (10) | 10 (10) | 8 (8) |
| H different cells, 5% + 5% | 8 (8) | 9 (9) | 9 (9) | 9 (9) |
| I different cells, 3% + 3% | 1 (1) | 1 (1) | 1 (1) | 7 (7) |
| J different cells, 25 cells per route, 10% + 10% | 10 (10) | 10 (10) | 10 (10) | 3 (3) |
| K same cells, 5% + 5%, both ways | 5 (5) | 6 (5) | 7 (7) | 8 (8) |
| L three maps, same cells, 7% each | 9 (5) | 9 (6) | 5 (3) | 4 (3) |
| M one map, 5% | 10 (10) | 10 (10) | 10 (10) | 8 (8) |
| total | 111 (105) | 114 (106) | 111 (106) | 90 (89) |

- **V1** (active centring, then the separation step) has 3 more clean fits than V0. No condition is more than 1
  worse. It does not reach the bar of 5.
- **V2** (ICA patterns as seeds) has the same total as V0. It loses on L (5 against 9).
- **V3** (ICA read out directly) is better on the weakest and shared cases (I: 7 against 1; K: 8 against 5). It
  is much worse elsewhere (J 3, C 5, F 8), so ICA alone does not replace the separation step.

**Decision: V0 stays.** The separation step keeps its quartimax seeds.

### Waking re-check

| data | after the ICA fix | before (`summary_restored.md`) |
|---|---|---|
| hc-11 laps, circular | 6/6 | 6/6 |
| hc-11 laps, linear | 5/10 | 5/10 |
| hc-5 wheel | 9/12 | 9/12 |
| planted laps W1–W5 (original criterion) | 10, 6, 7, 10, 10 | 10, 7, 7, 10, 10 |

No set loses more than 1 clean fit (W2: 6 against 7 on the original criterion; 10/10 on the variance share
before and after). The bar passes, so waking stays as it is.

### What is adopted

- **ICA:** the conjugation fix, with gradient ascent on unitary rotations as the engine of `cpca/ica.py`. This
  is a bug fix. The current pipeline's reactivation numbers with it (V0): 111/130 clean fits.
- **Active centring:** available as `center="active"`, not the default. It is clearly more sensitive, and it
  missed its false-positive bar by one hit.
- **Separation:** unchanged.
- **Not adopted:** the 1.5π arc and the pairwise merge rules (not tested here).

## The step 2 bar was badly designed (November 25)

The false-positive bar, "at 0%, no more hits than column centring", could not be passed reliably. It compared
against 0 hits, which demands zero false positives from a test run at the 5% level. Its 30 "0%" fits were
also not 30 independent tests: with nothing spiked in, a session's five draws use the same events and differ only
in random seeds. The single hit was one of Gatsby 08/02's five draws; the other four on the same data did not
pass. The bar is replaced by the check below, written before running. The step 2 decision above stands until
then.

## Step 4 design: false positives of active centring (written before running)

- **Data:** all 8 hc-11 sessions, two event sets:
  - the matched 1500 PRE events of the spike-in;
  - the events of rest on the maze (where column centring's outputs passed the order test on 87–98% of
    timing-shuffled copies, `docs/maze_rest.md`).
- **Shuffle:** for each, 20 timing-shuffled copies (`scripts/run_maze_rest.shuffle_timing`): within each event,
  the active cells swap spike trains. Which cells fire is kept; the order is destroyed.
- **Fits:** `cpca.fit` at π, ICA off, column and active centring. Score: the best output's order against the map,
  p < 0.05.
- **Bar:** active centring becomes the default for reactivation events if, in each event set, no more than
  8 of the 160 shuffled fits pass (5%). Column centring is reported alongside; it is not part of the bar.

## Step 5 design: the arc, π against 1.5π (written before running)

The arc is a basic choice, so the bar is strict. `encode`, `fit` and `separate` take `arc` (default π). Only
the arc changes; data, halves, seeds and scoring are those of step 3.

- **Planted reactivations A–M:** V0 (column centring) and V1 (active centring), each followed by the separation
  step, at both arcs.
- **Waking:** the current step (uncentred, waking mode) at both arcs, on planted laps W1–W5, hc-11 laps and hc-5.
- **Spike-in:** all 8 sessions at shares 0, 3, 5 and 10%, 5 draws, both centrings, both arcs.

**Bar.** 1.5π replaces π only if, for the centring that is the reactivation default after step 4, all of these
hold:

- **Planted:** at least 5 more clean fits of the 130, no condition more than 2 of 10 worse, F at least 9/10.
- **Waking:** no set loses more than 1 clean fit.
- **Spike-in:** hits at 3% and 5% together not lower (all 8 sessions). At 0%, no more than 5% of the 40 fits
  (2 hits).

Output: `results/adoption/arc_fp/`.

## Results of steps 4 and 5 (`scripts/run_arc_and_fp.py`, `results/adoption/arc_fp/summary.md`)

### Step 4: false positives (arc π)

Timing-shuffled fits whose best output passes the order test, with the median number of components kept:

| events | column | active |
|---|---|---|
| PRE, matched 1500 | 11/160 (4) | 7/160 (1) |
| rest on the maze | 45/160 (3) | 0/160 (0) |

Active centring passes the bar (at most 8/160 in each set). Column centring fails it on rest on the maze, where 28%
of shuffled fits pass.

### Step 5: the arc

Spike-in hits, all 8 sessions (40 fits per cell):

| arc | centring | 0% | 3% | 5% | 10% |
|---|---|---|---|---|---|
| π | column | 0 | 0 | 1 | 15 |
| π | active | 1 | 17 | 21 | 25 |
| 1.5π | column | 2 | 4 | 10 | 21 |
| 1.5π | active | 0 | 9 | 18 | 27 |

Planted reactivations: clean fits of 130, variance share (original criterion in brackets):

| centring | π | 1.5π |
|---|---|---|
| column | 111 (105) | 120 (100) |
| active | 114 (106) | 125 (101) |

At 1.5π, I (3% + 3%) rises from 1 to 8 and K from 6 to 8. On the original criterion, maps on shared cells fall:
for active centring, C from 9 to 3, D from 7 to 4 and L from 6 to 1.

Waking, current step, clean fits (original criterion):

| data | π | 1.5π |
|---|---|---|
| hc-11 laps, circular | 6/6 | 6/6 |
| hc-11 laps, linear | 5/10 | 3/10 |
| hc-5 wheel | 9/12 | 10/12 |
| planted laps W1, W2, W3, W4 | 10, 6, 7, 10 | 7, 1, 7, 10 |
| planted laps W5 (control, no sequence) | 10/10 | 8/10 |

**Decision on the arc: π stays.** For active centring, 1.5π passes on planted reactivations (+11) but fails on
waking data: hc-11 linear loses 2, W1 loses 3, W2 loses 5, and the no-sequence control W5 gains false outputs in
2 fits. It also fails on spike-in (3% and 5% together: 27 against 38). The wider arc helps weak routes on
different cells (I) and costs elsewhere, as expected from the exploration's own results.

### A problem the bars did not test: the PCA read-out after active centring

Making active centring the default in `cpca.fit` broke three tests (`tests/test_pipeline.py`, `test_pairs.py`,
`test_subspace.py`). The default was put back to column centring.

**What happens** (sim/routes, two routes on different cells, 30% replays, forward only):

- **Separation improves.** Each route sits on one component (80% of its weight, against about 50% with column
  centring).
- **The read-out order does not.** The best match to the map is |ρ| ≈ 0.5, against 1.0 with column centring.
  Alignment alone, or no centring, gives 1.0.

**Cause, partly understood.** A cell fires in replays at its own phase θ and otherwise in background events, whose
arrows average to a point b near π/2. Centring on active entries leaves the replay entries proportional to
e^{jθ} − b. In the components:

- the weights grow with distance from the route's middle (rank correlation 0.8–0.99);
- in some routes the phases spread over about 245° instead of 180°.

The 20% member threshold drops the weak middle cells. That opens the widest gap in the middle of the route, the
read-out cuts there, and the order comes out folded.

This explains only part of it. With all 40 route cells kept, ρ is +1.00 in the 3 of 8 routes that span about 245°.
In the other 5 (span 164–177°) it stays about −0.5: their phases themselves are folded, for a reason not yet
found. With the 20% threshold, ρ is about −0.5 in all 8.

**What this means for the results above.**

- **The separation step is not affected:** it refits templates on the raw arrows and reads them there. V1 (active
  centring, then the separation step) has 114 clean fits, against 111 for V0.
- **The spike-in and false-positive scores** read the PCA components directly. They measure whether the order
  test passes, not how good the order is. The detection gain is real, but the components' own read-out is
  distorted.

**Status.** Active centring passes its false-positive bar and is the more sensitive choice. It should be used only
as the input to the separation step, whose outputs are the method's outputs. Its PCA read-out should not be
reported as an order. Making it the default for reactivations is left open until this is decided.
