# Un-mixing complex PCA's components (main problem from November 5)

Code: `scripts/run_unmixing.py`, `cpca/rotation.py`. Results: `results/unmixing/`. Goal: `docs/goals.md`.

## Why

Complex PCA's kept components are mixtures of the routes (`docs/separation.md`). On the same planted data count PCA
mixes too, while count ICA unmixes the routes' membership (`docs/separation.md`, diagnostic 2). The goal needs one
route per output, chosen without the map. This page tests whether that is possible, and with which step.

## Design (fixed November 5 before running)

### Part 1: are the routes inside the kept components' span?

If they are, a rotation within the span can recover them. If they are not, no rotation can, and the problem lies
before the rotation (encoding, centring, the noise bound).

- **Data:** `sim/two_routes.py`, conditions A–E of `docs/separation.md`, 5 draws, all events.
- **Fits:** complex PCA centred and uncentred, kept components (shuffle null); count PCA, kept components
  (Marchenko–Pastur bound).
- **Targets per route:**
  - **membership:** the indicator of the route's cells, normalised;
  - **order:** the route's cells with phases π × position, normalised (complex fits only). This is the whole-route
    pattern; partial replays at a constant speed have the same order but other phase spacings, so it is approximate.
- **Statistic:** the share of the target inside the span of the kept components and their conjugates,
  ||Bᴴu||² (`cpca.containment`).
- **Null:** 500 random sets of 40 cells (random positions for the order target).
- **Passing:** a route is in the span if its share exceeds the null's 99th percentile. The span holds the routes if
  this is so for at least 90% of (route, draw) pairs. Medians of the share and the null are reported.

### Part 2: candidate un-mixing steps, scored per output

Same data and protocol as `docs/separation.md`:
- conditions A–F, 5 draws × 2 folds;
- fit on one half, outputs checked on the other half by the latency check;
- each output judged by `run_separation.judge`.

**Candidates.** Each gives complex weight vectors; none uses the map.

| Candidate | Weight vectors |
|---|---|
| PCA | centred complex PCA components (the reference of `docs/separation.md`) |
| ICA | centred complex PCA with complex ICA rotation |
| Quartimax | centred complex PCA components rotated within their span by the unitary rotation that maximises Σ\|w\|⁴, i.e. concentrates each component on as few cells as possible (`cpca/rotation.py`) |
| Count ICA + phase | count ICA assemblies (`baselines/count_assemblies.py`, on the training half); for each assembly's members, a complex template fitted on the raw arrows with the direction split (`cpca.direction.split_members`) gives the phases |

**Membership rules,** applied to each candidate's weight moduli:
- at least 20% of the largest (the current rule);
- above the mean + 2 SD (the count-assembly rule);
- **two-group split:** the cut of the sorted moduli that minimises the within-group variance (Otsu); members are the
  upper group.

**Output:** the weight vector restricted to its members.
- **Order:** the read-out of that vector.
- **Latency check:**
  - for the centred candidates, the held-out expression against 50 latency surrogates on centred, scaled held-out
    arrows;
  - for count ICA + phase, on raw arrows, because its template is uncentred.

  The surrogates are shared by all candidates within a fold.

**Measures per candidate, rule and condition:**
- the share of outputs passing;
- clean fits (every map represented, every output passing);
- the median coverage of passing outputs (members on the route ÷ route cells);
- outputs in F.

**A candidate solves a condition if:**
- at least 80% of its outputs pass;
- at least 8 of 10 fits are clean;
- the median coverage is at least 0.5, so a handful of cells does not count as a route;
- and in F at most 2 of 10 fits have any output.

**Expected:**
- **Quartimax and count ICA + phase:** help in A and B (routes on different cells), not in C and D (same cells),
  where sparsity cannot tell the maps apart; the templates (`docs/separation.md`) remain the reference there.
- **The 20% rule:** gives low purity whatever the rotation.

**Screen and confirmation.** Part 2 compares 12 combinations, so it is a screen. Any combination that solves A and B
is rerun on 5 fresh draws (new seeds) before it is adopted, and only the confirmation counts.

### Part 3: do the other methods separate? PP-Seq, seqNMF and complex PCA on the ladder, per output

The ladder's stored outputs (`results/ladder/*.pkl`; nothing is refitted) are re-scored per output on regenerated
planted maps (same seeds):
- **L0:** an output passes if its order matches the map (|Spearman| permutation p < 0.05, at least 6 members on the
  map).
- **L1–L6:** two maps on the same cells; an output passes if it matches exactly one.
- **Reported:** the share of outputs passing, and clean fits (both maps represented, every output passing).
- **No latency check:** the events are not stored with the outputs.

This tells whether PP-Seq or seqNMF give one sequence per output where complex PCA does not. The best-of-outputs
scoring of `docs/baselines.md` could not show this.

## Results (`results/unmixing/summary.md`)

### Part 1: the routes are inside the kept span

In every condition and draw (A–E; 10 route-draw pairs per condition, 5 in E), each route's target lies in the span
of the kept components and their conjugates far above random cell sets.

| Space | Order target: share vs random sets (medians) | Membership target |
|---|---|---|
| Complex PCA, centred (7–9 components) | 0.89–0.93 vs 0.14–0.20 | 0.77–0.87 vs 0.18–0.28 |
| Complex PCA, uncentred (8–10) | 0.87–0.93 vs 0.23–0.26 | 0.79–0.88 vs 0.37–0.40 |
| Count PCA (14–17) | – | 0.81–0.91 vs 0.39–0.41 |

**Reading.** The information is not lost. The kept components are a poor cut of a subspace that contains the routes,
so a rotation within the span can in principle recover them.

### Part 2: no candidate solves any condition

No candidate and rule reaches the bar (80% of outputs passing, 8 of 10 fits clean, coverage at least 0.5) in any
condition. The best results for routes on different cells:

| Condition | Candidate, rule | Outputs passing | Clean fits | Coverage |
|---|---|---|---|---|
| A | PCA, 20% rule (reference) | 2/82 (2%) | 0/10 | 0.84 |
| A | **Quartimax, 20% rule** | **26/50 (52%)** | 0/10 | 0.78 |
| A | Count ICA + phase, 20% rule | 64/147 (44%) | 0/10 | 0.23 |
| A | ICA, any rule | 0% | 0/10 | – |
| B | PCA, 20% rule (reference) | 9/87 (10%) | 0/10 | 0.82 |
| B | **Quartimax, 20% rule** | **32/52 (62%)** | 0/10 | 0.75 |
| B | Count ICA + phase, 20% rule | 61/146 (42%) | 0/10 | 0.20 |

- **The sparse rotation (quartimax) is the one clear gain.** It raises the share of one-route outputs from 2–10% to
  52–62% at three quarters of the route's cells, and halves the number of outputs.
- **Why its fits are not clean** (diagnostic after the results):
  - **Routes represented:** both routes have a passing output in 7/10 (A) and 10/10 (B) fits.
  - **Failing outputs:** about half are mixed, with purity near the bar (median 0.70–0.77). The other half are one
    route by membership but fail the order test.
  - **Duplicates:** 9–12 of the passing outputs duplicate a route already represented.
  - **A probable cause, not tested:** the order failures fit the split pair of balanced directions (`docs/method.md`,
    Section 2.7c of `docs/project_summary.md`). A real-valued component loses the order.
- **Count ICA + phase** separates the routes by membership only in fragments. Each route is cut into many small
  assemblies (12–15 outputs per fit, coverage about 0.2, 41–44 duplicate outputs).
- **Complex ICA** is worse than no rotation for routes on different cells (0–1%).
- **Same cells (C, D):** no candidate reaches the templates of `docs/separation.md` (76–84% of outputs passing, 6–7
  clean fits). Sparsity cannot tell two maps on the same cells apart, as expected.
- **No replays (F):** every candidate stays within at most 4 of 10 fits with any output.

### Part 3: PP-Seq does not give one sequence per output either

Stored ladder outputs (3 draws per rung), share of outputs matching exactly one map and clean fits:

| Rung | complex PCA, centred | PP-Seq | seqNMF |
|---|---|---|---|
| L0 one map | 5/6, 2/3 clean | 6/13, 0/3 | 2/26, 0/3 |
| L1 two maps, forward | 1/6, 0/3 | 6/10, 0/3 | 4/25, 0/3 |
| L2 two maps, both ways | 4/9, 0/3 | 10/22, 0/3 | 2/30, 0/3 |
| L4 + assemblies | 5/21, 0/3 | 13/19, 0/3 | 2/34, 0/3 |
| L5 + variable pacing | 6/9, 1/3 | 12/20, 0/3 | 0/24, 0/3 |
| L6 = H | 10/31, 0/3 | 7/12, 1/3 | 0/29, 0/3 |

- **PP-Seq** gives one map per output in 45–68% of its outputs on most rungs (20% on L3), with clean fits in 0–1 of 3.
  Its best output per map is near perfect (`docs/baselines.md`). The other outputs are pieces, volleys or blends.
- **seqNMF** is mostly mixtures (0–16%).
- **What this means:** the best-of-outputs comparison hid this for every method, not only complex PCA. Per output,
  none of the three methods meets the goal on the ladder.

### Reading

- **The routes are recoverable from the kept components** (Part 1), so un-mixing is a read-out problem.
- **A sparse rotation is the right kind of step,** but it is not enough on its own.
- **The remaining failures of this screen point to two specific fixes:**
  - outputs that are one route by membership but have lost the order (the real pair from balanced forward and backward plays);
  - duplicate outputs of a route already represented.

  Neither fix is in the candidates above. Both are in the pipeline developed next: an ordered template seeded from each sparse component, with backward events flipped, and duplicates merged. The confirmation of that pipeline is below.

## Why count ICA looked better, and what it actually does (November 5, diagnostic)

On condition A, count ICA's assemblies have 5–8 members each, all on one route, but each covers only 7–15% of the
route's positions (for example 0.31–0.43). Partial replays at a constant speed make neighbouring cells co-fire within
a 25 ms bin, so count ICA finds local stretches of each route, not routes. They are pure because they are small. Count
ICA therefore does not un-mix routes, and there is nothing route-level to borrow from it. What does carry over is the
idea of ICA in the count method: rotate the kept subspace so that each output is concentrated on few cells.

Inside the sparse (quartimax) rotation, each route of condition A appears as two components with 94–98% of their
weight on that route and circularity 0.95–0.99 (nearly real). This is the split pair of balanced directions
(`docs/project_summary.md`, Section 2.7c). The remaining 4–5 components are background assemblies that include route
cells of both routes.

## Development (development draws only, seed base 7000; `scripts/dev_unmixing.py`)

Each variant was scored per output with `run_separation.judge` on 5 development draws × 2 folds. None of these numbers
is a result; the confirmation below is.

| Step added | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| Sparse rotation, each component's members seed an ordered template (direction split on raw arrows), refined on its expressing events, duplicates merged | 77%, 5/10 | 69%, 3/10 | 0%, 0/10 | 8%, 0/10 | 100%, 10/10 | no outputs |
| + explaining away (largest template first; smaller ones lose claimed cells and are refitted on the rest) | 95%, 9/10 | 100%, 10/10 | 0% | 9% | 100%, 10/10 | no outputs |
| + latency check with 200 surrogates and Bonferroni over each fit's outputs | 100%, 10/10 | 100%, 10/10 | 0% | 9% | 100%, 10/10 | no outputs |
| + two orders on one template's cells (template step on its members), kept if both reproduce on the held-out half at similarity ≥ 0.97 | 100%, 10/10 | 100%, 10/10 | 70%, 7/10 | 57%, 4/10 | 100%, 10/10 | no outputs |

Entries are outputs passing and clean fits.

**Tried and dropped:**
- **Splits fitted from a fresh two-component PCA and checked by `mixture.heldout_check`:** too few events per inner
  half, so true splits were rejected.
- **Splits started from the global components:** false splits of single routes.
- **The global template step with the sparse templates as an extra start:** it picks mixed starts in A and B (10–20%).
- **The global template step alone** (`docs/separation.md`) on the same development draws: A 9%, B 8%, C 77% with 5/10
  clean, D 93% with 8/10, E 62% with 5/10.

**Thresholds set on development draws:**
- **Split reproduction ≥ 0.97.** On development draws, true splits of two maps reproduce at 0.986–0.995 with balanced
  event counts (about 150 each). False splits of one route reproduce at 0.28–0.956 with unbalanced counts (about
  170 against 45).
- **Merge similarity 0.5, refinement up to 10 iterations, largest-gap (Otsu) membership.** These were not tuned.

**The criterion for maps on the same cells (a correction, added after the development results).**
- **What happened:** in C and D, the split outputs match their own map at |ρ| 0.98–0.99 and the other map at 0.33–0.38.
  On those cells the two planted maps themselves correlate at 0.32–0.36.
- **Why it matters:** "p ≥ 0.05 for the other map" then fails even for an exact copy of one map. It measures how alike
  the maps are, not how mixed the output is.
- **The corrected criterion,** reported alongside the original: an output is one map if its order matches that map and
  its partial Spearman correlation with the other map, controlling for the matched map, is not significant
  (permutation, p ≥ 0.05). Both criteria are reported for every method.

## Confirmation design (fixed November 5 before running)

- **Pipeline** (`cpca/separate.py::separate`, settings of the last development row):
  1. centred complex PCA (shuffle null);
  2. quartimax rotation;
  3. seeded and refined templates;
  4. merge;
  5. explaining away;
  6. two-order split, kept at reproduction ≥ 0.97 on the held-out half;
  7. latency check on held-out raw arrows (200 surrogates, Bonferroni over outputs).
- **Data:** `sim/two_routes.py`, conditions A–F, 5 fresh draws (seed base 9000) × 2 folds. Neither this pipeline nor
  its thresholds have seen these draws.
- **Reference:** the global template step (`docs/separation.md`), on the same draws.
- **A condition is solved** if at least 80% of outputs pass, at least 8 of 10 fits are clean, and coverage is at least
  0.5. In F, at most 2 of 10 fits may have any output.
- **The un-mixing problem counts as solved on planted data** if A, B, C, D and E are solved and F passes, under the
  original criterion. If C or D pass only under the corrected criterion, that is reported as such.

## Confirmation results (fresh draws, seed base 9000; `results/unmixing/confirm_final.pkl`)

Pipeline of the confirmation design, outputs after the latency check:

| Condition | Outputs passing | Clean fits | Coverage | Solved |
|---|---|---|---|---|
| A: different cells, 10% + 10% | 20/21 (95%) | 9/10 | 0.85 | yes |
| B: different cells, 13% + 7% | 20/20 (100%) | 10/10 | 0.82 | yes |
| C: same cells, forward | 18/19 (95%) | 9/10 | 0.85 | yes |
| D: same cells, both ways | 16/20 (80%); corrected criterion 20/20 | 8/10; corrected 10/10 | 0.90 | yes (at the bar) |
| E: one map | 10/11 (91%) | 9/10 | 0.89 | yes |
| F: no replays | no outputs in any fit | 10/10 | – | yes |

**Reference on the same draws, the global template step:** A 16%, B 0%, C 90% (8/10 clean), D 73% (5/10), E 65%
(5/10).

**By the bar fixed before the run, the un-mixing problem is solved on this planted generator.** The waking map is not
used at any step: the outputs are chosen by the held-out reproduction and the latency check only.

## Stress test (descriptive, no bar; new draws)

**Purpose:** these conditions were not part of development. A first pass (seed base 11000) showed that three maps on
the same cells fail, because the split tried only two orders.

**The split was then generalised** (`split_orders_k`): it tries 2–4 orders and keeps the largest number whose templates
all reproduce at ≥ 0.97. This was developed on separate draws (seed base 13000) and run once on fresh draws (seed base
15000).

**Regression check:** the generalised pipeline was rerun on the A–F confirmation draws, a second use of those draws:
- A, B and E: 100% of outputs pass, 10/10 clean;
- C: 95%, 9/10;
- D: 80%, 8/10 (corrected criterion 100%, 10/10);
- F: no outputs.

Stress conditions, generalised pipeline, fresh draws (`results/unmixing/stress_confirm_final_k.pkl`):

| Condition | Outputs passing | Clean fits | Coverage | Fits with any output |
|---|---|---|---|---|
| G: three routes, different cells, 7% each | 29/31 (94%) | 9/10 | 0.85 | 10/10 |
| H: different cells, 5% + 5% | 18/20 (90%) | 8/10 | 0.80 | 10/10 |
| I: different cells, 3% + 3% | 8/10 (80%) | 1/10 | 0.80 | 7/10 |
| J: different cells, 25 cells per route, 10% + 10% | 20/20 (100%) | 10/10 | 0.84 | 10/10 |
| K: same cells, 5% + 5%, both ways | 11/15 (73%); corrected 10/15 | 5/10; corrected 4/10 | 0.93 | 10/10 |
| L: three maps, same cells, 7% each | 16/25 (64%); corrected 20/25 | 3/10; corrected 5/10 | 0.82 | 10/10 |
| M: one map, 5% | 10/10 (100%) | 10/10 | 0.82 | 10/10 |

**Reading.**
- **Routes on different cells separate robustly:** two or three routes, down to 5% replay each, and with 25 cells per
  route.
- **At 3% each, routes are often not detected** (no output in 3 of 10 fits, one route missing in most others). The
  outputs that appear are one route each. This is a detection limit, not mixing.
- **Maps sharing all their cells are harder:** 73% at 5% + 5%, and 64–80% with three maps. They do not reach the bar
  there.

## What this does and does not show

**It shows** that complex PCA's kept components contain the routes, and that a step after PCA turns them into one
ordered template per route, without the map, on this generator. The steps are:
1. sparse rotation;
2. ordered templates seeded from the rotated components and refined on their own events;
3. explaining away;
4. a reproduced split for orders on shared cells;
5. a corrected latency check.

**It does not show** that it works on real data:
- **one generator:** constant-speed partial replays, 8 ms jitter, random assemblies;
- **its realism is limited:** 40 route cells among 120, and 2500 events;
- **real reactivations are harder:** hc-11 sleep has 0–3.4% excess replay and 3–12 active place cells per event.

**Next, in order:**
- **real data with a known answer:** hc-11 waking laps, where the two running directions are two sequences on
  overlapping cells, and hc-5;
- **then sleep.**

The latency check is not appropriate for laps or wheel runs, where the sequence is identical in every event.

## Waking sequences (November 19)

`scripts/run_unmix_waking.py`, `results/unmix_waking/summary.md`. The confirmed pipeline, with the latency check
left off. Fit on one half of the events, scored on the other half's fields. An output passes when it matches
exactly one held-out map. A fit is clean when every map is represented and every output passes.

Two gates. **Confirmed:** a template needs 20 expressing events and the split needs 80, as on the planted runs.
**Open:** both minima are 4, so a recording with a handful of laps can form a template. The 0.97 reproduction cut
is unchanged. The template minimum is `min(20, min_events)`; the planted default stays 20.

| Data | Gate | Components | Templates |
|---|---|---|---|
| hc-11 laps, uncentred | 80 | 14/32 pass, 3/16 fits clean | 4/7 pass, 2/16 fits clean; no split |
| hc-11 laps, uncentred | 4 | 13/32 pass, 2/16 fits clean | 9/16 pass, 6/16 fits clean; no split |
| hc-5 wheel, uncentred | 80 | 10/16 pass, 7/12 fits clean | no template (halves have 3–13 runs) |
| hc-5 wheel, uncentred | 4 | 11/16 pass, 8/12 fits clean | 9/11 pass, 9/12 fits clean; no split |

The six clean lap fits at gate 4 are the one-direction (circular) sessions. No two-direction fit is clean: the step
returns one template, so the second running direction has no output. The split does run on those sessions. Its best
held-out similarity is 0.96–0.98, and it is not kept, because not every order in the set reaches 0.97. One template
then remains, and one template cannot represent both directions.

On the wheel, lowering the floor is enough. The step returns one template, and that template is the held-out time
sequence in 9 of 12 fits. The one half with only 3 runs still produces nothing. It does not split the single
sequence.

**Reading.** The event floor of 20 was a property of the planted generator, and it hid the wheel result. With it
lowered, one repeated sequence stays one output. Two running directions on the same cells do not separate: the
0.97 cut, set on development draws with about 150 events per order, rejects the split, and the surviving template
is only one of the two directions.

## Decision: the two running directions are two sequences (November 19)

**The decision.** Running a linear track in the opposite direction is not the same cells firing in reverse order: most
place cells are direction-selective, or fire at another position when running the other way. Reverse order is a
property of sleep reactivation (reverse replay), not of running. So the two running directions of a linear track
count as two separate sequences, and waking data are analysed without treating a sequence and its reverse as the
same.

**The data agree.** Measured on hc-11:
- **Positions:** cells with a field in both directions have them at unrelated positions in four of the five linear
  sessions (ρ 0.08–0.30), and at related positions only in the two Cicero linear sessions (0.69–0.81).
- **Effect on the earlier waking run:** it allowed time reversal, so it could merge the two directions into one
  template, which is what happened (see "Waking sequences").

**What carries over to sleep.** In sleep, reverse replay is real, so time reversal must still be allowed. A backward
replay of one direction and a forward replay of the other are then the same pattern of arrows only insofar as the two
directions' maps mirror each other. That is an identifiability limit of the data, to be measured per session before
two sleep outputs are expected.

## Waking design (fixed November 19 before running)

**Forward-only mode** (`conjugate=False`) in the un-mixing step, the template step and the merge similarity:
- each event matches a template only forward, |⟨s, q⟩|;
- no event is conjugated, and two templates are alike only if they are alike as they stand.

The planted reactivation conditions keep the default (time reversal allowed).

**Waking check** (replaces the latency check, which rejects a sequence identical in every event):
- **statistic:** a template's mean forward expression on the held-out events;
- **null:** 200 surrogates of the held-out events in which each cell's spike times in each event are circularly
  shifted by its own random amount. This destroys the sequence and keeps each cell's spike count;
- **correction:** Bonferroni over the fit's outputs;
- **pass:** p < 0.05.

**Development on planted laps** (`sim/laps.py`, new; development seeds only).
- **Generator:**
  - **laps:** each lap is one full traversal in one direction, lasting 2–3.5 s;
  - **firing:** each place cell fires a burst at its field (Poisson, mean 10 spikes, SD 6% of the lap) with
    probability 0.85 per pass, plus sparse spikes outside the field;
  - **population:** 60 place cells plus 20 cells without a field;
  - **direction 2:** positions are traversed in reverse time.

| Condition | Cells with fields in both directions | Their position correlation | Laps per direction |
|---|---|---|---|
| W1 | 70% (15% only one way, 15% only the other) | 0.2 | 30 |
| W2 | 70% | 0.7 | 30 |
| W3 | 70% | 0.2 | 10 |
| W4 | one direction only | – | 40 |
| W5 (control) | as W1, each cell's spikes circularly shifted in every lap | – | 30 |

- **Fits:** halves of the laps, stratified by direction, both folds, 5 draws, giving 10 fits per condition.
- **Score:**
  - an output passes if it matches exactly one direction's map, under the partial-correlation criterion (the
    original criterion is reported as well);
  - **clean** = every direction represented and every output passing;
  - coverage is reported.
- **Bar for going to real data:** W1–W4 each at least 80% of outputs passing and 8 of 10 fits clean; W5 at most 2 of
  10 fits with any output.

**Real data, run once after the planted bar is met:**

| Data | Fits | Satisfactory if |
|---|---|---|
| hc-11 linear tracks | 5 sessions × 2 folds = 10 | at least 7 of 10 fits clean, and at least 80% of outputs passing |
| hc-11 circular tracks (one running direction) | 3 × 2 = 6 | at least 5 of 6 clean |
| hc-5 wheel, aligned, repeat 0 | 6 × 2 = 12 | at least 9 of 12 clean |
| hc-5 shuffled control | 12 | at most 2 of 12 fits with any output passing the waking check |
| hc-11 laps with each cell's spikes circularly shifted in every lap (control) | 16 | at most 2 of 16 fits with any output |

The answer key is the place or time fields of the held-out half, as in `scripts/run_unmix_waking.py`.

## Waking development (planted laps, development seeds; `scripts/dev_waking.py`)

Outputs that pass the waking check; entries are the share of outputs that are one direction and clean fits, under the
variance-share criterion introduced below. W4 (one direction) was 10/10 throughout, and W5 (no sequence) had no
outputs throughout.

| Step | W1 (ρ 0.2) | W2 (ρ 0.7) | W3 (10 laps each) |
|---|---|---|---|
| Forward only (no conjugation) | 25%, 1/10 | 62%, 3/10 | 0%, 0/10 |
| + split kept when every template that survives reproduces | 83%, 7/10 | 94%, 7/10 | 0% |
| + laps clustered by pairwise similarity as an extra split start | 85%, 7/10 | 95%, 9/10 | 18%, 1/10 |
| + split templates keep only the cells they weight | 85%, 7/10 | 95%, 9/10 | 9%, 0/10 |
| Same, scored with the variance-share criterion | 100%, 10/10 | 100%, 10/10 | 18%, 1/10 |
| + reproduction bar from the data's own event count | 100%, 10/10 | 100%, 10/10 | 100%, 10/10 |

**What each step fixes.**
- **Forward only:** with time reversal allowed, the two directions could merge. On its own this is not enough: only two
  components pass the noise bound on laps (the mean lap and one more), so both seed the same blended template.
- **The split rule:** asked for K templates, the template step may return fewer, because it merges or drops some. The
  old rule rejected every run that returned fewer than K, which discarded correct two-direction splits (both
  templates at ρ 0.99 with their direction, reproduced at 0.99). Now the run with the most templates, all reproduced,
  is kept. This is a bug fix that applies to every mode.
- **Lap clustering:** every lap is a whole sequence, so laps of one direction resemble each other. This start applies
  in forward-only mode only.
- **Trimming:** after a split, each template keeps only the cells above its own two-group cut.
- **Event-count-aware reproduction:** each split template must reproduce at least as well as the unsplit template
  refitted on the same number of events from each half (median of 5 subsamples), with 0.97 as the ceiling. With about
  5 laps per direction per half, true splits reproduce at 0.92–0.97. In W4 no split candidate formed at all.

**The variance-share criterion (added after the waking development results).**
- **What happened:** outputs that matched their direction at ρ 0.99–1.00 failed the partial-correlation criterion. The
  little that is left after removing an almost perfect match can correlate significantly with the other map, even
  though it explains almost nothing.
- **The new criterion:** an output is one map if its order matches the map (p < 0.05) and adding the other map to a
  rank regression on the matched map raises the explained variance by less than 0.1.
- **Calibration:** on development outputs the gain was 0.000–0.004 for clean outputs and 0.26–0.30 for blends. Any cut
  from 0.01 to 0.25 gives the same verdicts.
- **Reporting:** all three criteria are reported.

## Waking confirmation (planted laps, fresh seeds; `results/unmixing/waking_confirm_waking.pkl`)

| Condition | Partial correlation (fixed) | Variance share | Original |
|---|---|---|---|
| W1 (ρ 0.2) | 95%, 9/10 | 100%, 10/10 | 100%, 10/10 |
| W2 (ρ 0.7) | 100%, 10/10 | 100%, 10/10 | 48%, 4/10 |
| W3 (10 laps each) | 95%, 9/10 | 100%, 10/10 | 100%, 10/10 |
| W4 (one direction) | 100%, 10/10 | 100%, 10/10 | 100%, 10/10 |
| W5 (no sequence) | 1 of 10 fits with an output | same | same |

- **The bar for real data is met under the fixed criterion.**
- **The original criterion fails W2,** where the two directions' positions correlate at 0.7, as expected.
- **No regression in reactivation mode:** the A–F confirmation and stress conditions give the same results as on
  November 5. The split-rule fix applies there; the other waking changes are off by default.

## Waking results on real data (run once; `scripts/run_waking_separation.py`, `results/waking_separation/summary.md`)

| Data | Bar | Result (partial-correlation criterion) | Met |
|---|---|---|---|
| hc-11 linear tracks | at least 7 of 10 fits clean, at least 80% of outputs passing | 5/10 clean, 13/17 outputs (76%) | no |
| hc-11 circular tracks | at least 5 of 6 clean | 6/6, 6/6 outputs | yes |
| hc-5 wheel | at least 9 of 12 clean | 9/12, 9/11 outputs | yes (at the bar) |
| hc-5 shuffled control | at most 2 of 12 with outputs | 0/12 | yes |
| hc-11 laps, circularly shifted (control) | at most 2 of 16 with outputs | 0/16 | yes |

The other two criteria give the same results, except the variance-share criterion on the linear tracks (12/17 outputs,
4/10 clean).

**Linear tracks per session** (descriptive, after the run; nothing was changed afterwards):

| Session | Result |
|---|---|
| Achilles 10/25 | both directions as separate outputs in both folds |
| Gatsby 08/02 | both directions as separate outputs in both folds |
| Cicero 09/01 | both directions in fold 1; one direction only in fold 2 (ρ 0.71 with it, 0.11 with the other) |
| Buddy | two outputs in both folds, but only 8–9 cells per direction can be scored; the outputs reach ρ 0.50–0.70 without significance, or are unordered (0.11) |
| Cicero 09/17 | 4–6 training laps per direction; no split forms, and the one output is the blend (ρ +0.48 and −0.61) |

**Reading.**
- **Before these changes, no linear-track fit had an output for each direction.** Now the two running directions come
  out as two separate outputs in every fit of the two sessions with enough laps and cells, and in one of two fits of
  Cicero 09/01.
- **One-direction data:** circular tracks and the wheel stay one output.
- **No-sequence controls:** no output at all.
- **The failures are the sessions with the least data:** 8–9 scorable cells per direction (Buddy), and 4–6 laps per
  direction in a training half (Cicero 09/17).
- **The bar for the linear tracks is not met.**

**Possible next steps,** none of them run:
- a split start that needs fewer laps, for example clustering held-out and training laps together;
- for sessions like Buddy, a score with more power, because the method's outputs there are not shown to be wrong.

Any change is to be developed on planted laps first, and the real data rerun only once.

## Initial real-data tests on local data (design fixed November 19 before running)

**Scope.** No method changes and no new datasets. The un-mixing step is used as it stands:
- **waking data:** the waking mode of `scripts/run_waking_separation.py`;
- **reactivations:** the reactivation mode confirmed on planted data (time reversal allowed, latency check).

The run is small and descriptive, with no pass bar. Code: `scripts/run_initial_real.py`; results:
`results/initial_real/`.

**Waking, figure-eight (DANDI 000552, 65 sessions × 2 folds).**
- **Events:** trials (one loop through the stem and one arm). Only spikes fired while running (above 5 units/s, as for
  the place fields) are kept, and time is counted in running time, so a pause does not stretch the sequence.
- **Split:** halves stratified by arm.
- **Answer key:** the held-out half's left-arm and right-arm place fields on the linearized maze.
- **The stem is its own shared segment.** Left and right trials are in the same place over the first 80 cm of the
  222 cm coordinate (2-D distance under 5 units), and 65–74 units apart along the arms.
  - **Cells:** a field centre below 85 cm makes a stem cell. A field at 85 cm or more on one arm only makes an arm
    cell.
  - **An output is an arm sequence** if at least 6 of its members are arm cells, at least 80% of those are on one arm,
    and their order matches that arm's field positions (|Spearman| permutation p < 0.05). Stem cells among its
    members do not count against it.
  - **An output is a stem sequence** if at least 80% of its members with a field are stem cells (at least 6) and
    their order matches the stem positions.
  - **Anything else fails.**
  - **Clean** = both arms represented by an arm sequence and every output passing.

**Waking, linear track (DANDI 001695, 15 sessions × 2 folds).** The same as hc-11 linear laps
(`scripts/run_waking_separation.py`): the two running directions are two sequences, scored with all three criteria.

**Reactivations (hc-11, 8 sessions; DANDI 001695, 15 sessions; 2 folds each).**
- **Events:** NREM bursts in POST and PRE, and immobility bursts during the maze epoch (rest on the maze).
- **Pipeline:** the confirmed reactivation pipeline.
- **Score:** each output must match exactly one running direction, using the three criteria, and the rotation search
  on circular tracks.
- **Reported:** outputs per epoch, how many match exactly one direction, and the chance level. PRE is the control.
- **Expectation:** low, given the measured replay content.

### Results (`results/initial_real/summary.md`)

**Waking, figure-eight (DANDI 000552): the arms do not separate.**
- **Outputs:** 133 outputs in 130 fits, none of them one arm. There is typically one output per fit, covering most of
  the cells (for example 115 of 124), ordered by time along the trial. That order matches both arms' fields
  (ρ 0.27–0.83).
- **The data are thin:** a training half has 2–7 trials per arm (median 11 trials in all), with 20–50 s of slow,
  interrupted running per trial.
- **The supervised check:** averaging each cell's time over its arm's training trials reaches only ρ 0.16–0.62
  against the held-out arm fields.
- **Reading:** the same failure as the hc-11 session with the fewest laps (Cicero 09/17). With a handful of events
  per route, no split forms.

**Waking, linear track (DANDI 001695).**
- **Overall:** 42/54 outputs are one direction (partial correlation; variance share 43/54), and 12/30 fits are clean.
- **By lap count:** no fit with fewer than 8–10 training laps per direction is clean. Above that, about half are.
- **Failures with many laps:** one direction missing, or one output failing.

**Reactivations: at chance, as expected.** Outputs passing the latency check that match exactly one running
direction (partial correlation), against the number expected by chance:

| Dataset | Rest on the maze | POST | PRE |
|---|---|---|---|
| hc-11 | 1 of 6 (0.6) | 3 of 16 (1.2) | 1 of 15 (1.2) |
| DANDI 001695 | 4 of 25 (2.4) | 0 of 25 (2.4) | 1 of 17 (1.6) |

There is no POST over PRE difference, and nothing clearly above chance.

**Reading.**
- **Waking:** separation works where each route has many traversals (hc-11 Achilles and Gatsby, part of 001695). It
  fails where there are only a few per route (the figure-eight, small sessions).
- **Reactivations:** separation adds nothing visible on these recordings, consistent with their low replay content.

## Patches removed from the model (November 19)

**What was moved.** The un-mixing steps are taken out of the `cpca` package and kept, unchanged, under
`experimental/`, so the results above stay reproducible:
- `separate.py` and `rotation.py`;
- `template_step.py`, the extended copy of the template step;
- the scripts `run_unmixing`, `dev_unmixing`, `dev_waking`, `run_waking_separation`, `run_initial_real` and
  `run_unmix_waking`;
- their test.

**What was restored.** `cpca/mixture.py` is back to its behaviour before November 19: no time-reversal switch, no
extra starts, no event-floor parameter.

**The model is now complex PCA itself** (`cpca.fit`), which this work did not modify. A patch returns only if it is
shown to be essential.

## Pure capabilities of complex PCA (design fixed November 19 before running)

**Model.** `cpca.fit` with no rotation, no templates and no split:
- the kept components (shuffle null), read out with members at 20% of the largest weight;
- the convention by the method's own rule: uncentred where the sequence is in every event (waking), centred for
  reactivations;
- a fit on one half of the events, both folds.

**Questions, each scored per route or per component on held-out answer keys:**

| Question | Test |
|---|---|
| 1. Detection | the number of kept components, with no-sequence data as the control |
| 2. Order present | is each route's order the best match of some component, against a null that takes the best over all components (`best_order`, `best_map_match`)? |
| 3. Routes inside the span | each route's pattern (members with phase π × position) inside the span of the kept components and their conjugates, against random cell sets (99th percentile) |
| 4. Separation, per component | is each component one route? Different cells: purity of at least 0.8 and order. Same cells: the variance-share criterion, with the original reported. Figure-eight: the stem rule |
| 5. Label-free selection (reactivations and planted only) | which components pass the latency check on the held-out half |

**Data:**
- **planted:** `sim/two_routes.py` A–F and `sim/laps.py` W1–W5, 5 fresh draws each (seed base 30000);
- **waking:** hc-5 wheel (aligned, repeat 0), hc-11 laps, DANDI 001695 laps, DANDI 000552 figure-eight trials;
- **reactivations:** hc-11 and DANDI 001695 POST, PRE and rest on the maze.

**Code:** `scripts/run_pure_capabilities.py`. **Results:** `results/pure/`. Descriptive, no bar.

### Results (`results/pure/summary.md`)

Pure complex PCA (`cpca.fit`), components read at 20% of the largest weight. Counts are routes or components over all
fits.

| Data | Components per fit | Order present (routes) | Inside the span (routes) | Components that are one route | Clean fits |
|---|---|---|---|---|---|
| Planted, two routes on different cells (A, B) | 9 | 20/20 | 20/20 | 4/90 | 0/10 |
| Planted, two maps on the same cells (C, D) | 7–8 | 16/20 | 20/20 | 30/77 | 0/10 |
| Planted, one map (E) | 7 | 5/5 | 5/5 | 17/35 | 0/5 |
| Planted, no replays (F) | 5 | – | – | 0/25 (24 of 25 fail the latency check) | – |
| Planted laps, two directions (W1–W3) | 2 | 30/30 | 30/30 | 15/30 (original criterion 0/30) | 0/15 |
| Planted laps, one direction (W4) | 1 | 5/5 | 5/5 | 5/5 | 5/5 |
| hc-5 wheel (one sequence) | 1 | 10/12 | 9/12 | 11/16 | 7/12 |
| hc-11 laps, circular (one direction) | 2 | 6/6 | 6/6 | 9/13 | 2/6 |
| hc-11 laps, linear (two directions) | 2 | 10/20 | 20/20 | 7/19 | 1/10 |
| DANDI 001695 laps (two directions) | 2 | 50/60 | 54/60 | 35/67 | 6/30 |
| DANDI 000552 figure-eight (two arms, shared stem) | 2 | 175/260 | 152/260 | 0/253 | 0/130 |

Reactivations, centred. "One route" is the share of components matching exactly one running direction; chance per
component is about 9.5%.

| Data | Order present | Inside the span | Components that are one route | One-route share among those passing the latency check |
|---|---|---|---|---|
| hc-11 rest on the maze | 10/26 | 18/26 | 20/55 (36%) | 20/48 |
| hc-11 POST | 2/26 | 10/26 | 14/89 (16%) | 14/83 |
| hc-11 PRE | 2/26 | 2/26 | 5/74 (7%) | 5/64 |
| DANDI 001695 rest on the maze | 26/60 | 39/60 | 51/154 (33%) | 48/141 |
| DANDI 001695 POST | 9/60 | 13/60 | 25/170 (15%) | 23/165 |
| DANDI 001695 PRE | 4/60 | 6/60 | 22/165 (13%) | 22/158 |

**Reading.**
- **Detection and representation work.** On planted and waking data, every route lies inside the kept span (100% on
  planted data, 90–100% on laps), and usually some component carries its order. The exception is the hc-11 linear laps,
  where only one of the two directions is the best match of a component.
- **One sequence:** one component that is the sequence in most fits (hc-5 11/16 components, one-direction laps).
- **Several sequences:** pure complex PCA does not give one route per component. Routes on different cells give 4/90,
  and the figure-eight 0/253. It does so in part where maps share cells (planted D, about half), and in a few DANDI
  001695 lap fits (6/30 clean).
- **Without the map, background structure is also kept.** On no-replay data, a median of 5 components per fit is kept;
  the latency check removes all but one in 25.
- **Reactivations:**
  - **The map's pattern lies in the kept span** more often after the maze than before it: 10/26 against 2/26 on
    hc-11, 13/60 against 6/60 on 001695. It is most often there at rest on the maze (18/26, 39/60).
  - **Single components** match one direction at about chance in PRE and in 001695 POST, at 16% in hc-11 POST, and at
    33–36% at rest on the maze.
  - **These checks do not exclude co-firing.** The span null uses random cells, and on maze rest the order test has
    been passed by co-firing before (`docs/maze_rest.md`).

## The separation step in the cPCA pipeline, simplified (November 24)

**What changed and why.** On November 19 the whole separation step was moved out of the model. The intended
instruction was narrower: simplify the step, and drop the stages that are not shown to be essential. The step is
back in `cpca/separate.py` (with `cpca/rotation.py`), reduced to the stages below. `experimental/` keeps the full
earlier version for the record.

**The step** (after `cpca.fit` on the training events):
1. **Rotation:** the kept components are rotated within their span so that each is concentrated on few cells
   (quartimax).
2. **Templates:** each rotated component's members seed an ordered template, refined on the events that express it.
3. **Merging and explaining away:** templates that are alike are merged, and a smaller template gives up cells
   claimed by a larger one.
4. **Split:** a template is split into 2–4 orders on its cells when every order is reproduced on the held-out events.
5. **Selection:** templates are kept if their held-out expression beats a timing null (Bonferroni over templates).

**Settings:** `reversal` and `min_events`.
- **`reversal` chooses the mode:**
  - **reactivations:** time reversal is allowed, and the null redraws each cell's spike times from its own timing;
  - **waking traversals:** no time reversal, and the null shifts every cell's spikes circularly.
- **Split rules follow the mode:**
  - **reactivations** (sparse, partial events): a split needs 4 × `min_events` active events and a reproduction
    similarity of at least 0.97;
  - **waking data** (every event a whole traversal): the events, grouped by similarity, are also tried as a split
    start; a split needs `min_events` events; and each order must reproduce as well as a single template estimated
    from its number of events (at most 0.97).

**Dropped as not essential:**
- trimming members after a split;
- the alternative split variants and starts;
- separate event floors for templates and splits.

**Each kept stage is essential somewhere.** Development draws, one stage switched off at a time (`scripts/run_separation_check.py`;
clean fits, original criterion):

| Condition | Full | Without rotation | Without refinement | Without explaining away | Without split | Without selection |
|---|---|---|---|---|---|---|
| A: different cells, 10% + 10% | 10/10 | 5/10 | 5/10 | 2/10 | 10/10 | 3/10 |
| B: different cells, 13% + 7% | 9/10 | 10/10 | 3/10 | 1/10 | 10/10 | 3/10 |
| C: same cells, forward | 6/10 | 6/10 | 4/10 | 1/10 | 0/10 | 1/10 |
| D: same cells, both ways | 4/10 | 6/10 | 0/10 | 4/10 | 0/10 | 0/10 |
| F: no replays (clean = no output) | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 0/10 |
| Planted laps W1, W2 (waking) | 3/10, 5/10 | 3/10, 5/10 | 0/10, 4/10 | 3/10, 5/10 | 0/10, 0/10 | 3/10, 5/10 |

**The waking split rules are also essential.** Without the event-group start and the event-count bar:
- planted laps with 10 per direction drop from 10/10 to 0/10 clean, and hc-11 linear laps from 5/10 to 1/10;
- with the event-count bar but without the event-group start, hc-11 linear laps reach 3/10.

**The fixed bar is essential for reactivations.** The event-count bar there causes false splits: A drops from 10/10
to 6/10 clean, B from 10/10 to 7/10, E from 10/10 to 8/10.

**Confirmation with the final code** (the same draws and seeds as before; clean fits; original / variance-share
criterion):

| Data | Result | Earlier full pipeline |
|---|---|---|
| Planted reactivations A, B, C, D, E | 10, 10, 9, 8 (10), 10 of 10 | the same |
| Planted no replays (F) | no output in 10/10 | the same |
| Planted laps W1, W2, W3, W4 | 10 (10), 7 (10), 7 (10), 10 of 10 | 9, 10, 9, 10 of 10 (partial criterion) |
| Planted laps, no sequence (W5) | no output in 10/10 | 9/10 |
| hc-11 linear laps | 5/10 | 5/10 |
| hc-11 circular laps | 6/6 | 6/6 |
| hc-5 wheel | 9/12 | 9/12 |

The simplified step keeps every confirmed result, with one setting for the kind of data and one event floor.
