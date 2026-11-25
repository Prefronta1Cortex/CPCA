# Re-scoring the real-data results per component (repair step 3)

Code: `scripts/rescore_components.py`. Results: `results/rescore/summary.md`. Goal and criterion:
`docs/goals.md`.

## Design (fixed November 5 before running)

Every result below is scored **per component**: the share of output components that are each one sequence. No
best-of-outputs.

**hc-5 wheel runs.**
- **Fits:** the existing cross-validated splits (`scripts/run_hc5.training_data`): 6 sessions × aligned and
  unaligned windows × 3 repeats × 2 folds. Complex PCA uncentred and centred, no rotation.
- **A component passes** if at least 6 of its members have a held-out time field and its member order matches them
  (|Spearman|, 1000 permutations, p < 0.05).
- **Expected:** one sequence, so ideally every component is that sequence.
- **No latency check:** a sequence played identically in every run is a latency pattern, and the check would
  reject it (`docs/readout_check.md`).

**hc-11 waking laps.**
- **Fits:** the splits of `docs/laps_hc11.md` (8 sessions × 3 splits × 2 folds); uncentred and centred.
- **Maps:** the two running directions are two maps on largely the same cells. A component passes if it matches
  exactly one direction's held-out place fields (single-output test of `best_map_match`, p < 0.05 for one map and
  p ≥ 0.05 for the other). On the circular tracks only one direction has fields, and the component must match it.
- **Reported:** per fit, whether each running direction is represented by a passing component.
- **No latency check,** for the same reason as hc-5.

**hc-11 sleep, POST and PRE.**
- **Fits:** all NREM bursts, split in half (2 folds), with complex PCA centred and the Step 3 templates (the two
  best methods of `docs/separation.md`).
- **Outputs:** the components or templates that pass the latency check on the held-out half.
- **An output passes** if it matches exactly one direction's place fields (as for the laps; place fields from the
  whole maze epoch).
- **Reported:** per session and epoch, the number of outputs, the number passing, and the share. PRE is the control.

## Results (`results/rescore/summary.md`)

**hc-5.** Uncentred complex PCA keeps a median of 1 component; 74% (aligned) and 86% (unaligned) of components are
the sequence. In 27 of 36 aligned fits and 30 of 36 unaligned fits, every component is the sequence. Centred: 12%
and 32%. For one sequence played in every event, the goal is met.

**hc-11 laps.**
- **Uncentred, circular tracks (one map):** 67–100% of components pass.
- **Uncentred, linear tracks:** 0–75% pass. Achilles 10/25 is 0/12, because every component matches both running
  directions.
- **Diagnostic (added after the results; first fold of the first two splits, refitted with a fixed seed):** on the linear tracks the components
  combine the two directions' sequences.
  - **The two components:** one orders cells by time since the start of the lap in both directions (|ρ| with one
    map and the other: 0.51 and 0.61, opposite signs, in Achilles 10/25). The other orders them by position in both
    (0.58 and 0.57, same sign). These are the sum and difference of the two sequences, as for two maps on the same
    cells in planted data.
  - **Not similar maps:** the two directions' field centres correlate at only ρ 0.30 in Achilles 10/25 (43 cells
    with a field in both directions).
  - **Smaller sessions:** Cicero 09/01 and Gatsby 08/02 show the same pattern more weakly (for example 0.40 and
    0.70), so the weaker match often misses p < 0.05 and the component passes.
- **Reading of the linear-track passes:** they overstate separation, because they partly reflect low power against
  the second map.

**hc-11 sleep.** Of the outputs that pass the latency check:

| Method | POST | PRE |
|---|---|---|
| Centred PCA | 18/87 (21%) match exactly one running direction; 8/8 sessions | 7/73 (10%); 5/8 sessions |
| Templates | 6/67 (9%) | 6/63 (10%) |

A random output passes "exactly one of two maps" about 9.5% of the time. Summed over the outputs (some are testable
on one map only), 8.2 of the 87 POST outputs are expected to pass by chance. 18 or more has p = 0.0013, treating
outputs as independent, which they are not within a session. Centred PCA in POST is therefore about twice chance,
with PRE at chance (7 against 6.9 expected). Templates are at chance in both.

**Reproducibility:** this script seeds with Python's `hash()`, which changes between processes. The two folds of a
sleep session may therefore use different random halves. Each fold is still a valid train/test split, but the exact
numbers will not reproduce.

**Reading.**
- **The goal is met for a single sequence in every event (hc-5)** and for one-map laps.
- **It is not met for two routes on different cells (planted),** nor for the two running directions of
  linear-track laps, which come out as their sum and difference.
- **It is not met for sleep reactivations.** Of the outputs that carry timing beyond per-cell latency, about a
  fifth match a waking direction in POST (twice chance); the rest do not match the waking map. Whether they are
  other sequences is not tested here.
