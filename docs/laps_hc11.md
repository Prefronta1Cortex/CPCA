# hc-11 waking laps: complex PCA against count PCA

Code: `scripts/run_laps_hc11.py`. Results: `results/laps_hc11/summary.md`.

## Why

hc-11's running laps resemble hc-5's wheel runs: each lap is a full traversal in one direction, with each place
cell firing at its own point along the track. The test asks whether complex PCA recovers the order on the same
cells whose sleep reactivations it fails on, and whether count PCA, which ignores timing, does too.

## Design (written on November 5 before running)

**Laps.** `cpca.hc11.laps`: continuous runs in one direction covering at least 60% of the track (13–81 per
session). They are split into halves stratified by running direction: 3 random splits × 2 folds, giving 6 fits
per session. Each method is fitted on one half.

**Answer key.** Place-field centres from the other half only (`place_centres` on the held-out laps, at least 20
spikes, a peak of at least 3 Hz), per running direction.

**Methods.**
- **Complex PCA, uncentred** (primary, because the sequence is in every lap) and **centred**. T is each lap's
  duration, and the rerun's read-out is used.
- **Count PCA:** z-scored spike counts per lap, components above the shuffle null, members ordered by weight,
  plus the seriation of its reduced space.
- **Averaging reference:** each cell's mean relative time within the lap over the training laps of each running
  direction. It uses the laps' running direction, so it is not label-free; it is a ceiling, as on hc-5.

**Score.** The best output's |Spearman| with the held-out place fields, with the best-of-outputs null. On the
circular tracks the null also searches rotations, because laps there start anywhere on the circle.

**Expected.** Complex PCA recovers the order in most fits. Count PCA does not, because every place cell fires in
every lap through its field, so counts carry little order. Averaging sets the ceiling.

## Results (`results/laps_hc11/summary.md`)

Fits (of 6 per session) whose best output matches the held-out place fields:

| Method | Linear tracks, 5 sessions | Circular tracks, 3 sessions | All |
|---|---|---|---|
| Complex PCA, uncentred | 19/30 | 18/18 | **37/48** |
| Complex PCA, centred | 15/30 | 11/18 | 26/48 |
| Count PCA | 5/30 | 10/18 | **15/48** |
| Averaging reference (uses lap direction) | 25/30 | 16/18 | 41/48 |

- **Complex PCA, uncentred, comes close to the averaging reference** without being told each lap's direction.
  Its median |ρ| is 0.65–0.82 per session, against the reference's 0.56–0.81.
- **The one session where it fails is Buddy 06/27,** with only 12–13 place cells per direction (0/6).
- **On the linear tracks count PCA mostly fails.** Every place cell fires in every lap, so spike counts carry
  little order.
- **On the circular tracks count PCA does better (10/18).** There laps are cut into one-track-length pieces
  starting anywhere, and pieces covering 60–100% of the circle make spike counts depend on which part of the track
  a piece covers.

**Reading.** On the same cells whose sleep reactivations no method decodes, complex PCA recovers the waking order
from timing in most fits, and count PCA does not. The failure on hc-11 sleep is specific to the reactivation
regime (few cells per event, rare and partial replays, co-firing), not to these cells or this recording. Like
hc-5, this is a real-data case where complex PCA's advantage comes from timing.
