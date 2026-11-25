# Checks A and B: the conjugate-pair read-out, the centred direction oracle, and the HMM on the G1 grid

Produced by `scripts/summarize_segments_pairs.py` from `scripts/run_segments_pairs.py`; design in `docs/segments.md` (addendum). Same planted events as `results/segments/summary.md`; one centred complex PCA fit per draw read out three ways. The uncentred direction oracle is copied from the main grid for comparison.

## Reverse fraction 0.5: whole-track order test, draws (of 5) passing

| ordered cells | share | standard read-out | adjacent pairs | all pairs | centred direction oracle | uncentred direction oracle (main grid) |
|---|---|---|---|---|---|---|
| 5 | 0% | 0/5 | 1/5 | 1/5 | 0/5 | 0/5 |
| 5 | 2% | 1/5 | 2/5 | 1/5 | 0/5 | 4/5 |
| 5 | 5% | 0/5 | 0/5 | 0/5 | 1/5 | 4/5 |
| 5 | 10% | 0/5 | 0/5 | 2/5 | 1/5 | 5/5 |
| 9 | 0% | 2/5 | 2/5 | 2/5 | 0/5 | 1/5 |
| 9 | 2% | 0/5 | 0/5 | 0/5 | 1/5 | 3/5 |
| 9 | 5% | 2/5 | 3/5 | 3/5 | 3/5 | 5/5 |
| 9 | 10% | 4/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| 13 | 0% | 2/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 13 | 2% | 0/5 | 2/5 | 2/5 | 3/5 | 4/5 |
| 13 | 5% | 4/5 | 3/5 | 3/5 | 5/5 | 5/5 |
| 13 | 10% | 3/5 | 5/5 | 5/5 | 3/5 | 5/5 |

## Reverse fraction 0.3: whole-track order test, draws (of 5) passing

| ordered cells | share | standard read-out | adjacent pairs | all pairs | centred direction oracle | uncentred direction oracle (main grid) |
|---|---|---|---|---|---|---|
| 5 | 0% | 0/5 | 1/5 | 1/5 | 0/5 | 0/5 |
| 5 | 2% | 1/5 | 0/5 | 0/5 | 1/5 | 2/5 |
| 5 | 5% | 2/5 | 1/5 | 2/5 | 2/5 | 4/5 |
| 5 | 10% | 3/5 | 1/5 | 1/5 | 4/5 | 5/5 |
| 9 | 0% | 2/5 | 2/5 | 2/5 | 0/5 | 1/5 |
| 9 | 2% | 0/5 | 2/5 | 2/5 | 1/5 | 3/5 |
| 9 | 5% | 3/5 | 3/5 | 3/5 | 3/5 | 5/5 |
| 9 | 10% | 4/5 | 4/5 | 4/5 | 5/5 | 5/5 |
| 13 | 0% | 2/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 13 | 2% | 2/5 | 2/5 | 4/5 | 2/5 | 4/5 |
| 13 | 5% | 4/5 | 2/5 | 5/5 | 4/5 | 5/5 |
| 13 | 10% | 5/5 | 5/5 | 5/5 | 4/5 | 5/5 |

## Whole-track test with the timing bar (30 shuffled refits read out the same way)

| reverse | cells | share | standard read-out | adjacent pairs | all pairs |
|---|---|---|---|---|---|
| 0.5 | 5 | 0% | 0/5 | 1/5 | 1/5 |
| 0.5 | 5 | 2% | 0/5 | 0/5 | 0/5 |
| 0.5 | 5 | 5% | 0/5 | 0/5 | 0/5 |
| 0.5 | 5 | 10% | 0/5 | 0/5 | 2/5 |
| 0.5 | 9 | 0% | 1/5 | 1/5 | 1/5 |
| 0.5 | 9 | 2% | 0/5 | 0/5 | 0/5 |
| 0.5 | 9 | 5% | 2/5 | 3/5 | 3/5 |
| 0.5 | 9 | 10% | 4/5 | 4/5 | 4/5 |
| 0.5 | 13 | 0% | 0/5 | 0/5 | 0/5 |
| 0.5 | 13 | 2% | 0/5 | 0/5 | 0/5 |
| 0.5 | 13 | 5% | 4/5 | 2/5 | 2/5 |
| 0.5 | 13 | 10% | 3/5 | 2/5 | 2/5 |
| 0.3 | 5 | 2% | 0/5 | 0/5 | 0/5 |
| 0.3 | 5 | 5% | 2/5 | 1/5 | 2/5 |
| 0.3 | 5 | 10% | 3/5 | 1/5 | 1/5 |
| 0.3 | 9 | 2% | 0/5 | 1/5 | 1/5 |
| 0.3 | 9 | 5% | 3/5 | 3/5 | 3/5 |
| 0.3 | 9 | 10% | 3/5 | 3/5 | 3/5 |
| 0.3 | 13 | 2% | 1/5 | 2/5 | 3/5 |
| 0.3 | 13 | 5% | 4/5 | 2/5 | 3/5 |
| 0.3 | 13 | 10% | 3/5 | 0/5 | 1/5 |

## Fragment criterion: bar 1, and bar 1 with the timing bar

| reverse | cells | share | standard read-out bar 1 | adjacent pairs bar 1 | all pairs bar 1 | centred direction oracle bar 1 | standard read-out both bars | adjacent pairs both bars | all pairs both bars |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 5 | 0% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 0.5 | 5 | 2% | 0/5 | 1/5 | 1/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 0.5 | 5 | 5% | 0/5 | 0/5 | 0/5 | 2/5 | 0/5 | 0/5 | 0/5 |
| 0.5 | 5 | 10% | 1/5 | 1/5 | 1/5 | 5/5 | 0/5 | 0/5 | 1/5 |
| 0.5 | 9 | 0% | 0/5 | 0/5 | 0/5 | 2/5 | 0/5 | 0/5 | 0/5 |
| 0.5 | 9 | 2% | 1/5 | 0/5 | 0/5 | 1/5 | 0/5 | 0/5 | 0/5 |
| 0.5 | 9 | 5% | 2/5 | 2/5 | 2/5 | 5/5 | 1/5 | 2/5 | 2/5 |
| 0.5 | 9 | 10% | 3/5 | 5/5 | 5/5 | 5/5 | 2/5 | 4/5 | 4/5 |
| 0.5 | 13 | 0% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 0.5 | 13 | 2% | 0/5 | 0/5 | 0/5 | 3/5 | 0/5 | 0/5 | 0/5 |
| 0.5 | 13 | 5% | 4/5 | 3/5 | 3/5 | 5/5 | 3/5 | 2/5 | 2/5 |
| 0.5 | 13 | 10% | 3/5 | 5/5 | 5/5 | 5/5 | 3/5 | 2/5 | 2/5 |
| 0.3 | 5 | 2% | 0/5 | 0/5 | 0/5 | 1/5 | 0/5 | 0/5 | 0/5 |
| 0.3 | 5 | 5% | 2/5 | 1/5 | 1/5 | 3/5 | 2/5 | 1/5 | 1/5 |
| 0.3 | 5 | 10% | 3/5 | 2/5 | 3/5 | 5/5 | 2/5 | 1/5 | 3/5 |
| 0.3 | 9 | 2% | 0/5 | 1/5 | 1/5 | 1/5 | 0/5 | 0/5 | 0/5 |
| 0.3 | 9 | 5% | 3/5 | 2/5 | 2/5 | 5/5 | 3/5 | 2/5 | 2/5 |
| 0.3 | 9 | 10% | 5/5 | 5/5 | 5/5 | 5/5 | 3/5 | 3/5 | 3/5 |
| 0.3 | 13 | 2% | 2/5 | 1/5 | 3/5 | 3/5 | 0/5 | 1/5 | 2/5 |
| 0.3 | 13 | 5% | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 | 3/5 | 3/5 |
| 0.3 | 13 | 10% | 5/5 | 5/5 | 5/5 | 5/5 | 3/5 | 0/5 | 1/5 |

## Controls (9 ordered cells per replay): whole-track passes, and fragment + timing

| control | n | standard read-out | adjacent pairs | all pairs | centred direction oracle | standard read-out fragment+timing | adjacent pairs fragment+timing | all pairs fragment+timing |
|---|---|---|---|---|---|---|---|---|
| assemblies_no_order | 5 | 3/5 | 5/5 | 5/5 | 5/5 | 1/5 | 0/5 | 0/5 |
| forward_only_5pct | 5 | 4/5 | 2/5 | 5/5 | 3/5 | 5/5 | 2/5 | 5/5 |

## Pass criteria (against the standard read-out on the same draws)

- False positives at share 0, standard read-out: 4/15 whole-track passes; with the timing bar 1/15.
- False positives at share 0, adjacent pairs: 3/15 whole-track passes; with the timing bar 2/15.
- False positives at share 0, all pairs: 3/15 whole-track passes; with the timing bar 2/15.
- False positives at share 0, centred direction oracle: 0/15 whole-track passes; with the timing bar –/15.
- adjacent pairs: does not beat the standard read-out. Cells gained by 3 or more draws (low-share or few-cell region): none; lost by 3 or more: none. Total passes over the 18 cells with replays: 40 against 38.
- all pairs: does not beat the standard read-out. Cells gained by 3 or more draws (low-share or few-cell region): none; lost by 3 or more: none. Total passes over the 18 cells with replays: 47 against 38.
- Assemblies without order, adjacent pairs: 0/5 under fragment + timing (criterion: at most 1 of 5).
- Assemblies without order, all pairs: 0/5 under fragment + timing (criterion: at most 1 of 5).

## Check B: the HMM (20 ms bins, 30 states) on 9 ordered cells per replay

| reverse | share | n | congruent at 95% (share of events) | at 99% | false alarms at 95% | recall of replays at 95% | AUC of congruence |
|---|---|---|---|---|---|---|---|
| 0.5 | 0% | 5 | 0.89 | 0.62 | 0.89 | nan | nan |
| 0.5 | 5% | 5 | 0.87 | 0.62 | 0.89 | 0.73 | 0.37 |
| 0.5 | 10% | 5 | 0.86 | 0.59 | 0.87 | 0.76 | 0.38 |
| 0.3 | 5% | 5 | 0.85 | 0.63 | 0.86 | 0.76 | 0.36 |
| 0.3 | 10% | 5 | 0.85 | 0.60 | 0.86 | 0.72 | 0.36 |
| assemblies_no_order | – | 5 | 0.88 | 0.65 | 0.88 | nan | nan |
| forward_only_5pct | – | 5 | 0.85 | 0.60 | 0.85 | 0.83 | 0.43 |
