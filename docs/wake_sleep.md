# Waking fit against sleep fit: a basic consistency check

Open with `Cmd+Shift+V` for formulas. Code: `cpca/hc11.py::laps`, `scripts/run_wake_sleep.py`, `scripts/summarize_wake_sleep.py`. Results: `results/wake_sleep/summary.md`; figure: `figures/wake_sleep.png`.

## Why

Suggested by the project mentor as a check that does not aim high: fit the same model separately to the waking data and to the sleep reactivation events, and see whether the two fits agree. Agreement does not prove that the sleep fit extracts useful information (both could share something trivial); no agreement where replay is expected would almost certainly mean something is wrong in the implementation.

## Design (written on October 28 before any result)

**Waking events.** Continuous runs in one running direction (pauses up to 0.5 s bridged) that cover at least 60% of the track; on circular tracks a longer run is cut into pieces of one track length. On the five linear tracks these are the laps from one end to the other (20–81 per session); on the three circular tracks, where tracking covers only 29–58% of the samples, there are 13–51. Each event is encoded with its own duration as the half-circle scale, all of a cell's spikes in it averaged into one arrow. Because the sequence is in every lap, the waking fit is uncentred (the regime rule); the centred waking fit is reported as well.

**Sleep events.** All population bursts of POST and of PRE sleep, centred complex PCA, as in `scripts/run_hc11.py` (the uncentred sleep fit is reported as well).

**The statistic.** For every pair of cells, their closeness in the waking fit's reduced space against their closeness in the sleep fit's reduced space ($\mathrm{Re}\,C_K$ normalised, all components above each fit's shuffle null; `cpca.topology.reduced_similarity`), Spearman over pairs, cell-permutation (Mantel) null with 2000 shuffles. Cells: pyramidal cells active in at least 5 waking events and in at least 1% of that epoch's sleep events. Closeness in $\mathrm{Re}\,C_K$ does not change when a replay runs backwards (conjugation), so forward and reverse replays count alike. No map is used.

**Checks around it.**

- The waking fit against the waking place map (correlation of direction-resolved rate maps, place cells only): the waking fit must reflect the map, otherwise the encoding of the laps is broken.
- Planted data (80 cells; 60 waking laps, each playing one of two maps on the same cells forward, as the two running directions do; 3000 sleep events; 10 draws per condition): sleep replays of the same two maps (as ladder rung L2, and as the realistic condition H); sleep replays of two other maps on the same cells; sleep without replays.

**Expected if the implementation is right.**

1. Planted data: waking and sleep fits agree (p < 0.05) in at least 9 of 10 draws with the same maps, in both sleep conditions; in at most 1 of 10 with other maps; no component (nothing to test) without replays.
2. hc-11: the waking fit matches the waking map in every linear-track session.
3. hc-11: waking and POST sleep fits agree in more sessions than chance (binomial against 5%), and more often than waking and PRE. Given the topology result (5 of 8 sessions, mostly co-firing), agreement is expected mainly in the sessions with many place cells.

A failure of 1 or 2 means an implementation problem; a failure of 3 alone would mean that the sleep fit does not reflect waking structure on these recordings, which is a result, not necessarily a bug.

## Results (`figures/wake_sleep.png`)

All three expectations are met.

1. **Planted data.** With the same maps the two fits agree in 10 of 10 draws with whole replays in sleep (median ρ 0.83) and in 10 of 10 with realistic sleep, partial replays and assemblies (0.34); with other maps on the same cells in 0 of 10 (median −0.02); without replays the sleep fit keeps no component, so there is nothing to compare.
2. **The waking fit reflects the waking map** in all 8 sessions (ρ 0.28–0.51, p ≤ 0.005), the circular tracks included.
3. **hc-11, waking against sleep.** POST: 4 of 8 sessions above chance (Achilles 10/25 ρ 0.17, Achilles 11/01 0.13, Cicero 09/10 0.25, Gatsby 08/02 0.14; binomial p = 0.0004); PRE: 1 of 8 (Cicero 09/10, 0.16; p = 0.34). POST exceeds PRE in 5 of 8 sessions (sign test p = 0.36). The one PRE agreement is in the session with the fewest waking events (13 runs on a circular track with poor tracking), which also has the highest POST value.

So the implementation passes the check: where the answer is known the fits agree exactly when they should, and on real data the sleep fit after the maze shares the waking fit's structure in half of the sessions, the same sessions that carry the other hc-11 results. As stated above, this does not show that the sleep fit captures sequences: the waking fit's closeness also contains co-firing, and the topology test found that co-firing carries most of the sleep structure.

Other conventions (`results/wake_sleep/summary.md`, POST / PRE sessions above chance): waking centred against sleep centred 5/8 and 2/8; waking uncentred against sleep uncentred 3/8 and 1/8; waking centred against sleep uncentred 3/8 and 2/8. Every convention gives more agreement after the maze than before; the uncentred sleep fit agrees least, as the regime rule predicts for sleep.
