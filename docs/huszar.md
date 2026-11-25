# Complex PCA on DANDI 000552 (Huszár et al., 2022), compared with hc-11

Open with `Cmd+Shift+V`. Code: `cpca/huszar.py` (loader), `scripts/run_huszar.py` (both datasets, the same measures), `scripts/summarize_huszar.py`. Results: `results/huszar/summary.md`; figure: `figures/huszar_vs_hc11.png`. Complex PCA only; seqNMF and PP-Seq are not run.

## Design (written on October 28 before the full run; the pilot of `docs/datasets.md` used two of these sessions)

**Data.** The 65 sessions with alternation trials (11 mice, figure-eight maze, left/right alternation with a 5 s delay). Per session: the task epoch is the one containing the first trials; PRE is the sleep epoch before it, POST the sleep epoch after it. The figure-eight is familiar to the mice (trained over days), unlike hc-11's novel tracks, so PRE is a weaker control here than on hc-11: replay of a familiar maze is expected in both sleeps.

*Amendment after the first run (before the results were summarized):* 22 files list only the first epoch in their epoch table and one lists none, although the recordings continue through the task and the sleep after it. For these the windows are taken from the trials: the task is the span of the trials, PRE everything before it, POST everything after it (only NREM is used, so any other activity in these windows does not enter). All 000552 sessions were rerun with the same definitions.

**Cells.** Units are not typed in the files: putative pyramidal cells are those firing below 8 Hz over the session. Place fields as on hc-11: rate maps over the linearized maze (40 bins), running faster than 5 cm/s, smoothed one bin; a field needs at least 40 spikes and a peak of at least 3 Hz. The linearization puts both arms on the same coordinates, so the two maps are built from left-arm and from right-arm trials separately, as hc-11 has one map per running direction.

**Events.** Population bursts of the putative pyramidal cells in NREM of each sleep epoch (`cpca.events.population_bursts`, as on hc-11). Waking events: the trials (start to end), as the laps on hc-11.

**Measures, identical code for both datasets** (hc-11 recomputed with it, so the comparison does not depend on earlier runs):

1. components above the shuffle null (centred);
2. component tests: each component's order against each map, |ρ| above its 95% null;
3. best output per session: the best component, map and sense against a null that takes the same best (`scripts/plot_hc11_sequences.best_map_match`);
4. topology: closeness in the reduced space against the similarity of the two cells' maps (Mantel);
5. waking against sleep: closeness in the waking fit (uncentred) against the sleep fit (Mantel);
6. direction labels: one template on all cells (Step 2), its held-out test, and its labels against each event's rank-order correlation with the map;
7. templates (Step 3): how many pass the held-out check, and how many of those match a map.

Sessions of one mouse are not independent; results are given per session and per mouse.

**Expected.** With 3–10 times as many place cells as most hc-11 sessions, measures 2–7 should be above chance in a larger share of POST sessions than on hc-11. PRE should be lower than POST, but less clearly than on hc-11 because the maze is familiar.

## Results (`figures/huszar_vs_hc11.png`, `results/huszar/summary.md`)

65 sessions, 11 mice; median 57 place cells in the larger map (20–178) against 24 (13–63) on hc-11; about half as many sleep events per epoch (median 1,690 against 3,400–4,000). The expectation was met for the population-level measures and not for the sequence-level ones.

- **Population level: strong in both sleeps.** The reduced space keeps the map's topology in 89% of POST and 78% of PRE sessions (hc-11: 75% and 12%); the sleep fit agrees with the waking fit in 80% and 74% (hc-11: 50% and 12%); a single template on all cells holds up on held-out events in 65% and 69% (hc-11: 38% and 38%). Averaged over mice the numbers are the same. With a familiar maze, map structure in the sleep before the task is expected, so the absence of a POST/PRE difference here is not a failure of the control; it also means these measures cannot separate experience from pre-existing structure on this dataset.
- **Sequence level: weak, and not better than hc-11 despite the cells.** Single components match a map in 9.9% of tests after the task and 6.2% before (chance 5%; hc-11: 15.8% and 2.9%); the best component per session beats its best-of-outputs null in 12% of POST sessions and 8% of PRE (hc-11: 25% and 0%); validated templates (Step 3) match a map in 6% and 12%. Panel C shows no relation between the number of place cells and the significance of the best component.
- **Direction labels are right where a template matches the map**: 178 of 189 significant replays labelled in the map's direction after the task and 260 of 268 before (94–97%; hc-11: 65 of 88).

So more cells did not, by themselves, make complex PCA recover individual replayed sequences: what it recovers robustly on both datasets is the population structure (who fires with whom, and in which phase relation), which the topology and waking-sleep checks see. On 000552 that structure is present before the task as well, as expected for a familiar maze. Possible reasons for the weak sequence level, not tested here: the two arms of the figure-eight share the stem (two orders on overlapping cells, Step 3's regime, where it validated few templates); mouse population bursts are short; and familiar-maze replay may be sparser per event. The 16 novel-environment explorations of this dataset remain the cleaner test of experience.
