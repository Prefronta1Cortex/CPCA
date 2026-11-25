# Candidate datasets beyond hc-11 (checked October 28)

Open with `Cmd+Shift+V`. Survey: `scripts/survey_huszar.py` (`results/datasets/huszar_sessions.csv`); pilot: `scripts/pilot_huszar.py` (`results/datasets/huszar_pilot.md`).

## What the method needs

Spike-sorted single units (complex PCA works on cells); many place cells recorded together (on hc-11 only the two Achilles sessions, with 50+ per direction, carry the results); many reactivation events; a waking map to score against; ideally a choice in the task and sleep before and after.

## The candidates

| dataset | sorted units | cells per session | events | choice | sleep before / after | size, access | verdict |
|---|---|---|---|---|---|---|---|
| hc-11 (Grosmark & Buzsáki 2016), current | yes | 13–36 place cells per direction in 6 of 8 sessions (77 in Achilles 10/25) | 1,500–6,000 bursts | none | yes, novel track | local | power-limited |
| Gillespie et al. 2021, DANDI 000115 (the mentor's link) | **no**: raw 30 kHz voltage only (128 channels, 32 tetrodes; no units table) | unknown until sorted | thousands of SWRs, 35–47% with remote replay | eight-arm win-stay lose-switch | yes (sleep / run / sleep / run / sleep) | 9.1 TB, 90–220 GB per session | ideal task, impractical: needs spike sorting from raw data |
| hc-6 (CRCNS) and its extension on Dryad (Denovellis et al. 2021: 10 rats incl. the Kay et al. 2020 animals) | yes (clustered cells, ripple times, multiunit) | not verified; Carr et al. 2012 report 292 CA1 cells pooled over 3 rats and all days, i.e. tens per day | many awake SWRs | W-track alternation | rest epochs between runs | 22 GB (CRCNS) / 72 GB (Dryad, CC0); Dryad blocks scripted downloads; CRCNS hc-6 asks that the data collector and L. Frank be authors | good task, cell counts to verify; authorship condition to clarify |
| **Huszár et al. 2022, DANDI 000552** | **yes** (units table, spike times) | **median 99 units in the 65 task sessions (34–376), 8,201 in total** | **median 4,725 ripples per session** | **figure-eight alternation: median 22 trials, accuracy 0.81, 50 sessions with at least 3 errors** | **~2 h before and after** | **12.6 GB processed NWB, CC-BY, streamable** | **best fit; mice; units not typed (interneurons by rate)** |

The Huszár sessions also include 16 explorations of novel environments (two linear tracks, three open fields, a figure-eight) between sleep periods, which would give a clean PRE/POST control like hc-11's.

## Pilot on two Huszár sessions (fixed in advance: most units in each of the two mice with most units)

The hc-11 pipeline unchanged: putative pyramidal cells (below 8 Hz), place fields on the linearized figure-eight (at least 40 spikes while running, peak at least 2 Hz), population bursts in NREM of the sleep before (PRE) and after (POST) the task, centred complex PCA, map tests per component and the best output against its best-of-outputs null, and the waking-against-sleep check with the alternation trials as waking events.

| session | place cells | epoch | events | components on the map | best output |ρ| (p) | waking vs sleep ρ (p) |
|---|---|---|---|---|---|---|
| e15-13f1 2022-01-17 (376 units) | 224 | PRE | 1,279 | 1/15 | 0.20 (0.28) | +0.08 (0.0005) |
| | | POST | 1,607 | 2/19 | 0.34 (0.004) | +0.17 (0.0005) |
| e13-26m1 2021-10-19 (271 units) | 153 | PRE | 2,201 | 0/11 | 0.18 (0.70) | +0.17 (0.0005) |
| | | POST | 1,512 | 3/12 | 0.26 (0.12) | +0.21 (0.0005) |

Three to ten times as many place cells per session as most hc-11 sessions. The direction of every result matches hc-11 (more map order after the task than before), with one session's best output above chance after the task and neither before. Waking and sleep fits agree in both epochs: the figure-eight is a familiar maze, so its structure is expected in sleep before the task too; the novel-environment sessions are the ones with a clean control. The place-cell criterion here (peak at least 2 Hz) is looser than hc-11's (3 Hz), and about two thirds of the putative pyramidal cells pass it (224 of 348, 153 of 248); it should be tightened, and cells typed properly, before any claim.

## Recommendation

Huszár et al. 2022 (DANDI 000552) as the next real dataset: sorted, large, small to download, with a choice and sleep around the task. Analysis plan: (1) the novel-environment sessions for the PRE/POST test with many more cells than hc-11; (2) the figure-eight sessions for the two-sequence problem (left and right arms share the stem; Step 3) and for choice (correct against error trials; replay of the chosen against the other arm before each trial); (3) the hc-11 analyses repeated unchanged, so that the comparison is about the data. The Dryad W-track data are the second choice once their cell counts are checked (a browser download of one rat, e.g. Remy, 3 GB); the Gillespie data only if sorted units become available.

## Follow-up (October 28)

Complex PCA run on all 65 task sessions with the hc-11 measures: `docs/huszar.md`. Population-level structure in most sessions of both sleeps; sequence-level recovery weak.
