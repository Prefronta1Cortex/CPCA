# Plan (version 2, October 2026)

Open with `Cmd+Shift+V` for formulas. This replaces the original plan; the history of what changed and why is in `lessons_learned.md`.

## Objective

A dimensionality-reduction method for hippocampal reactivation: a few components, each a set of cells with weights and relative firing times, found from sleep spikes alone, such that components correspond to routes (or route segments) run while awake. Advantages to preserve over sequence-detection methods: simplicity (one eigen-decomposition), promotability (the standard count-based assembly pipeline with time added), very few parameters, and no behavioural input. The waking map is the answer key, never an input.

## The pipeline

Population bursts in NREM → one arrow per cell per event, window = the event centred on its peak, $T$ = event duration (half-circle rule) → centre and scale per cell (sleep regime) → complex PCA → keep components above a shuffle null (equal to the Marchenko–Pastur bound for centred arrows) → read every kept component (weights, relative times) → label-free quality (split-half, span) → answer key once (member order vs place-field centre, per direction, with PRE sleep as control). Details in `method.md`; code in `cpca/pipeline.py::fit`.

## Correction and repair (November 5)

**Correction.** Results scored with the best-output test show that a sequence order is present among the components, not that routes are separated onto components. Goal and criteria: `docs/goals.md`.

**Repairs:**
1. **Read-out check** (`docs/readout_check.md`). Circularity fails, and a shuffle of spike trains between cells fails. A cross-validated latency null (each cell's own timing kept, coordination within events destroyed) rejects 97–100% of components from no-replay and map-ordered-assembly data while keeping 86–91% of route components. It is adopted for reactivations only: it would reject sequences played identically in every event.
2. **Separation, per component** (`docs/separation.md`). No method separates two routes on different cells: 0–13% of outputs pass, no clean fit. With two maps on the same cells, the Step 3 templates come close: 76–84% of outputs pass, 6–7 of 10 fits clean.
3. **Re-scoring, per component** (`docs/rescore.md`):
   - **hc-5:** meets the goal (uncentred; every component is the sequence in 27/36 aligned and 30/36 unaligned fits);
   - **hc-11 laps:** one-map circular tracks 67–100% of components pass; linear tracks 0–75%, where components are the sum and difference of the two running directions and the passes partly reflect low power against the second map;
   - **hc-11 sleep:** 21% of the latency-checked outputs match exactly one running direction in POST, against 10% in PRE (chance about 10%).

4. **Un-mixing** (`docs/unmixing.md`, main problem from November 5):
   - the routes lie inside the kept components' span (order target 0.87–0.93 against 0.14–0.26 for random cells);
   - no candidate step separates them cleanly. A sparse (quartimax) rotation raises one-route outputs from 2–10% to 52–62% for routes on different cells, with 0 clean fits;
   - per output, PP-Seq and seqNMF do not give one sequence per output either.
5. **Un-mixing step** (`cpca/separate.py`, `docs/unmixing.md`):
   - **the steps:** sparse rotation, ordered templates seeded from the rotated components and refined on their events, explaining away, a reproduced split for orders on shared cells, and a latency check corrected for the number of outputs;
   - **fresh planted draws:** 80–100% of outputs are one route, and 8–10 of 10 fits are clean in every replay condition, with no outputs without replays. No map is used;
   - **waking separation (November 19, forward-only mode):**
     - **planted laps:** the bar is met;
     - **hc-11 circular tracks:** 6/6 fits clean;
     - **hc-5 wheel:** 9/12 clean;
     - **controls:** no outputs;
     - **hc-11 linear tracks:** 5/10 clean, against a bar of 7/10. The directions separate in Achilles 10/25 and Gatsby 08/02, and fail in the smallest sessions (`results/waking_separation/`).
   - **waking test (November 19):** with the event floor lowered to 4, the hc-5 wheel stays one output and matches in 9 of 12 fits; the two directions of hc-11 linear laps do not separate (`results/unmix_waking/`).

6. **Separation step, simplified, back in the pipeline** (`cpca/separate.py`, `docs/unmixing.md`, November 24):
   - **the stages:** rotation, refined templates, merging and explaining away, a split with mode-specific reproduction rules, and a held-out timing check;
   - **evidence:** each stage is shown to be essential by switching it off;
   - **confirmed results kept:** planted reactivations 8–10/10 clean; planted laps 7–10/10; hc-11 linear laps 5/10, circular 6/6; hc-5 wheel 9/12; no outputs in the controls.

7. **Parts of the exploration adopted** (`docs/adoption.md`, November 25): the complex ICA fixed (conjugation
   errors; gradient ascent on unitary rotations is now the engine). Centring on active entries
   (`center="active"`) is available: it lowers the hc-11 spike-in threshold from about 10% to 3–5%, but missed its
   pre-set false-positive bar by one hit (1/30 against 0/30 at 0%), so it is not the default. The separation step
   is unchanged; no variant beat it by the pre-set margin. A proper false-positive check followed: active
   centring passes it (0–7 of 160 timing-shuffled fits pass, against 11–45 for column centring). But its own PCA
   read-out folds the order (middle cells of a route get small weights), so it is suited only as the input to the
   separation step; the default stays column centring until that is decided. The 1.5π arc was tested on the
   current pipeline and rejected (worse on waking data and spike-in).

## Status

Done:

- Library (`cpca/`), 68 tests, synthetic route generator (`sim/routes.py`), hc-11 loader and event detection, count PCA/ICA baseline.
- Synthetic benchmark (`results/synthetic/summary.md`, 10 draws per condition): all clean conditions recovered (median $|\rho|$ 0.97–1.00), including two routes in both directions at 15% replay and 15% participation; the realistic condition G recovered above chance in 80–90% of (route, draw) pairs at median $|\rho|$ 0.37–0.48. Difficulty ladder (condition H taken apart, `figures/synthetic_ladder.png`, `results/ladder/summary.md`; cPCA, seqNMF and PP-Seq on the same 1500 events, 3 draws per rung): with whole routes replayed (two maps on the same cells, both directions, assemblies, variable pacing) PP-Seq is essentially perfect (median $|\rho|$ 0.99–1.00) and centred cPCA recovers every map at 0.81–0.90; seqNMF drops once backward plays are present (0.26–0.47). Partial replays are hard for every method: on L3 and L6 (= H's settings) cPCA 0.59 and 0.48 (4/6, 5/6 maps), PP-Seq 0.35 and 0.39 (3/6 each, per-map values from 0.13 to 0.98), seqNMF at chance. PP-Seq's 0.89/0.81 on H in the matched rerun was one favourable draw. So the gap to PP-Seq is a constant precision gap on whole-route replays (present already with one map), not a failure specific to shared cells.
- hc-11 (`results/summary.md`, `figures/`): with centring, 20 of 120 (component, direction) tests match the place map in POST sleep (chance ≈ 6) vs 2 of 106 in PRE (chance ≈ 5); the effect is carried by the two sessions with 50+ place cells. Uncentred components are at chance in both (3/54, 1/44).
- Noise bound (October 21, `figures/noise_bound.png`): components are now kept against a per-cell shuffle null. It agrees with the textbook Marchenko–Pastur bound for centred, scaled arrows and corrects it for uncentred arrows, where the textbook bound kept 8–14 components per hc-11 session (16 on pure synthetic noise) against 2–6 by the shuffle. All cPCA results were recomputed; the earlier ones are in `results/_before_step0/`.
- Matched rerun after review (`docs/baselines.md`, "Corrections", `results/rerun/summary.md`): with identical 1,500-event subsets, a proper PP-Seq sweep and sampler, and a seqNMF K × L grid, PP-Seq matches or beats cPCA on planted routes (ahead in the realistic conditions G and H on that single draw; see the ladder below) while remaining at chance on hc-11 in both sleeps (2/47 POST, 2/44 PRE); seqNMF is at chance from 15% replay onward whatever K and L; cPCA keeps POST > PRE on the matched subsets (14/96 vs 6/74) and on all events (20/120 vs 2/106). The claim this supports is representation-level: cPCA's centred, event-normalised representation finds POST-specific map order on these recordings, which the other methods do not; it does not show that cPCA is a better sequence extractor in general, and PP-Seq is the more precise extractor of whole replayed routes.
- First method comparison with the authors' procedures (`docs/baselines.md`, `results/baselines/summary.md`, `figures/baselines_*.png`): seqNMF recovers planted routes as well as cPCA when replays are common (A–E) but not at 15% replay (F: 0.5 vs 0.97–0.99) and not in the realistic conditions; on hc-11 its one large factor per session is the synchronous burst (lag span 30–50 ms) and no factor orders place cells above chance. The Maboudi HMM detects planted replays (99–100%) with 29–40% false alarms; on hc-11 it calls 14–30% of events congruent in POST and 10–32% in PRE (not maze-specific), while its latent space, labelled with behaviour post hoc, decodes position far better from POST than PRE models in the two large sessions (0.21 vs 0.50 m; 0.10 vs 0.57 m), the same POST/PRE dissociation cPCA shows label-free. Bayesian decoding (map as input) marks 10–22% of POST and 8–22% of PRE events. PP-Seq (authors' Julia package, held-out hyperparameter selection as in the paper) recovers planted routes as well as cPCA when replays are common (A–E: 0.96–1.00) and one of two routes at 15% replay (F: 0.82 / 0.10); it models synchronous volleys as a zero-duration sequence type; on hc-11 its types are the population burst and small sub-bursts (spans 4–41 ms) and 1/30 map tests pass in POST, 1/25 in PRE. convSeq was not run (`docs/baselines.md`).

- Real-data benchmark with an answer key (`docs/benchmark_hc5.md`, `results/hc5/summary.md`, `figures/hc5_*.png`): hc-5 CA1 wheel sequences, six sessions, cross-fitted on held-out runs, cPCA vs seqNMF vs PP-Seq with each method's own settings procedure. Aligned windows: cPCA 32/36 fits recover the sequence, seqNMF 29/36, PP-Seq 14/36 (averaging the runs, ρ 0.71, is the ceiling). Unaligned windows: cPCA 35/36 (ρ 0.63), seqNMF 26/36, PP-Seq 13/36, averaging falls to ρ 0.47. Shuffled control at chance for all. cPCA about 50 ms per fit, seqNMF 50 s, PP-Seq 16 min. The upcoming left/right choice is not decodable from wheel activity in these sessions, so choice-specific sequences could not be tested.

- Topology test (October 22, `docs/topology.md`, `figures/topology_hc11.png`): closeness of every pair of place cells in the reduced space (all kept components) against the similarity of their waking rate maps (position × running direction), Mantel null. hc-11: 5/8 sessions above chance in POST, 1/8 in PRE; count PCA (no timing) about as good, so the topology in sleep is mostly co-firing. Validated: 0–10% false positives on null data; passes on co-firing alone (stated limit); where only timing carries the order (planted whole-route replays, hc-5) complex PCA passes and count PCA fails.

- Forward and backward plays (October 22, `docs/direction.md`, `figures/direction_*.png`): time reversal is complex conjugation, so one template $q$ matches forward events through $|\langle s,q\rangle|$ and backward ones through $|\langle s,\bar q\rangle|$; each component is split by alternating labels and template, gated per event by a within-event shuffle, tested on held-out events, and guarded (the template replaces the component's order only if they agree). Planted data, fresh draws: where one route is played both ways the order goes from 0.74–0.81 to 1.00 and every event's direction is right (median 100%, reverse share within 0.01); no reverse group in forward-only data (0.04–0.05) and nothing in a no-order control; partial replays are outside the model (labels 54–60%). hc-11: on Achilles 10/25 POST a template learned without the map labels 43 of 46 significant replays in the map's direction, and the forward and backward groups match the map with opposite signs (23 of 26 templates); where templates match the map only narrowly the labels are at chance, and the held-out test passes in the same three sessions in PRE as in POST. hc-5 has too few runs for the split.
- Sequences that share cells (October 22, `docs/sharing.md`, `figures/sharing_*.png`; acceptance criteria written before the fresh runs): when two maps share cells PCA returns their sum and differences; a step after PCA (one template per sequence started from the components and their ICA rotation, each event assigned to the template it matches either way round, refitted, and kept only if it holds up on held-out events) recovers both maps at median |ρ| 1.00 on fresh draws and on the ladder's own events (L1, L2, L4, L5; components 0.81–0.88, PP-Seq 0.99–1.00), changes nothing where sequences do not share cells, and validates nothing in the no-order control. Partial replays improve (ladder L3 0.59 → 0.70, L6 = H 0.48 → 0.81, PP-Seq 0.35 and 0.39) but come back as segments with some false templates. 3.5 s per fit against 116 min for PP-Seq. On hc-11 it adds almost nothing: 3 of 88 templates pass the held-out check (one on the map, Achilles 10/25 POST), because single sleep events carry too little order for a per-event test.

- Waking fit against sleep fit (October 28, `docs/wake_sleep.md`, `figures/wake_sleep.png`; mentor's suggestion, expectations written before the run): complex PCA fitted to the laps and to sleep bursts separately, pair closeness compared without a map. Planted data: agreement in 10/10 draws with the same maps (whole and partial replays), 0/10 with other maps; the waking fit matches the waking map in 8/8 hc-11 sessions; waking and sleep fits agree in 4/8 sessions after the maze and 1/8 before. The implementation passes the check; it does not by itself show that the sleep fit captures sequences.

- Candidate datasets (October 28, `docs/datasets.md`): the mentor's Gillespie et al. 2021 data (DANDI 000115) have no sorted units (raw voltage only, 9.1 TB); the Frank-lab W-track data (hc-6, Dryad) have sorted cells but unverified counts; Huszár et al. 2022 (DANDI 000552) has sorted units (median 99 per task session), figure-eight alternation with error trials, ~4,700 ripples per session and sleep before and after, in 12.6 GB. Pilot on two of its sessions with the hc-11 pipeline unchanged: 153–224 place cells, more map order after the task than before.

- Second dataset, complex PCA only (October 28, `docs/huszar.md`, `figures/huszar_vs_hc11.png`): DANDI 000552, 65 sessions, 11 mice, median 57 place cells (hc-11: 24), identical code and measures for both datasets. Population-level structure is strong in both sleeps (topology 89% / 78% of sessions, waking-sleep agreement 80% / 74%; the maze is familiar, so PRE carries it too); sequence-level recovery is weak and not better than on hc-11 (best component on the map in 12% / 8% of sessions; component tests 9.9% / 6.2% against 5% chance). More cells alone did not fix sequence extraction.

- One order from the whole reduced space (October 28, `docs/subspace.md`, `figures/subspace_hc11.png`; hc-11): spectral seriation and angular synchronisation of the reduced covariance do not beat the best single component (POST 1/8 and 2/8 against 2/8; PRE 0/8); the timing-free count PCA seriation 1/8. The topology of the reduced space does not become a global cell order.

- Supervised ceiling (October 28, `docs/ceiling.md`, `figures/ceiling.png`): replays chosen with half of the place cells, order scored on the other half. The diagnostic has power on planted data (+14 to +41 points over chance for whole-route replays, +1.7 for segment replays in busy events); on hc-11 and 000552 the excess is +0.3 and 0.0 points, under any event definition. Single sleep events in these recordings carry almost no map order; the plateau is the data at the per-event level, not selection or read-out. Planted segment replays also show a method weakness: given the right events, complex PCA as is (16–24%) loses to averaging and to direction-aligned fitting (92%).
- Is there enough replay? (`docs/replay.md`, `figures/replay_hc11.png`). The original paper's Bayesian replay analysis, reproduced on hc-11, confirms their finding: sequence content (rZ) is higher in POST than in PRE (7/8 sessions, pooled p = 0.002) and highest at rest on the maze. Our event windows carry the same content as the paper's events. But clear replays (significant under both shuffles) exceed the PRE level by a median of 0.8 and at most 3.4 percentage points of events. Adding replays to real PRE events shows that complex PCA needs about 10% ideal replays in the sessions with the most place cells, and more than 20% in small ones; real awake replays at 1–3% are never found. The plateau on hc-11 is the data's replay content, 3 to 10 times below what a label-free method needs, not a pipeline fault. Side findings that affect all methods alike: the paper's place-cell criterion keeps about 1.5 times as many cells as our answer key, and the rotation search on hc-11's circular track (a linearised one-way run) is unnecessary and costs power.
- hc-11 waking laps, cross-validated as on hc-5 (`docs/laps_hc11.md`).
  - **Result:** the place-field order on held-out laps is recovered by complex PCA uncentred in 37/48 fits, by count PCA in 15/48 (5/30 on linear tracks), and by the averaging reference that uses lap direction in 41/48.
  - **Reading:** on the same cells whose reactivations no method decodes, timing gives complex PCA a clear edge over counts. The sleep failure is specific to the reactivation regime.
- Complex PCA on DANDI 001695 (`docs/dandi001695.md`, `results/dandi001695/summary.md`).
  - **Data:** mice, Neuropixels 2.0, novel linear track; 15 sessions from 4 mice; about twice hc-11's place cells active per event.
  - **Primary result:** POST passes order + timing in 3/15 sessions, from 3 mice, against PRE 0/15. This meets the pre-registered bar, but only just.
  - **Secondary results do not corroborate it:** fragment + timing 1/15 against 0/15; ripple-gated 1/12 against 1/12; CA1 + CA3 1/15 against 1/15. Maze rest passes order 8/15 but timing 0/15.
  - **Replay content:** the decoder finds about 3% clear replays in both sleeps (7.7% at rest on the track).
  - **Reading:** a small, fragile improvement over hc-11.
- Sanity check, revised: are complex PCA's waking components found within its sleep components? (`docs/containment.md`, `results/containment/summary.md`).
  - **The question:** each waking component's share inside the span of the POST sleep components and their conjugates, against the sleep fit's cell identities permuted. There is no timing bar and no PRE, by design.
  - **Planted data:** passes for both maps, one map and segments (5/5 each); rejects other maps on the same cells (1/5).
  - **hc-11 POST:** 6/8 sessions, with 12/18 waking components contained individually.
  - **Replaces** the symmetric correlation of `docs/wake_sleep.md` and the earlier sleep-inside-waking design (kept in `results/containment/sleep_in_waking/`).
- The recurring-fragment criterion (`docs/fragment.md`, `results/fragment/summary.md`).
  - **The criterion:** order within one 40% stretch of the track, with at least 8 cells and a null that searches stretches too, plus a timing bar (the real fit must beat refits on spike trains shuffled within events).
  - **Planted data:** it recovers whole-route and segment replays (both bars in 4–5 of 5 draws, including the busy-event segment case) and rejects assemblies that follow the map without order (0/5). Count PCA passes bar 1 on those assemblies in 4 of 5 draws.
  - **hc-11:** nothing passes, in POST, PRE or maze rest. At hc-11's cell counts (25 route cells), even 20% planted segment replays clear both bars in only 1 of 4 draws, so the hc-11 result reflects low power, not absence.
  - **Reading:** the criterion is specific and should be used for claims, but it needs data with more cells per stretch.
- Awake rest on the maze (`docs/maze_rest.md`, `results/maze_rest/summary.md`).
  - **Primary result:** complex PCA (centred, best component) recovers the waking map in 5/8 sessions, against 0/8 in the matched PRE control. That includes all three sessions predicted to be above threshold (at least 10% clear replays and about 10 place cells per event).
  - **Count PCA** recovers the order as often as complex PCA under the same seriation read-out.
  - **A timing control, added after the results,** refits with spike trains shuffled among each event's active cells. It shows the order test can be passed by co-firing alone: in the Achilles sessions, shuffled fits match the map in 87–98/100. Order beyond co-firing is shown in 2–3/8 maze sessions (Gatsby 08/02, Cicero 09/10 or 09/17). The one sleep pass (Achilles 10/25 POST) is not shown to come from timing (timing p = 0.11).
  - **Consequence:** order claims on real data need this control. The earlier hc-11 order results should be re-read with it, and its calibration checked on planted data and hc-5.
- What any method could do on hc-11, and how much replay each needs (`docs/power.md`, `figures/power_hc11.png`).
  - **Step 0, perfect event choice:** the map-based decoder picks the clear replays with half of the place cells, and the other half is scored. This recovers the map in at most 3 of 10 splits in any session, while complex PCA without the map, scored on the same cells, reaches 5 of 10 in the best session. Knowing which events are replays does not beat complex PCA on these events.
  - **Step 1, the threshold for each method:** spike-in power test, 8 sessions.
    - Complex PCA needs about 10% ideal replays in the two largest sessions, and using all events changes little.
    - A new event-selecting variant (`cpca/sparse.py`) lowers the threshold 2 to 4 times on planted data. On real events it keeps the largest bursts or repeated co-firing, not the replays, and does worse; a cosine-matched version is worse still.
    - Rastermap needs far more replay. PP-Seq, refitted at a fixed setting, found no ideal replays even at 20%.
  - **Reading:** the hc-11 limit is shared by the label-free methods tested, and the data lie below it.
- Three more comparison methods (`docs/methods_hc11.md`, `figures/methods_hc11.png`): count PCA/ICA assemblies, SPOTDisClust (authors' code, pooled metric, with a fix to an argument-order bug in their pooled function) and Rastermap (authors' few-neuron settings, 10 ms bins). All three are first checked on planted data, where each recovers planted orders or assemblies and none reports anything without replays. On the same hc-11 events, scored by the same best-of-outputs order test, every method is at chance: complex PCA 1/8 POST and 0/8 PRE, Rastermap 1/8 and 0/8, seqNMF, PP-Seq and SPOTDis 0/8 in both. SPOTDis finds no clusters of events. Count ICA's most local assembly is significant in 2/8 POST and 1/8 PRE. Median time per fit: 0.3 s for count ICA, 0.5 s for complex PCA, 5.5 s for Rastermap, 18 s for SPOTDis, 117 s for seqNMF and 1073 s for PP-Seq.

Open (in order of value):

1. **Label-free identification of route components** (reframed: the topology test now checks the whole reduced space against the map; picking individual route components label-free may not be possible when replayed content has no recorded map). Candidates: a sequence score that rewards members spread over the half-circle *and* events whose score phase is consistent with the component's order; split-half reproducibility per component; a null specific to each component (its own members phase-randomised). Acceptance: on synthetic conditions F–H the route components rank in the top 2 by the score in ≥ 80% of draws.
2. **Direction inside a component.** Done (`docs/direction.md`); two different orders on the same cells done as well (`docs/sharing.md`). Remaining: partial replays (a segment-aware template), where the templates find segments and some false templates mixing them.
3. **Power.** Apply to a recording with several hundred simultaneous CA1 cells (public Neuropixels datasets); the POST/PRE difference should strengthen with the number of place cells. This is the single most valuable step for a biological result.
4. **Biology (if 3 succeeds).** Internal order of components across the night (early vs late POST), and by cell class (fast-firing rigid vs slow-firing plastic cells, following Grosmark & Buzsáki 2016 on these same sessions).
5. **Benchmark hygiene.** Done: 10 draws per condition, a variable-pacing condition (ladder L5). Still open: report the component-level false-positive rate of the answer-key test on PRE as the empirical null.
6. **A second real benchmark with several sequences.** hc-5 tests one dominant sequence per event. A dataset with two or more distinct sequences in the same recording (for example the Mizuseki et al. 2013 rat with separate left and right delay sequences, or convSeq's own real data) would test whether cPCA separates them onto different components on real spikes; convSeq should join that comparison through a thin adapter around the authors' model and loss.

## Decisions taken (with the evidence)

- Uncentred vs centred: both run; centred for sleep (minority replays), uncentred when a pattern is common to most events. Evidence: synthetic A–C vs G, hc-11 POST/PRE table.
- $T$: event duration, aligned on the peak. Per-event spike spread tested; no gain on hc-11 (centred: 22/112 vs 4/96 against 20/120 vs 2/106), kept as an option.
- Noise bound: shuffle null (permutation parallel analysis) for both conventions; the textbook Marchenko–Pastur bound is reported alongside. Evidence: `figures/noise_bound.png`.
- ICA: off by default; helps only when two routes share cells (synthetic E-type union), hurts with many components.
- Mixture/clustering of events, direction-split tests, and template-based detectors: removed from the core; they were answers to the wrong question (see `lessons_learned.md`). The direction split of October 22 (`cpca/direction.py`) is a different thing: it works inside one component, uses the conjugation identity, and has a per-event null, a held-out test and a guard, each checked on a no-order control.

## Publication framing

Methods paper: "the count-based assembly pipeline with time added": the half-circle rule, the uncentred/centred choice, components as assemblies with internal order, validation on planted routes and on hc-11 with PRE as control, comparison with count PCA/ICA (which finds the groups but not the order). Natural venues: *PLoS Computational Biology*, *eLife* Tools and Resources, *Journal of Neuroscience Methods*, *Neural Computation*. A biological paper requires item 3 above.

## Workspace layout

```text
cpca/        library: encode, decompose, ica, decode, evaluate, stats, events, hc11, hc5, pipeline, theory, topology
sim/         synthetic routes with a planted answer
baselines/   count PCA/ICA, seqNMF, HMM, PP-Seq (authors' Julia package), Bayesian decoding reference; count_assemblies, spotdis_baseline (third_party/spot), rastermap_baseline
scripts/     run_synthetic.py, run_hc11.py, summarize.py, plot_noise_bound.py; run_baselines/run_rerun/run_ppseq (comparisons); run_ladder.py, summarize_ladder.py; run_topology.py, summarize_topology.py; run_hc5.py, summarize_hc5.py
tests/       pytest (theory, library, pipeline, bound, hc5, topology)
docs/        method.md, plan.md, lessons_learned.md, baselines.md, benchmark_hc5.md, topology.md, rendering.md
results/     generated: synthetic/, hc11/, baselines/, rerun/, ladder/, hc5/, topology/, summary.md; _before_step0/ (results before the shuffle-null bound)     (pickles are not committed)
figures/     generated by the summarize scripts
data/, hc-11/, hc-5/  recordings (never committed)
```
