# Plan (version 2, October 2026)

Open with `Cmd+Shift+V` for formulas. This replaces the original plan; the history of what changed and why is in `lessons_learned.md`.

## Objective

A dimensionality-reduction method for hippocampal reactivation: a few components, each a set of cells with weights and relative firing times, found from sleep spikes alone, such that components correspond to routes (or route segments) run while awake. Advantages to preserve over sequence-detection methods: simplicity (one eigen-decomposition), promotability (the standard count-based assembly pipeline with time added), very few parameters, and no behavioural input. The waking map is the answer key, never an input.

## The pipeline

Population bursts in NREM → one arrow per cell per event, window = the event centred on its peak, $T$ = event duration (half-circle rule) → centre and scale per cell (sleep regime) → complex PCA → keep components above a shuffle null (equal to the Marchenko–Pastur bound for centred arrows) → read every kept component (weights, relative times) → label-free quality (split-half, span) → answer key once (member order vs place-field centre, per direction, with PRE sleep as control). Details in `method.md`; code in `cpca/pipeline.py::fit`.

## Status

Done:

- Library (`cpca/`), 49 tests, synthetic route generator (`sim/routes.py`), hc-11 loader and event detection, count PCA/ICA baseline.
- Synthetic benchmark (`results/synthetic/summary.md`, 10 draws per condition): all clean conditions recovered (median $|\rho|$ 0.97–1.00), including two routes in both directions at 15% replay and 15% participation; the realistic condition G recovered above chance in 80–90% of (route, draw) pairs at median $|\rho|$ 0.37–0.48. Difficulty ladder (condition H taken apart, `figures/synthetic_ladder.png`, `results/ladder/summary.md`; cPCA, seqNMF and PP-Seq on the same 1500 events, 3 draws per rung): with whole routes replayed (two maps on the same cells, both directions, assemblies, variable pacing) PP-Seq is essentially perfect (median $|\rho|$ 0.99–1.00) and centred cPCA recovers every map at 0.81–0.90; seqNMF drops once backward plays are present (0.26–0.47). Partial replays are hard for every method: on L3 and L6 (= H's settings) cPCA 0.59 and 0.48 (4/6, 5/6 maps), PP-Seq 0.35 and 0.39 (3/6 each, per-map values from 0.13 to 0.98), seqNMF at chance. PP-Seq's 0.89/0.81 on H in the matched rerun was one favourable draw. So the gap to PP-Seq is a constant precision gap on whole-route replays (present already with one map), not a failure specific to shared cells.
- hc-11 (`results/summary.md`, `figures/`): with centring, 20 of 120 (component, direction) tests match the place map in POST sleep (chance ≈ 6) vs 2 of 106 in PRE (chance ≈ 5); the effect is carried by the two sessions with 50+ place cells. Uncentred components are at chance in both (3/54, 1/44).
- Noise bound (October 21, `figures/noise_bound.png`): components are now kept against a per-cell shuffle null. It agrees with the textbook Marchenko–Pastur bound for centred, scaled arrows and corrects it for uncentred arrows, where the textbook bound kept 8–14 components per hc-11 session (16 on pure synthetic noise) against 2–6 by the shuffle. All cPCA results were recomputed; the earlier ones are in `results/_before_step0/`.
- Matched rerun after review (`docs/baselines.md`, "Corrections", `results/rerun/summary.md`): with identical 1,500-event subsets, a proper PP-Seq sweep and sampler, and a seqNMF K × L grid, PP-Seq matches or beats cPCA on planted routes (ahead in the realistic conditions G and H on that single draw; see the ladder below) while remaining at chance on hc-11 in both sleeps (2/47 POST, 2/44 PRE); seqNMF is at chance from 15% replay onward whatever K and L; cPCA keeps POST > PRE on the matched subsets (14/96 vs 6/74) and on all events (20/120 vs 2/106). The claim this supports is representation-level: cPCA's centred, event-normalised representation finds POST-specific map order on these recordings, which the other methods do not; it does not show that cPCA is a better sequence extractor in general, and PP-Seq is the more precise extractor of whole replayed routes.
- First method comparison with the authors' procedures (`docs/baselines.md`, `results/baselines/summary.md`, `figures/baselines_*.png`): seqNMF recovers planted routes as well as cPCA when replays are common (A–E) but not at 15% replay (F: 0.5 vs 0.97–0.99) and not in the realistic conditions; on hc-11 its one large factor per session is the synchronous burst (lag span 30–50 ms) and no factor orders place cells above chance. The Maboudi HMM detects planted replays (99–100%) with 29–40% false alarms; on hc-11 it calls 14–30% of events congruent in POST and 10–32% in PRE (not maze-specific), while its latent space, labelled with behaviour post hoc, decodes position far better from POST than PRE models in the two large sessions (0.21 vs 0.50 m; 0.10 vs 0.57 m), the same POST/PRE dissociation cPCA shows label-free. Bayesian decoding (map as input) marks 10–22% of POST and 8–22% of PRE events. PP-Seq (authors' Julia package, held-out hyperparameter selection as in the paper) recovers planted routes as well as cPCA when replays are common (A–E: 0.96–1.00) and one of two routes at 15% replay (F: 0.82 / 0.10); it models synchronous volleys as a zero-duration sequence type; on hc-11 its types are the population burst and small sub-bursts (spans 4–41 ms) and 1/30 map tests pass in POST, 1/25 in PRE. convSeq was not run (`docs/baselines.md`).

- Real-data benchmark with an answer key (`docs/benchmark_hc5.md`, `results/hc5/summary.md`, `figures/hc5_*.png`): hc-5 CA1 wheel sequences, six sessions, cross-fitted on held-out runs, cPCA vs seqNMF vs PP-Seq with each method's own settings procedure. Aligned windows: cPCA 32/36 fits recover the sequence, seqNMF 29/36, PP-Seq 14/36 (averaging the runs, ρ 0.71, is the ceiling). Unaligned windows: cPCA 35/36 (ρ 0.63), seqNMF 26/36, PP-Seq 13/36, averaging falls to ρ 0.47. Shuffled control at chance for all. cPCA about 50 ms per fit, seqNMF 50 s, PP-Seq 16 min. The upcoming left/right choice is not decodable from wheel activity in these sessions, so choice-specific sequences could not be tested.

Open (in order of value):

1. **Label-free identification of route components.** Candidates: a sequence score that rewards members spread over the half-circle *and* events whose score phase is consistent with the component's order; split-half reproducibility per component; a null specific to each component (its own members phase-randomised). Acceptance: on synthetic conditions F–H the route components rank in the top 2 by the score in ≥ 80% of draws.
2. **Direction inside a component.** Test whether the score phase $\arg a_{in}$ or the sign of the member-time slope separates forward and backward plays of the same route within a component; if so, report both directions per component.
3. **Power.** Apply to a recording with several hundred simultaneous CA1 cells (public Neuropixels datasets); the POST/PRE difference should strengthen with the number of place cells. This is the single most valuable step for a biological result.
4. **Biology (if 3 succeeds).** Internal order of components across the night (early vs late POST), and by cell class (fast-firing rigid vs slow-firing plastic cells, following Grosmark & Buzsáki 2016 on these same sessions).
5. **Benchmark hygiene.** Done: 10 draws per condition, a variable-pacing condition (ladder L5). Still open: report the component-level false-positive rate of the answer-key test on PRE as the empirical null.
6. **A second real benchmark with several sequences.** hc-5 tests one dominant sequence per event. A dataset with two or more distinct sequences in the same recording (for example the Mizuseki et al. 2013 rat with separate left and right delay sequences, or convSeq's own real data) would test whether cPCA separates them onto different components on real spikes; convSeq should join that comparison through a thin adapter around the authors' model and loss.

## Decisions taken (with the evidence)

- Uncentred vs centred: both run; centred for sleep (minority replays), uncentred when a pattern is common to most events. Evidence: synthetic A–C vs G, hc-11 POST/PRE table.
- $T$: event duration, aligned on the peak. Per-event spike spread tested; no gain on hc-11 (centred: 22/112 vs 4/96 against 20/120 vs 2/106), kept as an option.
- Noise bound: shuffle null (permutation parallel analysis) for both conventions; the textbook Marchenko–Pastur bound is reported alongside. Evidence: `figures/noise_bound.png`.
- ICA: off by default; helps only when two routes share cells (synthetic E-type union), hurts with many components.
- Mixture/clustering of events, direction-split tests, and template-based detectors: removed from the core; they were answers to the wrong question (see `lessons_learned.md`).

## Publication framing

Methods paper: "the count-based assembly pipeline with time added": the half-circle rule, the uncentred/centred choice, components as assemblies with internal order, validation on planted routes and on hc-11 with PRE as control, comparison with count PCA/ICA (which finds the groups but not the order). Natural venues: *PLoS Computational Biology*, *eLife* Tools and Resources, *Journal of Neuroscience Methods*, *Neural Computation*. A biological paper requires item 3 above.

## Workspace layout

```text
cpca/        library: encode, decompose, ica, decode, evaluate, stats, events, hc11, hc5, pipeline, theory
sim/         synthetic routes with a planted answer
baselines/   count PCA/ICA, seqNMF, HMM, PP-Seq (authors' Julia package), Bayesian decoding reference
scripts/     run_synthetic.py, run_hc11.py, summarize.py, plot_noise_bound.py; run_baselines/run_rerun/run_ppseq (comparisons); run_ladder.py, summarize_ladder.py; run_hc5.py, summarize_hc5.py
tests/       pytest (theory, library, pipeline, bound, hc5)
docs/        method.md, plan.md, lessons_learned.md, baselines.md, benchmark_hc5.md, rendering.md
results/     generated: synthetic/, hc11/, baselines/, rerun/, ladder/, hc5/, summary.md; _before_step0/ (results before the shuffle-null bound)     (pickles are not committed)
figures/     generated by the summarize scripts
data/, hc-11/, hc-5/  recordings (never committed)
```
