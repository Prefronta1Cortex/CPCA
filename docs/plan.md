# Plan (version 2, October 2026)

Open with `Cmd+Shift+V` for formulas. This replaces the original plan; the history of what changed and why is in `lessons_learned.md`.

## Objective

A dimensionality-reduction method for hippocampal reactivation: a few components, each a set of cells with weights and relative firing times, found from sleep spikes alone, such that components correspond to routes (or route segments) run while awake. Advantages to preserve over sequence-detection methods: simplicity (one eigen-decomposition), promotability (the standard count-based assembly pipeline with time added), very few parameters, and no behavioural input. The waking map is the answer key, never an input.

## The pipeline

Population bursts in NREM → one arrow per cell per event, window = the event centred on its peak, $T$ = event duration (half-circle rule) → centre and scale per cell (sleep regime) → complex PCA → keep components above the Marchenko–Pastur bound → read every kept component (weights, relative times) → label-free quality (split-half, span) → answer key once (member order vs place-field centre, per direction, with PRE sleep as control). Details in `method.md`; code in `cpca/pipeline.py::fit`.

## Status

Done:

- Library (`cpca/`), 40 tests, synthetic route generator (`sim/routes.py`), hc-11 loader and event detection, count PCA/ICA baseline.
- Synthetic benchmark (`results/synthetic/summary.md`): all clean conditions recovered (median $|\rho|$ 0.97–1.00), including two routes in both directions at 15% replay and 15% participation; the realistic condition recovered above chance in 83% of draws at median $|\rho| \approx 0.5$.
- hc-11 (`results/summary.md`, `figures/`): with centring, 21 of 118 (component, direction) tests match the place map in POST sleep (chance ≈ 6) vs 4 of 104 in PRE (chance ≈ 5); the effect is carried by the two sessions with 50+ place cells.
- Method comparison with the authors' procedures (`docs/baselines.md`, `results/baselines/summary.md`, `figures/baselines_*.png`): seqNMF recovers planted routes as well as cPCA when replays are common (A–E) but not at 15% replay (F: 0.5 vs 0.97–0.99) and not in the realistic conditions; on hc-11 its one large factor per session is the synchronous burst (lag span 30–50 ms) and no factor orders place cells above chance. The Maboudi HMM detects planted replays (99–100%) with 29–40% false alarms; on hc-11 it calls 14–30% of events congruent in POST and 10–32% in PRE (not maze-specific), while its latent space, labelled with behaviour post hoc, decodes position far better from POST than PRE models in the two large sessions (0.21 vs 0.50 m; 0.10 vs 0.57 m), the same POST/PRE dissociation cPCA shows label-free. Bayesian decoding (map as input) marks 10–22% of POST and 8–22% of PRE events. PP-Seq and convSeq were not run (reasons in `docs/baselines.md`).

Open (in order of value):

1. **Label-free identification of route components.** Candidates: a sequence score that rewards members spread over the half-circle *and* events whose score phase is consistent with the component's order; split-half reproducibility per component; a null specific to each component (its own members phase-randomised). Acceptance: on synthetic conditions F–H the route components rank in the top 2 by the score in ≥ 80% of draws.
2. **Direction inside a component.** Test whether the score phase $\arg a_{in}$ or the sign of the member-time slope separates forward and backward plays of the same route within a component; if so, report both directions per component.
3. **Power.** Apply to a recording with several hundred simultaneous CA1 cells (public Neuropixels datasets); the POST/PRE difference should strengthen with the number of place cells. This is the single most valuable step for a biological result.
4. **Biology (if 3 succeeds).** Internal order of components across the night (early vs late POST), and by cell class (fast-firing rigid vs slow-firing plastic cells, following Grosmark & Buzsáki 2016 on these same sessions).
5. **Benchmark hygiene.** More draws per condition; a condition with speed variability; report the component-level false-positive rate of the answer-key test on PRE as the empirical null.

## Decisions taken (with the evidence)

- Uncentred vs centred: both run; centred for sleep (minority replays), uncentred when a pattern is common to most events. Evidence: synthetic A–C vs G, hc-11 POST/PRE table.
- $T$: event duration, aligned on the peak. Per-event spike spread tested; no gain on hc-11 (12/160 vs 7/162), kept as an option.
- ICA: off by default; helps only when two routes share cells (synthetic E-type union), hurts with many components.
- Mixture/clustering of events, direction-split tests, and template-based detectors: removed from the core; they were answers to the wrong question (see `lessons_learned.md`).

## Publication framing

Methods paper: "the count-based assembly pipeline with time added": the half-circle rule, the uncentred/centred choice, components as assemblies with internal order, validation on planted routes and on hc-11 with PRE as control, comparison with count PCA/ICA (which finds the groups but not the order). Natural venues: *PLoS Computational Biology*, *eLife* Tools and Resources, *Journal of Neuroscience Methods*, *Neural Computation*. A biological paper requires item 3 above.

## Workspace layout

```text
cpca/        library: encode, decompose, ica, decode, evaluate, stats, events, hc11, pipeline, theory
sim/         synthetic routes with a planted answer
baselines/   count PCA/ICA (Lopes-dos-Santos et al. 2013)
scripts/     run_synthetic.py, run_hc11.py, summarize.py
tests/       pytest (theory, library, pipeline)
docs/        method.md, plan.md, lessons_learned.md, rendering.md
results/     generated: synthetic/, hc11/, summary.md     (pickles are not committed)
figures/     generated by scripts/summarize.py
data/, hc-11/  recordings (never committed)
```
