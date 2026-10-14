# Lessons learned (September–October 2026)

A record of what was tried before the current pipeline, what the numbers were, and what each step taught. Kept so the same detours are not taken twice. The code for the removed analyses is in the git history (commit "Snapshot before workspace revamp").

## What was tried, in order

1. **Uncentred complex PCA, 100 ms window on the population peak, first component read.** On hc-11 POST+PRE pooled, PC1 explained 15–27% of the trace, its weights matched participation ($\rho = 0.99$), its angles spanned 19–38°, and its order matched each cell's mean firing time relative to the peak ($\rho = 0.97$) and the cell's firing rate ($\rho \approx -0.3$), not the track. The off-diagonal of $C$ was 99% real.
2. **Co-activation null $C^{0}$, template from $C - C^{0}$, events selected by timing excess.** Correct mathematics (kept in `cpca/stats.py`), modest gain: whole-population order vs place map stayed near chance.
3. **Direction split by the sign of $\mathrm{Im}(a_1\overline{a_2})$.** Returned a 50/50 split with mirror-image orders on real data *and* on phase-shuffled data; an artefact without a null. Removed.
4. **Comparison against Bayesian decoding, rank-order, seqNMF, HMM, count PCA/ICA**, scored as whole-population order vs the place map. No label-free method beat chance on any session; Bayesian decoding (which takes the map as input) marked 9–22% of events as replay; forward/backward replays were split roughly 50/50. Fed only Bayesian-selected forward events, complex PCA recovered the order in 6/8 sessions.
5. **Mixture of templates (phase-invariant k-means), membership-blind variant.** Separated routes perfectly on clean synthetic data (where a single template could not), at chance on real sessions; a realistic synthetic (segments, two maps, assemblies, non-place cells) showed every method at chance at 15% and even 40% replay. seqNMF was the better extractor for partial-segment replays.
6. **Reorientation after discussion with the project mentor**: components should be routes; evaluation per component on its members; no waking input. Led to the two facts below and the current pipeline.

## The two facts that changed the design

- **Half-circle rule.** Routes separate on different components only when a route fills the half-circle. With 100 ms windows and activity within $\pm 12$ ms of the peak (a 40° arc) forward and backward of the same route are 92% parallel and merge; at 180° they are 3% parallel and separate. Using the event's own duration as $T$ fixes this with no parameter. The "forward/backward cancellation" treated as a fundamental hurdle for weeks was mostly this artefact.
- **Read all components.** 5–16 components exceed the noise bound in every session; only the first was ever examined. The route-like components are in the rest of the spectrum.

## Mistakes, named

- The evaluation target (whole-population order of PC1 vs the map) drove development toward a sequence detector; each fix added parameters and moved away from PCA. The question had to be re-read, not the answer re-optimised.
- "Never centre" was stated as a law; it is true for events with arbitrary clocks and false for events aligned on the population peak. The synthetic benchmark later showed that neither convention is right in all regimes: uncentred when the pattern is the majority, centred when it is a minority.
- $T$ was set to the window length, not to the pattern's extent.
- The first synthetic check of forward/backward mixing used a 94° arc and the conclusion was generalised.
- Real data were analysed before a synthetic benchmark with several routes existed; the realistic benchmark that explained the failures came last.
- A per-condition random seed used Python's `hash()`, which changes between runs, so one realistic result moved from 0.72/0.80 to 0.42/0.34 between two executions; seeds are now fixed and conditions are repeated.
- Timing-consistency $z$ was expected to identify route components label-free; it does not (route components rank 5–8 of 9 in the realistic condition). It is kept as a diagnostic, not a selector.

## The baseline comparison, redone (October)

The September comparison was criticised, correctly, for using the other methods inappropriately. It was redone with the authors' procedures (`docs/baselines.md`): seqNMF with the λ-crossover rule, penalty-driven factor count, held-out skewness significance and peak-lag read-out; the HMM with Maboudi's settings, congruence against transition shuffles and latent-state place fields; Bayesian decoding kept as the map-using reference; PP-Seq and convSeq excluded rather than re-implemented. Two of my own implementation mistakes surfaced and were fixed on the way: hmmlearn's element-wise Poisson likelihood made a fit take minutes (replaced by the identical matrix product), and empty flank bins made every event trivially "congruent" (removed, as the test concerns transitions between active states). Baselines validated on the planted routes before touching real data; conclusions in `docs/plan.md`.

## What held up

- The library mathematics (shift invariance, exact phases under partial coverage, the delay-line reading) and the test suite.
- $C^{0}$ and $\Delta_i$ as a closed-form co-activation null.
- The realistic synthetic generator as the development bed.
- Using PRE sleep as the control for the answer-key test: POST 21/118 vs PRE 4/104 is the first real-data result of the project.
