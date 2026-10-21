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

The September comparison was criticised, correctly, for using the other methods inappropriately. It was redone with the authors' procedures (`docs/baselines.md`): seqNMF with the λ-crossover rule, penalty-driven factor count, held-out skewness significance and peak-lag read-out; the HMM with Maboudi's settings, congruence against transition shuffles and latent-state place fields; Bayesian decoding kept as the map-using reference; convSeq excluded rather than re-implemented. PP-Seq was first excluded for lack of Julia; that hurdle was small (juliaup installs in a minute, the authors' package loads under Julia 1.13 unchanged) and the method was added through its own package rather than re-implemented. Practical notes: its masked-validation sampler is single-threaded, run time grows faster than linearly with events, and two jobs on one laptop contend badly, so the sweep runs on 600 events with a short schedule and the final fit on 1,500 events. Two of my own implementation mistakes surfaced and were fixed on the way: hmmlearn's element-wise Poisson likelihood made a fit take minutes (replaced by the identical matrix product), and empty flank bins made every event trivially "congruent" (removed, as the test concerns transitions between active states). Baselines validated on the planted routes before touching real data; conclusions in `docs/plan.md`.

## The matched rerun (October 19)

A reviewer's list of six comparison flaws (different windows and event sets per method, a PP-Seq sweep that never varied R and validated each candidate on a different mask, a mislabelled statistic, a membership rule wrongly attributed to the package, a one-bin shift in the seqNMF cost helper, a fixed K and L) was confirmed point by point against the sources and fixed. Two things I got wrong on the way: I first read the package's held-out statistic (per unit of masked neuron-time) as an uninformative total of 0.4 nats, when it is 5,700 nats in total, and I attributed PP-Seq's failures to model selection when the sampler schedule was the limiting factor. With the corrections PP-Seq became the strongest extractor on realistic planted data, and the conclusion about hc-11 had to be restated at the representation level: cPCA's centred, event-normalised representation finds POST-specific order there; the other methods, run properly, do not. Practical: PP-Seq costs 15 minutes to 1.5 hours per dataset on this laptop even after capping the sweep; the sweep candidates run as parallel single-threaded processes because the package's masked sampler is not multi-threaded.

## The hc-5 wheel benchmark (October 19)

The comparison was moved to real data with a known sequence (`docs/benchmark_hc5.md`): CA1 wheel-running sequences in hc-5, scored on held-out runs. Three things went wrong or nearly did, and each was caught before a figure was made:

- **The left/right test was planned before the data were checked.** The intended headline (does a method separate the sequence before a left choice from the one before a right choice) rests on a difference that a supervised decoder cannot find in these six sessions. Checking that first saved building a benchmark on nothing.
- **The first "unaligned" condition measured the wrong thing.** Windows that started after onset also cut off the start of the sequence; complex PCA scored the same with one common late start as with jittered starts, so its loss was the cut, not the misalignment. A direct control (same window length, no jitter) exposed it; windows now start before onset and contain the whole run.
- **The PP-Seq wrapper had a latent bug** (masks overrunning the end of the timeline when it is a whole number of mask lengths), which made every sweep candidate crash and the sweep silently fall back on its first candidate. Found because the first test fit returned types with no spikes; the runner now refuses a fit whose sweep mostly failed. A second PP-Seq behaviour, refits that end with no latent events, is handled by refitting the next-best candidate, in PP-Seq's favour.

Practical: on 10 cores, 9 single-threaded worker processes ran the 336 fits in about 3 hours; nearly all of it was PP-Seq (16 minutes per fit with the paper's schedule, against 50 s for a seqNMF fit and 3 ms for complex PCA).

## What held up

- The library mathematics (shift invariance, exact phases under partial coverage, the delay-line reading) and the test suite.
- $C^{0}$ and $\Delta_i$ as a closed-form co-activation null.
- The realistic synthetic generator as the development bed.
- Using PRE sleep as the control for the answer-key test: POST 21/118 vs PRE 4/104 is the first real-data result of the project.
