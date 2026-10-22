# hc-5 wheel benchmark: complex PCA, seqNMF, PP-Seq on real sequences

Open with `Cmd+Shift+V` for formulas. Code: `cpca/hc5.py` (data), `scripts/run_hc5.py` (fits, parallel), `scripts/summarize_hc5.py` (tables, figures). Results: `results/hc5/summary.md`; figures: `figures/hc5_sequences.png`, `figures/hc5_benchmark.png`, `figures/hc5_sensitivity.png`.

## Why this dataset

The comparison on hc-11 asked whether an order is specific to the maze just run (POST against PRE sleep). It is a weak test of which method extracts sequences better: nobody knows which sleep events are replays, and seqNMF often produced no testable factor. The question here is the direct one: **given the spikes of a few repetitions of a known sequence, without being told which cell fires when, does a method return the sequence?**

In hc-5 (Pastalkova et al., 2008; one rat, CA1 and entorhinal silicon-probe recordings) the rat alternates between the two arms of a figure-eight maze and, between laps, runs in a wheel for 13–20 s. CA1 cells fire in a reliable sequence during the wheel run, each at its own time since wheel onset (its *time field*). That sequence is the answer key. It is the same kind of data the seqNMF paper used for its hippocampal example (Mackevicius et al., 2019, rat 1), so the comparison is on ground the other methods were built for.

## Data

Six left/right alternation sessions (`i01_maze05.005`, `06.002`, `06.005`, `08.001`, `08.004`, `15.002`), read from `*_BehavElectrData.mat`:

- **Wheel runs**: the first segment of each run (`Laps.WhlTimeStartCW`/`WhlTimeEndCW`). All runs of a session are cut to one window length $W$ from onset, the 15th percentile of first-segment durations (segments under 10 s excluded); shorter runs are dropped. This gives one clock for every run of a session: $W$ = 13–20 s, 7–25 runs per session.
- **Cells**: principal cells (`Clu.isIntern = 0`) on shanks 5 and above (CA1 in the hc-5 channel map; shanks 1–4 are entorhinal cortex), with at least 10 spikes over the windows and spikes in at least 3 runs: 42–72 cells per session.
- **Arms**: laps through maze sections 3-5-7-9 are arm A, 4-6-8-10 arm B. The `TrialType` codes, which the data description does not define, agree with this in every session (1 and 4 are arm B, 2 and 3 arm A; 3 and 4 are alternation errors).

## What the data do not support

The plan was to make the headline whether a method separates the sequence before a left choice from the sequence before a right choice (Pastalkova et al. report choice-predictive wheel sequences). Checked first, with nothing fitted: on these six sessions neither single cells nor the population carry the upcoming choice during the wheel run. A map-difference test finds 0–3 choice-selective cells per session (about the 5% expected by chance), and a cross-validated logistic decoder of the upcoming (or the previous) arm from the population's wheel activity is at or below its shuffle null in all six sessions. With 4–13 runs per choice the test has little power, but a benchmark cannot be built on a difference that a supervised decoder cannot find. The left/right question was therefore dropped; the benchmark is about recovering the wheel sequence.

## Conditions

- **Aligned.** Every window starts at wheel onset, length $W$. The runs share a clock.
- **Unaligned.** Every window starts at its own random time in $[0, W/3)$ *before* onset and lasts $W + W/3$, so it always contains the whole run (plus the end of the lap before it), but the onset sits at a different place in every window. The runs no longer share a clock, which is the situation unsupervised sequence methods exist for: they should find the sequence without being told when it starts. Offsets are fixed per session and split and identical for every method.

  A first version started the windows *after* onset (offsets in $[0, W/3)$, length $2W/3$). It mixed two things: no shared clock, and the start of the sequence cut away (many cells fire in the first 2 s). Checked directly, complex PCA scored the same with one common late start for every run as with jittered starts (median $\rho$ 0.24 against 0.27), so its loss was the cut, not the misalignment; align-and-average lost to both (0.65 with a common start of the same length from onset, 0.48 with a common late start, 0.39 jittered). That version was discarded before any figure was made, and the windows now always contain the whole run.
- **Shuffled control.** Aligned windows; each cell's spikes in each training run circularly shifted by an independent random amount within the window. This destroys the order and keeps every cell's rate; every method should be at chance.

The aligned condition has a known trap: averaging the runs and reading each cell's peak already recovers the order, with no method at all. That number is reported as the **align-and-average reference**, the ceiling when windows are aligned. In the unaligned condition the same average is smeared, and the reference shows what knowing the onset is worth.

## Cross-fitting

Each method is fit on half of a session's runs and scored against time fields computed from the other half; the halves then swap. Three random splits per session (two for the shuffled control), so 6 fits per session and condition. The answer key never comes from runs a method was fitted on, so a method cannot score by fitting the noise that the key also contains.

**Time field** (the answer key): trial-averaged rate against time since onset on the held-out runs (0.25 s bins, Gaussian smoothing 0.5 s); the field is the peak time; a cell has a field if the peak is at least 2 Hz and it fires in at least a third of those runs.

## Methods and their settings, fixed before any score was seen

- **Complex PCA**, uncentred (headline) and centred. The wheel sequence occurs in every run, which is the regime where the method's own rule (`docs/plan.md`, decisions) is to leave the columns uncentred; centred is reported as the alternative. $T$ = window length (the half-circle rule), components above the Marchenko–Pastur bound, order = read-out times of members (weight at least 20% of the largest).
- **seqNMF** (the FeeLab Python port): λ by the authors' crossover rule ($2\lambda_0$), 200 ms bins, best of three restarts, each neuron's row scaled to unit standard deviation (as in the hc-11 comparison). Headline $K = 5$, $L$ = one window (the paper's guidance: $K$ above the expected number of sequences, $L$ about the length of one sequence); sensitivity grid $K \in \{3, 5, 10\}$, $L \in \{W/2, W\}$. Order = lag of each member's peak loading; members at least 20% of the factor's largest peak loading.
- **PP-Seq** (the authors' Julia package): the runs laid end to end with 3 s silent gaps, time unit 1 s (the paper's hippocampal analysis is in seconds). Hyperparameters by the paper's procedure: 20 random candidates (number of sequence types $R \in \{2, \dots, 8\}$, amplitudes, rates, neuron width, time warping up to 1.5) scored by the package's held-out log-likelihood on one shared speckled mask (7.5%), best candidate refit with the paper's sampler schedule (20 anneal stages × 100 sweeps from temperature 500, then 100 sweeps with 1,000 split-merge moves). Order = neuron offsets of members; headline membership is the per-type rule (response proportion at least 20% of the type's largest), the package's `sortperm_neurons` rule is reported alongside. A type that explains no spikes and has no latent events in the posterior is not counted as an output.

Every method sees the same windows, the same cells and the same spikes; none sees trial type or time fields; settings are chosen by each method's own procedure on its training half only.

## Score

For each fit, every output (component, factor, sequence type) with at least 6 members that have a time field is scored by the **signed** Spearman correlation between member order and time field; a sequence read backwards scores negative. Recovery is the best output's $\rho$. Its p-value comes from a null that permutes the time fields among all cells that have one, scores *every* output under the same permutation and keeps the best (`cpca.evaluate.best_order`), so a method gains nothing by producing many outputs. Also reported: coverage (the share of time-field cells that are members of the best output), precision (the share of outputs that pass on their own), the number of outputs, and run time per fit on one core, settings selection included.

A fit *recovers the sequence* if $p < 0.05$. Counts over fits carry 95% Clopper–Pearson intervals. Fits within a session share runs, so the session (median over its 6 fits) is the unit in the figures.

## Results

All 336 fits completed (6 sessions × 14 fits × 4 methods; tables in `results/hc5/summary.md`). The complex PCA fits were redone on October 21 with the shuffle-null noise bound (`docs/method.md` Section 4); before, with the textbook bound, they were 33/36 aligned, 36/36 unaligned and 1/12 shuffled. Fits that recover the sequence ($p < 0.05$) and median held-out $\rho$:

| | align-and-average (reference) | complex PCA | seqNMF | PP-Seq |
|---|---|---|---|---|
| aligned | 36/36, ρ 0.71 | **32/36 (89%)**, ρ 0.58 | 29/36 (81%), ρ 0.60 | 14/36 (39%), ρ 0.55 |
| unaligned | 32/36, ρ 0.47 | **35/36 (97%)**, ρ 0.63 | 26/36 (72%), ρ 0.52 | 13/36 (36%), ρ 0.43 |
| shuffled control | 0/12 | 0/12 | 0/12 | 0/12 |
| cells with a field in the best output | — | 40 of 40 | 30 of 40 | 13 of 40 |
| time per fit (one core) | — | 50 ms (noise bound included) | 50 s | 16 min |

What this says:

- **When the runs share a clock, nothing beats averaging them** (ρ 0.71, every fit). That is expected and is why the aligned condition alone cannot rank the methods.
- **Without a shared clock, complex PCA is the only method that keeps its performance.** Averaging drops from 0.71 to 0.47, seqNMF from 0.60 to 0.52, PP-Seq from 0.55 to 0.43; complex PCA goes from 0.58 to 0.63 and recovers the sequence in 35 of 36 fits. This is the property the method was built on: the lag matrix $C$ is unchanged by a common clock shift within an event (`docs/method.md`, Section 3), so its first component does not need the onset.
- **Reliability, not peak correlation, separates the methods.** When seqNMF recovers the sequence its ρ is as high as complex PCA's; it fails more often (81% and 72% of fits). PP-Seq splits the run into several short sequence types, each covering about a third of the cells; its best type has a ρ comparable to the others but few members, and with several types per fit its own chance level (best of its outputs) is high, so it clears it in about 40% of fits. Neither PP-Seq membership rule changes this (package rule: 15/36 and 14/36).
- **Every method is at chance on the shuffled control**, so none of the scores comes from firing rates alone. PP-Seq's median ρ on the shuffled data is 0.34 while 0 of 12 fits pass: that is the max-over-outputs null doing its job.
- **Settings.** seqNMF's headline cell ($K = 5$, $L$ = one window) is its best in the aligned condition and tied best in the unaligned one; a factor half a window long fails without alignment (9–13 of 36). Centred complex PCA is at chance when aligned (3/36), as the method's own rule predicts for a pattern present in every event; it partly recovers the sequence when unaligned (16/36), where centring no longer removes the whole sequence.
- **Cost.** Complex PCA fits in about 50 ms including its shuffle-null noise bound, three orders of magnitude faster than seqNMF and four faster than PP-Seq, with no settings to choose.

What it does not say: this is the regime that suits uncentred complex PCA (one dominant sequence in every event). On sleep-like planted data, where replays are a minority of events, PP-Seq is the more precise extractor when whole routes are replayed, and no method is reliable when only parts of routes are (`docs/baselines.md`, difficulty ladder). And the score rewards one output that spans the whole sequence; a method that tiles a sequence into pieces, as PP-Seq does here, is penalised for it (coverage 13 of 40 cells), which is a property of the score as much as of the method.

## Implementation notes

- **A bug in the PP-Seq wrapper**, found here and fixed (`baselines/ppseq/run_ppseq.jl`): masks were drawn over `max_time + 1e-6`, while the package builds the complementary masks over `max_time` and asserts that every mask fits. When the timeline length is a whole number of mask lengths (here: whole-second runs and gaps), the last mask overran and every masked candidate crashed, so the sweep silently fell back on its first candidate. Mask ends are now clipped to `max_time`. The hc-11 timelines were not whole multiples, no masks overran there (the runs would have crashed), and those results are unaffected.
- The runner refuses a PP-Seq fit in which more than half of the sweep candidates failed.
- **Empty PP-Seq refits.** In 16 of 84 fits the candidate with the best held-out score, refit with the long schedule, ended with no latent events (every spike assigned to the background). The package anneals by widening the prior on event amplitude, which makes new events unlikely at high temperature; when the chosen mean amplitude is far above the spikes in one run, the events die during annealing and are not reborn. The model then outputs nothing. Rather than score that as a miss, the next-best candidate is refit (up to two more), a deviation in PP-Seq's favour; `refit_attempts` records how many refits a fit needed. After this, 2 of 84 fits were still empty and are scored as no recovery.
- **PP-Seq is not reproducible run to run here.** Rerunning those 16 fits with identical seeds changed the selected candidate in 9 of them (for example a mean event amplitude of 960 instead of 2,696), so the package's sampler is not deterministic given the seed on this machine. Complex PCA and seqNMF are.
- Parallelism: one process per fit (`concurrent.futures`, 9 workers on 10 cores), every PP-Seq and seqNMF fit single-threaded. Random seeds are derived from the session, condition, split, fold and method names with CRC32, not Python's `hash()`.

## Limits

- One rat, six sessions, 7–25 runs per session: the unit of replication is small. The figures show sessions, not fits, for that reason.
- The wheel sequence is a single, dominant pattern that recurs in every run. It is the regime where uncentred complex PCA is expected to work, and it says nothing about minority patterns such as sleep replay.
- Time fields early in the run are common (many cells peak in the first 2 s), so part of every method's order is a coarse early-versus-late split.
