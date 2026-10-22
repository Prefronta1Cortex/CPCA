# Baseline methods: what was run, how, and what was not

Open with `Cmd+Shift+V` for formulas. Results: `results/baselines/summary.md`; figures: `figures/comparison_synthetic.png` (planted routes), `figures/comparison_hc11.png` (each method relative to its own chance level, POST vs PRE, per-session points, maze specificity), `figures/comparison_example.png` (one session, each method's output). Code: `baselines/`, `scripts/run_baselines.py`, `scripts/run_ppseq.py`, `scripts/summarize_baselines.py`.

## Rules

1. Same events and cells for every method (POST and PRE NREM population bursts, pyramidal cells). The place map is used only to score.
2. Each method is scored on **what it outputs**, not on a quantity it was never designed to produce:
   - complex PCA components and seqNMF factors give a cell order → Spearman between member order and the answer key, per component/factor and per running direction, against the 95th percentile of a cell-permutation null (identical test for both: `cpca.evaluate.order_vs_place`);
   - the HMM gives per-event congruence and a latent space → fraction of congruent events (and, on synthetic data, the AUC of congruence against the planted replay labels), plus the error of decoding running position through latent-state place fields vs. shuffled positions;
   - Bayesian decoding takes the map as input → fraction of events with a significant decoded trajectory; it is the reference ceiling, not a label-free competitor.
3. Every method was first run on the planted-route benchmark (`sim/routes.py`), so a failure on real data is attributable to the data.
4. PRE sleep (before the animal ever saw the maze) is the control for every map-based score.

## seqNMF (Mackevicius et al., 2019)

Implementation: the `seqnmf` Python port of the FeeLab toolbox (same convolutive NMF with the cross-orthogonality penalty). Procedure as in the paper: the penalty weight $\lambda$ is chosen by the crossover rule (fit over a range of $\lambda$, normalise reconstruction and cross-orthogonality costs, find the crossing $\lambda_0$, use $2\lambda_0$, the paper's value for its hippocampal example); $K = 5$ factors of length $L = 120$ ms (10 ms bins), the penalty removes unneeded factors; best of three restarts by reconstruction cost; significance by the toolbox's held-out skewness test (overlap of $W$ with 25% held-out events vs. circularly shifted held-out data, Bonferroni over $K$). A factor's cell order is the lag of each cell's peak loading in $W$, the quantity the toolbox sorts neurons by when it plots a factor; members are cells with peak loading at least 20% of the factor's maximum. Stated deviation: each neuron's row is divided by its standard deviation before fitting, otherwise a few high-rate cells own every factor. Events are capped at 1,500 training events per fit for run time.

## Hidden Markov model (Maboudi et al., 2018)

Implementation: `hmmlearn`'s Poisson HMM with the emission log-likelihood computed as one matrix product (identical to 10⁻¹⁴; hmmlearn's element-wise `poisson.logpmf` made a session take minutes). Settings as in the paper: 20 ms bins, Poisson emissions, 30 states, EM; cross-validated held-out likelihood against two surrogates (temporal shuffle per neuron, coherent time swap); **congruence**: an event's sequence likelihood under the fitted transition matrix vs. matrices whose off-diagonal entries are shuffled row-wise, congruent above 99% of 100 shuffles (95% also stored); **latent-state place fields**: running activity (100 ms bins, speed above 10 cm/s) decoded to state posteriors, $P(\text{state}\mid\text{position})$ estimated on half of the running data, position decoded on the other half, median error vs. shuffled positions. Stated deviations: empty bins inside an event are dropped (otherwise every event contains a quiet→active→quiet transition that the shuffle destroys and congruence is trivially high); states never visited get a uniform transition row; cross-validation uses 3 folds and 40 EM iterations for run time. The HMM does not output a cell order without behaviour, so it has no entry in the order-recovery table.

## Bayesian decoding (Zhang et al., 1998; Davidson et al., 2009; Grosmark & Buzsáki, 2016)

Direction-agnostic place maps (40 bins, occupancy ≥ 0.2 s, place cell = peak ≥ 3 Hz, ≥ 40 spikes), 20 ms bins over the full event, flat prior, weighted correlation between decoded position and time, column-cycle shuffle (500), events with at least five active place cells; significant at $p < 0.05$, and under the strict criterion $|R| > 0.5$ with at least five bins. Uses the map as input.

## PP-Seq (Williams, Degleris, Wang & Linderman, 2020)

Implementation: the authors' Julia package `PPSeq.jl` (installed into `baselines/ppseq/` as a Julia project; `baselines/ppseq/run_ppseq.jl` is a thin command-line wrapper, `baselines/ppseq_baseline.py` prepares the data and reads the output). Julia 1.13 via juliaup; the package's `DistributedSeqModel` runs the sampler on 4 threads.

Data: PP-Seq models one continuous recording, so the population-burst events are laid end to end with a 100 ms silent gap, in time units of 50 ms (one unit ≈ one replay-compressed traversal step, which brings the authors' hippocampal prior ranges, written for running-speed sequences, to a comparable scale). Spike times keep their position inside each event.

Hyperparameters, following Supplement F.2 of the paper: fixed $\gamma = 3$, $\phi = 1$, $\nu = 4$, offset prior mean 0; a random sweep over the number of sequence types $R$, the mean sequence amplitude (variance = mean), the background amplitude (variance = mean), the sequence rate, the neuron response width and time warping, each setting scored by the package's held-out statistic on one shared speckled 7.5% mask (`easy_sample_masked!`), the best setting refit on all spikes (`easy_sample!`). The current settings and the stated reductions relative to the paper are listed under "Corrections made in the matched rerun" below; the first version of this comparison (4 settings, $R \le 4$, no warping, separate masks per candidate) is described there as well.

Scoring, on PP-Seq's own outputs: each sequence type $r$ gives a cell order, the neuron offsets $b_{nr}$, with members the neurons whose response proportion $a_{nr}$ is at least 20% of the type's largest (the threshold the package's own `sortperm_neurons` uses) → `order_vs_place`, identical to cPCA and seqNMF. Each saved post-anneal sample gives latent events; an event window contains a sequence in the fraction of samples with a latent event inside it. PP-Seq represents a synchronous volley as a sequence type whose offsets coincide (span of a few ms), and on planted data it assigns such a type to every volley; per-event detection is therefore reported for *sequence types*, defined label-free as types whose member offsets span at least 20 ms, as well as for any type.

## Count PCA/ICA (Lopes-dos-Santos et al., 2013)

`baselines/count_pca_ica.py`: spike counts per event, z-scored, PCA, components above the Marchenko–Pastur bound, FastICA. Produces assemblies without order; reported as the number of assemblies and whether members' fields are compact (see `results/summary.md`).

## Not run, and why

- **convSeq** (Koshkin et al., 2024): the repository is research code (GPU device by default, YAML configuration, a trainer coupled to ground-truth plotting, 4,000-epoch runs, no packaged API); running it on these data would require rewriting its training loop.

(PP-Seq was initially excluded for lack of Julia; Julia was installed and the authors' package is now part of the comparison, see above.)

## Corrections made in the matched rerun (October 19)

After review, six flaws in the first comparison were fixed and the order-producing methods were rerun (`scripts/run_rerun.py`, results in `results/rerun/`, summary in `results/rerun/summary.md`):

1. **Same spikes for every method.** One fixed random subset of at most 1,500 events per dataset, each event a window of its own duration centred on the population peak (the cPCA window), used by cPCA, seqNMF and PP-Seq alike; cPCA is additionally reported on all events as before. Previously cPCA used all events with peak-centred windows while seqNMF and PP-Seq each drew their own 1,500 events with threshold-crossing windows (median interval overlap with the cPCA windows 86%).
2. **PP-Seq validation.** Every candidate setting is now scored on the *same* speckled 7.5% mask (mask seed separate from the sampler seed); previously each candidate had its own mask, so the held-out scores were not comparable. The package's statistic is reported as the package defines it, the baseline-subtracted held-out log-likelihood per unit of masked neuron-time; the earlier "per spike" division was a mislabel. The statistic is informative (a difference of 0.01 is 100–150 nats in total).
3. **PP-Seq sweep.** 20 candidates (was 4) with $R \in \{2,\dots,8\}$ drawn independently of the other hyperparameters (previously the four fixed draws tied $R$ to the other settings, so $R$ was never actually tested), time warping as a swept option (3 warp values, $w_F \in (1.2, 1.5)$, the paper's range), sequence rate capped at the paper's upper bound $10^{-1}$. Candidates run as three parallel single-threaded processes on 600 events with a short schedule.
4. **PP-Seq sampler.** Final fit with 12 annealing stages × 40 sweeps from temperature 200, then 60 sweeps with 20 split-merge moves (was 8 × 25 from 100, then 40 with 10; the paper: 20 × 100 from 500, then 100 with 1,000).
5. **PP-Seq membership.** Both rules are scored: the per-type rule (≥ 20% of the type's largest response proportion) and the package's `sortperm_neurons` rule (a neuron's largest response across types above the 20th percentile of those maxima, assigned to its preferred type); the earlier docstring wrongly attributed the first rule to the package.
6. **seqNMF.** Both costs in the λ-crossover rule use the package's own `reconstruct` (its lag index is applied at shift $l-1$, a MATLAB carry-over; my helper used shift $l$, a constant one-bin offset). A sensitivity grid $K \in \{5, 10, 15\} \times L \in \{60, 120, 200\}$ ms replaces the single $K = 5$, $L = 120$ ms; the cell $K = 15$, $L = 120$ ms (the paper's guidance: $K$ above the expected number of sequences, $L$ about one replay) is the headline and the best cell over the grid is reported as an optimistic bound.

Outcome. On planted routes the corrected PP-Seq is as good as or better than cPCA: perfect on A–E, one of two routes at 15% replay (F: 0.56 / 0.99), and in the realistic conditions clearly ahead at matched event counts (G: 0.31 / 0.54 and 0.60 / 1.00 against cPCA at chance on the same 1,500 events; H: 0.89 / 0.81 against 0.54 / 0.46). seqNMF is insensitive to $K$ and $L$: at chance from F onward. On hc-11, pooled over sessions, cPCA on the matched subsets has 14/96 map-matching tests in POST (15%, CI 8–23%) against 6/74 in PRE (8%, CI 3–17%); with all events 20/120 (17%, CI 10–25%) against 2/106 (2%, CI 0–7%) (cPCA recomputed on October 21 with the shuffle-null bound; before: 14/90 vs 5/74 and 21/118 vs 4/104); PP-Seq 2/47 (4%) against 2/44 (5%); seqNMF 1/6 against 0/6 (too few testable factors). PP-Seq's types on hc-11 are the burst and sub-bursts (offset spans 2–46 ms) in every session, under either membership rule.

## The real-data benchmark on hc-5 (October 19)

The comparison of the three order-producing methods was repeated on real sequences with an answer key: CA1 wheel-running sequences in hc-5, every method fit on half of a session's runs and scored on the other half, with aligned windows, unaligned windows and a shuffled control (`docs/benchmark_hc5.md`, `results/hc5/summary.md`, `figures/hc5_*.png`). Two changes to the PP-Seq wrapper came out of it and apply to any future PP-Seq run: mask ends are clipped to the timeline (a crash that made sweeps fall back on their first candidate whenever the timeline was a whole number of mask lengths; the hc-11 runs were not affected), and `ppseq_baseline.fit(refits_if_empty=...)` refits the next-best candidate when a refit ends with no latent events.

## The difficulty ladder (October 21–22)

Condition H combines two maps on the same cells, both directions, partial replays and assemblies, so it cannot say which difficulty separates the methods. The ladder (`sim/routes.py::LADDER`, `scripts/run_ladder.py`, `results/ladder/summary.md`, `figures/synthetic_ladder.png`) adds one difficulty at a time to "two maps on the same cells, both directions" (L2), with L0 (one map) and L1 (two maps, forward only) below it and L6 (= H's settings) at the top. Every rung: 80 cells (60 on the route), 1500 events, 40% replay; for each rung and draw one set of events is generated and all three methods get exactly those events; 3 draws per rung. Settings as in the matched rerun (cPCA components above the shuffle null; seqNMF λ-crossover rule on the K × L grid, headline K = 15, L = 120 ms; PP-Seq 20-candidate held-out sweep and the rerun's sampler, refit of the next-best candidate if a refit is empty, which no fit needed). Score per planted map: the best output's |Spearman|, recovered if it beats a null that takes the best of the method's outputs (both signs).

| rung | cPCA centred | PP-Seq | seqNMF |
|---|---|---|---|
| L0 one map, both directions | 0.90 (3/3) | 1.00 (3/3) | 0.97 (2/3) |
| L1 two maps, same cells, forward only | 0.88 (6/6) | 1.00 (6/6) | 0.99 (4/6) |
| L2 two maps, both directions | 0.83 (6/6) | 1.00 (6/6) | 0.47 (4/6) |
| L3 L2 + partial replays | 0.59 (4/6) | 0.35 (3/6) | 0.08 (0/6) |
| L4 L2 + assemblies | 0.85 (6/6) | 1.00 (6/6) | 0.36 (4/6) |
| L5 L2 + variable pacing | 0.81 (6/6) | 0.99 (6/6) | 0.26 (3/6) |
| L6 L2 + partial replays + assemblies (= H) | 0.48 (5/6) | 0.39 (3/6) | 0.15 (0/6) |

Median |ρ| over maps and draws (maps recovered). Three findings. (1) With whole routes replayed PP-Seq is essentially perfect on every rung and cPCA is 0.1–0.2 lower, a constant precision gap that is already there with one map (L0), so shared cells are not what costs cPCA. (2) Partial replays are hard for every method; there PP-Seq is very variable (per-map |ρ| from 0.13 to 0.98) and no longer ahead of cPCA, which recovers more maps. PP-Seq's 0.89 / 0.81 on H in the matched rerun was a single draw that happened to be favourable. (3) seqNMF degrades as soon as backward plays are present and fails on partial replays, whatever K and L.

## What was wrong with the earlier comparison (September)

Baselines were scored on whole-population order against the map, which none of them outputs; seqNMF ran with hand-picked $\lambda$, $K$, $L$ and a centre-of-mass read-out; the HMM was given a label-free "state seriation" that its authors never proposed; Bayesian decoding and rank order were treated as competitors although they take the map as input. The current comparison fixes each of these.
