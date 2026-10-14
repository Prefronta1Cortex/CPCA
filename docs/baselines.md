# Baseline methods: what was run, how, and what was not

Open with `Cmd+Shift+V` for formulas. Results: `results/baselines/summary.md`, `figures/baselines_*.png`. Code: `baselines/`, `scripts/run_baselines.py`, `scripts/summarize_baselines.py`.

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

## Count PCA/ICA (Lopes-dos-Santos et al., 2013)

`baselines/count_pca_ica.py`: spike counts per event, z-scored, PCA, components above the Marchenko–Pastur bound, FastICA. Produces assemblies without order; reported as the number of assemblies and whether members' fields are compact (see `results/summary.md`).

## Not run, and why

- **PP-Seq** (Williams et al., 2020): the authors' implementation is `PPSeq.jl`; Julia is not installed on this machine and re-implementing a Bayesian point-process model with MCMC inference would be exactly the kind of second-hand implementation that misrepresents a method.
- **convSeq** (Koshkin et al., 2024): the repository is research code (GPU device by default, YAML configuration, a trainer coupled to ground-truth plotting, 4,000-epoch runs, no packaged API); running it on these data would require rewriting its training loop.

Both are discussed from their published results in `docs/lessons_learned.md`.

## What was wrong with the earlier comparison (September)

Baselines were scored on whole-population order against the map, which none of them outputs; seqNMF ran with hand-picked $\lambda$, $K$, $L$ and a centre-of-mass read-out; the HMM was given a label-free "state seriation" that its authors never proposed; Bayesian decoding and rank order were treated as competitors although they take the map as input. The current comparison fixes each of these.
