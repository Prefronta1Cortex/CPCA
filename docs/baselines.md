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

Hyperparameters, following Supplement F.2 of the paper: fixed $\gamma = 3$, $\phi = 1$, $\nu = 4$, offset prior mean 0, no time warping (the paper found no consistent benefit on hippocampal data); a random sweep over the number of sequence types $R \in \{2, 3, 4\}$, the mean sequence amplitude (variance = mean), the background amplitude (variance = mean), the sequence rate and the neuron response width, each setting scored by the held-out log-likelihood of a speckled 7.5% mask (`easy_sample_masked!`), the best setting refit on all spikes (`easy_sample!`). Stated reductions relative to the paper: 4 sweep settings instead of 2,000, run on 600 events with a shortened schedule; the final fit on at most 1,500 events (the same cap as seqNMF) with 8 annealing stages × 25 sweeps from temperature 100, then 40 sweeps with 10 split-merge moves (paper: 20 × 100 from 500, then 100 sweeps with 1,000 moves). $R = 1$ is excluded because one type cannot represent two directions.

Scoring, on PP-Seq's own outputs: each sequence type $r$ gives a cell order, the neuron offsets $b_{nr}$, with members the neurons whose response proportion $a_{nr}$ is at least 20% of the type's largest (the threshold the package's own `sortperm_neurons` uses) → `order_vs_place`, identical to cPCA and seqNMF. Each saved post-anneal sample gives latent events; an event window contains a sequence in the fraction of samples with a latent event inside it. PP-Seq represents a synchronous volley as a sequence type whose offsets coincide (span of a few ms), and on planted data it assigns such a type to every volley; per-event detection is therefore reported for *sequence types*, defined label-free as types whose member offsets span at least 20 ms, as well as for any type.

## Count PCA/ICA (Lopes-dos-Santos et al., 2013)

`baselines/count_pca_ica.py`: spike counts per event, z-scored, PCA, components above the Marchenko–Pastur bound, FastICA. Produces assemblies without order; reported as the number of assemblies and whether members' fields are compact (see `results/summary.md`).

## Not run, and why

- **convSeq** (Koshkin et al., 2024): the repository is research code (GPU device by default, YAML configuration, a trainer coupled to ground-truth plotting, 4,000-epoch runs, no packaged API); running it on these data would require rewriting its training loop.

(PP-Seq was initially excluded for lack of Julia; Julia was installed and the authors' package is now part of the comparison, see above.)

## What was wrong with the earlier comparison (September)

Baselines were scored on whole-population order against the map, which none of them outputs; seqNMF ran with hand-picked $\lambda$, $K$, $L$ and a centre-of-mass read-out; the HMM was given a label-free "state seriation" that its authors never proposed; Bayesian decoding and rank order were treated as competitors although they take the map as input. The current comparison fixes each of these.
