# Label-free method comparison

Open with `Cmd+Shift+V` for rendered formulas.

## Rules of the comparison

1. Every method sees spikes only: POST NREM population events (edges at the mean rate, peak above mean + 3 SD, 50–500 ms), pyramidal cells. No behavioural information enters any method.
2. The waking place map (place-field centre of each cell, per running direction, from the maze epoch) is the **answer key**. It is used once, at the end, to score the cell order a method produced. Nothing is selected, tuned, or filtered with it.
3. Template-based detectors (Bayesian decoding, rank-order correlation) are therefore not in the comparison: they take the map as input.
4. Every method is first run on synthetic events with a planted order (`run_synthetic.py`). A method that recovers the planted order there is being used correctly; what it then does on real data reflects the data.
5. Score: best $|\rho|$ (Spearman) between the method's cell order and the answer key, maximised over the two running directions and over the method's templates or factors. The chance level is the 95th percentile of the **same maximum** after shuffling cell identities, so a method with more templates is not favoured. For circular mazes the best of 12 rotations of the track is taken, for the data and for every shuffle alike.

## Methods

- **cPCA, one template.** The existing pipeline: 100 ms window on the population peak, mean arrow per cell, template read from $C$, from $C - C^{0}$ (co-activation null removed), and from $C$ restricted to events with positive timing excess. `cpca/`, `runs/hc11_post/post_timing.py::templates`.
- **cPCA, mixture of templates (new, `cpca/cluster.py`).** $C$ averages the lag arrows $s_{ik}\overline{s_{il}}$ over *all* events, so two routes, or one route played forward and backward, cancel. The mixture assigns each event to one of $K$ templates by $|\langle s_i, p_k\rangle|^2 / \|s_i\|^2$ (blind to the event's clock start and size), then refits each template as the top eigenvector of the lag matrix of **its own events only**, and alternates until assignments stop changing (k-means on the complex unit sphere). Templates are read from $C - C^{0}$ of each group. $K \in \{2, 3, 4, 6\}$ is chosen without labels by split-half stability (fit on two halves, match templates, average $|\langle p, q\rangle|$). Do not centre within events first: a forward event and its reverse then differ only by a global sign, which the assignment ignores (documented in the module and checked in `tests/test_cluster.py`).
- **seqNMF** (Mackevicius et al., 2019; `seqnmf` PyPI package). Events binned at 10 ms, zero-padded by $L$ bins and concatenated; each neuron divided by its standard deviation (otherwise one high-rate cell owns every factor); $K = 3$ factors of length $L = 12$ bins (120 ms), $\lambda = 0.001$, 80 iterations, best of 3 restarts by final cost. A factor's cell order is the time centre of mass of its $W$ column, for cells whose peak loading is at least 20% of the factor's maximum.
- **Poisson HMM** (Maboudi et al., 2018; `hmmlearn`). Events binned at 20 ms; number of states (20 or 35) chosen by held-out log-likelihood on 20% of events; label-free seriation of states by the Fiedler vector of the symmetrised transition graph; a cell's position is its rate-weighted mean state rank. Note: the original paper displayed its latent map by anchoring states to position with behaviour; that step is not allowed here, so the seriation is the label-free substitute.
- **Count PCA/ICA** (Lopes-dos-Santos et al., 2013). Event spike counts, z-scored, PCA with the Marchenko–Pastur bound, FastICA. Gives assembly membership, not order; reported as the number of assemblies and whether the top members of an assembly have neighbouring place fields.

## Files

- `harness.py`: data loading (reuses `scripts/analyze_hc11_window.py` and `runs/hc11_post/post_timing.py`), answer key, scoring, the five methods, the synthetic generator.
- `run_synthetic.py`: four planted-order conditions, results in `results/synthetic.pkl`.
- `run_real.py <session>`: one hc-11 session, results in `results/real_<session>.pkl`.
- `summarize.py`: `results/summary.md` and `figures/*.png`.

## Environment

The `CPCA` conda environment plus `hmmlearn` and `seqnmf`. The `seqnmf` package imports `pkg_resources`, so `setuptools<80` is needed. The runs reported here used a throwaway virtual environment built on the `CPCA` interpreter:

```bash
python -m venv --system-site-packages /tmp/cpca_bl
/tmp/cpca_bl/bin/pip install hmmlearn seqnmf "setuptools<80"
/tmp/cpca_bl/bin/python run_synthetic.py
for s in Achilles_10252013 ...; do /tmp/cpca_bl/bin/python run_real.py $s; done
/tmp/cpca_bl/bin/python summarize.py
```

## Results

See `results/summary.md` (tables) and `figures/` (`synthetic_benchmark.png`, `real_sessions.png`, `mixture_K_selection.png`).
