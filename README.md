# Complex PCA of spike phase

Assemblies with internal timing from hippocampal reactivation events, found from sleep spikes alone. Each spike becomes an arrow whose angle is its time within the event; one complex eigen-decomposition then returns components whose weights say which cells belong together and whose angles say in what order they fire. The waking place map is used only afterwards, as the answer key.

- `docs/method.md`: the method, plainly, with the two design facts that matter (half-circle scale, read all components) and the evaluation protocol.
- `docs/plan.md`: objective, status, open problems, decisions and their evidence.
- `docs/lessons_learned.md`: what was tried before and why it was dropped.
- `docs/rendering.md`: open the docs with `Cmd+Shift+V` so formulas render.

## Results in one paragraph

Components are kept against a shuffle null (each cell's arrows permuted across events), which agrees with the textbook Marchenko–Pastur bound for centred arrows and corrects it for uncentred ones (`figures/noise_bound.png`, `docs/method.md` Section 4). On synthetic events with planted routes the pipeline recovers every route in all clean conditions (10 draws each; median |Spearman| 0.97–1.00 between a component's member times and the route positions, 0.87–0.92 when two routes share half their cells), including two routes played in both directions in 15% of events with 15% of cells firing per event. A difficulty ladder that takes the realistic condition H apart one difficulty at a time, with all three methods on the same events (`figures/synthetic_ladder.png`, `results/ladder/summary.md`), shows that two maps on the same cells, assemblies and variable pacing cost complex PCA little (median |ρ| 0.81–0.90, every map recovered) while PP-Seq is essentially perfect on those rungs (0.99–1.00); partial replays are hard for every method, and there PP-Seq loses its lead (median 0.35–0.39 against complex PCA's 0.48–0.59, 3 of 6 maps recovered against 4–5 of 6). On the eight hc-11 sessions (Grosmark & Buzsáki 2016), with centred components and the event duration as the half-circle scale, 20 of 120 (component, direction) tests match the place map in sleep after the novel maze (POST; chance ≈ 6) against 2 of 106 in sleep before it (PRE; chance ≈ 5); the effect is carried by the two sessions with 50 or more place cells. Uncentred components are at chance in both sleeps (3/54, 1/44), as the regime rule predicts. Against the other methods run with their authors' procedures on identical event subsets (`docs/baselines.md`, `results/rerun/summary.md`): PP-Seq, properly swept, matches or beats cPCA on planted routes when whole routes are replayed (its earlier lead on the realistic condition H came from a single draw and does not hold across three matched draws on the ladder) but finds no map-ordered sequence type on hc-11 in either sleep; seqNMF is at chance from 15% replay onward whatever K and L; the Maboudi HMM detects sequential events in both sleeps alike. On these recordings cPCA's centred, event-normalised representation is the one that finds POST-specific map order (14/96 tests in POST vs 6/74 in PRE on the matched subsets); this is a statement about the representation on this dataset, not a general ranking of the methods as sequence extractors. One-figure summary: `figures/comparison_hc11.png`; planted routes: `figures/comparison_synthetic.png`; outputs side by side: `figures/comparison_example.png`. Full tables: `results/summary.md`, `results/synthetic/summary.md`, `results/baselines/summary.md`; figures: `figures/`.

**Real sequences with an answer key (hc-5, `docs/benchmark_hc5.md`).** CA1 wheel-running sequences in six sessions of one rat, each method fit on half of a session's runs and scored on the other half. With windows aligned on wheel onset, complex PCA recovers the sequence in 32/36 fits, seqNMF in 29/36, PP-Seq in 14/36, and plain averaging of the runs does best (ρ 0.71). When every window starts at its own random time, complex PCA is the only method that keeps its performance (35/36 fits, ρ 0.63; seqNMF 26/36, PP-Seq 13/36, averaging drops to ρ 0.47), because its lag matrix ignores each event's clock. All methods are at chance on a shuffled control. A fit takes about 50 ms for complex PCA (noise bound included), 50 s for seqNMF, 16 min for PP-Seq. This is the regime that suits the method (one sequence in every event); on planted minority replays of whole routes PP-Seq is the more precise extractor. Figures: `figures/hc5_sequences.png`, `figures/hc5_benchmark.png`, `figures/hc5_sensitivity.png`.

## Layout

```text
cpca/        library (encode, decompose, ica, decode, evaluate, stats, events, hc11, hc5, pipeline, theory)
sim/         synthetic routes with a planted answer
baselines/   count PCA/ICA, seqNMF, HMM (Maboudi), PP-Seq (authors' Julia package), Bayesian decoding reference
scripts/     run_synthetic.py, run_hc11.py, summarize.py, plot_noise_bound.py; run_ladder.py, summarize_ladder.py (difficulty ladder); run_hc5.py, summarize_hc5.py (wheel benchmark); comparison runners
tests/       pytest
docs/        method, plan, lessons learned, baselines, benchmark_hc5, rendering
results/     generated tables (pickles are not committed)
figures/     generated by the summarize scripts
data/        session files: hc-11 (data/hc11_sessinfo/...), hc-5 (data/hc5/*_BehavElectrData.mat, extracted from the archives); never committed
hc-11/, hc-5/  raw archives; never committed
```

## Quick start

```bash
conda create -n CPCA --override-channels -c conda-forge python=3.12 numpy scipy matplotlib scikit-learn h5py pytest pip
conda activate CPCA
pip install -e .
pytest -q

python scripts/run_synthetic.py        # ~3 min on 9 cores: planted routes, 8 conditions + 7 ladder rungs x 10 draws
python scripts/run_hc11.py             # ~1 min on 8 cores: all sessions, POST and PRE, 4 variants
python scripts/summarize.py            # tables -> results/summary.md, figures -> figures/
python scripts/plot_noise_bound.py     # figures/noise_bound.png: textbook bound vs shuffle null
.venv/bin/python scripts/run_ladder.py && .venv/bin/python scripts/summarize_ladder.py   # ~2 h on 9 cores: cPCA, seqNMF, PP-Seq on the ladder
python scripts/plot_component_circle.py  # components and data on the unit half-circle

# method comparison (extra packages: python -m venv --system-site-packages .venv; .venv/bin/pip install hmmlearn seqnmf "setuptools<80";
# PP-Seq needs Julia: curl -fsSL https://install.julialang.org | sh, then julia --project=baselines/ppseq -e 'using Pkg; Pkg.instantiate()')
.venv/bin/python scripts/run_baselines.py synthetic     # ~25 min, resumable per condition: seqNMF, HMM, Bayes, count PCA/ICA, cPCA
.venv/bin/python scripts/run_baselines.py hc11          # ~1.5 h, all sessions, POST and PRE
python scripts/run_rerun.py synthetic --methods=cpca,ppseq   # matched rerun: fixed 1500-event subsets, corrected PP-Seq (~15 min to 1.5 h per dataset)
.venv/bin/python scripts/run_rerun.py synthetic --methods=seqnmf   # seqNMF K x L grid on the same subsets
python scripts/run_rerun.py hc11 --methods=cpca,ppseq; .venv/bin/python scripts/run_rerun.py hc11 --methods=seqnmf
.venv/bin/python scripts/summarize_rerun.py             # results/rerun/summary.md, figures/comparison_synthetic.png, figures/comparison_hc11.png
.venv/bin/python scripts/summarize_baselines.py         # results/baselines/summary.md and figures/comparison_example.png (first comparison; HMM, Bayes)

# hc-5 wheel benchmark (real sequences with an answer key; docs/benchmark_hc5.md)
for f in hc-5/*.tar; do tar -xf "$f" -C data/hc5 --include='*_BehavElectrData.mat'; done
.venv/bin/python scripts/run_hc5.py --workers=9          # ~3 h on 10 cores (PP-Seq is nearly all of it), resumable
.venv/bin/python scripts/run_hc5.py --workers=9 --redo-empty-ppseq
.venv/bin/python scripts/summarize_hc5.py                # results/hc5/summary.md, figures/hc5_sequences.png, hc5_benchmark.png, hc5_sensitivity.png
```

Minimal use of the method on your own events:

```python
import numpy as np
from cpca import fit

# times_ms: (events, cells, spikes) in ms from each event window's start, NaN where no spike
# T_ms: each event's duration (the half-circle rule), or one number for all events
result = fit(times_ms, T_ms, center=True, scale=True)
result["n_kept"]                      # components above the noise bound (shuffle null; result["n_mp"]: textbook count)
result["readouts"][0]["times_ms"]     # relative firing time of each member cell in component 1
result["readouts"][0]["weights"]      # its weight (participation)
```
