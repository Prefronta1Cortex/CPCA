# Complex PCA of spike phase

Assemblies with internal timing from hippocampal reactivation events, found from sleep spikes alone. Each spike becomes an arrow whose angle is its time within the event; one complex eigen-decomposition then returns components whose weights say which cells belong together and whose angles say in what order they fire. The waking place map is used only afterwards, as the answer key.

- `docs/method.md`: the method, plainly, with the two design facts that matter (half-circle scale, read all components) and the evaluation protocol.
- `docs/plan.md`: objective, status, open problems, decisions and their evidence.
- `docs/lessons_learned.md`: what was tried before and why it was dropped.
- `docs/rendering.md`: open the docs with `Cmd+Shift+V` so formulas render.

## Correction (November 5): what the results mean

Most results below use a best-output test: the best of a method's components against the map. They show that a sequence's order is **present among** the components. They do **not** show that each component is one route.

Scored per component (`docs/goals.md`, `docs/separation.md`, `docs/rescore.md`):
- **hc-5 wheel runs:** complex PCA meets the one-sequence-per-component goal.
- **Planted data with two routes on different cells:** no tested method separates them.
- **Planted data with two maps on the same cells:** a template step after PCA comes close in most fits.
- **hc-11 sleep:** about a fifth of the outputs that carry timing beyond each cell's latency match a waking running direction after the maze, about twice chance; before the maze, at chance.

**Statistics.** Session counts treat sessions as independent and are not corrected for the many tests run. For the paper they need reporting per animal, and with correction.

## Results in one paragraph

Components are kept against a shuffle null (each cell's arrows permuted across events), which agrees with the textbook Marchenko–Pastur bound for centred arrows and corrects it for uncentred ones (`figures/noise_bound.png`, `docs/method.md` Section 4). On synthetic events with planted routes the pipeline recovers every route in all clean conditions (10 draws each; median |Spearman| 0.97–1.00 between a component's member times and the route positions, 0.87–0.92 when two routes share half their cells), including two routes played in both directions in 15% of events with 15% of cells firing per event. A difficulty ladder that takes the realistic condition H apart one difficulty at a time, with all three methods on the same events (`figures/synthetic_ladder.png`, `results/ladder/summary.md`), shows that two maps on the same cells, assemblies and variable pacing cost complex PCA little (median |ρ| 0.81–0.90, every map recovered) while PP-Seq is essentially perfect on those rungs (0.99–1.00); partial replays are hard for every method, and there PP-Seq loses its lead (median 0.35–0.39 against complex PCA's 0.48–0.59, 3 of 6 maps recovered against 4–5 of 6). On the eight hc-11 sessions (Grosmark & Buzsáki 2016), with centred components and the event duration as the half-circle scale, 20 of 120 (component, direction) tests match the place map in sleep after the novel maze (POST; chance ≈ 6) against 2 of 106 in sleep before it (PRE; chance ≈ 5); the effect is carried by the two sessions with 50 or more place cells. Uncentred components are at chance in both sleeps (3/54, 1/44), as the regime rule predicts. Against the other methods run with their authors' procedures on identical event subsets (`docs/baselines.md`, `results/rerun/summary.md`): PP-Seq, properly swept, matches or beats cPCA on planted routes when whole routes are replayed (its earlier lead on the realistic condition H came from a single draw and does not hold across three matched draws on the ladder) but finds no map-ordered sequence type on hc-11 in either sleep; seqNMF is at chance from 15% replay onward whatever K and L; the Maboudi HMM detects sequential events in both sleeps alike. On these recordings cPCA's centred, event-normalised representation is the one that finds POST-specific map order (14/96 tests in POST vs 6/74 in PRE on the matched subsets); this is a statement about the representation on this dataset, not a general ranking of the methods as sequence extractors. Per session, scoring only each method's best output against a null that takes the best over all of its outputs, running directions and senses, the bar is stricter: complex PCA passes in 1 of 8 POST sessions (Achilles 10/25) and 0 of 8 PRE, seqNMF and PP-Seq in none (`figures/hc11_sequences.png`, waking place fields sorted by each method's sleep order, with a random order as the chance reference; `results/hc11_sequences/summary.md`). One-figure summary: `figures/comparison_hc11.png`; planted routes: `figures/comparison_synthetic.png`; outputs side by side: `figures/comparison_example.png`. Full tables: `results/summary.md`, `results/synthetic/summary.md`, `results/baselines/summary.md`; figures: `figures/`.

**Waking fit against sleep fit (`docs/wake_sleep.md`).** A basic consistency check: the same model fitted to the laps on the maze and to sleep bursts, compared pair by pair without any map. On planted data the two fits agree in 10 of 10 draws when sleep replays the waking maps and in 0 of 10 when it replays other maps; the waking fit matches the waking place map in all 8 hc-11 sessions; and waking and sleep fits agree in 4 of 8 sessions after the maze (binomial p = 0.0004) against 1 of 8 before it. Figure: `figures/wake_sleep.png`.

**A second dataset (`docs/huszar.md`).** On 65 sessions of DANDI 000552 (Huszár et al. 2022; mice, figure-eight alternation; median 57 place cells against 24 on hc-11), with identical code and measures: the population structure of the reduced space matches the waking map and the waking fit in most sessions, in the sleep before the task as well as after it (the maze is familiar); single components match the map only slightly above chance (9.9% of tests after the task, 6.2% before). Figure: `figures/huszar_vs_hc11.png`.

**Topology of the reduced space (`docs/topology.md`).** Taking all kept components together, pairs of place cells that are close in complex PCA's reduced space of sleep are also close on the waking map (position × running direction) in 5 of 8 hc-11 sessions after the maze and 1 of 8 before it. Count PCA, which ignores timing, does about as well on hc-11, so the topology in sleep is mostly co-firing; where only timing carries the order (planted whole-route replays, the hc-5 wheel) complex PCA passes and count PCA fails. Figure: `figures/topology_hc11.png`; the checks on planted data and hc-5 are in `results/topology/summary.md`.

**Forward and backward plays (`docs/direction.md`).** Playing an event backwards conjugates its arrows, so one complex template describes a route in both directions and each event can be labelled by which of the two it matches. On planted data where one route is played both ways this restores the order (median |ρ| 0.74–0.81 → 1.00) and labels every event's direction correctly (median 100%, reverse share within 0.01); it finds no reverse group in forward-only data and no order in a no-order control; partial replays are outside the model. On hc-11 a template learned without the map labels 43 of 46 significant replays of Achilles 10/25 POST in the map's direction, and the forward and backward groups match the map with opposite signs; in sessions with few place cells the labels are at chance or no template matches the map. Figures: `figures/direction_synthetic.png`, `figures/direction_hc11.png`.

**Sequences that share cells (`docs/sharing.md`).** When two maps play over the same cells, PCA returns their sum and differences. A step after PCA fits one template per sequence, assigns each event to the template it matches (either way round), refits, and keeps only templates that hold up on held-out events. On the difficulty ladder's own events, the ones PP-Seq and seqNMF were given, it recovers both maps at median |ρ| 1.00 where whole routes are replayed, as PP-Seq does (components alone: 0.81–0.88), and leads on partial replays (0.70 and 0.81 against PP-Seq's 0.35 and 0.39, though as segments of the maps), at 3.5 s per fit against PP-Seq's 116 min. It changes nothing where sequences do not share cells and finds nothing in a no-order control. On hc-11 it adds almost nothing: single sleep events carry too little order for templates to pass the held-out check. Figures: `figures/sharing_synthetic.png`, `figures/sharing_ladder.png`, `figures/sharing_hc11.png`.

**Six methods on the same hc-11 events (`docs/methods_hc11.md`).** Count PCA/ICA assemblies, SPOTDisClust and Rastermap, each run with its authors' code and settings and checked on planted data first, join complex PCA, seqNMF and PP-Seq on identical sleep events, scored by the same best-of-outputs test against the waking map. No method recovers the map reliably. Complex PCA and Rastermap each pass in 1 of 8 sessions after the maze and none before it; seqNMF, PP-Seq and SPOTDis pass in none; SPOTDis finds no clusters of events. Complex PCA is among the fastest (0.5 s per fit, PP-Seq 18 min). Figure: `figures/methods_hc11.png`.

**Is there enough replay in hc-11 sleep? (`docs/replay.md`).** The original paper's Bayesian replay analysis, reproduced here, confirms that sleep after the maze carries more of the maze's order than sleep before it (7 of 8 sessions; pooled p = 0.002), and our event windows carry the same content. But clear replays exceed the level before the maze by at most 3.4% of events (median 0.8%). Adding known amounts of replay to real sleep events shows that complex PCA needs about 10% even with ideal replays and many place cells per event. hc-11 sleep therefore has 3 to 10 times too little replay for a method that is not given the map. Figure: `figures/replay_hc11.png`.

**Awake rest on the maze, and a caveat on order results (`docs/maze_rest.md`).** On rest periods on the maze, where replay is common, complex PCA recovers the waking map in 5 of 8 sessions, against 0 of 8 in sleep before the maze. A timing control (each event's spike trains shuffled among its active cells) shows that the order test can be passed by co-firing alone. Order beyond co-firing is shown in 2–3 of the 8 sessions, and not for the earlier sleep pass of Achilles 10/25. The hc-11 order results above should be read with this in mind until they have been rerun with the timing control.

**hc-11 waking laps (`docs/laps_hc11.md`).** Fitted on half of the running laps and scored against place fields from the other half, complex PCA recovers the order in 37 of 48 fits, close to an averaging reference that is told each lap's direction (41/48). Count PCA, which ignores timing, does so in 15 of 48, and in 5 of 30 on the linear tracks. The same cells whose sleep reactivations no method decodes are ordered correctly from their waking timing.

**A larger dataset: DANDI 001695 (`docs/dandi001695.md`).** These are mouse Neuropixels recordings around a novel linear track: 15 sessions from 4 mice, with about twice hc-11's place cells active per event. After the track, complex PCA recovers order that is beyond co-firing in 3 of 15 sessions (three mice), against 0 of 15 before it. This is a small improvement over hc-11, but fragile: the fragment criterion, the ripple-gated events and the CA1 + CA3 analysis do not reproduce it. The decoder finds only about 3% clear replays in either sleep.

**Sanity check: waking components within sleep components (`docs/containment.md`).** What complex PCA finds on the waking laps is found again, beyond chance, among its components on sleep after the maze in 6 of 8 hc-11 sessions (12 of 18 waking components individually). On planted data the test passes when sleep replays the waking maps, one of them or segments of them, and fails when sleep replays other maps on the same cells.

**A stricter order criterion (`docs/fragment.md`).** A new test asks for map order within one 40% stretch of the track, and requires the real fit to beat refits on events whose spike timing was shuffled. On planted data it finds whole and partial replays and is not fooled by co-firing that merely follows the map. On hc-11 nothing passes, but at hc-11's cell counts the test has little power even for strong planted replay. It needs recordings with more place cells.

**Could any method do better on hc-11? (`docs/power.md`).** Two checks say no.
- **Perfect event choice.** Even when a map-based decoder picks the replays, averaging those events recovers the map no more often than complex PCA does without the map.
- **Each method's threshold.** On real sleep events with known amounts of replay added, complex PCA needs the least replay of the methods tested (about 10% in the largest sessions). Rastermap, PP-Seq at a fixed setting and a new event-selecting variant of complex PCA need more. The variant helps on planted data, but on real events it locks onto large bursts and repeated co-firing.

Figure: `figures/power_hc11.png`.

**Real sequences with an answer key (hc-5, `docs/benchmark_hc5.md`).** CA1 wheel-running sequences in six sessions of one rat, each method fit on half of a session's runs and scored on the other half. With windows aligned on wheel onset, complex PCA recovers the sequence in 32/36 fits, seqNMF in 29/36, PP-Seq in 14/36, and plain averaging of the runs does best (ρ 0.71). When every window starts at its own random time, complex PCA is the only method that keeps its performance (35/36 fits, ρ 0.63; seqNMF 26/36, PP-Seq 13/36, averaging drops to ρ 0.47), because its lag matrix ignores each event's clock. All methods are at chance on a shuffled control. A fit takes about 50 ms for complex PCA (noise bound included), 50 s for seqNMF, 16 min for PP-Seq. This is the regime that suits the method (one sequence in every event); on planted minority replays of whole routes PP-Seq is the more precise extractor. Figures: `figures/hc5_sequences.png`, `figures/hc5_benchmark.png`, `figures/hc5_sensitivity.png`.

## Layout

```text
cpca/        library (encode, decompose, ica, decode, evaluate, stats, events, hc11, hc5, pipeline, theory, topology, direction, mixture)
sim/         synthetic routes with a planted answer
baselines/   count PCA/ICA, seqNMF, HMM (Maboudi), PP-Seq (authors' Julia package), Bayesian decoding reference
scripts/     run_synthetic.py, run_hc11.py, summarize.py, plot_noise_bound.py; run_topology.py, summarize_topology.py (topology test); run_direction.py, summarize_direction.py (forward/backward plays); run_sharing.py, summarize_sharing.py (sequences sharing cells); run_ladder.py, summarize_ladder.py (difficulty ladder); run_hc5.py, summarize_hc5.py (wheel benchmark); comparison runners
tests/       pytest
docs/        method, plan, lessons learned, baselines, benchmark_hc5, topology, direction, sharing, wake_sleep, rendering
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
python scripts/run_topology.py && python scripts/summarize_topology.py   # ~1.5 min on 10 cores: topology test (hc-11, synthetic, hc-5)
python scripts/run_direction.py && python scripts/summarize_direction.py # ~7 min on 10 cores: forward/backward plays (synthetic, hc-11)
python scripts/run_sharing.py && python scripts/summarize_sharing.py     # ~11 min on 10 cores: sequences sharing cells (synthetic, ladder, hc-11)
.venv/bin/python scripts/run_ladder.py && .venv/bin/python scripts/summarize_ladder.py   # ~2 h on 9 cores: cPCA, seqNMF, PP-Seq on the ladder
python scripts/plot_component_circle.py  # components and data on the unit half-circle
python scripts/plot_hc11_sequences.py     # ~10 s on 10 cores: waking fields sorted by each method's sleep order (hc-11, matched subsets)
python scripts/run_wake_sleep.py && python scripts/summarize_wake_sleep.py   # ~30 s on 10 cores: waking fit against sleep fit (hc-11, planted data)

# method comparison (extra packages: python -m venv --system-site-packages .venv; .venv/bin/pip install hmmlearn seqnmf "setuptools<80";
# PP-Seq needs Julia: curl -fsSL https://install.julialang.org | sh, then julia --project=baselines/ppseq -e 'using Pkg; Pkg.instantiate()')
.venv/bin/python scripts/run_baselines.py synthetic     # ~25 min, resumable per condition: seqNMF, HMM, Bayes, count PCA/ICA, cPCA
.venv/bin/python scripts/run_baselines.py hc11          # ~1.5 h, all sessions, POST and PRE
python scripts/run_rerun.py synthetic --methods=cpca,ppseq   # matched rerun: fixed 1500-event subsets, corrected PP-Seq (~15 min to 1.5 h per dataset)
.venv/bin/python scripts/run_rerun.py synthetic --methods=seqnmf   # seqNMF K x L grid on the same subsets
python scripts/run_rerun.py hc11 --methods=cpca,ppseq; .venv/bin/python scripts/run_rerun.py hc11 --methods=seqnmf
.venv/bin/python scripts/summarize_rerun.py             # results/rerun/summary.md, figures/comparison_synthetic.png, figures/comparison_hc11.png
.venv/bin/python scripts/summarize_baselines.py         # results/baselines/summary.md and figures/comparison_example.png (first comparison; HMM, Bayes)

# three more methods (docs/methods_hc11.md; .venv/bin/pip install numba hdbscan rastermap; SPOTDis code in baselines/third_party/spot)
.venv/bin/python scripts/validate_methods.py --workers=10  # planted checks -> results/methods_hc11/validation.md
.venv/bin/python scripts/run_methods_hc11.py --workers=8   # ~5 min; needs results/hc11_sequences (scripts/plot_hc11_sequences.py)
.venv/bin/python scripts/summarize_methods_hc11.py       # figures/methods_hc11.png, results/methods_hc11/summary.md

# replay in hc-11 sleep (docs/replay.md): the paper's Bayesian analysis, then replays added to real events
python scripts/run_replay_hc11.py --workers=8            # ~4 min; results/replay/
python scripts/run_spikein_hc11.py --workers=8           # ~2 min; results/spikein/
python scripts/summarize_replay.py                       # figures/replay_hc11.png, results/replay/summary.md

# awake rest on the maze (docs/maze_rest.md)
.venv/bin/python scripts/run_maze_rest.py --workers=8                                  # complex PCA, count PCA, PRE control; ~30 s
.venv/bin/python scripts/run_maze_rest.py --timing --epoch=maze --workers=8            # timing control, 100 shuffles; ~5 min
.venv/bin/python scripts/run_maze_rest.py --timing --epoch=maze --uncentred --workers=8
.venv/bin/python scripts/run_maze_rest.py --timing --epoch=post --workers=8
python scripts/summarize_maze_rest.py                                                  # results/maze_rest/summary.md

# DANDI 001695 (docs/dandi001695.md; data in data/dandi001695, 15 behaviour sessions, 2.6 GB)
.venv/bin/python scripts/run_dandi001695.py --areas=CA1 --workers=9        # ~25 min
.venv/bin/python scripts/run_dandi001695.py --areas=CA1,CA3 --workers=9    # ~3 h
python scripts/summarize_dandi001695.py                                     # results/dandi001695/summary.md

# sanity check: waking components within sleep components (docs/containment.md)
.venv/bin/python scripts/run_containment.py synthetic --workers=9   # planted checks; ~1 min
.venv/bin/python scripts/run_containment.py hc11 --workers=9        # POST; ~10 s
python scripts/summarize_containment.py                             # results/containment/summary.md

# the recurring-fragment criterion (docs/fragment.md)
.venv/bin/python scripts/run_fragment.py validate --workers=9   # planted checks; ~5 min
.venv/bin/python scripts/run_fragment.py hc11 --workers=9       # POST, PRE, maze rest; ~25 min
python scripts/check_fragment_power.py                          # power at hc-11 size; ~2 min
python scripts/summarize_fragment.py                            # results/fragment/summary.md

# what any method could do (docs/power.md)
python scripts/run_oracle_hc11.py --workers=8            # Step 0, perfect event choice; ~2 min
.venv/bin/python scripts/run_power_hc11.py --workers=9   # Step 1: complex PCA on all events, event-sparse, Rastermap; ~25 min
.venv/bin/python scripts/run_power_hc11.py --methods=sparse_norm --workers=4
.venv/bin/python scripts/run_power_hc11.py --methods=ppseq --workers=2   # 8 fits, ~1 h
python scripts/summarize_power.py                        # figures/power_hc11.png, results/power/summary.md

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
