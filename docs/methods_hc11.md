# Three more comparison methods on hc-11

Count PCA/ICA assemblies, SPOTDisClust and Rastermap join complex PCA, seqNMF and PP-Seq. All six get the
same events and are scored by the same functions against the waking map. Figure: `figures/methods_hc11.png`;
tables: `results/methods_hc11/summary.md`; checks on planted data: `results/methods_hc11/validation.md`.

## What each method is given

The rerun's matched subsets (`scripts/run_rerun.py`): per session and sleep epoch, the same 1500 population
bursts and the same spike times (at most 60 spikes per cell per event) for every method. No method sees the
waking session; the place map is used only for scoring. Complex PCA, seqNMF (K = 15, L = 120 ms) and PP-Seq
are the outputs scored for `figures/hc11_sequences.png`, reused unchanged.

## The three methods and every choice made for them

**Count PCA/ICA assemblies** (Lopes-dos-Santos et al. 2013; `baselines/count_assemblies.py`). The events
are binned at 25 ms and concatenated, each cell's counts are z-scored, the number of assemblies is the number
of eigenvalues above the Marchenko–Pastur bound, and FastICA in that subspace gives the assembly weights.
Members are cells with weight above the mean plus 2 SD. The method ignores spike order within a bin, so
it gives groups of co-firing cells and no order.

**SPOTDisClust** (Grossberger et al. 2018; the authors' code, `baselines/third_party/spot`, MIT licence,
commit 5d8d8ca; adapter `baselines/spotdis_baseline.py`). Spike times are expressed as fractions of each
event's duration (epoch length 1). Event-to-event dissimilarity is SPOTDis, and HDBSCAN clusters the events
at the paper's settings (min_cluster_size = min_samples = 10). Two choices were made on planted data before
looking at hc-11:
- *Pooled metric.* The pairwise metric (`SPOTD_xcorr`) could not separate patterns in sparse events.
  There, a replay's nearest neighbours were volleys, and HDBSCAN called every event noise. The authors'
  pooled metric (`SPOTD_xcorr_pooled`) separates them: 91–99% of an event's 10 nearest neighbours share its
  route and direction.
- *A fix to the authors' pooled code.* `xcorr_pooled_spotdis_cpu_` passes the shorter list first to
  `signature_emd_`, which requires the longer list first. It therefore raises an error for any two events
  with unequal spike counts, and its own line is marked TODO. Our copy orders the arguments as their
  pairwise metric does. Event pairs with no comparison are left as NaN instead of dividing by zero, and are
  set to the largest distance before clustering. Nothing else is changed.

Each cluster of events becomes a template. Members are the cells that fire more often in the cluster's
events than in the rest (one-sided Fisher exact test, p < 0.01). A member's time is its median first-spike
fraction in the cluster's events.

**Rastermap** (Stringer et al. 2024; `rastermap` 1.0; adapter `baselines/rastermap_baseline.py`). These
are the settings of the authors' notebook for fewer than 200 neurons, which itself uses hc-11:
n_clusters = None, n_PCs = min(64, cells − 1), locality = 0.1, time_lag_window = 15, grid_upsample = 0,
z-scored activity. One change: the events are binned at 10 ms instead of the notebook's 200 ms. Replay is
about 20 times faster than running, and 10 ms keeps the lag window (150 ms) at the time scale of an event.
The output is one ordering of all active cells.

## Scoring (the same for all methods)

- **Order test** (`scripts/plot_hc11_sequences.best_map_match`). This is |Spearman| between an output's
  member times and the field centres, for outputs with at least 6 place cells. It takes the best over
  outputs, running directions and senses, and its p-value comes from 1000 permutations of the field centres
  that take the same best. A method therefore gains nothing from returning more outputs.
- **Assembly locality** (`cpca.evaluate.best_compactness`). This is descriptive only: it asks whether some
  output's cells have fields close together, using the same best-of-outputs null. A route-wide sequence
  covers the whole track and scores at chance by construction, so this measure describes count ICA's
  assemblies and does not rank sequence methods.
- **Runtime of one fit**, one CPU core per fit:
  - complex PCA: timed again here;
  - seqNMF: its K = 15, L = 120 ms fit, including the λ search;
  - PP-Seq: the whole run, including the sweep that chooses its setting.

## Checks on planted data (`scripts/validate_methods.py`)

There were 1500 events of 80 cells and 3 draws per condition:
- D: two routes on different cells;
- L0: one map, both directions;
- L2: two maps on the same cells;
- a control with no replays.

Results:
- **Complex PCA** recovers every order: |ρ| 1.00 on D, 0.82 on L0, 0.81 on L2.
- **Rastermap** recovers every order on D (0.99) and L0 (0.56), and 2 of 3 on L2 (0.37).
- **SPOTDis** recovers 2 of 3 on D (0.98) and every order on L0 (1.00) and L2 (0.99).
- **Count ICA** finds a significantly local assembly for every map, with the members' field distance at
  0.11–0.16 of chance.
- **The no-replay control:** no method reports anything. Rastermap's single order is at chance (0/3), and
  the others return no outputs.

## Results on hc-11

- **Order test, sessions with p < 0.05, after the maze (POST) and before it (PRE):**
  - complex PCA 1/8 and 0/8;
  - Rastermap 1/8 and 0/8;
  - seqNMF, PP-Seq and SPOTDis 0/8 and 0/8;
  - random order 0/8 and 0/8.

  Chance is 5% of sessions, so 3 or more of 8 would be needed for p < 0.01. No method reaches that.
- **SPOTDis** finds no clusters in 12 of 16 epochs. In the other 4 it finds two small clusters of 11–28
  events each, which no cell is enriched in. This is not because hc-11 events are sparser than the planted
  ones: they have a median of 21 active cells per event against 15, and two events share 5 cells in both.
  hc-11 simply has no dense groups of near-identical events. Each event's 10th-nearest neighbour is at a
  SPOTDis of 0.030, against 0.015 for planted replays.
- **Count ICA** finds 5–16 assemblies per epoch. The most local one is significant in 2/8 POST and 1/8 PRE
  sessions.
- **Runtime, median per fit:**
  - count ICA 0.3 s;
  - complex PCA 0.5 s;
  - Rastermap 5.5 s;
  - SPOTDis 18 s;
  - seqNMF 117 s;
  - PP-Seq 1073 s.

This agrees with the supervised ceiling (`docs/ceiling.md`). Single sleep events on hc-11 carry almost no
map order, and adding methods built on different principles does not change that. Complex PCA stays among
the best and is among the fastest. None of the six recovers the waking map from sleep reliably on these
recordings.
