# Complex PCA of spike phase for hippocampal reactivations: a full account of the work to 5 November 2026

Internal summary for checking, not for submission. Open with `Cmd+Shift+V` so the formulas render. Each number comes
from the document or results file named next to it. Numbers computed for this summary are marked *(computed for this
summary)*. Where an earlier statement turned out wrong, the correction is given in place and listed again in Section 8.

---

## Abstract

Hippocampal place cells fire in sequences while an animal runs along a track. During later sleep and rest, brief
population bursts sometimes replay these sequences in compressed form, forward or backward. Replay is normally
detected with a Bayesian decoder built from the waking place map, which can only find what was recorded awake.

We developed a label-free alternative: complex PCA of spike phase. Each spike becomes a unit complex number whose
angle is its time within the burst. Each cell gets one averaged complex number per burst. The eigenvectors of the
cells' Hermitian covariance (the lag matrix) are read out as groups of cells with relative firing times.

**The goal (fixed on 5 November).** From reactivation events alone, the method should output components that are each
one route or sequence. Success is measured per component.

**Where the goal is met:**
- **One sequence in every event.** On real wheel-running sequences (hc-5), uncentred complex PCA usually returns one
  component, and that component is the sequence in 74–86% of cases.
- **Fast.** It recovers the sequence without a shared clock (35 of 36 held-out fits) at about 50 ms per fit.
- **Waking laps.** On hc-11 waking laps it recovers the order where count PCA does not (37 of 48 fits against 15).

**Where the goal is not met:**
- **Several sequences.** On planted data with two routes on different cells, no tested variant gives one component per
  route: 0–13% of outputs are one route, and no fit is clean. With two maps on the same cells, a template step after
  PCA comes close (76–84% of outputs, 6–7 of 10 fits clean).
- **Linear-track laps.** The two running directions come out as their sum and difference.
- **hc-11 sleep.** After the maze, 21% of the outputs that carry event-specific timing match exactly one running
  direction, about twice chance. Before the maze the share is at chance.

**Order present among the components.** Most results before 5 November were scored by the best component against the
map. That shows the order is present among the components, not that routes are separated:
- On planted data it was present in all clean conditions.
- On hc-11 sleep it was weak and mostly explained by co-firing (cells active in the same events), not by timing.
- hc-11 sleep contains little replay: clear replays exceed the pre-maze level by at most 3.4% of events, while the
  method needs about 10% under ideal conditions.

---

## 1. Background and question

### 1.1 Replay

- **Place cells and replay.** A place cell fires when the animal is at one position on a track (its place field), so
  a run along the track produces a sequence of cells. In sleep and quiet rest, CA1 produces population bursts lasting
  50–500 ms. In some of them the cells fire in the order of their place fields, about 20 times faster than running.
  This is replay. It can run forward or backward, and it can cover only part of the track (a partial replay).
- **Standard detection.** A Bayesian decoder uses the waking place fields to estimate the position represented in
  each 20 ms bin of a burst. A burst counts as a replay if the decoded position moves steadily, beyond shuffle nulls
  (Grosmark & Buzsáki 2016 for hc-11). The decoder requires the map as input.

### 1.2 Label-free methods

Methods that look for repeated sequences without the map:
- **seqNMF** (Mackevicius et al. 2019): convolutive non-negative matrix factorisation.
- **PP-Seq** (Williams et al. 2020): a Bayesian point-process model of sequence events.
- **Hidden Markov models** (Maboudi et al. 2018).
- **SPOTDisClust** (Grossberger et al. 2018): clustering events by a spike-time distance.
- **Rastermap** (Stringer et al. 2024): one ordering of all cells.
- **Count PCA/ICA assembly analysis** (Lopes-dos-Santos et al. 2013): groups of co-firing cells from spike counts,
  with no order.

The idea of this project is to keep the simplicity of count PCA (one eigendecomposition, almost no parameters) and
add timing by encoding spike times as complex phases.

### 1.3 The goal as agreed (`docs/goals.md`, 5 November)

- **Goal.** From reactivation events, without the waking map, complex PCA (with any separation step it needs) outputs
  components that each correspond to one route or sequence. The map is used only afterwards, to check them.
- **Success.** The share of output components that are each one sequence. A component that mixes two sequences fails.
- **Reporting.** Per component and per route. Always keep "order present among the outputs" apart from "routes
  separated onto outputs".

**History.** Until 5 November most results were scored with a best-of-outputs test: the best component, map and sense
against a null that takes the same best (Section 4). That test answers "is the order present somewhere among the
outputs?". It does not answer the goal. The separation failure was found when the user inspected the components
directly. Section 5.11 re-scores the main results per component.

---

## 2. The method

Code: `cpca/` (`encode`, `decompose`, `decode`, `pipeline.fit`). Full description: `docs/method.md`.

### 2.1 Events

- **Bursts.** The pooled spikes of pyramidal cells give a population rate. A burst runs between crossings of the mean
  rate, and its peak must exceed the mean plus 3 SD. Bursts last 50–500 ms. Only NREM sleep is used for sleep.
- **Windows.** Each burst's window is centred on its population peak and lasts as long as the burst.
- **Epochs (hc-11).** PRE is sleep before the animal first saw the maze, and serves as the control. MAZE is the
  running session. POST is sleep after it.

### 2.2 Encoding: one complex number per cell per event

- **Spikes.** A spike at time $t$ (ms from the window's start) in an event of duration $T$ becomes the unit complex
  number $e^{j\pi t/T}$, whose angle is between 0 and $\pi$.
- **Cells.** A cell's spikes in the event are averaged into one complex number $s_{ik}$ (event $i$, cell $k$). A tight
  burst gives a long vector at the burst's phase, scattered spikes a short one, and a silent cell 0.
- **Why $T$ is the event's duration (the half-circle rule).** Two different orders of the same cells become nearly
  orthogonal vectors only if their phases spread over a wide arc.
  - With a 100 ms scale for spikes within ±12 ms of the peak (a 40° arc), forward and backward plays of one route are
    92% parallel and PCA merges them.
  - With the event's own duration (a 180° arc) they are 3% parallel and PCA separates them, with eigenvalues in the
    planted 60:40 ratio (`docs/method.md`, Section 2).

### 2.3 The lag matrix, centred or not

- **The covariance.** $C_{kl} = \frac{1}{M}\sum_i s_{ik}\overline{s_{il}}$ is Hermitian. Each entry averages, over
  events, a complex number whose angle is the lag between cells $k$ and $l$ and whose length is how consistent that lag
  is. A common time shift of an event multiplies all its entries by one phase, which cancels in $C$, so $C$ does not
  need a shared clock across events.
- **Uncentred.** $C$ as written. The first component then reflects participation, meaning which cells fire often.
  This is right when the pattern is in nearly every event (wheel runs, laps).
- **Centred and scaled.** Subtract each cell's mean complex value across events and divide by its root-mean-square
  length (the complex analogue of z-scoring). Components then describe how events differ. This was chosen for sleep,
  where replays are a minority of events.
- **A side effect of centring.** A silent cell's 0 becomes a value no spike can produce. Centred components therefore
  partly encode "did not fire". This lets co-firing contrasts look like order (Section 5.5).

### 2.4 Components and how many are kept

- **Components.** Eigendecomposition $C p_n = \lambda_n p_n$. Cell $k$'s weight in component $n$ is $|p_{nk}|$.
  Its relative time is $\tau_{nk} = \frac{T}{\pi}\arg p_{nk}$, up to one offset per component.
- **The noise bound (a shuffle null).** Each cell's values are permuted across events independently, 100 times.
  This keeps everything about each cell and destroys coordination between cells. A component is kept if its eigenvalue
  exceeds the 95th percentile of the shuffled data's largest eigenvalue.
  - For uncentred data the first component is always kept, and later ones are tested against the shuffle's second
    eigenvalue.
  - The textbook Marchenko–Pastur bound agrees with the shuffle for centred, scaled data. It keeps far too many
    uncentred components: 8–14 per hc-11 session against 2–6 by the shuffle, and 16 on pure noise
    (`figures/noise_bound.png`). It was replaced on 21 October.

### 2.5 Reading a component

- **Members.** Cells with weight at least 20% of the component's largest weight.
- **Order.** The members' angles are placed on the circle and cut at the widest empty gap. The angles are then
  unwrapped into relative times, and their ranking is the component's order.
- **Consequence.** Any component becomes an "order", including one whose members sit at two opposite phases (0 and
  $\pi$), which is a contrast between two groups rather than a sequence. Section 5.11 adds a check for this.

### 2.6 Optional steps after PCA

- **Complex ICA rotation** (`cpca/ica.py`). This rotates the kept components to make them sparser, which can split a
  union of two routes. It was off by default in nearly every result before November.
- **Direction split** (`cpca/direction.py`, `docs/direction.md`). Playing an event backwards maps $t$ to $T-t$, and
  $e^{j\pi(T-t)/T} = -\overline{e^{j\pi t/T}}$. **Time reversal is complex conjugation**, up to a common phase that
  cancels.
  - A template $q$ therefore matches forward plays through $|\langle s, q\rangle|$ and backward plays through
    $|\langle s, \bar q\rangle|$.
  - The split alternates two steps: label each event's direction, and refit the template on the direction-aligned
    events.
  - Its checks:
    - a per-event gate against a shuffle within the event;
    - a held-out binomial test;
    - a guard, so that the template replaces the component's order only if they agree (|Spearman| ≥ 0.6).
- **Templates for sequences that share cells** (`cpca/mixture.py`, `docs/sharing.md`; called "Step 3" in the docs).
  - **Starts:** one template per kept component, started from the components and from three ICA rotations.
  - **Fit:** each event is assigned to the template it matches best either way round, if it beats a within-event
    shuffle. Each template is refitted on its events with backward ones conjugated. Small templates are dropped and
    near-identical ones merged.
  - **Kept:** only templates that reproduce on both halves of the events.
- **Conjugate-pair read-out** (`cpca/pairs.py`). From each pair of components it takes the combination whose phases are
  most spread around the circle. It is meant for the case in Section 2.7c.

### 2.7 Three properties of PCA that limit separation

These are mathematical facts, and they explain most of Section 5.11.

- **a. Near-equal eigenvalues leave the components undetermined.** If two patterns are replayed about equally often,
  their eigenvalues are nearly equal. Any rotation of the two eigenvectors within their plane is then almost as good a
  solution, so noise decides which mixture comes out. Background structure (assemblies, per-cell latencies) adds
  further components of similar size that can mix in too.
- **b. Two maps on the same cells are not orthogonal.** With positions uniform on the half-circle,
  $\mathbb{E}\,e^{j\pi(p_1-p_2)} = (2/\pi)^2 \approx 0.4$. PCA then returns their sum and difference, not the maps
  (`docs/sharing.md`).
- **c. Balanced forward and backward plays split into a pair.** A forward replay adds $qq^H$ to $C$ and a backward one
  adds its conjugate. In equal numbers the replay term is $\mathrm{Re}(qq^H) = cc^T + ss^T$, with $q = c + js$.
  - This is a near-degenerate pair of real-valued components, each with about half the eigenvalue.
  - A single component of the pair has phases 0 or $\pi$ only, so its order is lost. The pair together still spans
    $q$.
  - An earlier statement that balanced directions "cancel" was wrong: they split, they don't cancel
    (`docs/segments.md`, addendum).

---

## 3. Data

### 3.1 Planted data (the routes are known)

| Generator | What it plants | Used for |
|---|---|---|
| `sim/routes.py`, conditions A–H | 80 cells, 3000 events. A–F are clean: one or two routes, forward and backward, 15–100% replay. G and H are realistic: two direction maps, partial replays, assemblies, non-place cells, 15% and 40% replay | First benchmark, comparisons |
| `sim/routes.py`, ladder L0–L6 | 80 cells (60 on the route), 1500 events, 40% replay. L0: one map, both directions. L1: two maps on the same cells, forward only. L2: as L1, both directions. L3–L6 add partial replays, assemblies, variable pacing, or partial replays with assemblies (L6 = H) | Which difficulty costs which method |
| `sim/segments.py` | 120 cells, 2500 events, constant-speed partial replays of one route. Per-cell background participation (5–30%) and latency, 6 random assemblies, windows that follow the activity, 5–13 ordered cells per replay, 0–10% replay | Direction analyses, read-out check |
| `sim/two_routes.py` | As `segments.py`, with two maps of 40 cells: on different or the same cells, at given replay shares | Separation benchmark (Section 5.11) |

### 3.2 Recordings

| Dataset | Animals, sessions | Cells | Events | Role |
|---|---|---|---|---|
| **hc-11** (Grosmark & Buzsáki 2016) | 4 rats (Achilles, Buddy, Cicero, Gatsby), 8 sessions. Five 1.6–2 m linear tracks and three circular tracks, all novel; PRE, MAZE, POST | 13–77 place cells per running direction (median about 24); only the two Achilles sessions have 50 or more | About 1,500–6,000 NREM bursts per epoch | Main sleep dataset |
| **hc-5** (Pastalkova et al. 2008) | 1 rat, 6 sessions | 42–72 CA1 cells | 7–25 wheel runs of 13–20 s per session | Real sequence in every event, with an answer key (time fields) |
| **DANDI 000552** (Huszár et al. 2022) | 11 mice, 65 sessions, familiar figure-eight maze | Median 57 place cells (20–178) | Median about 1,690 bursts per epoch | Second dataset, more cells |
| **DANDI 001695** (González & Vöröslakos 2026) | 4 mice, 15 sessions, novel 1.1 m linear track, Neuropixels 2.0 | 44–173 CA1 pyramidal cells; 11–72 place cells per direction | 444–2,523 bursts per epoch | Twice hc-11's place cells active per event |

**The answer key on hc-11.** Place-field centres from running faster than 5 cm/s, per running direction. A field needs
at least 40 spikes and a 3 Hz peak. The paper's criterion keeps about 1.5 times as many cells (`docs/replay.md`).

---

## 4. The tests, and what each checks

The tests differ in what they can tell apart, and most of the corrections in this project came from reading one as
another.

| Test | Question it answers | What can pass it without a sequence | Code |
|---|---|---|---|
| **Order test, one output** | Do this output's members fire in the map's order? Members' relative times are compared with their field centres (\|Spearman\|), against field centres permuted among cells with a field. Circular tracks search 12 rotations, for data and null alike. At least 6 members with a field | Co-firing that follows the map; per-cell latencies that follow the map | `cpca.evaluate.order_vs_place` |
| **Best-of-outputs test** | Is the map's order present among a method's outputs? The best output, map and sense, against a null that takes the same best, so extra outputs gain nothing | As above, and it says nothing about the other outputs | `scripts/plot_hc11_sequences.best_map_match` |
| **Per-component criterion** (the goal) | Is each output one sequence? Planted: at least 80% of the members on a route are on the same route, and the order matches it. Real: matches exactly one running direction | On real data, low power against the second map (Section 5.11) | `scripts/run_separation.py`, `scripts/rescore_components.py` |
| **Timing control** | Is the order more than co-firing? The real fit must beat refits on events whose spike trains are shuffled among each event's active cells (who fires is kept; who fires when is destroyed) | Per-cell latency, meaning each cell firing early or late in every burst | `scripts/run_maze_rest.shuffle_timing` |
| **Latency check** | Does an output carry timing that varies between events? Its expression on held-out events must beat events in which each cell's relative spike times are redrawn from that cell's own pooled times | Nothing seen so far. It also rejects a sequence played identically in every event, so it is used only for reactivations | `cpca/readout_check.py` |
| **Fragment criterion** | Is there map order within one 40% stretch of the track (at least 8 members), beyond a null that also searches stretches, plus the timing control? | Credits partial replays; co-firing fails the timing part | `cpca/fragment.py` |
| **Topology (Mantel)** | Are cells that are close in the reduced space also close on the waking map (direction-resolved rate maps)? Spearman over cell pairs, against permuted cell identities | Co-firing of neighbouring cells | `cpca/topology.py` |
| **Waking against sleep** | Do the waking-lap fit and the sleep fit place cell pairs alike? | Co-firing | `scripts/run_wake_sleep.py` |
| **Containment** | Does each waking component lie in the span of the sleep components (and their conjugates), beyond permuted cell identities? | Co-firing (by design: it is a sanity check) | `cpca/containment.py` |
| **Bayesian replay decoder** | Which events replay the map? Uses the map as input | – | `cpca/replay.py` |
| **Spike-in power test** | How much replay does a method need? Turns a share of real PRE events into ideal replays and rescores | – | `scripts/run_spikein_hc11.py` |

**PRE as control.** PRE was recorded before the animal saw the maze, so a map match in PRE is chance or pre-existing
structure. A result specific to experience needs POST above PRE.

---

## 5. Results

### 5.1 Planted data, order present (best output per route)

- **Clean conditions A–F** (`results/synthetic/summary.md`; 10 draws, 80 cells, 3000 events). Some component recovers
  every route, at median |ρ| 0.97–1.00 for routes on different cells and 0.87–0.92 for routes sharing half their cells.
  This includes condition F: two routes, both directions, 15% replay, 15% of cells per event.
- **Centring.** In A (every event a replay), centred PCA keeps nothing, because the route is the mean pattern.
  Uncentred recovers it at 1.00.
- **Realistic conditions G and H.** Median |ρ| 0.37–0.56. A route is above chance in 70–100% of (route, draw) pairs.
- **How these numbers were scored** (`scripts/run_synthetic.py::score_routes`):
  - each route separately gets the component that correlates best with it, so the same component may be picked for
    both routes;
  - "above chance" compares that component with its own null, which does not account for having picked the best of
    k components (k is 2–12).
  - **Consequence:** the percentages above chance are inflated, most for G and H, where |ρ| is low. The clean
    conditions (|ρ| 0.97–1.00) are far above any null.
  - **The same per-output null** is used for the planted comparison in the matched rerun (`scripts/run_rerun.py::score_orders`),
    for every method alike.
  - **The later tests correct for picking the best:** the ladder, hc-5 and the hc-11 per-session tests use a null that
    takes the best over all outputs. They still let one output serve two routes.
- **Ladder** (`results/ladder/summary.md`; 3 draws; centred complex PCA, median |ρ| and maps recovered):

| Rung | complex PCA | PP-Seq | seqNMF |
|---|---|---|---|
| L0 one map, both ways | 0.90 (3/3) | 1.00 (3/3) | 0.97 (2/3) |
| L1 two maps, same cells, forward | 0.88 (6/6) | 1.00 (6/6) | 0.99 (4/6) |
| L2 as L1, both ways | 0.83 (6/6) | 1.00 (6/6) | 0.47 (4/6) |
| L3 L2 + partial replays | 0.59 (4/6) | 0.35 (3/6) | 0.08 (0/6) |
| L4 L2 + assemblies | 0.85 (6/6) | 1.00 (6/6) | 0.36 (4/6) |
| L5 L2 + variable pacing | 0.81 (6/6) | 0.99 (6/6) | 0.26 (3/6) |
| L6 partial + assemblies (= H) | 0.48 (5/6) | 0.39 (3/6) | 0.15 (0/6) |

**Reading.**
- **Whole routes:** PP-Seq is essentially perfect, and complex PCA trails by 0.1–0.2. The gap is already there with
  one map (L0).
- **Partial replays:** these are hard for every method.
- **An earlier claim:** "PP-Seq is clearly ahead on H" came from one favourable draw (0.89/0.81). It doesn't hold over
  three draws.

**The template step** (`docs/sharing.md`) closes the whole-route gap: 1.00 on L1, L2, L4 and L5, as PP-Seq. On partial
replays it gives 0.70 (L3) and 0.81 (L6), but as several segment templates, with some false templates that mix
segments. It takes 3.5 s per fit against PP-Seq's 116 min.

### 5.2 Comparison with other methods (`docs/baselines.md`, `docs/methods_hc11.md`)

Each method was run with its authors' code and procedure and first checked on planted data. On hc-11, all were given
the same 1,500-event subsets and scored by the same functions.

**Comparison setup:**
- **The matched rerun (19 October)** fixed six flaws found in review: different event sets per method; a PP-Seq sweep
  that never varied the number of sequence types; per-candidate masks; a mislabelled statistic; a wrongly attributed
  membership rule; and a one-bin shift in the seqNMF cost helper.
- **The PP-Seq wrapper** had a mask bug (`docs/benchmark_hc5.md`, implementation notes), now fixed.
- **SPOTDis:** the authors' pooled function passes its arguments in the wrong order. It is fixed in our copy.

**Results on hc-11 sleep:**
- **Best-of-outputs test per session (POST / PRE):**
  - complex PCA 1/8 and 0/8 (Achilles 10/25);
  - Rastermap 1/8 and 0/8;
  - seqNMF, PP-Seq and SPOTDis 0/8 and 0/8;
  - a random order 0/8.
- **SPOTDis** finds no clusters of events in 12 of 16 epochs.
- **Count ICA's most local assembly** is significant in 2/8 POST and 1/8 PRE.
- **Runtime per fit:**
  - count ICA 0.3 s;
  - complex PCA 0.5 s;
  - Rastermap 5.5 s;
  - SPOTDis 18 s;
  - seqNMF 117 s;
  - PP-Seq 1073 s.
- **Component-level tests,** all events, centred complex PCA: 20 of 120 (component, direction) tests match the map in
  POST (chance about 6) against 2 of 106 in PRE (about 5). This is carried by the two Achilles sessions. On the matched
  subsets it is 14/96 against 6/74. PP-Seq gives 2/47 against 2/44.
- **HMM:** it calls 14–30% of events congruent in POST and 10–32% in PRE, so it is not maze-specific.

### 5.3 Real sequences present in every event

**hc-5 wheel runs** (`docs/benchmark_hc5.md`).
- **Design:** fit on half of a session's runs, score against time fields from the other half. 6 sessions × 3 splits
  × 2 folds = 36 fits per condition. Best-of-outputs test.
- **Results:**

| Windows | Averaging the runs (reference) | complex PCA, uncentred | seqNMF | PP-Seq |
|---|---|---|---|---|
| Aligned on wheel onset | 36/36, ρ 0.71 | 32/36, ρ 0.58 | 29/36, ρ 0.60 | 14/36, ρ 0.55 |
| Unaligned (random start before onset) | 32/36, ρ 0.47 | 35/36, ρ 0.63 | 26/36, ρ 0.52 | 13/36, ρ 0.43 |
| Shuffled control | 0/12 | 0/12 | 0/12 | 0/12 |

- **Reading:** complex PCA is the only method that keeps its performance without a shared clock, because the lag
  matrix ignores each event's clock. It takes 50 ms per fit, against 50 s for seqNMF and 16 min for PP-Seq.
- **Not testable:** the upcoming left/right choice could not be decoded from wheel activity in these sessions, even
  by a supervised decoder, so choice-specific sequences were not tested.

**hc-11 waking laps** (`docs/laps_hc11.md`). Same design, with laps (runs in one direction over at least 60% of the
track), 6 fits per session.

| Method | Linear, 5 sessions | Circular, 3 sessions | All |
|---|---|---|---|
| complex PCA, uncentred | 19/30 | 18/18 | 37/48 |
| complex PCA, centred | 15/30 | 11/18 | 26/48 |
| count PCA | 5/30 | 10/18 | 15/48 |
| Averaging, told each lap's direction | 25/30 | 16/18 | 41/48 |

- **Reading:** on the same cells whose sleep reactivations no method decodes, the waking order is recovered from
  timing, and count PCA mostly fails on linear tracks.
- **Per-component scoring** changes this picture for linear tracks (Section 5.11).

### 5.4 hc-11 sleep, order tests

- **Component-level tests and per-session best outputs:** as in Section 5.2.
- **The read-outs of the whole reduced space** (`docs/subspace.md`) give POST / PRE sessions passing:
  - spectral seriation 1/8 and 0/8;
  - angular synchronisation 2/8 and 0/8;
  - the best single component on all events 2/8 and 0/8.

  No read-out beats the best single component, and they disagree on which sessions pass.

### 5.5 The timing control, and what it changed (`docs/maze_rest.md`)

- **Awake rest on the maze,** where the decoder finds the most replay (2–18% clear replays per session):
  - centred complex PCA's best component matches the map in 5/8 sessions, against 0/8 in a matched PRE control;
  - count PCA seriation matches it as often as complex PCA seriation (4/8 each).
- **The timing control was added after these results,** because count PCA matched complex PCA:
  - **co-firing alone passes the order test:** in the two Achilles sessions, refits on timing-shuffled events match
    the map in 87–98 of 100 shuffles;
  - **order beyond co-firing** appears in 2–3/8 sessions: Gatsby 08/02 under both conventions, plus Cicero 09/10 or
    09/17 under one;
  - **the one sleep pass** (Achilles 10/25 POST, p = 0.005) has a timing p of 0.11, so it is not shown to come from
    timing.
- **Likely mechanisms:**
  - **centred:** silent cells enter the covariance, components contrast groups that fire in different events (such as
    the two ends of the track), and the read-out turns the sign contrast into early and late;
  - **uncentred:** the shuffle keeps each event's timing profile.
- **Consequence:** every hc-11 order result above, apart from this control, is "order present, possibly from
  co-firing".

**Fragment criterion** (`docs/fragment.md`):
- **Planted data:** it passes whole and partial replays (4–5/5 draws) and rejects map-ordered assemblies without order
  (0/5).
- **hc-11:** nothing passes in POST, PRE or maze rest.
- **Power:** at hc-11's size (25 route cells), even 20% planted segment replays pass in only 1 of 4 draws, so the hc-11
  null result is uninformative.

### 5.6 Population-level structure

These measures use all kept components at once and check co-firing as much as order.

- **Topology** (`docs/topology.md`):
  - **centred complex PCA:** above chance in 5/8 POST and 1/8 PRE sessions;
  - **count PCA,** which has no timing: 5/8 and 1/8, so the sleep topology is mostly co-firing;
  - **where only timing can carry the order** (planted whole-route replays; hc-5), complex PCA passes and count PCA
    fails (hc-5 uncentred 35/36 against 4/36).
- **Waking fit against sleep fit** (`docs/wake_sleep.md`):
  - **planted data:** agreement in 10/10 draws with the same maps and 0/10 with other maps;
  - **hc-11:** 4/8 POST and 1/8 PRE (binomial p = 0.0004 for POST).
- **Containment of waking components in the sleep span** (`docs/containment.md`, final design):
  - **planted data:** passes for both maps, one map and segments (5/5 each); fails for other maps on the same cells
    (1/5); passes for map-ordered co-firing (5/5) by design;
  - **hc-11 POST:** 6/8 sessions, 12 of 18 waking components contained individually;
  - **the shares are modest:** 0.45–0.69 against 0.35–0.60 by chance.

  This is a sanity check of the implementation, not evidence of sequences.

### 5.7 How much replay hc-11 sleep contains, and how much the method needs

- **The paper's Bayesian analysis reproduced** (`docs/replay.md`):
  - sequence content is higher in POST than PRE (7/8 sessions; pooled one-sided p = 0.002) and highest at rest on the
    maze;
  - our burst windows carry the same content as the paper's events;
  - clear replays (significant under both shuffles) are 2–8% of POST events;
  - **POST exceeds PRE by a median of 0.8 and at most 3.4 percentage points.**
  - **Deviation:** the files have no LFP, so the paper's ripple condition can't be applied. The replay share may be
    diluted by up to about a factor of two.
- **A supervised ceiling** (`docs/ceiling.md`). Half of the place cells select events whose order matches the map, and
  the other half is scored. The excess of selected events over chance is:
  - +0.3 points in hc-11 POST and 0.0 in PRE, and 0.0 on 000552;
  - against +14 to +41 points for planted whole-route replays.

  Single events in these recordings rarely carry whole-event map order.
- **Spike-in power test** (`docs/replay.md`, `docs/power.md`). A share of real PRE events is turned into ideal replays:
  each active place cell keeps its spike count, and its spikes move to its field position, forward or reverse.
  - **Centred complex PCA:**
    - needs about 10% in the two Achilles sessions (9/10 draws pass at 10%, 1/10 at 5%);
    - needs more than 20% in sessions with 3–5 active place cells per event;
    - real awake replays added at 1–3% are never found.
  - **Uncentred:** the same threshold, with fewer false positives at 0% (0/10 against 2/10).
  - **Rastermap:** 1/10 at 10%.
  - **PP-Seq, at a fixed setting:** 0/2 even at 20%.
  - **The threshold compared with the data:** about 10% needed with ideal replays, against an observed excess of
    0.8–3.4%. The ideal replays make this an optimistic threshold.
- **Perfect event choice** (`docs/power.md`, Step 0). The decoder picks the replays with half of the cells, and the
  chosen events are averaged on the other half. This recovers the map in at most 3/10 splits in any session, while
  complex PCA without the map, scored on the same cells, reaches 5/10 in the best session. Knowing which events are
  replays does not give a better order on these events.

### 5.8 Direction of replay

- **The direction split** (Section 2.6; `docs/direction.md`):
  - **planted, one route played both ways:** it restores the order (median |ρ| 0.74–0.81 → 1.00) and labels every
    event's direction correctly (median 100%);
  - **planted controls:** no reverse group in forward-only data, nothing in a no-order control;
  - **partial replays:** labels are at chance (54–60%);
  - **hc-11, Achilles 10/25 POST:** a template learned without the map labels 43 of 46 significant replays in the
    map's direction;
  - **hc-11, small sessions:** labels are at chance, or no template matches the map.
- **Direction balance on hc-11** (`docs/segments.md`): the forward share of clear replays is 0.47–0.66 in POST and
  0.50–0.83 on maze rest. Sleep is close to balanced, which is the case of Section 2.7c.
- **Label-free alternatives on the partial-replay grid** (`docs/segments.md`; five development variants and a formal
  grid of 5–13 ordered cells per replay at 2–10% replay):
  - **No label-free variant beat complex PCA.** The variants were a pairwise-order SVD, a PCA of standardised spike
    times, a gated refit, a standardised lag matrix, a direction EM, and a lead–lag component.
  - **Planted directions as an oracle:** the uncentred fit recovers the order in nearly every grid cell (29/30 in
    development).
  - **Why the oracle wins:** a centred oracle gains only about a factor of two in replay share. The uncentred oracle's
    large margin comes from a read-out that needs the true directions.
  - **Real hc-11 events:** forward-only spike-ins have the same 10% threshold as balanced ones, so direction balance
    is not what limits the centred pipeline there.
  - **Conjugate-pair read-out:** the only label-free gain on real events (5/10 at 5% against 1/10). It seriates
    co-firing as readily as timing, so it needs the timing control.
- **The HMM on the grid:** it calls 85–89% of events congruent, with false alarms at 0.86–0.89, so it does not
  separate replays there.

### 5.9 Other datasets

- **DANDI 000552** (`docs/huszar.md`; 65 sessions, familiar maze):
  - **population-level measures:** strong in both sleeps (topology 89% POST and 78% PRE; waking-sleep agreement 80%
    and 74%), as expected for a familiar maze;
  - **sequence-level measures:** weak and no better than hc-11 (component tests 9.9% POST and 6.2% PRE, chance 5%;
    best output 12% and 8% of sessions);
  - **cell count:** no relation between the number of place cells and the best component's significance.
- **DANDI 001695** (`docs/dandi001695.md`; pre-registered bar: at least 3 of 15 POST sessions passing the order test
  and its timing control, above PRE, in more than one mouse):
  - **primary outcome:** POST 3/15 (three mice; binomial p = 0.036) against PRE 0/15, which meets the bar at its
    smallest margin;
  - **secondary outcomes do not reproduce it:** fragment criterion 1/15, ripple-gated events 1/12 against 1/12, CA1 +
    CA3 1/15 against 1/15;
  - **replay content:** clear replays are about 3.4% POST and 3.3% PRE, so there is no POST excess;
  - **reading:** a small, fragile improvement; twice hc-11's cells per event did not move the method clearly above
    threshold.

### 5.10 A sanity check of the implementation

Containment and the waking-against-sleep agreement (Section 5.6) pass where expected on planted data. On hc-11 they
pass in most POST sessions. The implementation is therefore not broken. Neither check says whether the shared
structure is order or co-firing.

### 5.11 Scored per component (the goal; 5 November)

**A first planted check** (`docs/goals.md`; 4 draws; "both routes on separate, well-matched components"):

| Version | Two routes, different cells | Two maps, same cells, forward | Same, both directions |
|---|---|---|---|
| Centred, no rotation | 2/4 | 2/4 | 1/4 |
| With ICA rotation | 1/4 | 4/4 | 2/4 |

**Step 1: a read-out check** (`docs/readout_check.md`; `sim/segments.py`, 6 draws). Each design was written down
before it was run.
1. **Circularity failed.** Circularity is $|\sum w^2| / \sum |w|^2$ over members: 1 for real-valued components.
   Route and non-route components had the same values (0.81–0.86).
2. **A spike-train shuffle between cells failed.** It kept 98–100% of components, even from no-replay data, because
   every cell has its own latency within bursts, and the shuffle destroys that.
3. **The latency check passed for three versions,** with the bar fixed in advance: keep at least 80% of route
   components and reject at least 90% of components from the controls (no replays; map-ordered assemblies).

| Version | Route components kept | Control components rejected |
|---|---|---|
| Centred PCA | 91% | 98–100% |
| Centred ICA | 86% | 100% |
| Uncentred ICA | 86% | 97–100% |
| Uncentred PCA | 76% (fails) | 97–100% |

It also keeps 28–97% of non-route components in replay data, which are a separation problem rather than a read-out
problem.

**Step 2: separation benchmark** (`docs/separation.md`, `results/separation/summary.md`).
- **Data:** `sim/two_routes.py`, two maps of 40 cells among 120, partial replays of 50–100% of the route,
  background structure, 5 draws × 2 folds. Outputs are those passing the latency check on held-out events.
- **Criterion:**
  - maps on different cells: at least 6 members on one map, at least 80% of map members on that map, and order
    p < 0.05;
  - maps on the same cells: exactly one map matches.
- **Clean fit:** every map represented and every output passing.

| Condition | Centred PCA | Centred ICA | Uncentred ICA | Templates |
|---|---|---|---|---|
| A: different cells, 10% + 10% | 2/88 pass, 0 clean | 0/90, 0 | 0/84, 0 | 1/30, 0 |
| B: different cells, 13% + 7% | 10/86, 0 | 0/89, 0 | 0/77, 0 | 4/30, 0 |
| C: same cells, forward | 22/64, 0 | 25/70, 0 | 41/75, 0 | 19/25, 7 |
| D: same cells, both ways | 34/59, 1 | 38/74, 0 | 40/81, 0 | 26/31, 6 |
| E: one map, 20% | 32/65, 0 | 44/67, 2 | 27/72, 0 | 11/18, 4 |
| F: no replays (clean = no outputs) | 6 | 8 | 8 | 10 |

- **Diagnostic** (added after the results), centred components in A:
  - they spread their weight over both routes (for example 0.43 and 0.32) and off-route cells (0.15–0.48);
  - they have 50–100 members of 120 cells;
  - even with one map (E), only the first component is dominated by the route (90–94% of its weight).

  The failures are real mixing, not a scoring artefact.

**Count PCA and ICA on the same planted data** (`docs/separation.md`, diagnostic 2; membership only):
- **Count PCA mixes the two routes too.** Count ICA unmixes them: both routes land on a one-route assembly in 5/5 fits.
- **Complex PCA with its 20% membership rule** has 74–78 of 120 cells per component and is mixed by membership in
  every component. With the count methods' sparser rule (mean + 2 SD), both routes are on a one-route output in 2–3
  of 5 fits.

**Step 3: real data re-scored per component** (`docs/rescore.md`, `results/rescore/summary.md`).
- **hc-5** (best-of-outputs replaced by per component; no latency check, since the sequence is identical in every
  run):

| Windows | Convention | Components that are the sequence | Fits where every component is |
|---|---|---|---|
| Aligned | uncentred | 34/46 (74%) | 27/36 |
| Aligned | centred | 3/25 (12%) | 3/36 |
| Unaligned | uncentred | 36/42 (86%) | 30/36 |
| Unaligned | centred | 11/34 (32%) | 11/36 |

- **hc-11 laps,** uncentred: a component passes if it matches exactly one running direction's held-out fields.
  - **circular tracks** (Achilles 11/01, Cicero 09/10, Gatsby 08/28; one direction): 12/12, 6/9 and 10/12 components,
    28/33 in all *(total computed for this summary)*;
  - **linear tracks:** Achilles 10/25 0/12, Buddy 3/12, Cicero 09/01 9/12, Cicero 09/17 6/11, Gatsby 08/02 7/12;
    25/59 in all *(total computed for this summary)*.

  A diagnostic after the results corrects how these linear-track numbers read:
  - **The components combine the two running directions.** One orders cells by time since the start of the lap
    (opposite signs against the two maps; Achilles 10/25: 0.51 and 0.61). The other orders them by position (same
    sign; 0.58 and 0.57). These are the sum and difference of Section 2.7b, on real data.
  - **The two maps are not alike:** field centres correlate at only ρ 0.30 over 43 cells.
  - **Smaller sessions** show the same pattern more weakly, so the second map often misses significance and the
    component passes. **The linear-track passes overstate separation.**
- **hc-11 sleep,** all NREM bursts, 2 folds. Outputs passing the latency check, scored by "matches exactly one
  running direction":

| Epoch | Method | Pass / outputs | Expected by chance | Sessions with at least one |
|---|---|---|---|---|
| POST | centred PCA | 18/87 (21%) | 8.2 | 8/8 |
| PRE | centred PCA | 7/73 (10%) | 6.9 | 5/8 |
| POST | templates | 6/67 (9%) | 6.1 | 6/8 |
| PRE | templates | 6/63 (10%) | 5.9 | 5/8 |

- **The chance rate** is about 9.5% per output (exactly one of two tests at p < 0.05). The probability of 18 or more
  in POST is 0.0013, treating outputs as independent, which they are not within a session.
- **Per session in POST:** 6 of 18 outputs in Achilles 10/25, and 1–3 in the others.
- **The latency check** was calibrated on planted data only; whether it removes all co-firing structure on hc-11 has
  not been checked.

---

## 6. Discussion

### 6.1 What is established

1. **One dominant sequence in every event is recovered, on real data, by uncentred complex PCA:**
   - hc-5: per component 74–86%, every component correct in 27–30 of 36 fits; 35/36 best-output fits without a shared
     clock;
   - hc-11 circular-track laps: 28/33 components;
   - it takes about 50 ms per fit with no settings to choose. On hc-5 it beats seqNMF and PP-Seq without a shared
     clock. On the laps (where only count PCA was compared) it clearly beats count PCA.
2. **On planted reactivation-like data with whole-route replays,** each route's order is present among the components.
   With the template step for maps on shared cells, the best outputs match PP-Seq at a fraction of the cost.
3. **On hc-11 sleep, POST differs from PRE** in several measures:
   - component map tests: 20/120 against 2/106;
   - topology: 5/8 against 1/8;
   - waking-sleep agreement: 4/8 against 1/8;
   - per-component exactly-one-map: 18/87 against 7/73.

   On topology, count PCA does as well, and on maze rest the order test is passed by co-firing. Most of the POST
   structure is who fires with whom.
4. **hc-11 sleep carries little replay** beyond the pre-maze level (at most 3.4% of events). That is below the
   spike-in threshold of the best label-free method tested (about 10% with ideal replays). Perfect event selection did
   not beat complex PCA on these events.

### 6.2 What is not established

1. **The goal itself:** separating several routes onto one component each. No variant tested does it for routes on
   different cells (0–13% of outputs). For maps on the same cells, only the template step comes close.
2. **Order beyond co-firing in hc-11 sleep:** 0/8 sessions with the timing control. On DANDI 001695, 3/15 POST pass,
   which is fragile.
3. **That more cells help:** DANDI 000552 and 001695 did not show it.
4. **That the per-component sleep result (21% in POST) is made of sequences:** it is "timing beyond per-cell latency
   plus a match to one direction". The other 79% of outputs are not interpreted.

### 6.3 Why separation fails: measured and hypothesised

- **Measured:**
  - components spread weight over both routes and background cells in planted data;
  - the linear-track laps come out as a sum and difference;
  - maps on the same cells give sum and difference (Section 2.7b, `docs/sharing.md`).
- **Hypothesised, not tested:**
  - in planted condition A the cause is near-equal eigenvalues of the two routes plus background components of similar
    size (Section 2.7a);
  - the broad membership (50–100 of 120 cells at the 20% threshold) adds to it.

  ICA rotation did not help there (0/90). The cause of its failure is not known.

### 6.4 Limitations of the evidence

- **Statistics:**
  - Sessions are treated as independent, but hc-11 has 4 rats (2, 1, 3 and 2 sessions), DANDI 001695 has 4 mice and
    000552 has 11.
  - Many tests were run without multiple-comparison correction.
  - Per-animal reporting and corrections are still to be done for anything going into a paper.
- **Criteria added after results:** these are marked in the docs:
  - the timing control at maze rest;
  - the per-component purity diagnostic;
  - the lap diagnostic in this summary;
  - the direction-balance analysis.
- **Generators:** one per question; results may depend on its choices (constant speed, 8 ms jitter, assembly model).
- **Reproducibility:**
  - `scripts/rescore_components.py` seeds with Python's `hash()`, which changes between processes, so its exact
    numbers will not reproduce (to be fixed with CRC32, as in the other scripts);
  - PP-Seq's sampler is not deterministic given the seed on this machine.
- **The answer key:** our place-cell criterion keeps fewer cells than the paper's, and the circular-track rotation
  search costs power. Both apply to all methods alike.
- **The latency check:** calibrated on one generator only.

### 6.5 Open questions

1. **Separating routes on different cells.** Can it be done label-free at all at realistic replay shares, or is the
   rotation ambiguity irreducible with second-order statistics? Candidates not yet tested:
   - sparser rotations or other read-outs;
   - statistics beyond pairs;
   - latent-trajectory models.
2. **The other 79% of POST outputs.** They pass the latency check but don't match the waking map. Whether they are
   reproducible sequences of another kind is the subject of the draft design in `docs/sequence_discovery.md`, which has
   not been run.
3. **Data with more replay per event.** Rest on the maze, early POST, or recordings with hundreds of place cells.
   Not to be started without a decision.

---

## 7. Parameters that were chosen, and why

| Choice | Value | Evidence |
|---|---|---|
| Phase scale $T$ | event duration, window centred on the peak | Section 2.2; per-event spike spread gave no gain on hc-11 |
| Centring | centred for sleep, uncentred when the pattern is in every event | Synthetic A–C against G; hc-5; laps. On sleep spike-ins both have the same threshold, so the choice does not limit sleep results (`docs/power.md`, addendum) |
| Noise bound | shuffle null, 100 shuffles, 95th percentile | `figures/noise_bound.png` |
| Members | weight at least 20% of the largest | Not tuned; broad membership is a suspected contributor to mixing (Section 6.3) |
| ICA | off by default before November | It helped only for a union of routes sharing cells in the first benchmark |

---

## 8. Corrections made during the project

| Date | What was wrong | What changed |
|---|---|---|
| September | The evaluation scored the first component's whole-population order against the map; baselines were scored on quantities they don't produce | Read all components; baselines rerun with the authors' procedures (`docs/lessons_learned.md`) |
| September | $T$ was the window length, so forward and backward plays merged | The half-circle rule |
| September | A direction split by the sign of $\mathrm{Im}(a_1\overline{a_2})$ had no null and gave 50/50 splits on shuffled data | Removed |
| September | Seeds from Python's `hash()` moved a result from 0.72/0.80 to 0.42/0.34 between runs | CRC32 seeds and repeated draws (the new `rescore_components.py` repeats this mistake; Section 6.4) |
| 19 Oct | Six flaws in the method comparison | Matched rerun (`docs/baselines.md`) |
| 19 Oct | PP-Seq wrapper mask bug | Fixed; affected hc-5 only |
| 21 Oct | The Marchenko–Pastur bound kept noise components for uncentred data; uncentred hc-11 went from 14/134 to 3/54 | Shuffle null |
| 21 Oct | "PP-Seq is ahead on H" came from one draw | Ladder with 3 matched draws |
| 22 Oct | A direction test that could not fail (its null was 1.00) | Held-out per-event test |
| 29 Oct | The order test was read as evidence of timing; co-firing passes it | Timing control; fragment criterion |
| Early Nov | "Balanced directions cancel" | They split into a degenerate pair (Section 2.7c) |
| Early Nov | "Every label-free model family fails" | Restricted to the families actually tested |
| Early Nov | Containment was first designed symmetric, then sleep-inside-waking | Final design: waking inside sleep |
| Early Nov | The replay shortfall was called a biological limit, and the 3–10× figure was overstated | It is a measured shortfall on these recordings, with an optimistic ideal-replay threshold |
| 5 Nov | The first planted benchmark scored each route by its best component against that component's own null (selection not corrected; one component could serve two routes) | Stated in Section 5.1; later tests use a max-over-outputs null |
| 5 Nov | Best-of-outputs scoring hid that components are not one route each | Goal fixed per component; read-out check, separation benchmark, re-scoring |
| 5 Nov | The linear-track lap failures were attributed to similar maps (ρ 0.30 measured) | They are sum-and-difference mixtures of the two directions (Section 5.11) |
| 5 Nov | The sleep chance rate was quoted as "about 10%, p ≈ 0.002" without computing it | Computed: 8.2 expected, p = 0.0013 |

---

## 9. Where to find things

- **Goal:** `docs/goals.md`. **Method:** `docs/method.md`. **Status and decisions:** `docs/plan.md`. **History:**
  `docs/lessons_learned.md`.
- **Per-component work:**
  - `docs/readout_check.md` with `cpca/readout_check.py` and `scripts/calibrate_readout.py`;
  - `docs/separation.md` with `sim/two_routes.py` and `scripts/run_separation.py`;
  - `docs/rescore.md` with `scripts/rescore_components.py`.
- **Planted benchmarks:**
  - `results/synthetic/summary.md`, `results/ladder/summary.md`;
  - `docs/sharing.md`, `docs/direction.md`, `docs/segments.md`.
- **Real data:**
  - hc-5: `docs/benchmark_hc5.md`;
  - hc-11 laps: `docs/laps_hc11.md`;
  - hc-11 sleep: `results/summary.md`, `docs/methods_hc11.md`, `docs/maze_rest.md`, `docs/fragment.md`,
    `docs/topology.md`, `docs/wake_sleep.md`, `docs/containment.md`, `docs/subspace.md`;
  - replay content and power: `docs/replay.md`, `docs/ceiling.md`, `docs/power.md`;
  - other datasets: `docs/huszar.md`, `docs/dandi001695.md`, `docs/datasets.md`.
- **Comparisons:** `docs/baselines.md`, `docs/methods_hc11.md`.
