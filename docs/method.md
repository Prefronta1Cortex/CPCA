# Complex PCA of spike phase: the method

Open with `Cmd+Shift+V` (classic Markdown preview) so the formulas render; see `rendering.md`.

## 1. Goal

Reactivation events in the sleeping hippocampus are brief bursts in which many cells fire. We want a small number of components, each describing a group of cells **and the order in which they fire**, obtained from the sleep spikes alone. Ideally each component corresponds to a route (or a piece of a route) the animal ran while awake. The waking place map is used only afterwards, as the answer key, to check whether a component's internal order matches the track. The method should stay as simple as the standard count-based assembly analysis it extends, with as few tunable parameters.

## 2. One arrow per cell per event

Take one event (a population burst) and one cell. Each spike at time $t$ (ms from the start of the event window) becomes a unit arrow

$$
e^{\,j\pi\,t/T},
$$

an arrow of length 1 whose angle is proportional to the spike time. The cell's spikes in the event are averaged into one arrow $s_{ik}$ (event $i$, cell $k$): a tight burst gives a long arrow at the burst time, scattered spikes give a short arrow, and a silent cell gives the number 0.

**The scale $T$ is the half-circle rule.** With $T$ equal to the event's duration and the window centred on the population peak, the event's extent maps onto the angle range $[0, \pi]$ and the peak sits at $\pi/2$ in every event. Two different orders of the same cells are then nearly orthogonal as complex vectors and PCA can give each its own component. If $T$ is much longer than the activity (for example a 100 ms scale for spikes within $\pm 12$ ms of the peak), every pattern sits in a small arc near one angle, different orders are nearly parallel, and PCA merges them. Checked on simulated routes played forward and backward (60/40): with a 40° arc the two directions are 92% parallel and land on one component; with a 180° arc they are 3% parallel and land on two components whose eigenvalues are in the 60:40 ratio.

## 3. The matrix and its covariance

Stack the arrows into $S$ (events by cells). The covariance

$$
C_{kl} = \frac{1}{M}\sum_{i=1}^{M} s_{ik}\,\overline{s_{il}}
$$

is Hermitian. Each entry is the average over events of the *lag arrow* between cells $k$ and $l$: its length says how consistently the pair keeps the same relative timing, its angle says the typical lag and who leads. Multiplying by the conjugate makes a common shift of an event's clock cancel, so $C$ does not depend on where each event's window starts.

Two conventions, and the data decide which to use:

- **Uncentred.** $C$ as written. Its first component is dominated by participation (which cells tend to fire; in sleep data its weights match participation at $\rho = 0.99$ and its angles sit in a $20°$ arc). Routes appear in the following components. This is right when the pattern of interest is common to most events (synthetic conditions where every event is a replay: perfect recovery, while centring removes the pattern entirely).
- **Centred and scaled.** Subtract each cell's mean arrow across events and divide by its root-mean-square length, the complex analogue of z-scoring in count-based assembly analysis. Components then describe how events differ from one another. This is right when replays are a minority variation on a common background, which is the sleep regime: on hc-11 the centred components match the place map in POST sleep in 20/120 tests (chance ≈ 6) against 2/106 in PRE, while the uncentred components are at chance in both (3/54 and 1/44). Centring requires aligned events (shared clock), which the peak-centred window provides.

**Silent cells.** A cell that does not fire in an event is the number 0. Uncentred, a zero contributes nothing to $C$. Centred and scaled, it becomes the cell's mean arrow reversed and divided by the cell's spread, an arrow pointing into the lower half-plane that no spike time can produce, so a centred component carries a weak "did not take part" term opposite in direction to the cell's firing. Measured on hc-11 POST (Achilles 10/25, Gatsby 08/28): silent entries carry about 10% of each cell's centred variance (interquartile range 4–15%) and 12–13% of the kept components' event-score energy, and component weights are only weakly related to participation ($r$ between −0.10 and +0.21). In the hc-5 wheel runs a typical cell fires in 98% of runs, and the fits there are uncentred, so the term does not arise.

The two conventions suit different data, not one better and one worse: uncentred is the right choice when the sequence is in nearly every event (synthetic conditions A–C; the hc-5 wheel runs, `docs/benchmark_hc5.md`), centred when it is a minority of events (sleep).

## 4. Components

Eigen-decomposition $C\,p_n = \lambda_n p_n$ gives components $p_n$ (one arrow per cell) with eigenvalues $\lambda_n \ge 0$. For cell $k$ in component $n$: $|p_{nk}|$ is its weight (participation), and its relative firing time is

$$
\tau_{nk} = \frac{T}{\pi}\arg p_{nk}
$$

up to one common offset per component. **Read all kept components, not the first.** In hc-11 POST sleep 5 to 11 centred components pass the noise bound per session; the route-like ones are rarely the first.

**How many components are real (the noise bound).** Noise alone spreads the eigenvalues out, so a component counts only if it is larger than the largest eigenvalue noise can produce. $C$ is Hermitian, so its eigenvalues are real. The bound is set by a shuffle null (permutation parallel analysis; Horn, 1965; Buja & Eyuboglu, 1992): each cell's arrows are permuted across events independently, 100 times, which keeps everything about each cell (participation, arrow lengths, mean arrow, phase distribution, the half-circle) and destroys only the coordination between cells; a component is kept if its eigenvalue exceeds the 95th percentile of the shuffle's largest noise eigenvalue (`cpca/decompose.py::n_above_null`). For uncentred arrows the shuffle keeps each cell's mean arrow, so its first eigenvalue is the mean pattern rather than noise; the first component is then always kept (in the hc-5 wheel runs the mean pattern *is* the sequence) and later components are tested against the shuffle's second eigenvalue.

The textbook Marchenko–Pastur bound $\bar\lambda\,(1+\sqrt{N/M})^2$, the largest eigenvalue of independent noise with the same variance in every cell, gives the same count as the shuffle for centred and scaled arrows (zero mean, unit variance per cell; hc-11 Achilles 10/25 POST: 1.42 × the mean eigenvalue for both), complex values and half-circle notwithstanding. It fails for uncentred arrows, whose cells have very different variances (participation) and a non-zero mean: on independent synthetic noise it keeps 16 components where the shuffle keeps only the mean pattern, and on hc-11 it kept 8–14 uncentred components per session where the shuffle keeps 2–6. `figures/noise_bound.png` shows both cases. The uncentred hc-11 and synthetic results before October 21 were computed with the textbook bound (`results/_before_step0/`).

Each event gets a complex score $a_{in} = \sum_k s_{ik}\overline{p_{nk}}$ on each component: its length is how strongly the event expresses the component, its angle the event's clock shift.

Why PCA can separate routes at all: PCA separates directions that are (nearly) orthogonal and have different variances. Routes on different cells are orthogonal; one route forward and backward are orthogonal when the half-circle is filled (Section 2); routes sharing cells with different orders are not, and PCA then returns their union in one component. For that case a complex ICA rotation (`cpca/ica.py`) splits the union; in the synthetic benchmark it helped only there and is therefore off by default.

## 5. Reading a component

Members are the cells with weight at least 20% of the component's largest weight. Their angles are placed on the circle, the widest empty gap is cut, and the angles are unwrapped from there into relative times (ms). The same numbers are a delay-line neuron in the sense of Haessig et al. (2020): synaptic weight $|p_{nk}|$ and delay $D - \tau_{nk}$, so that the members' spikes arrive together when the event plays in the component's order.

`figures/component_on_the_circle.png` (from `scripts/plot_component_circle.py`) shows the picture: on synthetic data each cell's event arrows cluster at one angle and the component's arrow for that cell points there. On hc-11 the cells' mean arrows all sit within 5–8° of the population peak; a centred component's arrows describe *deviations* from that common direction, so after centring the angles are relative and their spread is amplified. Reading them as an order is valid; reading them as milliseconds is not. A read-out that projects each entry onto the direction at right angles to the cell's mean arrow (the linearised timing deviation) gives the same POST/PRE result (17/118 vs 5/104 against 21/118 vs 4/104, both computed with the earlier textbook bound), so the simpler angle read-out is kept.

## 6. Evaluation

Label-free, from the sleep spikes only:

- number of components above the noise bound;
- split-half reproducibility: fit on two random halves of the events, match components one-to-one by $|\langle p, q\rangle|$, report the mean overlap;
- angular span of each component's members (a synchronous volley has a small span; a sequence a large one);
- a timing-consistency $z$ (component energy against phase-randomised events). **Caveat:** this does not identify route components (see Section 8); it is kept as a diagnostic.

Answer key, used once at the end, per component and per running direction: Spearman correlation between the members' relative times and their place-field centres, against the 95th percentile of the same statistic after permuting field centres among all cells with a field (for circular tracks, the best of 12 rotations for data and null alike). Compactness of the members' fields on the track is reported alongside. **PRE sleep (before the animal saw the maze) is the control**: a method that matches the map in POST but not PRE is seeing the experience, not the track geometry or the analysis.

Co-activation null: $C^{0}$ (the lag matrix expected when each event's arrows are reassigned at random among the cells that fired, computed in closed form) and the per-event timing excess $\Delta_i$ are kept in `cpca/stats.py` as diagnostics of "is there timing beyond who fired".

## 7. Parameters

Event detection (edges at the mean population rate, peak above mean + 3 SD, 50–500 ms); the half-circle scale (event duration, default) ; uncentred or centred (data regime, Section 3); member threshold (20% of the maximum weight). The noise bound (95th percentile of 100 shuffles), the mean-arrow encoding and the read-out have no tuned parameters. ICA is optional.

## 8. Known limits and open problems

- **Which components are routes, without the answer key.** In the realistic synthetic condition the route components rank 5–8 of 9 by timing consistency; in hc-11 the map-matching components have eigenvalue shares of 1.2–3.8% and are spread through the spectrum. A label-free criterion for "this component is a sequence of a minority of events" is the main open problem.
- **Direction mixing inside a component.** Members of one component can come from both running directions; the per-direction map check partly accounts for this but the read-out does not separate them.
- **Statistical power** follows the number of place cells: sessions with 50+ place cells show the POST/PRE difference clearly; sessions with 13–25 are at chance in both epochs.
- **Spans beyond the half-circle.** Component angles can span more than $\pi$ (the eigenvector accumulates lags across cells that never fire together); the read-out unwraps them, but the time axis is then only ordinal.
