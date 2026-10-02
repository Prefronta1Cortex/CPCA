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
- **Centred and scaled.** Subtract each cell's mean arrow across events and divide by its root-mean-square length, the complex analogue of z-scoring in count-based assembly analysis. Components then describe how events differ from one another. This is right when replays are a minority variation on a common background, which is the sleep regime: on hc-11 it roughly doubles the number of components whose order matches the place map in POST sleep (21/118 vs 14/134 tests above chance) while PRE sleep stays at chance in both. Centring requires aligned events (shared clock), which the peak-centred window provides.

## 4. Components

Eigen-decomposition $C\,p_n = \lambda_n p_n$ gives components $p_n$ (one arrow per cell) with eigenvalues $\lambda_n \ge 0$. For cell $k$ in component $n$: $|p_{nk}|$ is its weight (participation), and its relative firing time is

$$
\tau_{nk} = \frac{T}{\pi}\arg p_{nk}
$$

up to one common offset per component. The number of components to keep is the number of eigenvalues above the Marchenko–Pastur bound $\bar\lambda\,(1+\sqrt{N/M})^2$, the largest eigenvalue pure noise of the same size would produce. **Read all kept components, not the first.** In hc-11 POST sleep 5 to 14 components pass the bound per session; the route-like ones are rarely the first.

Each event gets a complex score $a_{in} = \sum_k s_{ik}\overline{p_{nk}}$ on each component: its length is how strongly the event expresses the component, its angle the event's clock shift.

Why PCA can separate routes at all: PCA separates directions that are (nearly) orthogonal and have different variances. Routes on different cells are orthogonal; one route forward and backward are orthogonal when the half-circle is filled (Section 2); routes sharing cells with different orders are not, and PCA then returns their union in one component. For that case a complex ICA rotation (`cpca/ica.py`) splits the union; in the synthetic benchmark it helped only there and is therefore off by default.

## 5. Reading a component

Members are the cells with weight at least 20% of the component's largest weight. Their angles are placed on the circle, the widest empty gap is cut, and the angles are unwrapped from there into relative times (ms). The same numbers are a delay-line neuron in the sense of Haessig et al. (2020): synaptic weight $|p_{nk}|$ and delay $D - \tau_{nk}$, so that the members' spikes arrive together when the event plays in the component's order.

## 6. Evaluation

Label-free, from the sleep spikes only:

- number of components above the noise bound;
- split-half reproducibility: fit on two random halves of the events, match components one-to-one by $|\langle p, q\rangle|$, report the mean overlap;
- angular span of each component's members (a synchronous volley has a small span; a sequence a large one);
- a timing-consistency $z$ (component energy against phase-randomised events). **Caveat:** this does not identify route components (see Section 8); it is kept as a diagnostic.

Answer key, used once at the end, per component and per running direction: Spearman correlation between the members' relative times and their place-field centres, against the 95th percentile of the same statistic after permuting field centres among all cells with a field (for circular tracks, the best of 12 rotations for data and null alike). Compactness of the members' fields on the track is reported alongside. **PRE sleep (before the animal saw the maze) is the control**: a method that matches the map in POST but not PRE is seeing the experience, not the track geometry or the analysis.

Co-activation null: $C^{0}$ (the lag matrix expected when each event's arrows are reassigned at random among the cells that fired, computed in closed form) and the per-event timing excess $\Delta_i$ are kept in `cpca/stats.py` as diagnostics of "is there timing beyond who fired".

## 7. Parameters

Event detection (edges at the mean population rate, peak above mean + 3 SD, 50–500 ms); the half-circle scale (event duration, default) ; uncentred or centred (data regime, Section 3); member threshold (20% of the maximum weight). The noise bound, the mean-arrow encoding and the read-out have no free parameters. ICA is optional.

## 8. Known limits and open problems

- **Which components are routes, without the answer key.** In the realistic synthetic condition the route components rank 5–8 of 9 by timing consistency; in hc-11 the map-matching components have eigenvalue shares of 1.2–3.8% and are spread through the spectrum. A label-free criterion for "this component is a sequence of a minority of events" is the main open problem.
- **Direction mixing inside a component.** Members of one component can come from both running directions; the per-direction map check partly accounts for this but the read-out does not separate them.
- **Statistical power** follows the number of place cells: sessions with 50+ place cells show the POST/PRE difference clearly; sessions with 13–25 are at chance in both epochs.
- **Spans beyond the half-circle.** Component angles can span more than $\pi$ (the eigenvector accumulates lags across cells that never fire together); the read-out unwraps them, but the time axis is then only ordinal.
