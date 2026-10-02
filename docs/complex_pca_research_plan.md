# Complex PCA of Spike Timing for Hippocampal Reactivation: Research Plan

Version 1 (18 Sep 2026). Companion to the Cursor plan file. This is the full, referenceable document; the plan file is the short checklist.

---

## 0. How to read this document with rendered formulas

Cursor has three different markdown renderers and they disagree about math. This file is written for the **classic Markdown preview**, which is the only editor surface in Cursor that renders formulas.

1. Open this file in the editor.
2. Press `Cmd+Shift+V` (or `Cmd+K` then `V` for a side-by-side preview).
3. Do **not** use the "Preview" tab of Cursor's native markdown editor, and do not read the `.plan.md` file for formulas: neither renders math (confirmed by Cursor staff as of version 3.13).

If a formula still shows as raw text, check that the setting `markdown.math.enabled` is `true` (it is by default; your `settings.json` does not override it) and that the built-in extension `markdown-math` is enabled. Appendix B has the full cheat-sheet, including the rules for chat and notebooks.

Test line: if you can read $e^{j\theta} = \cos\theta + j\sin\theta$ as a typeset formula, rendering works.

---

## 1. The idea in one paragraph

The paper (Haessig et al., 2020, *Front. Neurosci.* 14:420, Section 3.2.3) turns each sensor's spike time into a point on a circle, a complex number whose angle is the spike time. A vector of such numbers describes one spatio-temporal pattern. The authors fit these vectors to known labels with least squares, and read the fitted complex weights back out as a synaptic weight (its length) and a synaptic delay (its angle). The proposal here is to keep the encoding but drop the labels: build a matrix of these complex numbers from many hippocampal reactivation events recorded during rest, and run principal component analysis (PCA) on it. Because the numbers are complex, each principal component carries, for every neuron, both *how much* it participates (length) and *when* it fires relative to the others (angle). Standard PCA on spike counts can only give the first of these. If the timing inside reactivation events is consistent, complex PCA should recover the order of neurons in the replayed sequence without any behavioural information, in one closed-form computation.

---

## 2. Background you need (kept basic)

### 2.1 Complex numbers as clock hands

A complex number $z = r e^{j\theta}$ is an arrow (a clock hand) of length $r$ pointing at angle $\theta$. Here $j$ is the imaginary unit ($j^2 = -1$; the paper uses $j$, engineers' notation). Three facts are used throughout:

- Multiplying two arrows multiplies their lengths and **adds** their angles: $r_1 e^{j\theta_1} \cdot r_2 e^{j\theta_2} = r_1 r_2\, e^{j(\theta_1 + \theta_2)}$.
- The **conjugate** $\overline{z} = r e^{-j\theta}$ is the mirror image (same length, angle flipped). So $z_1 \overline{z_2}$ has angle $\theta_1 - \theta_2$: it is an arrow that points at the **difference** of the two angles.
- Adding arrows that point the same way gives a long arrow; adding arrows that point in random directions gives a short one. So the length of a sum measures **agreement** of angles.

If we set the angle proportional to a spike time, then "difference of angles" becomes "difference of spike times" (a lag), and "length of a sum of lag-arrows across many events" becomes "how consistent that lag is across events". This is the whole trick.

### 2.2 PCA in one paragraph

Given a table with one row per observation and one column per variable, PCA finds the direction (a weight vector over the columns) along which the rows vary most, then the next best direction at right angles to the first, and so on. Each direction is an eigenvector of the variables' covariance matrix; the amount of variation along it is the eigenvalue. In neuroscience, when rows are time bins and columns are neurons, the top eigenvectors are "cell assemblies": groups of neurons that go up and down together. PCA works unchanged with complex numbers if the covariance is formed with a conjugate: $C_{kl} = \frac{1}{M}\sum_i s_{ik}\overline{s_{il}}$. Such a matrix is *Hermitian* (it equals its own conjugate transpose), which guarantees real non-negative eigenvalues and eigenvectors that are orthogonal. The eigenvectors themselves are complex, so every entry has a length and an angle.

### 2.3 Hippocampal reactivation (replay)

When a rat runs on a track, hippocampal place cells fire in a spatial order. During subsequent rest and sleep, during brief high-frequency events called sharp-wave ripples (about 50 to 150 ms long), the same cells fire again in the same (or reversed) order, compressed roughly 10 to 20 times in time. These events are called reactivation or replay. Standard analysis either (a) finds groups of co-active cells (assemblies) from spike counts, ignoring order, or (b) checks order against a template built from the running period (place fields), which requires behaviour and cannot be used when there is no clean behavioural template.

---

## 3. What the paper did, in plain words

Setup: 8 vibration sensors on a plate; a finger tap at one of 32 positions (8 angles $\times$ 4 distances) makes a wave that reaches each sensor at a slightly different time (differences of about 1 ms). Only the first spike of each sensor is used. 10 taps per position gives 320 patterns.

Encoding (their Eq. 9): for sensor $k$ with first-spike time $t(k)$, and $t_0$ the time of the earliest spike of that tap,

$$
s(k) = \exp\!\left( j\,\frac{\pi\,(t(k) - t_0)}{T} \right) \tag{P1}
$$

where $T$ is the longest possible arrival-time difference ($T = 2 r_{\max}/c$, their Eq. 10). Dividing by $T$ and multiplying by $\pi$ maps all times into the angle range $[0, \pi]$, i.e. **half a circle**. They note this half-circle choice is necessary "to avoid geometric inconsistencies": on a full circle, a very early and a very late spike would land at the same angle.

Fitting (their Eq. 11): for each target position $p$, find complex weights $\hat{w}_p(k)$ that make the weighted sum $\sum_k \hat{w}_p(k) s_i(k)$ equal to 1 for taps at $p$ and 0 for taps elsewhere, by least squares (solved with the pseudo-inverse).

Reading out (their Eq. 12): a complex weight is not something a synapse can hold, so they split it:

$$
w_p(k) = |\hat{w}_p(k)|, \qquad d_p(k) = \frac{T}{\pi}\arg \hat{w}_p(k) \tag{P2}
$$

The length becomes the synaptic weight, the angle becomes a synaptic delay. Why this works: if the weight's angle is the negative of the spike's angle, the product is a real positive number, i.e. the delayed spikes all arrive at the same moment and add up (their Eq. 5, $d_p(k) = d^*_p - t_p(k)$). The fitted neuron is therefore a **coincidence detector with learned delays**, and the least-squares fit also learns to trust reliable sensors more (larger $|\hat{w}|$). Result: 100% recovery of both angle and distance, whereas the purely analytic method got 73.4% on distance and the fixed-delay coincidence detector got 37.5%, because the sensors had systematic timing biases that the regression absorbed.

Stated limitations (their Section 3.2.3): one spike per channel; needs labels; no hidden units.

---

## 4. The proposal: replace the labels with variance

### 4.1 Step 1: cut the rest period into events

Each "observation" is one candidate reactivation event $i = 1, \dots, M$: a window of spiking activity during rest. Standard choices are sharp-wave ripples detected in the LFP, or population burst events (PBEs) detected as peaks of the summed firing of all cells above about 3 standard deviations, lasting 50 to 500 ms. Cell $k = 1, \dots, N$ fires $n_{ik}$ spikes in event $i$ at times $t_{ikm}$.

### 4.2 Step 2: turn spikes into clock hands (encoding)

For event $i$ and neuron $k$:

$$
s_{ik} = \sum_{m=1}^{n_{ik}} \exp\!\left( j\,\frac{\pi\,(t_{ikm} - t_{i0})}{T} \right), \qquad s_{ik} = 0 \ \text{ if } n_{ik} = 0 \tag{1}
$$

Here $t_{i0}$ is a reference time for the event (its centre of mass, or the ripple peak; Section 4.6 shows the choice does not matter) and $T$ is a fixed window at least as long as the longest event, so that any two spikes in one event are less than half a circle apart. This is the paper's encoding with two changes that Section 6 discusses in detail: several spikes are summed into one arrow (H1), and silent neurons get the zero arrow (H2). Collect all $s_{ik}$ into an $M \times N$ complex matrix $S$.

### 4.3 Step 3: the lag matrix

$$
C_{kl} = \frac{1}{M}\sum_{i=1}^{M} s_{ik}\,\overline{s_{il}} \tag{2}
$$

Read one entry. For events where both cells fire once, $s_{ik}\overline{s_{il}} = e^{\,j\pi (t_{ik} - t_{il})/T}$ is a unit arrow whose angle is the **lag** between cell $k$ and cell $l$ in that event. $C_{kl}$ is the average of these lag-arrows over events. Its **length** says how consistent the lag is (and how often the pair is co-active); its **angle** says the typical lag and its sign says who leads. The **real part** is large when the pair fires together (lag near zero); the **imaginary part** is non-zero only when one cell reliably leads the other. Standard count-based PCA sees, in effect, only the real part. The diagonal $C_{kk}$ is the mean squared arrow length of cell $k$, i.e. its participation. In code: `C = (S.T @ S.conj()) / M`.

Two derived quantities are useful on their own:

$$
\rho_{kl} = \frac{\left|\sum_i s_{ik}\overline{s_{il}}\right|}{\sum_i |s_{ik}|\,|s_{il}|} \in [0, 1] \tag{3}
$$

the **lag reliability** of the pair (1 = identical lag in every co-active event; 0 = random), and

$$
\eta = \frac{\|\mathrm{Im}\,C\|_F}{\|C\|_F} \tag{4}
$$

the **timing share**: the fraction of the matrix that lives in consistent lead/lag relations. If $\eta$ is at the level of a within-event time shuffle (Section 6, H8), timing adds nothing beyond co-activation and the hypothesis is false for that dataset.

### 4.4 Step 4: eigen-decomposition and read-out

Solve $C\,p_n = \lambda_n p_n$ for $n = 1, \dots, N$, eigenvalues sorted from largest. Each eigenvector $p_n$ is a **template pattern**, and its share of the total is $\lambda_n / \sum_m \lambda_m$. For neuron $k$ in pattern $n$:

- **participation** $= |p_{nk}|$;
- **relative firing time** $\tau_{nk} = \dfrac{T}{\pi}\arg p_{nk}$, up to one common offset per pattern.

The paper's neuron-with-delays interpretation carries over exactly: the detector weights are the mirror image of the template, $\hat{w}_{nk} = \overline{p_{nk}}$, so the synaptic weight is $|p_{nk}|$ and the synaptic delay is $d_{nk} = -\tau_{nk} + \text{const}$, chosen so that every delayed spike arrives at the same moment (this is the paper's Eq. 5 and Eq. 12 applied to $\overline{p_n}$).

Each event gets a complex **score** per pattern:

$$
a_{in} = \sum_{k} s_{ik}\,\overline{p_{nk}} \tag{5}
$$

$|a_{in}|$ is how strongly event $i$ expresses pattern $n$ (the analogue of "reactivation strength"), and $\arg a_{in}$ is the time shift of event $i$ relative to the template (it absorbs the arbitrary reference $t_{i0}$). In code: `A = S @ P.conj()`.

One convention is needed because an eigenvector times any unit arrow is still an eigenvector: rotate each $p_n$ so that its strongest entry (largest $|p_{nk}|$) has angle zero. Never anchor on a weak entry (Section 6, H6).

### 4.5 A worked example with numbers

Three neurons, true relative times 0, 20, 40 ms, window $T = 100$ ms, so true angles $0^\circ, 36^\circ, 72^\circ$. Each event randomly drops each neuron with probability 0.3 and jitters each spike by 3 ms (standard deviation). 400 events. Result of the computation (run today, seed 0):

- eigenvalues $1.75, 0.24, 0.20$: one clear pattern, the rest is noise;
- recovered relative times from the angles of $p_1$: $0.0, 19.6, 39.9$ ms;
- participation $|p_1| = 0.57, 0.58, 0.58$: all three equally reliable.

Ordinary count-PCA on the same events would return one component with three equal positive weights and no information about the 20 ms steps.

### 4.6 Four properties verified numerically today

These are the load-bearing claims. All were checked with a short NumPy script (results quoted; the script will become `tests/test_theory.py` in Phase 1).

- **Property A: the event reference time does not matter.** Shifting all spikes of an event by any amount multiplies its row of $S$ by a unit arrow $e^{j\phi_i}$, and $s_{ik}\overline{s_{il}}$ is unchanged because $e^{j\phi_i}\overline{e^{j\phi_i}} = 1$. So $C$, its eigenvalues and eigenvectors are exactly invariant to per-event misalignment. Verified: maximum change in $C$ under random shifts $= 2.7 \times 10^{-15}$. Consequence: we do not need to align events to ripple onset or to the first spike. Caveat: this holds only for the **uncentered** matrix of Eq. 2; subtracting a column mean destroys it (Section 6, H13).
- **Property B: order is recovered under jitter and dropout.** See the worked example above.
- **Property C: when the window is much longer than the event, the method reduces to ordinary PCA on spike counts.** As $T \to \infty$ all angles go to zero and $s_{ik} \to n_{ik}$, the spike count. Verified: $\max |S - \text{counts}| = 10^{-9}$ at $T = 10^9$. So complex PCA **contains** the standard method (uncentered version) as the special case "ignore timing", which makes the comparison in Section 9 clean.
- **Property D: the eigenvector angles are exact even when each event shows only a piece of a longer sequence.** 30 neurons, 12 ms apart (full sequence 348 ms), window $T = 150$ ms, each event showing a random 100 ms piece, 60% participation, 3 ms jitter. The recovered angles matched the true ones to within numerical precision for all 28 neurons that appeared in at least one event; the two edge neurons never appeared and had zero length (their angle is undefined). This is the known result that the top eigenvector of a matrix of pairwise relative angles recovers the angles themselves ("angular synchronization", Singer 2011), and it means the template can be longer than one event as long as the pieces overlap. Reading the order then requires **unwrapping** the angles (adding multiples of $2\pi$ where consecutive neurons jump), see H3 and H12.

Two more facts from the same checks are the basis of hurdles H4 and H8:

- **Forward and reverse versions of the same sequence cancel each other's timing** in $C$. With forward-only events, $\eta = 0.58$ and eigenvalues $5.2, 0.2, 0.2$. With a 50/50 mix of forward and reverse, $\eta = 0.03$ ($C$ becomes almost real) and eigenvalues $4.0, 1.4, 0.2$: the timing information has not disappeared, it has moved into a **pair** of components. The sign of $\mathrm{Im}(a_{i1}\overline{a_{i2}})$ separated forward from reverse events with 100% accuracy, and re-running complex PCA on each group recovered the lags with the right sign ($+14.5, +29.9, \dots$ ms vs $-14.6, -29.9, \dots$ ms, true step 15 ms).
- **Shuffling spike times within events (keeping who fired) destroys the imaginary part and lowers the top eigenvalue** ($\eta$: $0.51 \to 0.06$; $\lambda_1$: $2.36 \to 1.83$) while leaving the diagonal of $C$ unchanged. This is the specific null model for "does timing add anything".

### 4.7 Relation to the paper's least squares

Least squares finds weights that make a chosen set of events light up: $\hat{w}_p = S^{+} y_p$ with $y_p$ the 0/1 labels. PCA finds the weights that make the *whole dataset* light up the most, $p_1 = \arg\max_{\|p\| = 1} \sum_i |a_{i1}|^2$, then the next best weights at right angles, and so on. Equivalently, PCA is least squares in which the targets $y$ are unknown and fitted jointly with the weights (the best low-rank approximation of $S$). Same encoding, same decoding into weights and delays, labels replaced by variance. What is lost is the ability to point a neuron at a *chosen* pattern; what is gained is not needing to know the patterns in advance, which is exactly the situation in sleep.

A further link, for the theory paper (Section 10): the online version of PCA, Oja's rule, in the complex domain reads $\Delta p_k = \eta\,(s_k \overline{a} - |a|^2 p_k)$ with $a$ the neuron's own complex output. Translated to spikes, "rotate the input arrow back by the neuron's own output angle" means "compare the input spike time with the neuron's own firing time and shift the synaptic delay accordingly", a local, unsupervised delay-learning rule of the kind the paper's Section 3.2.5 asks for. This is a conjecture to be tested in simulation, not yet a result.

---

## 5. Why this should capture more temporal information than current methods

### 5.1 What current methods see

- **Count-based PCA/ICA** (Lopes-dos-Santos et al., 2013; Peyrache et al., 2009): bin spikes (typically 20 to 25 ms), z-score, PCA, keep components above the random-matrix noise bound, un-mix with ICA. Output: groups of co-active cells. Order inside a bin is invisible by construction; order across bins is blurred by the bin edges. In the language of Eq. 2 they see $\mathrm{Re}\,C$ only.
- **Template methods** (Bayesian decoding, rank-order correlation): order-aware, but the order comes from behaviour, not from the rest data. They cannot discover a sequence that has no behavioural template, and they inherit any error in the place-field estimate.
- **Unsupervised sequence finders** (seqNMF, PP-Seq, convSeq, HMM): order-aware and label-free, but iterative, non-convex, with several hyper-parameters (number of patterns, pattern length, penalties or priors), need binning (seqNMF, convSeq) or heavy sampling (PP-Seq, Julia), and take minutes to hours.

Complex PCA sits in the gap: label-free and order-aware like the third group, closed-form and nearly parameter-free like the first.

### 5.2 Back-of-envelope numbers for hippocampal replay

Typical values: ripple duration 50 to 150 ms (choose $T \approx 200$ to 300 ms); compression 10 to 20 times; spike timing jitter relative to the replayed trajectory of order 10 to 15 ms; each pyramidal cell fires in roughly 10 to 30% of ripples; a session yields 50 to 150 pyramidal cells and 1,000 to 5,000 ripples.

- **Jitter.** An angle jitter of $\sigma_\theta = \pi\sigma_t/T$ multiplies the length of each lag-arrow by about $e^{-\sigma_\theta^2}$. Computed: 10 ms jitter at $T = 200$ ms keeps 0.98; 15 ms at 250 ms keeps 0.97; 30 ms at 250 ms keeps 0.87. Timing precision is **not** the bottleneck.
- **Speed variability.** If an event runs 30% faster or slower, a 50 ms lag changes by 15 ms: the lag-arrow length drops to 0.99. A 150 ms lag with 50% speed variation drops to 0.86. Speed variation matters only for long lags (H5).
- **Sample size.** With $N = 100$ cells and $M = 2{,}000$ events, the noise eigenvalues of $C$ fall below about $(1 + \sqrt{N/M})^2 \approx 1.5$ times the per-cell variance (random-matrix bound, same formula as used by Lopes-dos-Santos). A pair co-active in 3% of events has 60 lag samples, enough to estimate its lag reliability with a standard error of about $1/\sqrt{60} \approx 0.13$.
- **What will actually limit us:** dropout (each event contains a small subset of cells), heterogeneity (different events replay different segments, directions, or environments), and shared cells between sequences. Property D says partial events are fine if pieces overlap; H4 handles direction; multiple environments produce multiple components.

### 5.3 Verdict on plausibility

Plausible. The mathematics is sound and verified on toy data; the physiological numbers say the timing signal should survive the encoding; the unknown is how much consistent within-event timing real reactivations contain beyond co-activation, and Eq. 4 with the time-shuffle null measures exactly that. A null result would itself be informative.

---

## 6. Hurdles and solutions

Each hurdle: what the problem is, why it matters, the proposed solution(s), and how we will test it. Decisions to be made during execution are marked **D**.

### H1. A neuron fires several spikes in one event

The paper encodes exactly one spike per channel. Hippocampal cells burst.

Solution: Eq. 1 sums the arrows of all spikes of a cell in the event. A tight burst gives one long arrow at the burst centre; two spikes far apart partially cancel, producing a short arrow that honestly says "this cell's timing is ambiguous here". Variants to compare (**D1**): raw sum (rate-weighted, length up to $n_{ik}$), mean (length between 0 and 1, pure timing concentration), first spike only (the paper's choice; fragile to a stray early spike). Limitation to state: a cell with two true positions in the sequence (two place fields) will be garbled; PP-Seq and seqNMF share this limitation.

Test: synthetic events with bursts and double-field cells; measure order recovery for each variant.

### H2. Most neurons are silent in any given event

In a ripple only 5 to 15% of recorded cells fire, so $S$ is mostly zeros.

Why it matters: a zero contributes nothing to $C_{kl}$, which is the desired behaviour ("no vote"), but it means $|C_{kl}|$ mixes two things: how often the pair is co-active and how consistent their lag is.

Solution: keep Eq. 2 for the decomposition (it is guaranteed to have non-negative eigenvalues) and report Eq. 3 separately as the pure timing measure. Optional per-cell normalization (divide column $k$ of $S$ by its root-mean-square length) to stop very active cells dominating, the analogue of z-scoring in Lopes-dos-Santos (**D2**).

Test: vary participation probability in simulation from 0.05 to 0.8; plot order recovery and the eigenvalue gap.

### H3. Choice of reference time and window, and wrap-around

Two spikes more than $T$ apart map to angles more than $\pi$ apart and become ambiguous (the paper's "geometric inconsistency").

Solution: Property A removes the reference-time problem entirely. For $T$, use a fixed value at least the longest event duration (e.g. 1.5 times the 99th percentile of event lengths). Then all within-event lags are unambiguous. Across events the template may still span more than $\pi$ (Property D), which is fine: read the order by unwrapping the eigenvector angles along the chain of strongly connected cells. Diagnostic: if unwrapping is ambiguous (weak links), report the order only within well-connected blocks.

Test: sweep $T$ from 0.5 to 5 times the event length; confirm a plateau of good recovery.

### H4. Forward and reverse replay, and sequences that share cells

Shown in Section 4.6: a 50/50 forward/reverse mix makes $C$ almost real and splits the sequence across two components. The same happens whenever two patterns share cells with different relative timing (the two running directions on a linear track, exactly the case PP-Seq reported on the hc-11 data).

Solution (two-step, both parts verified today on toy data): (i) run complex PCA to find the significant subspace (top $K$ components); (ii) separate the mixture inside that subspace by clustering the complex scores (or complex ICA, mirroring the real ICA step of Lopes-dos-Santos), then re-run complex PCA within each cluster to get clean templates. A cheap diagnostic for a hidden pair is a small $\eta$ (Eq. 4) together with two eigenvalues well above the noise bound.

Test: synthetic mixtures with forward/reverse ratio from 0/100 to 50/50 and with partially overlapping cell sets; measure per-pattern order recovery after the clustering step.

### H5. Events run at different speeds (time warping)

A faster replay scales all lags, which is not a rotation and so is not absorbed by Property A.

Solution: (a) normalize each event's time axis by its own duration (e.g. the spread between its 10th and 90th percentile spike times) before mapping to angles, making the encoding speed-invariant at the cost of assuming every event covers a similar fraction of the sequence; (b) or fit one scale factor per event from a small grid by maximizing $|a_{i1}|$ (cheap: one matrix product per grid value); (c) or accept the loss, which Section 5.2 shows is small for lags below about 100 ms (**D3**).

Test: synthetic speed variation of 0 to 3 times; compare (a), (b), (c).

### H6. Eigenvector length mixes coverage with reliability

Property D showed that when events cover different parts of a long sequence, $|p_{k}|$ follows how often cell $k$ appears (a smooth bump over the sequence), not only how reliable it is. Anchoring the angle convention or the unwrapping on a cell with small $|p_k|$ gives garbage (this is what produced a spurious factor 1.7 stretch in the first version of today's check, until the anchor was moved to the strongest cell).

Solution: always anchor on the strongest cell; exclude cells with $|p_k|$ below a threshold (e.g. 10% of the maximum, or below the shuffle level) from order read-out; report Eq. 3 as the reliability measure.

### H7. Very active cells and interneurons

Interneurons fire in almost every ripple with weak sequence timing and would dominate $C$.

Solution: restrict to putative pyramidal cells (standard practice, based on waveform and rate); apply the per-cell normalization of H2 if needed.

### H8. How many components are real, and does timing add anything

Solution, three null models, each answering a different question:

- **Within-event time shuffle**: permute spike times among the cells that fired in the event. Keeps who fired, destroys when. Compare $\lambda_1$ and $\eta$ (Eq. 4) with the shuffle distribution. This is the test of the central hypothesis.
- **Cell-identity shuffle across events**: destroys co-activation as well; gives the "nothing at all" baseline, comparable to the random-matrix bound.
- **Split-half cross-validation**: fit templates on half the events, measure explained variance and score correlation on the other half; guards against eigenvalue inflation.

Report the number of components above the shuffle 99th percentile.

### H9. No ground truth on real data

Solution: use behaviour as an external check, but only *after* the analysis of rest data is complete. Prediction 1: the unwrapped angles of the top template from **sleep alone** correlate with place-field centre position measured during running (circular-linear correlation against a time-shuffle null). Prediction 2: events with large $|a_{i1}|$ are the ones the Bayesian decoder scores as replay (report an ROC curve). Prediction 3: post-task sleep expresses the task template more strongly than pre-task sleep. Precedent that unsupervised recovery of the map from sleep is achievable: Maboudi et al. 2018 (HMM) and Williams et al. 2020 (PP-Seq), both on hc-11-type data.

### H10. Events must be segmented first

Unlike seqNMF or PP-Seq, which run on continuous data, the method needs event windows.

Solution: default to PBE detection from summed firing (no LFP needed); ripple detection when LFP is available; and a sliding-window variant (windows of length $T$ stepped every $T/4$ over the whole rest period) as a segmentation-free alternative, at the cost of correlated rows.

### H11. Patterns change over hours of sleep

Solution: run the analysis in blocks of events (e.g. 500 at a time) and compare templates across blocks by the angle between subspaces and by matching cells' relative times; compare pre- and post-task sleep explicitly.

### H12. Practical read-out of order

Angles live on a circle. Converting the top template into a ranked cell order requires: fix the convention (H6), unwrap along strongly connected cells (H3), then rank by unwrapped time. Report a confidence per cell from Eq. 3 and from the split-half agreement of its angle.

### H13. Do not mean-centre

Standard PCA subtracts each column's mean. Here that would (a) turn "silent" into a non-zero arrow and (b) break Property A. Use the uncentered matrix of Eq. 2. If a centred variant is ever wanted, it must be done after aligning events to a meaningful common reference (e.g. ripple peak), and the loss of Property A must be accepted (**D4**).

---

## 7. What already exists (honest novelty assessment)

- **Complex PCA on continuous signals** exists under the names "complex empirical orthogonal functions" (climate science, Horel 1984; Hannachi et al. 2007) and was applied to fMRI to find travelling waves (Bolt et al. 2022, *Nat. Neurosci.*). There the complex numbers come from the Hilbert transform of a band-passed signal, so the phase is defined relative to an oscillation cycle, not relative to an event. Our encoding is the paper's direct time-to-angle map, needs no filtering and works on single events. To our knowledge (web check 18 Sep 2026), complex PCA has **not** been applied to phase-encoded spike times.
- **Angular synchronization** (Singer 2011): recovering angles from noisy pairwise angle differences via the top eigenvector of a Hermitian matrix. Our Eq. 2 is exactly this matrix with the differences supplied by spike lags. This gives us existing recovery guarantees to cite, and it means the mathematics is not new; the application and the connection to the paper's neuron-with-delays are.
- **Phasor networks / complex-valued neural networks** (Hirose 1992; Reichert and Serre 2013; Frady and Sommer 2019): the paper itself cites these for the time-as-angle idea.
- **Lag-aware assembly detection**: Russo and Durstewitz 2017 (CAD) find assemblies with arbitrary lags via pairwise statistics; SPOTDis (Grossberger, Battaglia, Vinck 2018) clusters events by spike-time pattern dissimilarity. Neither produces a closed-form component with per-cell delays.
- **Unsupervised sequence detection**: seqNMF (Mackevicius et al. 2019), PP-Seq (Williams et al. 2020), convSeq (Koshkin et al. 2024), HMM congruence (Maboudi et al. 2018). These are the strongest competitors and the comparison set.
- **Replay-detection methodology**: Tingley and Peyrache 2020 review the pitfalls of template-based replay scoring (shuffle choice, false positives), a useful frame for arguing that a label-free method is valuable.

Position to take in writing: a known mathematical tool (angular synchronization / complex PCA), a known encoding (the paper's), combined for the first time on spike data, with a direct mapping to a spiking neuron with learned delays, validated against three families of existing methods.

---

## 8. Trade-offs summary

Advantages:

- Closed form: one Hermitian eigen-decomposition; deterministic; runs in well under a second for 100 cells and 5,000 events.
- No time binning; timing enters as continuous angles.
- No event alignment needed (Property A).
- Participation and timing in one object; the standard count method is the special case $T \to \infty$ (Property C).
- Built-in, specific null model for "does timing matter" (Eq. 4 with the time shuffle).
- Direct interpretation as a coincidence-detector neuron with learned weights and delays (the paper's Eq. 12), and a candidate local learning rule (complex Oja).
- Few knobs: $T$, the encoding variant (D1), normalization (D2), number of components (from the null).

Disadvantages:

- Linear and low-rank: mixed patterns that share cells land in a shared subspace and need a second un-mixing step (H4), like real PCA before ICA.
- One timing summary per cell per event; cells with two positions in a sequence are garbled (H1).
- Needs event windows or a sliding window (H10).
- Angles are only known modulo a full turn; order requires unwrapping and a connected chain of cells (H3, H12).
- Sensitive to speed variability for long lags (H5).
- Zero means both "silent" and "unknown timing" (H2); $|C_{kl}|$ mixes co-activation frequency with lag reliability.
- No probabilistic model of spike counts, so no likelihoods, no per-event posterior uncertainty of the kind PP-Seq provides.

---

## 9. Validation strategy

### 9.1 Order of evidence

1. **Theory tests** (Section 4.6) as automated unit tests.
2. **The paper's own task, simulated**: an 8-sensor array with 32 tap positions and realistic jitter. Reproduce the complex least-squares result (100% recovery), then show that complex PCA on the same 320 patterns clusters the 8 angles and orders the sensors' arrival times with no labels. This is the cleanest demonstration that "least squares to PCA" works, and a figure any reader of the original paper will understand.
3. **Synthetic replay** with controls: number of cells and events, participation, jitter, speed variability, forward/reverse ratio, partial coverage, shared cells, background Poisson spikes, one versus several sequences.
4. **Real data**, once chosen (Section 11, Phase 5).

### 9.2 Baselines (all three families requested)

- **Count-based PCA/ICA** (Lopes-dos-Santos 2013; Peyrache 2009 reactivation strength). Own implementation (short). Compare assembly membership (overlap of high-weight cells) and show what is added: the order.
- **Template methods** (Bayesian decoding with line-fit score; rank-order correlation). Own implementation. Used both as a baseline and as the ground-truth proxy on real data (H9).
- **Unsupervised sequence finders**: seqNMF (Python package available), HMM (hmmlearn, following Maboudi 2018), convSeq (PyTorch, GitHub), PP-Seq (Julia; optional, run only if time permits).

### 9.3 Metrics

- Order recovery: circular-linear correlation between recovered relative times and true (or place-field) positions; Spearman rank correlation after unwrapping.
- Event detection: ROC area of $|a_{i1}|$ against ground-truth (synthetic) or Bayesian-decoder (real) labels.
- Timing content: $\eta$ (Eq. 4) and $\lambda_1$ against the within-event time shuffle.
- Membership agreement with count-PCA/ICA: overlap of top-weight cells.
- Cost: wall-clock time, number of hyper-parameters, need for binning, handling of speed variability.

---

## 10. Value if successful, and what can be published

### 10.1 Value

- **Methodological**: a ten-line, label-free, closed-form way to get temporally ordered assemblies with a principled significance test, generalizing the most widely used assembly method. Easy to adopt; that alone tends to get cited.
- **Scientific**: a direct answer to "how much consistent within-event timing do reactivations contain beyond co-activation", per session, per sleep stage, pre versus post learning, without a behavioural template. If the answer is "a lot", it opens fine-timescale questions (does internal order sharpen with sleep, does it depend on ripple frequency, does it degrade with age or in disease models). If "little", that constrains models of replay.
- **Theoretical / neuromorphic**: the paper's authors explicitly wanted local, unsupervised delay learning (their Section 3.2.5). Complex Oja's rule on this encoding is a candidate, and each principal component *is* one of their delay-line neurons. This closes a loop with the source paper.
- **Beyond hippocampus**: any event-based population data with sequences (songbird HVC, cortical up-states, zebrafish, spinal CPGs).

Realistic risk: the strongest competitors (PP-Seq, HMM) may recover order at least as well on real data; the case then rests on speed, simplicity, the interpretable neuron mapping and the null model, which is still a publishable position.

### 10.2 Papers

1. **Methods paper (primary)**: "Complex-valued PCA of spike phases: label-free extraction of temporally ordered cell assemblies". Content: Sections 4 to 6, the tap-array reproduction, synthetic benchmarks against all three baseline families, one real dataset. Venues: *PLoS Computational Biology*, *eLife* (Tools and Resources), *Journal of Neuroscience Methods*, *Neural Computation*; the machine-learning venues where PP-Seq and convSeq appeared (NeurIPS / ICML workshops or main track) if the benchmark story is strong.
2. **Biology paper (conditional on a positive real-data result)**: "Fine-timescale order within hippocampal reactivations across sleep". Venues: *Journal of Neuroscience*, *Cell Reports*, *Hippocampus*, *eNeuro*.
3. **Theory / neuromorphic paper**: "Every complex principal component is a coincidence detector with learned delays: complex Oja's rule as local unsupervised delay learning". Venues: *Frontiers in Neuroscience* (Neuromorphic Engineering, the source paper's own section), *Neuromorphic Computing and Engineering*, *Neural Computation*, ICONS.
4. **Fallback (negative result)**: "Within-event spike timing adds little beyond co-activation in hippocampal reactivation: a complex-PCA test". *eNeuro* publishes negative results; still useful to the field.

---

## 11. Execution roadmap

Everything below is Python. Environment: new conda environment `cpca` (Python 3.11; numpy, scipy, matplotlib, scikit-learn, pandas, jupyter, pytest). Do not use the system `python3` (blocked by the Xcode licence prompt on this machine) or the existing `RLPP` environment (keep projects separate).

Proposed workspace layout:

```text
Complex PCA/
  docs/complex_pca_research_plan.md   this document
  environment.yml
  cpca/            library: encode.py, decompose.py, stats.py, decode.py, mixture.py, viz.py
  sim/             tap_array.py (paper's task), replay_sim.py (synthetic replay)
  baselines/       count_pca_ica.py, template_replay.py, seqnmf_wrap.py, hmm_wrap.py
  data/            raw and converted data (git-ignored)
  scripts/         one script per figure / experiment
  notebooks/       exploratory, math renders with $...$
  tests/           pytest, including test_theory.py from Section 4.6
```

Dataset-agnostic input format (so the data decision can wait):

```text
spikes.npz    times_s (float), unit_ids (int)
units.csv     unit_id, cell_type (pyr/int), region, optional place_field_centre
events.csv    start_s, end_s, kind (ripple/PBE), epoch
epochs.csv    name (pre_sleep/run/post_sleep), start_s, end_s
position.csv  time_s, x, y                (optional, for validation only)
```

### Phase 0: setup (half a day)

Create the environment, layout, `environment.yml`, a `.cursor/rules` file with the math-delimiter rules of Appendix B, and a check that this document renders (open with `Cmd+Shift+V`).

### Phase 1: theory and toy demonstrations (1 to 2 weeks)

- `tests/test_theory.py`: Properties A to D and the two mixture/shuffle facts as assertions.
- `sim/tap_array.py`: simulate the paper's 8-sensor task (their geometry, 32 positions, 10 repeats, systematic per-sensor bias plus jitter). Reproduce the complex least squares of their Eq. 11 and 12; then run complex PCA and cluster the scores.
- A notebook that walks through Section 4 with figures: arrows on the unit circle for one event, the lag matrix as an image (length and angle), the template as arrows, the read-out to weights and delays.

Acceptance: least squares reproduces 100% angle recovery on the simulated array; complex PCA clusters the 8 angles with adjusted Rand index at least 0.9 without labels; all theory tests pass.

### Phase 2: core library (1 week)

`encode.py` (spikes to $S$, variants D1, D2, H5a), `decompose.py` (Eq. 2, eigen-decomposition, phase convention, scores), `stats.py` (three null models of H8, random-matrix bound, split-half), `decode.py` (unwrapping, ranking, delays and weights per Eq. P2), `mixture.py` (clustering / complex ICA in the top-$K$ subspace, H4), `viz.py`.

Acceptance: unit tests; a full run on 100 cells and 5,000 events in under 2 seconds.

### Phase 3: synthetic replay benchmark (2 weeks)

`sim/replay_sim.py` with all controls of Section 9.1 item 3. Sweep each control; produce degradation curves for order recovery, event ROC and $\eta$. Decide D1 to D4 from these curves.

Acceptance: with jitter at most 15 ms, participation at least 0.2, 500 or more events and a single forward sequence, circular-linear correlation at least 0.9 and ROC area at least 0.9; documented behaviour in the forward/reverse and shared-cell cases after the H4 step.

### Phase 4: baselines (2 weeks)

Own implementations of count PCA/ICA and template scoring; wrappers for seqNMF and HMM; convSeq and PP-Seq as stretch goals. Run every method on the same synthetic sweeps; table of metrics and run times.

### Phase 5: real data (3 to 4 weeks, after the data decision)

Default if no own data: CRCNS hc-11 (Grosmark and Buzsaki 2016; free account required; rats on novel linear and circular tracks with pre- and post-task sleep; the dataset PP-Seq used). Convert to the input format above; detect PBEs; keep pyramidal cells; run pre-sleep, post-sleep and awake rest separately; validate per H9; run all baselines.

Acceptance: recovered order from post-task sleep correlates with place-field position with $p \lt 0.01$ against the time-shuffle null in at least one session; $\eta$ above shuffle; explicit comparison table against baselines.

### Phase 6: write-up (ongoing)

Figures from `scripts/`; decide which of the papers in Section 10.2 the results support.

---

## 12. Open decisions for you

- Data source for Phase 5 (own recordings vs hc-11 vs both). Nothing before Phase 5 depends on it.
- D1 to D4 are to be settled by the Phase 3 curves, unless you have a preference now.
- Whether PP-Seq (Julia) is worth installing for the comparison, or whether seqNMF, HMM and convSeq suffice.

---

## Appendix A. Notation

- $N$: number of neurons (cells); index $k, l$.
- $M$: number of events; index $i$.
- $t_{ikm}$: time of the $m$-th spike of cell $k$ in event $i$; $n_{ik}$: its spike count; $t_{i0}$: event reference time.
- $T$: window that maps time to half a circle, $\theta = \pi t / T$.
- $s_{ik}$: complex arrow for cell $k$ in event $i$ (Eq. 1); $S$: the $M \times N$ matrix.
- $C$: $N \times N$ lag matrix (Eq. 2), Hermitian, positive semi-definite.
- $\lambda_n, p_n$: eigenvalue and eigenvector (template) $n$; $p_{nk}$: entry for cell $k$.
- $\tau_{nk} = (T/\pi)\arg p_{nk}$: relative firing time of cell $k$ in template $n$.
- $\hat{w}_{nk} = \overline{p_{nk}}$: detector weight; $|p_{nk}|$: synaptic weight; $d_{nk}$: synaptic delay.
- $a_{in}$: complex score of event $i$ on template $n$ (Eq. 5).
- $\rho_{kl}$: lag reliability (Eq. 3); $\eta$: timing share (Eq. 4).
- $\overline{z}$: complex conjugate; $\|\cdot\|_F$: Frobenius norm (square root of the sum of squared entry lengths).

## Appendix B. Getting formulas to render in Cursor

Findings from the Cursor documentation, staff forum answers and the renderer source (checked 18 Sep 2026). Cursor's own docs say nothing about math; the rules below come from the renderers themselves.

Surfaces and what they render:

- **Chat panel (agent replies)**: renders `\(...\)` inline and `\[...\]` or `$$...$$` display. Does **not** render `$...$` (intentionally, so that prices like \$5 are not parsed as math). Do not nest `\(` inside `\[...\]`; it breaks the rest of the message.
- **Classic Markdown preview** (`Cmd+Shift+V`, VS Code webview): renders `$...$` and `$$...$$` through the built-in `markdown-math` extension (KaTeX). Does **not** render `\(...\)` or `\[...\]`. Settings: `markdown.math.enabled` (default true), `markdown.math.macros`.
- **Cursor native "Preview | Markdown" tab and the WYSIWYG editor**: render **no** math at all (staff-confirmed, no ETA).
- **Plan editor (`.plan.md`)**: same native renderer, renders no math. This is why the plan file writes formulas in code font and points here.
- **Jupyter notebook markdown cells**: same as the classic preview: `$...$` and `$$...$$`.

Rules for `$` in the classic preview (from the renderer's grammar):

- The opening `$` must not be preceded by a letter, digit, backslash or another `$`, and must be followed by a non-space. The closing `$` must be preceded by a non-space and not followed by a letter or digit. So write `the window $T$ is` (renders) rather than `window$T$is` or `$x$2` (both stay raw).
- Put `$$` display blocks on their own lines with a blank line before and after.
- A literal dollar sign is `\$`.
- Do not put formulas inside back-ticks or code fences; they are shown literally there.
- Avoid a bare `<` inside math (it can be read as HTML); use `\lt`, `\le`.

Recommended user rule (Cursor Settings, Rules), so that future chats render:

```text
Math in chat: use \( ... \) inline and $$ ... $$ for display; never single $.
Math in .md and .ipynb files: use $ ... $ inline and $$ ... $$ display; never \( \).
Never nest delimiters. Put $$ blocks on their own lines with blank lines around them.
```

Optional: for sharing outside Cursor, export with pandoc using `--katex` or `--mathjax` (pandoc is not installed on this machine at the moment).

## Appendix C. References

- Haessig G, Milde MB, Aceituno PV, Oubari O, Knight JC, van Schaik A, Benosman RB, Indiveri G (2020). Event-based computation for touch localization based on precise spike timing. *Front. Neurosci.* 14:420.
- State L (2019). Training delays in spiking neural networks. Master's thesis, MPI for Mathematics in the Sciences, Leipzig (the source of the complex-weight method).
- Lopes-dos-Santos V, Ribeiro S, Tort ABL (2013). Detecting cell assemblies in large neuronal populations. *J. Neurosci. Methods* 220:149-166.
- Peyrache A, Khamassi M, Benchenane K, Wiener SI, Battaglia FP (2009). Replay of rule-learning related neural patterns in the prefrontal cortex during sleep. *Nat. Neurosci.* 12:919-926.
- Mackevicius EL et al. (2019). Unsupervised discovery of temporal sequences in high-dimensional datasets, with applications to neuroscience. *eLife* 8:e38471 (seqNMF).
- Williams AH, Degleris A, Wang Y, Linderman SW (2020). Point process models for sequence detection in high-dimensional neural spike trains. *NeurIPS* (PP-Seq).
- Koshkin R et al. (2024). convSeq: fast and scalable method for detecting patterns in spike data. arXiv:2402.01130.
- Maboudi K, Ackermann E, de Jong LW, Pfeiffer BE, Foster D, Diba K, Kemere C (2018). Uncovering temporal structure in hippocampal output patterns. *eLife* 7:e34467 (HMM).
- Grossberger L, Battaglia FP, Vinck M (2018). Unsupervised clustering of temporal patterns in high-dimensional neuronal ensembles using a novel dissimilarity measure. *PLoS Comput. Biol.* 14:e1006283 (SPOTDis).
- Russo E, Durstewitz D (2017). Cell assemblies at multiple time scales with arbitrary lag constellations. *eLife* 6:e19428 (CAD).
- Tingley D, Peyrache A (2020). On the methods for reactivation and replay analysis. *Phil. Trans. R. Soc. B* 375:20190231.
- Grosmark AD, Buzsaki G (2016). Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences. *Science* 351:1440-1443 (dataset CRCNS hc-11).
- Davidson TJ, Kloosterman F, Wilson MA (2009). Hippocampal replay of extended experience. *Neuron* 63:497-507 (Bayesian decoding of replay).
- Foster DJ, Wilson MA (2006). Reverse replay of behavioural sequences in hippocampal place cells during the awake state. *Nature* 440:680-683 (rank-order).
- Singer A (2011). Angular synchronization by eigenvectors and semidefinite programming. *Appl. Comput. Harmon. Anal.* 30:20-36.
- Bolt T et al. (2022). A parsimonious description of global functional brain organization in three spatiotemporal patterns. *Nat. Neurosci.* 25:1093-1103 (complex PCA of fMRI).
- Hannachi A, Jolliffe IT, Stephenson DB (2007). Empirical orthogonal functions and related techniques in atmospheric science: a review. *Int. J. Climatol.* 27:1119-1152.
- Frady EP, Sommer FT (2019). Robust computation with rhythmic spike patterns. *PNAS* 116:18050-18059.
- Oja E (1982). A simplified neuron model as a principal component analyzer. *J. Math. Biol.* 15:267-273.
