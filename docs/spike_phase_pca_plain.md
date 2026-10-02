# Complex spike-phase PCA, explained

This is a plain-language reading of the harder mathematics note (the one that starts "Promoting the complex-weight method of Haessig et al."). It assumes you already followed `docs/math_guide.md`: a spike time is an arrow, $s_k\overline{s_l}$ is a time gap, and $C$ is the average of those gaps.

Open this file and press `Cmd+Shift+V` so the formulas render.

Test line: if you can read $e^{j\theta} = \cos\theta + j\sin\theta$ as a typeset formula, rendering works.

The numbers quoted below are the simulation results already collected in that note. They are not new runs.

---

## What the hard note is actually saying

Three ideas, in order:

1. The paper's complex weight **is** a neuron with a delay on each input. The peak voltage of that neuron is the length of one complex number. This remains true when a cell fires more than once.
2. Replacing the paper's labels with "which pattern makes the voltage largest on average" is principal component analysis. The awkward parts (half-circle, subtracting a mean, one spike only) have specific fixes.
3. Once you have the patterns, a second, smaller calculation says whether two patterns are **stitched together with a stable gap**, or whether they merely happen to show up in the same events.

---

## 0. The few symbols that keep coming back

- One event is a short window of spikes. There are $M$ events and $N$ cells.
- $\omega = \pi/T$. A full turn of the arrow takes time $2T$, because the angle is $\omega t = \pi t/T$.
- $s_{ik}$ is cell $k$'s arrow in event $i$. The table of all of them is $S$, with events as rows and cells as columns.
- The gap table is

$$
C_{kl} = \frac{1}{M}\sum_i s_{ik}\,\overline{s_{il}}.
$$

- A template $p$ is one arrow per cell. $|p_k|$ is how much cell $k$ takes part. The angle of $p_k$, divided by $\omega$, is when cell $k$ fires relative to the others.
- The score of event $i$ on that template is one number,

$$
a_i = \sum_k s_{ik}\,\overline{p_k}.
$$

$|a_i|$ will turn out to be a peak voltage. $\arg a_i$ will turn out to be the time of that peak.

---

## 1. The paper's neuron, without the labels

### 1.1 What the paper did

Eight sensors, one spike each. Encode

$$
s(k) = e^{\,j\omega(t(k)-t_0)}, \qquad \omega = \pi/T,
$$

with $t_0$ the earliest spike, so every arrow sits on the upper half of the circle. For each target place, find complex weights $\hat w$ that make $\sum_k \hat w(k)\,s(k)$ close to $1$ for taps at that place and close to $0$ otherwise. That is least squares: $\hat w = S^{+} y$, where $y$ is the list of 0s and 1s.

Then split each weight into a length and an angle. The length is a synaptic strength. The angle is a delay. The neuron fires when the delayed spikes land together.

### 1.2 Why a cosine bump makes this exact

Give cell $k$ a strength $w_k \ge 0$, a delay $d_k$, and a bump shaped like a cosine, $\cos(\omega u)$. One spike at time $t_k$ contributes

$$
w_k\cos\big(\omega(t - t_k - d_k)\big).
$$

Write the delay as "wait a common time $D$, then undo this cell's own firing time $\tau_k$":

$$
d_k = D - \tau_k.
$$

The expression inside the cosine becomes

$$
\omega(t-D) - \omega t_k + \omega\tau_k.
$$

A cosine is the horizontal part of a spinning arrow, $\cos\phi = \mathrm{Re}(e^{j\phi})$. So this one spike contributes the horizontal part of

$$
w_k\, e^{\,j\omega(t-D)}\, e^{-j\omega t_k}\, e^{\,j\omega\tau_k}.
$$

Group the pieces that belong to the synapse, $p_k = w_k e^{\,j\omega\tau_k}$, and the piece that belongs to the spike, $e^{\,j\omega t_k}$. Adding every spike of every cell, the voltage is

$$
v(t) = \mathrm{Re}\Big[e^{\,j\omega(t-D)}\,\overline{a}\Big], \qquad a = \sum_k \hat x_k\,\overline{p_k},
$$

where $\hat x_k = \sum_m e^{\,j\omega t_{km}}$ is the sum of the spike arrows of cell $k$. That sum is exactly $a = \langle \hat x, p\rangle$, the score from section 0.

A spinning arrow's horizontal part is largest when the arrow points right, and the height of that peak is the arrow's length. So

$$
\max_t v(t) = |a|, \qquad \text{the peak happens at } t^{*} = D + \frac{\arg a}{\omega}.
$$

Checked on 6 cells with 1–4 spikes each: the highest voltage found by scanning time was $7.255738$, and $|a|$ was $7.255738$. The peak time matched $D + \arg(a)/\omega$ up to one full period $2T$.

What this buys:

- $|a|$ is not an abstract score. It is the peak voltage of the delay-line neuron. $\arg a$ is when that peak happens.
- The sum over spikes never assumed "only one spike". The paper's one-spike rule was a choice of encoding. The neuron already accepts $\hat x_k = \sum_m e^{\,j\omega t_{km}}$.
- A cosine goes negative. A bump $\tfrac12(1+\cos)$ stays non-negative and adds an extra count of spikes on top of $\tfrac12|a|$. The timing part is still $|a|$.

### 1.3 Why least squares needs a shared clock

The paper's target $y_i$ is a plain $0$ or $1$, a real number. For the score $a_i$ to land on a positive real number for every "yes" example, those examples must all sit at the same absolute angle. That is why each pattern is anchored on its first spike.

If each event is given a random clock instead, the arrows point every which way, their average is about $0$, and the fit collapses.

Checked: forward versus reverse sequences, 600 events each. Least squares with a first-spike clock got $98.2\%$ right, with weight length $0.42$. The same spikes with a random clock got $50\%$ (chance), and the weight length fell to $0.11$.

PCA does not need that shared clock. Its score is allowed to be complex, and $\arg a_i$ soaks up each event's clock. Section 2 is that statement. One warning from the same check: if the random shift is so large that spikes cross a full turn **inside** the event, the arrows from one cell no longer mean one time, and the eigenvalue gap collapses. That is the window rule in section 5, not a failure of the shift-invariance itself.

---

## 2. Drop the labels: ask for the strongest average peak

Least squares makes a chosen set of events produce a large voltage. Drop the choice. Ask for the unit-length template that makes the **average squared peak** as large as possible:

$$
p_1 = \arg\max_{\|p\|=1}\ \frac{1}{M}\sum_i |a_i|^2, \qquad a_i = \sum_k s_{ik}\,\overline{p_k}.
$$

Expand the square. The double sum over cells is exactly

$$
\frac{1}{M}\sum_i |a_i|^2 = p^{\mathsf H} C p,
$$

with $C$ the gap table from section 0. The unit vector that makes $p^{\mathsf H} C p$ largest is the top eigenvector of $C$. The next one is the best pattern at right angles to the first, and so on. Eigenvalues $\lambda_n$ rank them.

Same fact, other wording: find a complex label $y_i$ and a template $p$ together so that the outer product $y\,p^{\mathsf H}$ is as close as possible to $S$. The best rank-one answer is $p = p_1$ and $y_i = a_i$. Two things changed relative to the paper:

- nobody hands you $y_i$; $|a_i|$ is estimated from the data;
- $y_i$ is complex, so each event may have its own clock. That clock is $\arg a_i$, which is the time the detector neuron fires (section 1.2).

Shifting every spike in event $i$ by $\delta_i$ multiplies that row of $S$ by $e^{\,j\omega\delta_i}$. In the product $s_k\overline{s_l}$ the extra factors are $e^{\,j\omega\delta_i}$ and $e^{-j\omega\delta_i}$, so they cancel. $C$ and its eigenvectors do not change. Only $\arg a_i$ moves, by $\omega\delta_i$. This holds when the shift does not push spikes across a full turn inside the event.

Apply the paper's read-out to eigenvector $n$: synaptic strength $|p_{nk}|$, delay $D_n - \tau_{nk}$ with $\tau_{nk} = \arg(p_{nk})/\omega$. By section 1.2, that neuron's peak on event $i$ is $|a_{in}|$, and it peaks at $D_n + \arg(a_{in})/\omega$. Complex PCA is a closed-form list of the paper's delay-line neurons, ordered by how much of the data each one explains.

On a simulated 20-cell sequence (6 ms steps, 40% of cells active per event, 5 ms jitter, bursts, 1500 events, no labels) the recovered order matched the true order perfectly (Spearman $1.000$). Timing error was about $2$–$4$ ms. The top eigenvalue was about $9$–$12$ times the second.

---

## 3. The half-circle is a rule about $T$, not a place arrows must sit

Writing the angle as $\pi t/T$ with times between $0$ and $T$ is the same circle as a full turn of length $2T$. The only real requirement is that a time gap and "that gap minus one full period" must not become the same arrow for any gap you care about. That is a limit on how small $T$ may be (section 5). It does not require every arrow to sit on the upper half-plane. PCA never uses a fixed "zero angle = start of the window" (section 2).

---

## 4. Do not subtract the average arrow

Ordinary PCA subtracts each column's average. Here the column average $\bar s_k = \frac{1}{M}\sum_i s_{ik}$ is the wrong object, for four reasons.

1. A silent cell is the number $0$: no evidence. After subtraction it becomes $-\bar s_k$, an arrow pointing away from the cell's usual time, and that fake arrow enters every gap.
2. If every pattern was anchored on its first spike, all arrows lie in the upper half-plane. Their average then points upward mostly because of that anchoring, which is an accident of which cell fired first.
3. Subtracting the average breaks the shift cancellation of section 2. The centered gap table equals $C_{kl} - \bar s_k\overline{\bar s_l}$, and the second term changes when you spin the events.
4. Complex data have a second table, built **without** the conjugate: average of $s_{ik}s_{il}$. Standard complex PCA ignores that table, which is legitimate when the table is zero ("proper" data). Half-circle arrows make it nonzero. The leftover is absolute phase: where the pattern sits in the window. That is the arbitrary clock again.

The clean replacement is: give each event a random spin $e^{\,j\phi_i}$, with $\phi_i$ uniform on the circle, and call the spun arrows $\tilde s$. Then, on average,

- each cell's mean arrow is $0$ (spins point all ways);
- the gap table is still $C$, because a spin cancels in $s\overline{s}$;
- the no-conjugate table averages to $0$, so the spun data are proper.

So the uncentered $C$ **is** the centered covariance, once you treat the clock start as meaningless. That is the reference-invariant way to "center". You do not subtract $\bar s$. You refuse to privilege a clock.

---

## 5. What is still in the way: cells that fire together, regardless of order

After section 4, a different nuisance remains, and it lives in the **lengths**, not in the angles. A crowded event has many long arrows. Cells that are often co-active get a large **real** $C_{kl}$ even if their order is random. One synchronous burst, everybody at once, is a nearly real rank-one block. That block is the analogue of "mean firing rate" in ordinary PCA.

Remove it by asking what $C$ would be if timing carried nothing.

Null, inside one event: keep the set of cells that fired, and keep the bag of arrows, but hand the arrows out at random. For event $i$ let $A_i$ be the active cells, $n_i$ how many, $z_a$ their arrows, and $b_i$ a 0/1 indicator of who fired. Two real numbers describe the bag:

$$
\delta_i = \frac{1}{n_i}\sum_{a\in A_i}|z_a|^2,
$$

the average arrow power, and

$$
\gamma_i = \frac{\big|\sum_{a} z_a\big|^2 - \sum_{a}|z_a|^2}{n_i(n_i-1)}
$$

(with $\gamma_i = 0$ if fewer than two cells fired). $\gamma_i$ is large when the arrows in the event already point the same way: a tight synchronous volley. It is smaller when the arrows are spread around the circle: a sequence.

Under a random hand-out, every active pair has expected product $\gamma_i$, and each active cell has expected power $\delta_i$. Average over events:

$$
C^{0} = \frac{1}{M}\sum_i\Big[\gamma_i\big(b_i b_i^{\mathsf T} - \mathrm{diag}\,b_i\big) + \delta_i\,\mathrm{diag}\,b_i\Big].
$$

$C^{0}$ is real and symmetric. No shuffle loop is required; it is one pass over the events. Then

$$
C^{\Delta} = C - C^{0}.
$$

Because $C^{0}$ is real, the vertical part is untouched: $\mathrm{Im}\,C^{\Delta} = \mathrm{Im}\,C$. Lead and lag stay. The horizontal part loses the co-activation that the null predicts. If the random hand-out were the truth, $C^{\Delta}$ would be $0$ on average.

Checked against 400 actual shuffles: the closed form missed by at most $0.0012$ on entries of size about $0.47$, which is the shuffle's own noise. On the sequence data, eigenvalues of $C$ were about $[3.87,\ 0.32,\ 0.31,\ldots]$; of $C^{\Delta}$, $[2.12,\ -0.01,\ -0.02,\ldots]$. A real shuffle's top eigenvalue of $C^{\Delta}$ sat below $0.10$ on $99\%$ of draws, so $2.12$ is far above "no timing".

### Which matrix to read

- **Templates, by default, from $C$.** It has no negative eigenvalues, so the neuron story of section 1.2 is exact. When every gap is a consistent angle, $C_{kl}$ is a positive weight times $e^{\,j(\theta_k-\theta_l)}$, and the top eigenvector's angles are the firing times.
- **Significance, from $C^{\Delta}$.** Count eigenvalues of $C^{\Delta}$ above the shuffle's 99th percentile. $p^{\mathsf H} C^{\Delta} p$ is the extra size along $p$ beyond co-activation.
- **When co-activation swamps the template, switch.** If the top eigenvector of $C$ has all cells crammed into a small arc, timing lost to the synchronous block. $C^{\Delta}$ is then the better template, with one caveat: it also subtracts a bit of the sequence itself (a sequence's arrows do not average to exactly zero), so angles move slightly. The repair is: keep events whose timing excess $\Delta_i$ (section 7) is positive, recompute $C$ on those events only, and take that eigenvector. Each piece was checked; the full loop in one pipeline was not.

The swamp, in numbers. 60% real sequence events, 40% synchronous bursts (70% of cells inside 25 ms). From $C$: order still perfect, but times were wrong by 24 ms and the pattern only spanned $44^\circ$ instead of the true $180^\circ$ (everyone pulled together). From $C^{\Delta}$: order still perfect, error 8 ms, span $177^\circ$. A generalized eigenproblem $C p = \lambda C^{0} p$ was worse (27 ms) and is not the thing to use.

---

## 6. Choosing the window $T$

Let $L$ be the longest within-event gap you need to resolve (use the 99th percentile of event length). Sweeping $T/L$ on the same sequence data:

| $T/L$ | pattern span | order recovered? | time error |
|---|---|---|---|
| 10 | $18^\circ$ | yes | 5.5 ms |
| 1 | $180^\circ$ | yes | about 2 ms |
| 0.7 | $257^\circ$ | yes | 0.5 ms |
| 0.55 | $327^\circ$ | yes | 0.6 ms |
| 0.51 | $353^\circ$ | no | 57 ms |

What this means:

1. The eigenvector's angles, read **on the circle**, stay within about $0.07$ rad of the truth for every $T$ in the sweep, even when the pattern runs past a full turn. Wrapping does not break the decomposition. It breaks the step that turns angles into milliseconds.
2. That step is safe while the cells leave an empty gap somewhere on the circle (span below $360^\circ$), i.e. $T > L/2$. Cut the circle in the widest empty gap and unwrap from there. Anchoring on the strongest cell, which the earlier plan did, fails once the pattern spans more than half a turn. The paper's rule $T \ge L$ is safe, and stricter than necessary.
3. Time error gets **smaller** as $T$ shrinks toward $L/2$. A stray spike knocks an arrow by some angle; converting angle to time multiplies by $T/\pi$. A smaller $T$ turns the same angular knock into fewer milliseconds. Pure jitter of the spike clock does not have this $T$ factor. Stop shrinking $T$ when the empty gap disappears.
4. A huge $T$ flattens every arrow onto the positive real axis, $s_{ik}$ becomes the spike count, and the vertical share of $C$ goes to $0$. That is ordinary count PCA.

Use $T$ between $0.7L$ and $L$. If each event is only a piece of a longer sequence, recover order by adding short neighbor gaps, as in the basic guide, rather than by one cut of the circle.

---

## 7. More than one spike

The arrow for cell $k$ in an event is the Fourier coefficient of its spikes at frequency $\omega$:

$$
\hat x = \sum_m e^{\,j\omega t_m}.
$$

Three ways to turn that into the entry of $S$:

- **Sum.** $s = \hat x$. Matches the cosine neuron spike for spike. A cell that fires five times can have length 5, so rate leaks into the template weights.
- **Mean.** $s = \hat x / n$ when the cell fired $n$ times. Length is between $0$ and $1$: a tight burst has length near $1$ at the burst center; scattered spikes shrink toward $0$ and automatically count less. This is the default.
- **First spike only.** The paper's choice. One early stray spike sets the cell's time.

On the bursty 20-cell simulation, all three got the order exactly. Time error: first spike 4.1 ms, sum 2.3 ms, mean 3.7 ms. Prefer the mean. Use the sum when you need the neuron story to be literal.

### A cell with two times

If a cell fires at 30 ms and again at 90 ms, one frequency can only store the circular average. The template put that cell at 59 ms, with a weight $1.27$ times the typical cell. The pairwise reliability

$$
\rho_{kl} = \frac{\big|\sum_i s_{ik}\overline{s_{il}}\big|}{\sum_i |s_{ik}|\,|s_{il}|}
$$

dropped to $0.81$ for that cell, against $0.98$ for its neighbors. A low $\rho$ is the flag.

Higher frequencies $2\omega, 3\omega, \ldots$ can show two peaks. A clock shift multiplies frequency $h$ by $e^{\,jh\omega\delta}$, so a matrix that mixes frequencies is **not** shift-invariant. Do it in two passes:

1. Run the one-frequency PCA. $\phi_i = \arg a_{i1}$ is the detector's firing time. Spin event $i$ back by $-\phi_i$.
2. On the aligned events, average $e^{-jh(\omega t - \phi_i)}$ for $h = 0,1,\ldots,H$. The reverse series is each cell's timing profile relative to the template. $h = 0$ is the spike count.

With $H = 8$ and the strongest 70% of events, single-field cells landed within 0.5 ms of their true times. The double-field cell showed peaks at 33 ms and 86 ms (truth 30 and 90).

---

## 8. $|a_i|$ is a bad detector of sequences

$|a_i|$ is the peak voltage. A crowd of simultaneous spikes drives that peak just as hard as a true sequence: enough input at once crosses the threshold, delays or not. Magnitude (how many cells) and alignment (whether they followed the template) are mixed in one number.

Use the same null as section 5, now for a single event. Shuffle the arrows among the cells that fired, and compute the expected $|a_i|^2$. The formula is the one-event version of $C^{0}$ contracted with $p$; you do not need to write it out to use the idea. The timing excess is

$$
\Delta_i = |a_i|^2 - \big(\text{that expectation}\big).
$$

Averaged over events, $\frac{1}{M}\sum_i \Delta_i = p^{\mathsf H} C^{\Delta} p$. So $\Delta_i$ is the per-event piece of the matrix in section 5.

A perfectly synchronous volley has $\gamma_i = \delta_i = 1$ and $\Delta_i = 0$, whatever its size. A template-ordered event has $|a_i|$ equal to the sum of the participating weights, while the shuffle expectation stays smaller, so $\Delta_i$ is large and positive.

On the 60/40 sequence-versus-burst data, with the template taken from $C^{\Delta}$:

- raw $|a_i|$ separated the two kinds of events with area under the curve $0.544$ (useless; chance is $0.5$);
- $|a_i|$ divided by the event's total arrow length reached $0.877$;
- $\Delta_i$ reached $0.967$.

A score that only deletes the diagonal "cell with itself" terms still scored bursts higher than sequences. Synchronous cross-terms are the problem, and $\Delta_i$ removes them. Report $\Delta_i$ as the sequence strength, not $|a_i|$.

---

## 9. Merged experiences versus two sequences in the same event by chance

### 9.1 Three possibilities

Events sometimes contain cells from sequence $A$ and sequence $B$.

- **Coincidence.** Both are present, but the time offset between them wanders from event to event, on the scale of the whole event.
- **Merge.** $B$ follows $A$ (or the reverse) with one gap $g$ that stays put, jittered only as much as the spikes inside a sequence.
- **One sequence.** $A$ and $B$ were never two blocks. There is no seam in $|C|$.

### 9.2 What a stable gap does to $C$

Order the cells of $A$, then the cells of $B$. The diagonal blocks of $C$ are each sequence's own gap table. The off-diagonal block is the average of the **gap arrows between the two blocks**. Its length $\beta$ is "how often both appear" times "how stable the gap is". Its angle $\psi$ is $\omega$ times the average gap.

If the gap is stable, the top eigenvector is the stitched pattern: $A$'s arrows, then $B$'s arrows rotated by that gap. The second eigenvector is the "A minus B" pattern. The two eigenvalues split apart by an amount proportional to $\beta$.

If the gap wanders, the gap arrows point different ways and average toward $0$, so $\beta \to 0$. The two eigenvalues tie, and any mixture of "A" and "B" is as good an eigenvector as the pure ones. You cannot read a junction off that tie.

If they were one sequence all along, $|C|$ has no block boundary to find.

Checked on 3000 events: 40% pure $A$, 40% pure $B$, 10% merged with a 25 ms gap, 10% coincident with the offset thrown uniformly between $-80$ and $+80$ ms. Average $|C|$ inside a block was $0.168$; across the seam, $0.043$. The **phase** of the cross-block still matched the merged template with coherence $0.993$. The top eigenvector ordered all 30 cells as $A$ then $B$ (Spearman $0.94$) but read the gap as $-40$ ms, not $+25$ ms. The coincident events pull the average gap toward $0$. The eigenvector reports an **average** lag. It does not say whether that average came from one stable junction or from a mixture. The next statistic does.

### 9.3 The gap jitter, event by event

1. Find the two blocks from $|C|$ alone: cells linked by a large $|C_{kl}|$ form a graph; the standard spectral cut of that graph (its Fiedler vector) splits it. In both simulated datasets the split recovered $A$ and $B$ exactly. Clustering events in the top score-plane, as in the research plan, is the alternative when the blocks share cells.
2. Take the top eigenvector of $C$ inside each block. Those are the sub-templates $p_A$ and $p_B$.
3. Score every event on both, $a_{iA}$ and $a_{iB}$. The relative phase

$$
\psi_i = \arg\big(a_{iA}\,\overline{a_{iB}}\big)
$$

is $\omega$ times the gap in that event (up to one fixed offset). A clock spin of the whole event hits both scores the same way and cancels in the product.

4. Among events that express both templates, the concentration is

$$
\kappa_{AB} = \frac{\big|\sum_i a_{iA}\,\overline{a_{iB}}\big|}{\sum_i |a_{iA}|\,|a_{iB}|} \in [0,1].
$$

$\kappa = 1$ means the gap was identical every time. For a roughly bell-shaped scatter of gaps, $\kappa = e^{-\sigma_\psi^2/2}$, so the gap's standard deviation in milliseconds is

$$
\hat\sigma_g = \frac{T}{\pi}\sqrt{-2\ln\kappa_{AB}}.
$$

Call it a merge when $\hat\sigma_g$ is on the scale of ordinary spike jitter (about 5–15 ms for replay), and a coincidence when $\hat\sigma_g$ is on the scale of the whole event. If $\psi_i$ has several clumps, fit one circular bump per clump and apply the same rule to each. The clump's mean angle, divided by $\omega$, is that junction's gap.

Checked:

- Merged events alone: $\kappa = 0.996$, so $\hat\sigma_g = 7.3$ ms. The simulated within-sequence jitter was 5 ms.
- Coincident events alone: $\kappa = 0.849$, so $\hat\sigma_g = 45.6$ ms. The uniform $\pm 80$ ms offsets really do have standard deviation $46.2$ ms. The formula still worked.
- In the mixture, one tight clump of $\psi_i$ held 134 events, every one of them truly merged. The broad remainder was the coincident class.
- Peaks in a smoothed histogram are not enough. The coincident-only control grew two fake bumps. The width $\hat\sigma_g$ is the test.
- The eigenvalue split is not the test either. $(\lambda_1-\lambda_2)/\lambda_3$ was $5.8$ for the merged mix and $7.3$ for a coincident mix. Bounded random offsets still have $\kappa \approx 0.85$, so $\beta$ does not vanish and the eigenvalues still split.

### 9.4 One level up, not yet run

If $A$ is the $M \times K$ table of complex scores on $K$ templates, then $G = \frac{1}{M} A^{\mathsf T}\overline{A}$ is a gap table **between templates**. Its eigenvectors are candidate stitched trajectories: delay-line neurons whose inputs are the first-level neurons. This was not simulated.

Shared cells (a cell in both $A$ and $B$, with two times) break the clean block split. Cluster events first, then run the gap test inside each cluster, and use the two-pass harmonics of section 7 for the shared cells. That case was not simulated either.

---

## 10. The procedure, in the order you would code it

1. Set $T$ to somewhere between $0.7$ and $1.0$ times the 99th percentile of event length.
2. Encode each cell in each event as the **mean** phasor at $\omega = \pi/T$ (or the sum, if the neuron must be literal). Optionally rescale each cell by its root-mean-square length. Keep pyramidal cells.
3. Build $C = \frac{1}{M} S^{\mathsf T}\overline{S}$. Build $C^{0}$ from the closed form in section 5. Set $C^{\Delta} = C - C^{0}$.
4. Eigenvalues of $C^{\Delta}$. Keep as many components as sit above the shuffle's 99th percentile of the top eigenvalue.
5. Templates from the eigenvectors of $C$. If the top one is crammed into a small arc, take templates from $C^{\Delta}$ instead, then recompute $C$ on events with $\Delta_i > 0$ and read the template from that.
6. For each template: weights $|p_k|$; cut the circle at the widest empty gap to get times $\tau_k$; drop cells whose weight is under 10% of the largest; delays are $D - \tau_k$.
7. For each event: score $a_{in}$, sequence strength $\Delta_{in}$, clock shift $\arg a_{in}$.
8. Merging: split the $|C|$ graph, score the two sub-templates, compute $\kappa$ and $\hat\sigma_g$. Report the gap and the jitter per clump.
9. Optional: align events by $\arg a_{i1}$ and accumulate harmonics $h = 0,\ldots,8$ for cells that look double-peaked.

Cost is one $N\times N$ eigen-decomposition plus one pass over events. For 100 cells and 5000 events that is well under a second.

---

## 11. What this fixes, and what it leaves open

Fixed, with the delay-line neuron still intact:

- no labels, and no shared clock;
- the mean-arrow problem (random spin, section 4) and the co-activation problem ($C^{0}$, section 5);
- more than one spike per cell;
- a window rule: $T > L/2$, cut at the empty gap, smaller $T$ is sharper until that limit;
- an event score, $\Delta_i$, that stays quiet on a synchronous pile-up;
- a gap-jitter test, $\hat\sigma_g$, for a stable junction between two sequences.

Not settled:

- two patterns that **share cells with different times** still land in one subspace and need a clustering step before the templates are clean;
- an event that is a sped-up or slowed-down copy scales every gap, and a spin does not absorb that;
- $C^{\Delta}$ nudges angles (the test span was $177^\circ$ against a true $180^\circ$); the "recompute $C$ on events with $\Delta_i > 0$" repair was not run as one pipeline;
- harmonic profiles inherit whatever pass 1 got wrong;
- shared-cell merges, and the multi-clump fit, are specified and not simulated;
- a template longer than one full turn still needs neighbor-gap addition; one cut of the circle fails there on purpose;
- the simulations used 40–50% of cells active in an event. Real ripples are closer to 5–15%. The shuffle threshold on $C^{\Delta}$ is what has to carry that, and it is the first thing to check on data.
