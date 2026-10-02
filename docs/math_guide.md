# The math, one step at a time

This is a companion to `docs/complex_pca_research_plan.md` and to the functions in `cpca/theory.py`. It walks through the calculation with small numbers. It does not try to prove the linear algebra.

Open this file and press `Cmd+Shift+V` so the formulas render. The native Preview tab will show the dollar signs as raw text.

Test line: if you can read $e^{j\theta} = \cos\theta + j\sin\theta$ as a typeset formula, rendering works.

Every number below was checked against `cpca/theory.py`.

---

## Step 1. What the calculation is for

A reactivation event is a short burst of spikes from many cells. Suppose three cells fire in the order A, then B, then C, in every event. We want a method that reads that order back out, using only the spike times inside the events.

The method has four moves:

1. Turn each spike time into an arrow.
2. From those arrows, measure the time gap between every pair of cells.
3. Average those gaps across events.
4. Ask which pattern of arrows shows up most strongly.

That pattern is one principal component. Its arrow lengths say how much each cell takes part. Its arrow angles say when each cell fires relative to the others.

---

## Step 2. An arrow is a length and an angle

A complex number is an arrow in the plane. It has a length and a direction.

$$
z = x + jy
$$

$x$ is how far the arrow sticks out horizontally. $y$ is how far it sticks up. $j$ is a marker that means "this part is the vertical one". The only rule you need is

$$
j^2 = -1.
$$

The same arrow can be written by its length $r$ and its angle $\theta$:

$$
z = r(\cos\theta + j\sin\theta) = r e^{j\theta}.
$$

$e^{j\theta}$ is just a short way to write "length 1, angle $\theta$". Angles in these formulas are in radians. If degrees are easier: $180^\circ$ is $\pi$ radians, so $36^\circ$ is $\pi/5$ radians.

Examples:

- Angle $0$: the arrow points right. The number is $1$.
- Angle $90^\circ$, which is $\pi/2$: the arrow points up. The number is $j$.
- Angle $180^\circ$, which is $\pi$: the arrow points left. The number is $-1$.
- Angle $360^\circ$, which is $2\pi$: the arrow is back to pointing right. The number is $1$ again.

So angles are a circle. After one full turn you are at the same arrow.

The length is written $|z|$. For $z = x + jy$,

$$
|z| = \sqrt{x^2 + y^2}.
$$

The angle is written $\arg z$. It is "which way the arrow points".

---

## Step 3. Multiplying arrows adds their angles

If you multiply two arrows, the lengths multiply and the angles add:

$$
r_1 e^{j\theta_1} \cdot r_2 e^{j\theta_2} = r_1 r_2 \, e^{j(\theta_1 + \theta_2)}.
$$

That is the whole reason this encoding works. A time gap is a difference of angles, and a difference of angles is what falls out when arrows are multiplied in the right way.

---

## Step 4. The conjugate flips the angle

The conjugate of $z = r e^{j\theta}$ is the same arrow mirrored across the horizontal axis:

$$
\overline{z} = r e^{-j\theta}.
$$

Same length, opposite angle. In coordinates, $x + jy$ becomes $x - jy$. You flip the sign of the vertical part.

Multiplying by a conjugate subtracts angles:

$$
z_1 \overline{z_2} = r_1 r_2 \, e^{j(\theta_1 - \theta_2)}.
$$

The angle of $z_1 \overline{z_2}$ is "angle of the first, minus angle of the second". Once angles stand for times, that difference is a time gap.

---

## Step 5. Adding arrows measures agreement

Add arrows by adding their horizontal parts and adding their vertical parts, the way you add forces.

- Arrows that point the same way add into a long arrow.
- Arrows that point opposite ways cancel.
- Arrows that point in random directions add into a short arrow.

The length of a sum is large when the angles agree. That is how we will tell "this time gap keeps happening" from "this time gap is all over the place".

---

## Step 6. Turn one spike time into an arrow

Pick a window length $T$, in milliseconds. For a spike at time $t$, and a reference time $t_0$, the arrow is

$$
s = \exp\left(j \pi \frac{t - t_0}{T}\right).
$$

This is `encode_times` in `cpca/theory.py`. The length is $1$. All of the information is in the angle:

$$
\theta = \pi \frac{t - t_0}{T}.
$$

Worked numbers. Let $T = 100$ ms and $t_0 = 0$.

A spike at $0$ ms:

$$
\theta = 0, \qquad s = 1.
$$

A spike at $20$ ms:

$$
\theta = \pi \frac{20}{100} = \frac{\pi}{5}.
$$

$\pi/5$ radians is $36^\circ$. So

$$
s \approx 0.809 + 0.588j.
$$

The arrow has length $1$ and points $36^\circ$ up from the horizontal.

Check the time formula backwards. If you have the angle and want the time,

$$
t - t_0 = \frac{T}{\pi} \theta.
$$

Plug in $\theta = \pi/5$ and $T = 100$:

$$
t = \frac{100}{\pi} \cdot \frac{\pi}{5} = 20 \text{ ms}.
$$

This backward step is `relative_times`.

---

## Step 7. One full turn takes $2T$ milliseconds

Look at the angle formula again:

$$
\theta = \pi \frac{t}{T}.
$$

A full turn is $\theta = 2\pi$. Solve for $t$:

$$
2\pi = \pi \frac{t}{T} \quad \Rightarrow \quad t = 2T.
$$

With $T = 100$ ms, a spike at $0$ ms and a spike at $200$ ms become the same arrow. Both are the number $1$.

A spike at $t = T = 100$ ms has angle $\pi$, which is the number $-1$. That is half a turn, the opposite direction from $0$.

This is why $T$ must be at least as long as the longest gap inside one event. If two spikes in the same event are more than $2T$ apart, they land on top of each other and the gap is lost. If they are more than $T$ apart, the angle is more than half a turn, and "B is 120 ms after A" starts to look similar to "B is a bit before A" once you only have the arrow. Inside one event, keep every gap shorter than $T$.

Across many events the recovered pattern can still be longer than $2T$. Step 24 explains how, by adding short gaps end to end.

---

## Step 8. Several spikes from one cell become one arrow

A cell can fire more than once in one event. Encode each spike, then add the arrows. That is still `encode_times`: it sums.

Same time, twice. Two copies of the same arrow:

$$
e^{j\theta} + e^{j\theta} = 2e^{j\theta}.
$$

The angle stays at the spike time. The length becomes $2$. A tight burst is one strong arrow at the middle of the burst.

Spikes far apart. With $T = 100$, spikes at $0$ and at $100$ ms:

$$
1 + (-1) = 0.
$$

They point opposite ways and cancel. The length $0$ is the method saying "this cell's timing in this event is not one moment".

Spikes a quarter turn apart. Spikes at $0$ and $50$ ms, $T = 100$. The second angle is $\pi/2$, so the second arrow is $j$:

$$
1 + j.
$$

The length is $\sqrt{2} \approx 1.41$, which is less than $2$. The angle is $45^\circ$, halfway between them. The cell gets one arrow, weaker than a tight burst, pointing at the middle.

A cell that does not fire gets the number $0$. The zero arrow has no direction. In the averages below it adds nothing. Silence is "no vote", not "fired at time zero". Time zero would have been the arrow $1$.

---

## Step 9. One event is one row of arrows

Suppose two cells, A and B, and $T = 100$ ms, $t_0 = 0$.

Event 1: A at $0$ ms, B at $20$ ms.

$$
s_A = 1, \qquad s_B \approx 0.809 + 0.588j.
$$

Event 2: the same gap, but the whole event happens $10$ ms later. A at $10$ ms, B at $30$ ms.

$$
s_A \approx 0.951 + 0.309j, \qquad s_B \approx 0.588 + 0.809j.
$$

Stack the events as rows and the cells as columns. This table is the matrix $S$. It has one row per event and one column per cell. `encode_matrix` builds it. A missing spike is stored as "not a number" and becomes $0$ in $S$.

$$
S \approx
\begin{bmatrix}
1 & 0.809 + 0.588j \\
0.951 + 0.309j & 0.588 + 0.809j
\end{bmatrix}
$$

---

## Step 10. Shifting a whole event only spins that row

Take event 1 and add $10$ ms to both spikes. You get event 2. Every angle in the row grows by the same amount,

$$
\Delta\theta = \pi \frac{10}{100} = 18^\circ.
$$

Spinning a whole row by the same angle is multiplication by one unit arrow:

$$
\text{row}_2 = e^{j\pi/10} \cdot \text{row}_1.
$$

You can check: $e^{j\pi/10} \approx 0.951 + 0.309j$, and

$$
(0.951 + 0.309j) \cdot 1 \approx 0.951 + 0.309j,
$$

$$
(0.951 + 0.309j)(0.809 + 0.588j) \approx 0.588 + 0.809j.
$$

That is event 2. The time gap inside the event did not change. Only the clock start did.

If you also move the reference $t_0$ by the same $10$ ms, the angles do not change at all. $t - t_0$ is the same number. The row itself is unchanged.

You do not need to line events up on a ripple peak before encoding. Step 12 shows why the gap matrix ignores a per-event spin.

---

## Step 11. The gap between two cells is one arrow

For a single event, the gap arrow from cell $l$ to cell $k$ is

$$
s_k \, \overline{s_l}.
$$

Its angle is $\theta_k - \theta_l$, which is the time of $k$ minus the time of $l$.

Event 1, gap of B relative to A:

$$
s_B \, \overline{s_A} \approx (0.809 + 0.588j) \cdot 1 = 0.809 + 0.588j.
$$

Angle $36^\circ$, which converts back to $20$ ms.

Event 2, same pair. Conjugate of $s_A$ flips the sign of the vertical part:

$$
\overline{s_A} \approx 0.951 - 0.309j.
$$

Multiply by $s_B \approx 0.588 + 0.809j$. The product is again

$$
0.809 + 0.588j.
$$

Same gap arrow, even though event 2 was shifted by $10$ ms. The extra spin $e^{j\phi}$ on the row meets its own conjugate $e^{-j\phi}$ and cancels:

$$
(e^{j\phi} s_k) \, \overline{(e^{j\phi} s_l)} = s_k \, \overline{s_l}.
$$

---

## Step 12. Average the gap arrows. That average is $C$

Do this for every event, then average. For cells $k$ and $l$,

$$
C_{kl} = \frac{1}{M} \sum_{i=1}^{M} s_{ik} \, \overline{s_{il}}.
$$

$M$ is the number of events. $s_{ik}$ is the arrow for cell $k$ in event $i$.

In code, with $S$ the events-by-cells table:

```text
C = (S.T @ S.conj()) / M
```

That is `lag_matrix`. The `.T` swaps rows and columns so the sum runs over events. `.conj()` is the conjugate. The division is the average.

For the two events above, both gap arrows are $0.809 + 0.588j$, so the average is the same arrow. The full $2 \times 2$ matrix is

$$
C \approx
\begin{bmatrix}
1 & 0.809 - 0.588j \\
0.809 + 0.588j & 1
\end{bmatrix}.
$$

Read it entry by entry.

- $C_{AA} = 1$. This is the average of $|s_A|^2$. Each event had one spike from A, and a single spike has length $1$, so $1^2 = 1$.
- $C_{BB} = 1$, for the same reason.
- $C_{BA} \approx 0.809 + 0.588j$. Angle about $+36^\circ$. B fires about $20$ ms after A, and the length is $1$, so the gap was the same in both events.
- $C_{AB}$ is the conjugate of $C_{BA}$. It is the same gap read in the other direction: A fires $20$ ms before B.

The diagonal is participation: how strong each cell's arrows were, on average. It does not know order.

The off-diagonal length is large when the pair is often co-active and their gap keeps the same sign. The off-diagonal angle is the typical gap. A positive angle of $C_{BA}$ means B leads in the sense of our formula $s_B \overline{s_A}$: B's time minus A's time is positive, so B fires later.

The real part of an off-diagonal entry is large when the two cells fire near the same time (angle near $0$, arrow nearly horizontal). The imaginary part, the vertical part, is large when one cell reliably fires earlier than the other.

---

## Step 13. The timing share $\eta$

Some of $C$ is horizontal (real) and some is vertical (imaginary). The timing share is

$$
\eta = \frac{\|\mathrm{Im}\, C\|_F}{\|C\|_F}.
$$

$\|C\|_F$ means: square the length of every entry, add those squares, take the square root. It is the size of the whole matrix. The top is the same thing computed on the vertical parts only. $\eta$ is near $0$ when every gap arrow is horizontal, which means "no consistent lead or lag". $\eta$ is larger when lead and lag are a real part of the matrix.

`timing_share` computes this.

$\eta$ depends on $T$. If you pick a huge $T$, every angle $\pi t / T$ is tiny, every arrow is almost horizontal, and $\eta$ is tiny even if the order is perfect. Always compare $\eta$ with a shuffle at the same $T$ (Step 27). A small $\eta$ by itself is not a verdict.

---

## Step 14. Leave the column mean alone

Ordinary PCA often subtracts the average of each column before it starts. Here that average is an arrow, and subtracting it does two bad things.

A silent cell is $0$. Subtracting a nonzero average turns silence into some other arrow, so a cell that did not fire starts casting a vote.

A per-event spin leaves $C$ unchanged, as Step 12 showed. After you subtract column averages, that cancellation stops. Two copies of the same events, shifted differently, produce different matrices.

Use the plain average in Step 12. That is the uncentered lag matrix.

---

## Step 15. What the principal component is asking

$C$ is a table of every pair. A principal component compresses that table into one pattern: an arrow per cell.

The pattern is a list $p$, one complex number per cell. Try it on every event. The score of event $i$ is

$$
a_i = \sum_k s_{ik} \, \overline{p_k}.
$$

In words: take each cell's arrow in the event, spin it by the opposite of the pattern's angle, weight it by the pattern's length, and add. If the event's angles match the pattern, every term points to the right and $a_i$ is long. If the event's angles are wrong, the terms point different ways and $a_i$ is short.

The first principal component is the pattern $p$ that makes $\sum_i |a_i|^2$ as large as possible, among patterns with total length $1$. The next component is the best pattern that is "sideways" from the first, and so on.

You do not search for $p$ by trial and error. For a matrix built the way $C$ is built, the patterns are the eigenvectors of $C$, and the scores-squared add up to the eigenvalues. `eigh_desc` computes them and sorts the eigenvalues from large to small. Each eigenvector is one column of arrows, one arrow per cell.

Two facts that make the output easy to read:

- The eigenvalues are ordinary real numbers, and they are $\ge 0$. A large eigenvalue means that pattern accounts for a lot of the gap table.
- The arrows in one eigenvector are defined only up to a global spin. If $p$ is an eigenvector, so is $e^{j\alpha} p$. Spinning the whole pattern does not change the time gaps inside it. Step 17 fixes the spin so the numbers are comparable.

---

## Step 16. The $2 \times 2$ example, all the way through

From Step 12,

$$
C \approx
\begin{bmatrix}
1 & 0.809 - 0.588j \\
0.809 + 0.588j & 1
\end{bmatrix}.
$$

The eigenvalues are $2$ and $0$. One pattern explains the table. The other explains nothing.

The eigenvector that belongs to $2$, before any cleanup, comes out of the solver as approximately

$$
\begin{bmatrix} -0.707 \\ -0.572 - 0.416j \end{bmatrix}.
$$

Both entries have length $0.707$. The negative signs are a global spin of half a turn. They are not a statement that the cells are "negative".

---

## Step 17. Anchor the pattern on its strongest cell

`anchor_phase` finds the entry with the biggest length and spins the whole pattern so that entry points right (angle $0$, real and positive).

In Step 16 the two lengths are equal, so the first cell is used. Its angle is $180^\circ$. Spinning everyone by $-180^\circ$ is multiplication by $-1$:

$$
p \approx
\begin{bmatrix}
0.707 \\
0.572 + 0.416j
\end{bmatrix}.
$$

Cell A now sits at angle $0$. Cell B sits at angle $36^\circ$.

If two cells tie for the longest arrow, the computer's choice of which one becomes angle $0$ can flip. The time gaps between cells stay the same. Read gaps, or call `aligned_phase_error`, when you compare two patterns.

---

## Step 18. Read the angles as times

$$
\tau_k = \frac{T}{\pi} \arg p_k.
$$

With $T = 100$:

- Cell A: angle $0$, so $\tau_A = 0$ ms.
- Cell B: angle $36^\circ = \pi/5$, so $\tau_B = 20$ ms.

The pattern says B fires $20$ ms after A. That is the gap we put in.

The lengths $|p_A| = |p_B| \approx 0.707$ say the two cells were equally reliable. They are equal here because each cell fired once in every event, with a fixed gap. In a long sequence where each event only covers a piece, a cell that appears more often gets a longer arrow. Length mixes "how often" with "how consistent". The angle is the time.

---

## Step 19. The score of one event

The score is Step 15's formula, `scores` in code:

$$
a_i = \sum_k s_{ik} \, \overline{p_k}.
$$

Use event 1, $s_A = 1$, $s_B \approx 0.809 + 0.588j$, and the anchored $p$ from Step 17. Then $\overline{p_B} \approx 0.572 - 0.416j$, and

$$
a \approx (1)(0.707) + (0.809 + 0.588j)(0.572 - 0.416j) \approx 1.414.
$$

The imaginary part is $0$. The score is real and positive. Its size $1.414$ equals $|p_A| + |p_B|$. Every cell's arrow lined up with the pattern, so the lengths added in full.

$|a_i|$ is how strongly event $i$ matches the pattern. $\arg a_i$ is the leftover clock shift of that event. Event 2 would get the same length and an angle of $18^\circ$, because the whole event was $10$ ms later and we did not remove that shift. The match is just as good.

The delay-line reading is the same arithmetic. Give cell $k$ a synaptic weight $|p_k|$ and a delay chosen from $-\arg p_k$. Multiplying the spike arrow by $\overline{p_k}$ is "apply that weight and delay". When the delays are right, the delayed spikes line up and the sum is real and positive. The component is a coincidence detector with a learned delay on each cell.

---

## Step 20. Three cells, with jitter and missing spikes

The picture stays the same with more cells. Put true times at $0$, $20$, and $40$ ms, $T = 100$. On each event, drop each cell with probability $0.3$, and add a random timing error of a few milliseconds. Also spin each event by a random clock shift.

$C$ is now an average of noisy gap arrows. Arrows for the true gap still point roughly the same way, so they add. The eigenvector angles come back within about $1$ ms of $0$, $20$, and $40$, after anchoring. The three lengths stay close to each other. One eigenvalue is several times larger than the other two: one pattern, plus small noise.

A method that only counts spikes, and never looks at time, can still see that these three cells are a group. It cannot see the $20$ ms steps. The angles are the extra information.

---

## Step 21. A very large $T$ forgets time on purpose

As $T$ grows, $\pi t / T$ shrinks toward $0$, so every arrow flattens onto the positive horizontal axis. A cell that fired $n$ spikes then has

$$
s \approx n,
$$

a positive real number, the spike count. $C$ becomes the ordinary table of "how often did these cells fire in the same events", and $\eta$ falls toward $0$.

At $T = 10^9$ ms the difference between $S$ and the spike-count table is already smaller than one part in a million. Complex PCA contains spike-count PCA as the special case "ignore timing". The comparison is fair: same events, same uncentered average, timing switched off by the choice of $T$.

---

## Step 22. Forward and reverse

Suppose the cells also sometimes fire in the opposite order, C then B then A.

In a forward event the gap arrow for a pair points one way, say $+36^\circ$. In a reverse event the same pair points the opposite way, $-36^\circ$. Average those two unit arrows:

$$
\frac{e^{j\pi/5} + e^{-j\pi/5}}{2} = \cos 36^\circ \approx 0.809,
$$

which is a horizontal arrow. The vertical parts cancelled. $C$ becomes almost real, so $\eta$ drops, even though every individual event was well ordered.

The timing did not disappear. It moved into a pair of components. Each event still gets two scores, $a_1$ and $a_2$. The sign of the vertical part of $a_1 \overline{a_2}$,

$$
\mathrm{Im}\!\left(a_1 \overline{a_2}\right),
$$

is positive for one order and negative for the other. In a clean simulation this separates forward from reverse on more than $99\%$ of events. Split the events by that sign, run the average-and-eigenvector steps again on each half, and the angles come back with the right sign: about $+15$ ms per step on one half and $-15$ ms per step on the other, if that was the true spacing.

A single component cannot represent both orders at once. A mixture of the two orders looks like "no timing" if you only look at $\eta$.

---

## Step 23. Each event can show only a piece of a longer sequence

Suppose $30$ cells sit $12$ ms apart, so the full order is $348$ ms long, and $T = 150$ ms. Each event shows a random $100$ ms piece, and inside that piece each cell fires with probability $0.6$, with $3$ ms of jitter.

A cell that never falls inside any window has a column of zeros in $S$. Its row and column in $C$ are zero. Its eigenvector entry is $0$. The angle is meaningless because a zero arrow has no direction. In this setup the first and last cells are never covered, and the other $28$ are.

For the cells that do fire, the gap arrows still agree: a pair that is $12$ ms apart has angle

$$
\pi \frac{12}{150} = 14.4^\circ
$$

whenever both appear. The top eigenvector's angles match those true angles, up to one global spin. With zero jitter the match is exact up to rounding error (far past the 6th decimal in milliseconds). With $3$ ms jitter the leftover error is under about $0.5$ ms.

Why exact when there is no jitter: every entry has the form

$$
C_{kl} = w_{kl} \, e^{j(\theta_k - \theta_l)},
$$

with $w_{kl}$ a plain positive number (how often the pair co-occurred). That is a real-number table, with a spin $e^{j\theta_k}$ attached to each cell. The eigenvector keeps those spins. Its lengths are whatever the real-number table says, which follows how often each cell was covered. Its angles are the spike angles.

The eigenvalue list is a different story. Because different events cover different pieces, the table is not a single pattern plus noise. The second eigenvalue can be a large fraction of the first. The angles of the top component are still right. A big gap after the first eigenvalue is not guaranteed for a long sequence seen in pieces.

---

## Step 24. Sorting angles is not the order, once the sequence wraps

Angles live on a circle of circumference $2T$. With $T = 150$ ms, times $0$ and $300$ ms are the same arrow.

Take cells at $0$, $100$, $200$, and $300$ ms. Their angles are $0^\circ$, $120^\circ$, $240^\circ$, and $360^\circ$. $360^\circ$ is the same direction as $0^\circ$. If you sort the cells by angle, the cell at $300$ ms lands next to the cell at $0$ ms. That sort is wrong.

The fix uses the short gaps, which do not wrap. Neighbors are $100$ ms apart, and $100 < T$, so each neighbor gap is an unambiguous arrow of $120^\circ$, well short of a half turn. Read the angle of $C_{k, k-1}$, convert it with $T/\pi$, and add:

$$
0 \rightarrow 100 \rightarrow 200 \rightarrow 300.
$$

`test_adjacent_lags_reconstruct_unwrapped_order` checks this addition on the $12$ ms chain. You only need a chain of pairs whose gaps are shorter than $T$ and whose arrows are long enough to trust. Cells with length $0$ stay out of the chain.

---

## Step 25. Empty events shrink the scale and leave the pattern

A row of zeros is an event where nobody fired. It adds $0$ to every sum in $C$. It does increase $M$, the count you divide by.

If you append empty rows, every entry of $C$ is multiplied by the same fraction (old $M$, divided by new $M$). Eigenvalues scale by that same fraction. The arrows in the eigenvectors keep the same lengths and the same angle gaps.

Two identical real events, plus one empty event, multiply the eigenvalues by $2/3$. The recovered times stay put.

---

## Step 26. A spike-time shuffle is the "timing does not matter" test

Keep, in each event, the list of which cells fired. Reassign the spike times at random among those cells. Then build $C$ again.

Who fired is unchanged, so each cell still has the same arrow lengths. The diagonal of $C$, which is the average of those squared lengths, stays the same.

The gaps are now random. Gap arrows point every which way and their average gets short. The vertical part of $C$ shrinks, so $\eta$ drops. The top eigenvalue drops, because a single tidy pattern no longer fits.

This is `permute_observed_times`. It is the right shuffle when each cell fires at most once in an event.

---

## Step 27. Bursts need a different shuffle

If a cell fired twice, "swap the times among cells" changes how the arrows add inside a cell, so it changes arrow lengths. Then the diagonal of $C$ moves, and you can no longer say the shuffle touched timing alone.

After encoding, each cell in each event is already one arrow $s_{ik}$ with a length and an angle. The matching shuffle spins each nonzero arrow by a fresh random angle and leaves the length alone:

$$
s_{ik} \leftarrow |s_{ik}| \, e^{j\phi_{ik}}, \qquad \phi_{ik} \text{ random}.
$$

That is `randomize_phases`. Lengths stay, so the diagonal of $C$ stays. Consistent gaps are destroyed, so $\eta$ drops. Use this version whenever cells burst.

Compare $\eta$ and the top eigenvalue with this shuffle at the same $T$. That is the test of "does timing add anything beyond who fired together".

---

## Step 28. The whole calculation in order

1. Cut the recording into events. For each event, list the spike times of each cell.
2. Choose $T$ at least as long as the longest gap you care about inside one event.
3. Encode. One arrow per cell per event: sum `exp(j π (t - t0) / T)` over that cell's spikes. Silence is $0$.
4. Build $C$ by averaging $s_k \overline{s_l}$ over events. Do not subtract column averages.
5. Eigenvectors of $C$, largest eigenvalue first.
6. Spin each eigenvector so its longest arrow has angle $0$.
7. Read $|p_k|$ as participation and $(T/\pi) \arg p_k$ as a relative time.
8. If the times along a chain of strong pairs jump by a full turn, add the short neighbor gaps instead of sorting the angles.
9. Score each event with $\sum_k s_{ik} \overline{p_k}$. The length is the match. The angle is that event's clock shift.
10. If forward and reverse are mixed, split events using the sign of $\mathrm{Im}(a_1 \overline{a_2})$ and repeat steps 4–7 on each half.
11. Shuffle (Step 26 or 27) and check that $\eta$ and the top eigenvalue fall. If they do not, the angles are not carrying a repeated order.

The functions for steps 3–7 and 9 and 11 are in `cpca/theory.py`. The checks are in `tests/test_theory.py`.
