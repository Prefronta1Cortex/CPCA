# Topology test: does the reduced space keep the shape of the waking map?

Open with `Cmd+Shift+V` for formulas. Code: `cpca/topology.py` (measures and test), `scripts/run_topology.py` (all data, parallel), `scripts/summarize_topology.py` (figures, table). Results: `results/topology/summary.md`; figure: `figures/topology_hc11.png`.

## Why

Picking out route components without the map may be impossible: some replayed content has no recorded map, or no spatial meaning at all. What can be checked is whether the reduced space as a whole preserves the topology of the map: whether two cells that are close in the reduced space are also close on the map, where "the map" includes running direction (heading), so it is not one line. The test uses all kept components at once, not one component at a time.

## The two closenesses

**In the reduced space.** With the kept components $V_K$ and eigenvalues $\Lambda_K$, the reduced covariance is $C_K = V_K \Lambda_K V_K^{H}$. For cells $k$ and $l$,

$$
r_{kl} = \frac{\mathrm{Re}\,(C_K)_{kl}}{\sqrt{(C_K)_{kk}\,(C_K)_{ll}}},
$$

the cosine between the two cells' points in the reduced space. It is high when the cells load on the same components *and* at similar phases (similar firing times), negative when they sit at opposite phases, and it does not change when a component is multiplied by an arbitrary phase. Variants: $|(C_K)_{kl}|$ instead of the real part ignores timing and keeps co-membership; $K$ = all components gives the full-space (complex) correlation of the arrows, no reduction.

**On the map.** Each place cell's waking rate maps in the two running directions (40 bins each, smoothed, wrapping on circular tracks) are concatenated, and two cells' closeness is the correlation of these vectors. Heading is therefore a dimension of its own (a cell with a field at one place in one direction only is far from a cell at the same place in the other direction), cells with several fields are handled, and no field centre has to be chosen. Place cells are those with a field in at least one direction (the same answer key as everywhere else).

**The statistic.** The Spearman correlation between the two closenesses over all pairs of place cells. Pairs are not independent (each cell is in many pairs), so the null permutes which cell has which map, 2000 times, and recomputes the correlation (Mantel test). A session counts if $p < 0.05$; across sessions, the number of sessions that count is compared with the binomial expectation of 5%.

## Representations compared

From the same sleep events: complex PCA centred (headline: the sleep regime), the same components with timing ignored, complex PCA with all components, complex PCA uncentred, and count PCA (z-scored spike counts per event, components above the same shuffle null; Lopes-dos-Santos et al., 2013), which knows who fires together and nothing about timing.

## Validation first (tables in `results/topology/summary.md`)

Planted data, 80 cells, 3000 events, maps made of Gaussian bumps at the planted positions (one map per planted map):

- **Calibration.** With no topology to find (synchronous volleys only; assemblies unrelated to the map; 30 draws each) the representations report topology in 0–10% of draws, consistent with 5% for 30 draws. Centred cPCA keeps no component on volleys alone, so there is nothing to test.
- **What the test cannot separate.** When neighbouring cells on the map fire together without any order (assemblies made of contiguous stretches of the map), every representation reports the topology, count PCA included (30/30). The test measures topology, not sequence; co-firing of neighbours is enough to pass it.
- **Where timing is what carries it.** When every replay recruits cells along the whole route, co-firing says nothing about the map: count PCA fails (1/10 on one map in both directions, 1/10 on two maps on the same cells), while complex PCA finds the topology in every draw (ρ 0.69–0.82). With partial replays, co-firing carries the map again and count PCA succeeds as well.
- **Real positive control: hc-5.** In the wheel runs every cell fires in every run, so only timing can order them. Fit on half of the runs and scored against the time fields of the other half: uncentred complex PCA keeps the time topology in 35/36 fits aligned and 35/36 unaligned; count PCA in 4/36 and 1/36; timing ignored, 2/36. (Centred complex PCA keeps no component in most aligned fits, as expected when the sequence is in every run.)

## Results on hc-11 (`figures/topology_hc11.png`)

- **The reduced space keeps the map's topology in sleep after the maze, not before.** Centred complex PCA, reduced space: 5 of 8 sessions above chance in POST (ρ 0.15–0.31; binomial p = 1.5 × 10⁻⁵ for 5 of 8) against 1 of 8 in PRE (p = 0.34). POST exceeds PRE in 6 of 8 sessions (sign test p = 0.15); the exceptions are Buddy 06/27, the smallest session (18 place cells), where PRE is above chance, and Cicero 09/17, at chance in both. In the largest session (Achilles 10/25) the waking map's block pattern is visible in the reduced space of POST sleep and not of PRE (`figures/topology_hc11.png`, B and C).
- **Most of it is who fires with whom.** Count PCA, without any timing, does about as well (5/8 POST, 1/8 PRE). Within complex PCA timing still adds: the phase-aware closeness beats the timing-blind one in most POST sessions and by a wide margin in two (Achilles 11/01: 0.31 against 0.24; Gatsby 08/28: 0.21 against 0.04).
- **The reduction keeps the topology.** All components give about the same (6/8 POST, 1/8 PRE); the kept components lose nothing measurable. Uncentred components carry less (4/8 POST).

## What this does and does not show

It shows that the space complex PCA reduces sleep to is organised like the waking map after the experience and not before it, a population-level statement that does not depend on picking components. It does not show that the organisation is sequential: on hc-11 a representation with no timing at all does as well, and the planted control shows the test passes on co-firing alone. Where timing is the only information (planted whole-route replays, hc-5) complex PCA passes and count PCA fails, so the timing is used when it is there. Seven of eight sessions have 18–38 place cells, so per-session power is limited and the across-session count is the main evidence.
