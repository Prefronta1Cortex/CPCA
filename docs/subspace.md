# Reading the order from the whole reduced space (option 1)

Open with `Cmd+Shift+V`. Code: `cpca/subspace.py`, `scripts/run_subspace.py`. Results: `results/subspace/summary.md`; figure: `figures/subspace_hc11.png`.

## Why

On both datasets the reduced space as a whole is organised like the waking map (topology, waking-sleep agreement), but single components match the map only slightly above chance, and not more often with more place cells (`docs/huszar.md`). If the order is spread over the kept components rather than carried by one of them, a read-out that uses all of them at once should recover it.

## Design (written on October 28 before running; hc-11 only)

All read-outs come from the same centred complex PCA fit on all NREM population bursts of each session and sleep epoch (as in `scripts/run_hc11.py`), components above the shuffle null. Each read-out gives one order of the cells (one output), so each is tested exactly like the best component, with the best-of-outputs null over both running directions and both senses (`scripts/plot_hc11_sequences.best_map_match`, 1000 permutations; the per-component read-out has k outputs and its null accounts for that).

| read-out | what it uses |
|---|---|
| **best component** (current) | each kept component's phases, best of k |
| **subspace seriation** | the closeness of every pair of cells in the reduced space ($\mathrm{Re}\,C_K$ normalised, phase and co-membership), cells ordered along its leading non-trivial graph-Laplacian direction (spectral seriation) |
| **subspace phase synchronisation** | the phase of every pair in $C_K$: the cells' times are the angles of the leading eigenvector of $C_K$ normalised by each cell's total coupling ($D^{-1/2} C_K D^{-1/2}$, angular synchronisation), read out like a component |
| *direction-blind seriation* (control) | the same seriation on $|C_K|$ normalised. Not timing-free: with forward and backward plays mixed, $|C_K|$ between two cells falls with the cosine of their lag, so it keeps the order but loses its direction (found on planted data before running hc-11) |
| *count PCA seriation* (control) | the same seriation on count PCA's reduced space (spike counts per event, no timing) |

On the three circular tracks the seriation order is the angle of the two leading non-trivial Laplacian directions (a ring), on the linear tracks the leading one; this uses the track's shape, not the map. Cells: all pyramidal cells; the test scores the cells with a field.

**Expected if the order is spread over the subspace.** A subspace read-out (seriation or phase synchronisation) recovers the map in more POST sessions than the best component (2 of 8 with these measures), still in few PRE sessions. **If timing is what carries it**, the timing-aware read-outs beat count PCA seriation, the only control with no timing at all; if they do not, the subspace order is co-firing, which the topology test already showed.

## Results (`figures/subspace_hc11.png`, `results/subspace/summary.md`)

The expectation was not met. Sessions above chance, POST / PRE: best component 2/8 / 0/8; subspace seriation 1/8 / 0/8; subspace phase synchronisation 2/8 / 0/8; direction-blind seriation 3/8 / 1/8; count PCA seriation 1/8 / 0/8. No read-out of the whole subspace beats the best single component, and none separates from the others by more than one session in eight. The read-outs also disagree on which sessions pass (best component: Achilles 10/25 and Cicero 09/01; phase synchronisation: Achilles 10/25 and Gatsby 08/28; direction-blind seriation: Buddy 06/27, Cicero 09/17, Gatsby 08/02), which is what weak effects near threshold look like.

So on hc-11 the order is not hidden in the subspace waiting for a better read-out: the topology result (which pairs of cells are close) does not translate into a single cell order that matches the map. This is consistent with the topology being carried mostly by co-firing of cells with nearby fields, which gives local neighbourhoods but no global sequence. Option 1 is closed on hc-11; the decoder-selected ceiling (option 3) is the test that would say whether the sleep events contain enough order for any read-out.
