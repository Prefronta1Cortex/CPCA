# Where is the order lost? A supervised ceiling (option 3)

Open with `Cmd+Shift+V`. Code: `scripts/run_ceiling.py`. Results: `results/ceiling/summary.md`; figure: `figures/ceiling.png`. A diagnostic, not part of the method: it uses the waking map to choose events.

## Question

On hc-11 and DANDI 000552 complex PCA recovers the map's order from sleep only slightly above chance, and neither more cells (`docs/huszar.md`) nor a read-out of the whole subspace (`docs/subspace.md`) changes that. Three explanations predict different things: (1) the events carry the order but most events are not replays, so the order is diluted (selection is the bottleneck); (2) the replays are there but complex PCA does not extract their order (the method is the bottleneck); (3) the sleep events carry little order of the map at all (the data are the limit).

## Design (written on October 28 before running)

**Boundaries.** Per session, sleep epoch (PRE, POST) and map (hc-11: the two running directions; 000552: left-arm and right-arm trials), the place cells with a field in that map are split at random into two halves, A and B (5 splits per map).

- **Selection uses A only.** An event is a *replay* of the map if at least 5 A cells fire in it and the rank correlation between their mean spike times and their field centres beats the 95th percentile of the same correlation after shuffling which A cell has which centre (200 shuffles; on circular tracks the best of 12 rotations, for data and shuffles alike; `cpca.evaluate.order_vs_place`). The sign of that correlation gives the replay's direction.
- **Fitting and scoring use B only.** The A cells are removed before any fit. Every read-out is scored on the B cells against their field centres in that map, either sense, with the best-of-outputs null (`scripts/plot_hc11_sequences.best_map_match`, 500 permutations). No cell is used both to choose events and to score them.
- **Matched random control.** The same number of events drawn at random among the events in which at least 5 A cells fire, analysed identically.

**Read-outs on a set of events E** (selected, or random):

1. complex PCA centred on E, best component;
2. complex PCA uncentred on E, best component;
3. direction-aligned complex PCA: events with a negative A correlation conjugated (played back forwards), uncentred, first component;
4. averaging: each B cell's relative time in each event of E (flipped for backward events), averaged over E; one order;

and, as the baseline, 5. the standard pipeline: complex PCA centred on all events, best component (B cells scored).

**Reading.** Per (session, epoch, map): median |ρ| over the 5 splits and the share of splits with p < 0.05; a map counts for a session if most splits pass. Sets with fewer than 30 selected events are reported but not scored.

- Averaging on selected events recovers B's order, complex PCA on the same events does not: the method is the bottleneck (explanation 2).
- Complex PCA on selected events recovers it (well above the random control and the baseline): the events carry the order and selecting them is what is missing (explanation 1).
- Neither does, or both are no better than random: the events carry little map order (explanation 3).
- The share of events selected (POST against PRE) measures how much replay of the map there is at all.

Both datasets: hc-11 (8 sessions) and DANDI 000552 (65 sessions).

*Additions before the results were read:* each split's own chance rate (the same selection with the A cells' fields shuffled), because the per-event test is conservative and its false-positive rate depends on the data; and planted checks of the diagnostic itself (whole-route replays at 10% and 25%, weaker replays recruiting 30% of the route, no replays, and 10% replays of 40% segments in busy events where 40% of all cells fire).

## Results (`figures/ceiling.png`, `results/ceiling/summary.md`, `results/ceiling/events_summary.md`)

**The diagnostic works.** With planted whole-route replays, half of the place cells choose 14–41 percentage points more events than chance, and every read-out recovers the other half's order. With planted segment replays in busy events the excess falls to +1.7 points, yet the chosen events still carry the order: averaging and the direction-aligned fit recover it in 92% of splits (random events: 24–32%).

**On both real datasets almost no event replays the map.** The excess over chance is +0.3 points in hc-11 POST, 0.0 in hc-11 PRE and 0.0 in both sleeps of 000552, far below even the hard planted segment case; the chosen events give the other half's order no more often than random events. Changing the event definition does not change this (first spike instead of mean, a 150 ms window around the burst peak, the dataset's own ripples: excess +0.2 to +0.6 points after the task, 0 before; `scripts/run_ceiling_events.py`). The real events are not short of cells: they have as many or more active place cells than the planted ones (000552: median 30–40 per event).

**Reading.** Explanation 3: in these sleep recordings, single events whose cells fire in the waking map's order are rare (well under 1% of events, at the per-event resolution of this test), so no read-out of the events, supervised or not, can recover much order; the population-level structure (who fires with whom, `docs/topology.md`, `docs/wake_sleep.md`) is what is there. Two qualifications: the test sees whole-event order, and planted segment replays already show that it is weak for partial replays (real replays could be shorter or sparser still); and it uses half of the place cells, so a decoder using all cells (Bayesian, with a movement model) is more sensitive than this test.

**For the method.** The planted segment case isolates a real weakness: given the right events, complex PCA as is (centred 24%, uncentred 16%) loses to simple averaging and to the direction-aligned fit (92%). Partial replays in both directions defeat it unless each event's direction is aligned first, which Step 2 does label-free but only for whole-route replays.
