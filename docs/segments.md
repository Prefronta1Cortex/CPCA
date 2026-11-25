# A label-free replay model beyond complex PCA: what was tried, and the direction oracle

Code: `sim/segments.py` (constant-speed partial replays), `scripts/run_segments.py` (gate G1 with the direction
oracle), `scripts/summarize_segments.py`, `scripts/check_enrichment.py` (label-free enrichment on hc-11).
Results: `results/segments/`, `results/enrichment/`. Written November 4, 2026.

## The task

Find a label-free model that takes reactivation events and returns the replayed sequences: an order of cells, which
events are replays, and their direction; the waking map is used only to score. The gate is G1 of the brief: planted
constant-speed partial replays in both directions, swept over replay share and cells per replay; the model must beat
complex PCA in part of the low-share or few-cell regime, keep false positives near 5%, fail on map-ordered assemblies
without order, and pass the timing control.

## Development checks (run before this design was written, in `/tmp`; not evaluations)

Five candidate formulations were tried on single draws of a constant-speed partial-replay generator (the prototype of
`sim/segments.py`: 100 cells, 2000 events, segments of 30–50% of the route, both directions 50/50, 15% background
participation, 8 ms jitter). Each is reported here because its failure is informative; none was tuned.

1. **Pairwise-order SVD (Kendall-kernel PCA).** Each event becomes a vector over cell pairs (sign or normalised lag of
   each co-active pair); time reversal is a sign flip of that vector, so a rank-1 SVD should give each event's direction
   (left singular vector) and the consensus order (right singular vector), with the within-event shuffle as the null.
   Result: the top component's events were all replays (AUC 0.85, top-20 loadings 20/20 replays), but its pair pattern
   was orthogonal to the planted order (overlap 0.05) and direction accuracy was at chance (0.56). The SVD is dominated
   by a direction-invariant structure (which cells sit at an event's extremes), and the signed order signal is diluted by
   the square of a partial replay's coverage of the map's pairs (15 of 300 pairs per replay). Rejected.
2. **PCA of within-event standardised spike-time deviations** with silence treated as missing (no participation term,
   per-cell centring to remove latency structure). Direction-invariant by construction. Result: forward-only 10% replays
   with 25 route cells, where complex PCA reads |ρ| 0.99, gave 0.19. Deviations depend on a per-event clock (the mean
   of the active cells' times), which is noisy with 20 active cells; complex PCA's pairwise lag products are clock-free.
   Rejected.
3. **Direction-resolved refit with a per-event gate.** Label each event's direction by the sign of its correlation with
   a seed order, keep events whose correlation beats a within-event permutation null (p < 0.05), refit the order from
   direction-corrected lags (HodgeRank) or standardised times (averaging). Result: seeded with the true order plus noise
   (ρ 0.72–0.77), the refit did not improve it (0.50–0.78); seeded with the exact truth it degraded it (0.66–0.82). At
   5% replays with ~8 route cells per replay, the gate admits ~95 of 1900 noise events and 20–50% of the ~100 replays
   (precision 0.05–0.30). Only at ~15 cells per replay did the gate reach precision 0.3 at recall 0.5. Rejected: the
   per-event evidence is too weak to bootstrap, as the project found before for templates and mixtures.
4. **Standardised complex lag matrix**: per co-active pair, the lag arrow minus the event's own shuffle expectation,
   summed over co-active events and divided by its noise standard deviation (a z-score per pair; Hermitian; eigh;
   shuffle null). Result on two draws per condition: eigenvalues above the null in some conditions, but the phase and
   seriation read-outs were no better than complex PCA's (e.g. 10% forward-only: 0.11/0.33 against 0.79/0.35). Rejected
   as no gain.
5. **Direction EM** (the project's own split in `cpca/direction.py`, without gate and guard): alternate per-event labels
   (forward if |⟨s,q⟩| ≥ |⟨s,q̄⟩|) and the leading eigenvector of the direction-aligned lag matrix over all events;
   seeds: the kept centred components, the top-3 components regardless of the bound, 10 random orders, the truth;
   objective: the aligned top eigenvalue. Result (6 draws × 5 conditions): the objective is the same from every seed,
   including on data with no replays and on timing-shuffled data (1.18–1.33 throughout), label accuracy from label-free
   seeds is 0.51–0.66, and the truth seed "passes" even on shuffled data. Every order is a self-consistent fixed point,
   because the 95% non-replay events can be flipped to agree with any template; the objective carries no information
   about which fixed point is right. Rejected.

**The direction oracle (the finding that organises the rest).** In the same 6 draws × 5 conditions, the direction-aligned
uncentred lag matrix over *all* events, with each event's planted play direction given and nothing else (non-replay
events get a random sign), recovers the order (whole-track test, p < 0.05) in 29/30 draws: 5/6 at 5% with 8.6 route
cells per replay, 6/6 at 10% with 5.2, 6/6 at 2% with 21.5, 6/6 at 3% with 12.8. Complex PCA (centred, best component)
passes 0–2/6 in the same draws. The pooled second-order statistic therefore has enough signal; what is lost is the play
direction of each event, and checks 3 and 5 show it cannot be estimated label-free at these cell counts.

## Design of the formal check (fixed before running)

**Generator** (`sim/segments.py`). 120 cells; `n_route` of them on one route with positions uniform in [0, 1]; the
rest never replay. Every cell has its own background participation (uniform in 0.05–0.30) and a fixed latency offset
(normal, SD 8% of the event duration), and cells belong to 6 random assemblies (a group is drawn per event; its
members participate 3 times more, others half as much; as in `sim/routes.py`). Per event: a segment length L uniform
in 0.3–0.5 of the route and a speed (250 ms per route length, times uniform 0.8–1.25) set the activity span L × speed,
and the window is the span plus 10% margins on each side, so **the window follows the activity and the speed is constant
within a replay**. With probability `replay_frac` the event is a replay: a segment start uniform in [0, 1 − L], a
direction (backward with probability `reverse_frac`), and each route cell in the segment fires with probability 0.6 at
the time its position is reached (1–3 spikes, 8 ms jitter); all other active cells fire 1–3 spikes at normal(centre +
latency, span/4). Non-replay events draw L and speed the same way, so duration does not mark replays. Cells per replay
is set through `n_route` (≈ 0.24 n_route). 2,500 events per draw.

**Grid.** Cells per replay 5, 9, 13 (n_route 21, 37, 54) × replay share 0, 2, 5, 10% × reverse fraction 0.5 and
0.3 (the real-data value measured in `docs/direction.md`: 0.24–0.29); 5 draws per cell; share 0 is shared between the
balances. Two controls, 5 draws each at 9 cells per replay: map-ordered assemblies without order (assemblies are 6
contiguous stretches of the route, no replays), and forward-only replays at 5% (the regime the imaginary part was made
for).

**Outputs scored per draw** (orders over cells; the map is used only here):
- complex PCA, centred, all kept components (the reference; `scripts/run_rerun.run_cpca`);
- complex PCA, uncentred, all kept components;
- the lead–lag component: the leading eigenvector of j·Im(C) of the raw arrows (co-firing-free; carries only the
  direction-imbalanced part of the signal);
- the direction EM of check 5 from the kept centred components (best objective; for the record);
- the direction oracle: the aligned uncentred leading eigenvector over all events, planted directions given;
- the event-and-direction oracle: the same on the replay events only.

**Scores.** Whole-track order test (`scripts/plot_hc11_sequences.best_map_match`, 500 permutations) and the fragment
criterion's bar 1 (`cpca/fragment.py`, 500 permutations), per output set. Timing control (bar 2) for complex PCA centred
and the lead–lag component: 30 refits on events whose spike trains were shuffled among each event's active cells; timing
p = share of shuffled fits at least as good (whole-track statistic). Per-event direction accuracy on true replays for the
EM and for the oracle (labels = sign of F − B against the oracle template, to show what a correct template buys), and the
share of events the EM calls forward.

**Pass criteria (G1).** A candidate beats complex PCA if, in at least one grid cell with share ≤ 5% or cells per
replay ≤ 9, it passes the whole-track test in at least 3 more of 5 draws than complex PCA, and in no grid cell does
complex PCA pass in 3 more draws than it. False positives: at share 0, whole-track p < 0.05 in at most 2 of 15 draws.
Assemblies without order: at most 1 of 5 draws passes both fragment bars. Timing: among the draws a candidate passes,
timing p < 0.05 in at least 4 of 5 where there are 5 passes.

**Expected (written before running).** The direction oracle passes nearly everywhere at ≥ 9 cells per replay and from
5% at 5 cells; complex PCA centred passes at 10% with ≥ 9 cells and little below; at reverse fraction 0.3 complex PCA
improves and the lead–lag component passes where the imbalanced signal clears its own edge (≥ 9 cells, 5–10%); the EM
does not improve on its seed. If no label-free candidate beats complex PCA, G1 fails for every family tried, and the
reason is the direction ambiguity quantified by the oracle.

## Label-free enrichment on hc-11 (fixed before running)

If directions cannot be estimated per event, the remaining label-free lever on real data is the replay share itself:
features of an event that need no map (duration, number of active cells, spread of spike times in the window, peak of
the population rate, number of spikes) may be enriched for replays. Using the paper's Bayesian scores already computed
on our POST bursts (`results/replay/<session>.pkl`, "ours"), `scripts/check_enrichment.py` measures, per session, the
AUC of each feature for "clear replay" (p < 0.05 under both shuffles) and the replay share in the top 30% and 50% of
events by each feature, against the share in all events. The map is used only to label events for this measurement.
Pass: a feature whose top-50% subset at least doubles the clear-replay share in the two largest sessions. Expected: the
spread of spike times helps (replays spread spikes over the event; volleys do not); the others little.

## Results

### G1 grid (`results/segments/summary.md`, `figures/segments_g1.png`): no label-free candidate beats complex PCA; the direction oracle does, everywhere

Measured ordered cells per replay: 5.0, 9.0–9.5, 12.4–13.4 (events have a median of 18–20 active cells). Draws of 5
whose best output passes the whole-track order test:

| reverse | ordered cells | share | complex PCA centred | lead-lag | direction EM | direction oracle | event + direction oracle |
|---|---|---|---|---|---|---|---|
| 0.5 | 5 | 2 / 5 / 10% | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 1 / 1 | 4 / 4 / 5 | 5 / 5 / 5 |
| 0.5 | 9 | 2 / 5 / 10% | 0 / 2 / 4 | 0 / 0 / 1 | 0 / 0 / 0 | 3 / 5 / 5 | 5 / 5 / 5 |
| 0.5 | 13 | 2 / 5 / 10% | 0 / 4 / 3 | 0 / 0 / 1 | 0 / 1 / 0 | 4 / 5 / 5 | 5 / 5 / 5 |
| 0.3 | 5 | 2 / 5 / 10% | 1 / 2 / 3 | 0 / 2 / 1 | 1 / 0 / 0 | 2 / 4 / 5 | 5 / 5 / 5 |
| 0.3 | 9 | 2 / 5 / 10% | 0 / 3 / 4 | 1 / 1 / 0 | 0 / 1 / 1 | 3 / 5 / 5 | 5 / 5 / 5 |
| 0.3 | 13 | 2 / 5 / 10% | 2 / 4 / 5 | 1 / 1 / 0 | 2 / 1 / 1 | 4 / 5 / 5 | 5 / 5 / 5 |

- **Pass criteria.** Complex PCA uncentred, the lead–lag component and the direction EM each fail the "beats complex PCA"
  criterion: no grid cell with 3 or more draws gained, and 5–8 cells with 3 or more lost. G1 fails for every label-free
  family tried here and in the development checks.
- **The direction oracle passes where complex PCA does not:** at 2% it passes 3–4 of 5 against complex PCA's 0–2; at 5
  ordered cells per replay it passes 4–5 of 5 at 5% and 10% against 0–3. Knowing which events are replays as well
  (second oracle) gives 5/5 everywhere, including 2% with 5 cells.
- **False positives.** At share 0, complex PCA centred passes the whole-track test in 4 of 15 draws (27%): with
  assemblies and per-cell latency offsets the whole-track order test is not calibrated on its own, as `docs/maze_rest.md`
  found on real data. With the timing bar it falls to 1 of 15. The other candidates: 0–1 of 15.
- **Assemblies without order:** complex PCA centred passes the whole-track test in 3/5 draws, uncentred in 5/5, the
  lead–lag component in 4/5 (its component follows the within-assembly latency order and the assemblies follow the map);
  the fragment criterion with its timing bar rejects them (complex PCA 0/5, lead–lag 1/5); both oracles 0/5.
- **Timing control.** Where complex PCA passes at 5–10% the passes are timing-specific (whole-track + timing 3–4 of 5
  where the whole-track test alone gives 4–5); at 2% nothing is timing-specific.
- **Forward-only replays at 5%, 9 cells:** complex PCA 4/5 with the timing bar (against 2–3/5 when directions are
  balanced), the oracle 5/5. The direction mix is what separates the easy from the hard regime.
- **Per-event directions.** The EM's labels are at chance (0.50–0.70) and it labels 89–96% of events "forward" (it
  settles on the latency structure common to every event). Labels read from the oracle's correct template reach only
  0.64–0.89 at these cell counts, which is why a correct map cannot be bootstrapped from its own labels either.

### Enrichment on hc-11 (`results/enrichment/summary.md`): fails

- **The features predict the decoder's own power, not maze replay.** In POST the AUCs for clear replay are 0.55–0.71
  (duration, number of active cells, spikes, interquartile range of spike times); in PRE, where there is no maze to
  replay, they are 0.55–0.68. Longer, larger events simply have more bins and cells for the Bayesian test.
- **The top half of events by the best feature raises the clear share 1.3–1.6 times** (Achilles 10/25: 5.6% → 8.0% by
  the interquartile range; Achilles 11/01: 4.7% → 7.5% by spikes), not 2 times; the POST − PRE excess goes from +1.3 to
  +2.6 and from +2.9 to +5.4 points. Since the pooled signal scales with the share times the square root of the event
  count, keeping half of the events for 1.5 times the share gains about nothing (1.0–1.1 times).
- Correlations of the features with rZ are 0.00–0.12. No label-free event feature enriches replay usefully on hc-11.

### The direction balance of hc-11's replays (descriptive, added after the oracle result; not pre-registered)

From the decoder's clear replays in `results/replay/` (forward = decoded position moving in the template's running
direction), the forward share in POST sleep is 0.47–0.66 across the 8 sessions (|2p − 1| = 0.00–0.32), in PRE
0.43–0.63, and on maze rest 0.50–0.83. The part of the replay share that survives direction cancellation in any pooled
lag statistic, |2p − 1| × clear share, is 0.0–1.5% of events in POST and 0.7–10.6% on maze rest (Cicero 09/17 10.6%,
Gatsby 08/02 5.5%, Buddy 3.3%, Cicero 09/01 2.9%; the two Achilles sessions 0.0–0.7%). The clear-replay sets include
the test's false positives (PRE level 1.2–7.7%), which are balanced and dilute these estimates, so the true replays may
be somewhat more one-sided than this. This is consistent with where complex PCA's timing-specific passes were found
(`docs/maze_rest.md`: Gatsby 08/02, Cicero 09/10 or 09/17) and with its failure in the Achilles sessions' sleep despite
their cell counts.

## Reading

**G1 fails, and the reason is identified.** At hc-11-like cell counts (5–13 ordered cells per replay, ~20 active cells
per event) and low shares (2–5%), the pooled lag statistic contains enough signal to recover the map, but only once each
event's play direction is known: forward and backward plays of the same segment are conjugates of each other and cancel
in every label-free second-order statistic, leaving a direction-invariant proximity structure that carries order weakly
and is confounded with co-firing. The direction cannot be recovered per event because a single event with 5–13 ordered
cells among 20 active ones does not carry enough evidence even against the correct map (labels 0.64–0.89), and an EM
over directions has every order as a self-consistent fixed point. The sign-invariant alternative (pair space) is diluted
by the square of each partial replay's coverage. These are properties of the data regime, not of complex PCA, which
remains the best label-free read-out tested (it also gains most from a direction imbalance: 4/5 at 5% with 9 cells when
replays are forward-only, 2–3/5 when balanced).

**On hc-11 the direction balance is close to even** (forward share 0.47–0.66 in POST), so the coherent part of the
replay share is 0–1.5% of events, below what any pooled statistic needs; on maze rest the imbalance is larger in exactly
the sessions where complex PCA's timing-specific passes were found.

**What this leaves open, label-free:** only more ordered cells per event (the second oracle's 5/5 at 2% shows that the
events, not the shares, are the constraint), which is the data lever (DANDI 001695, under test in `docs/dandi001695.md`).
A direction prior from behaviour on awake rest (forward replays before departure, backward after arrival; Diba &
Buzsáki 2007) would supply the missing sign without the place map, but it is waking information and needs a decision on
whether the protocol allows it. Label-free event features do not enrich replay (above).

**Limitations.** One generator (constant speed, segments of 30–50%, 8 ms jitter, 2,500 events, 120 cells), 5 draws per
cell, one replay map; the oracles' advantage is a lower bound on what directions are worth, because they use the
uncentred read-out without any tuning. Development checks 1–5 were single or double draws and are reported as such.

## Addendum (November 4, after review; design written before running)

**Corrections to the reading above.**
- Balanced directions do not cancel the replay signal in the lag matrix. A forward replay contributes q·qᴴ and a backward
  one its conjugate, so in equal numbers the replay term is Re(q·qᴴ) = c·cᵀ + s·sᵀ with q = c + js: the antisymmetric
  part cancels, the symmetric part keeps the order, split over a near-degenerate pair of real components that each carry
  about half of the eigenvalue. Any single component of the pair read out alone has lost the order (its phases are 0 or
  π); q itself is still in the pair's span. The information is split and harder to read, not destroyed.
- "Every label-free model family" was too broad: the families tested are five second-order or EM variants and a lead–lag
  component. The Maboudi HMM and PP-Seq were not run on this grid, and no latent-trajectory model or statistic beyond
  pairs (triplets) was tried.
- The oracle is uncentred while complex PCA is centred with best-of-outputs scoring. Random-sign conjugation of the
  non-replay events also removes consistent latency structure, which centring removes too, so part of the oracle's margin
  may be centring rather than direction.
- The DANDI 001695 figures quoted in the first report were read from a run in progress; the final values
  (`docs/dandi001695.md`, parallel session) are 3–26 place cells active per event (median 12) and 2.0–6.1% clear
  replays after the track (median 3.4%), 3.3% before, 7.7% at rest on the track.

**Check A: the conjugate-pair read-out on the same G1 grid** (`cpca/pairs.py`, `scripts/run_segments_pairs.py`; the
data of every grid cell are regenerated from the same seeds). For complex PCA's kept centred components, each pair of
components gives one extra output: the unit combination in their span whose phases are most spread around the circle,
i.e. least like a real vector, found in closed form as the combination that zeroes the doubled-phase resultant
|Σ_k u_k²| (Takagi factorisation of the 2 × 2 pseudo-covariance). For a degenerate pair this is q or q̄, either of which
gives the order up to reversal. Two variants: adjacent pairs in eigenvalue order (primary; a degenerate pair has adjacent
eigenvalues; k − 1 outputs) and all pairs (secondary; k(k − 1)/2 outputs). Scoring as before: whole-track test and
fragment bar 1 with the best-of-outputs null over the variant's outputs, and the timing bar (30 shuffled refits with the
same read-out). Also scored: the standard read-out (single components, as in the grid) recomputed on the same data as a
check, and the **centred direction oracle**: the planted directions applied as conjugation, then the centred fit with the
standard read-out and best-of-outputs scoring, so that the oracle and complex PCA differ only in the direction labels.
Pass criteria as for G1 (3 or more draws gained in at least one low-share or few-cell cell and none lost by 3, false
positives at most 2 of 15 at share 0, assemblies without order at most 1 of 5 under fragment + timing). Expected: the
pair read-out recovers the read-out loss but not the halved eigenvalue, so it gains most at 5–10% with balanced
directions and closes part of the gap to the oracle; the centred oracle sits between complex PCA and the uncentred
oracle if centring contributes, and equals the uncentred oracle if it does not.

**Check B: the HMM on a slice of the grid (G4).** `baselines/hmm_baseline.py` with its own settings (20 ms bins, 30
states) on 9 ordered cells per replay, shares 0, 5 and 10%, both balances, 5 draws: the share of events called congruent
(the false-alarm rate at share 0) and the AUC of congruence against the planted replay labels. The HMM gives no cell
order without behaviour and is not scored on order.

**Check C: forward-only against balanced replays on real hc-11 events** (`scripts/run_spikein_direction.py`). The
spike-in power test of `docs/replay.md` (the matched 1500 PRE events, ideal replays of the waking map, f = 1–20%, 5
draws, 8 sessions), with the sense drawn per event as before (balanced) and with every planted replay forward. Read-outs:
complex PCA centred, standard (the published numbers are reproduced for the balanced case), the conjugate-pair read-out
(adjacent pairs; all pairs), and the direction oracle on real events (the planted backward replays conjugated, random
signs for the other events), uncentred and centred. Scored by `best_map_match` as in the power test. Expected: the
forward-only threshold lies below the balanced one (about 10% in the two largest sessions), measuring what balance costs
on real events; the pair read-out lowers the balanced threshold if the split-pair reading is the loss on real events too;
the oracles give the floor.

### Results of the addendum

**Check A** (`results/segments/pairs_summary.md`, `figures/segments_pairs.png`; identical events to the main grid):

| reverse | cells | share | standard | adjacent pairs | all pairs | centred oracle | uncentred oracle |
|---|---|---|---|---|---|---|---|
| 0.5 | 5 | 2 / 5 / 10% | 1 / 0 / 0 | 2 / 0 / 0 | 1 / 0 / 2 | 0 / 1 / 1 | 4 / 4 / 5 |
| 0.5 | 9 | 2 / 5 / 10% | 0 / 2 / 4 | 0 / 3 / 5 | 0 / 3 / 5 | 1 / 3 / 5 | 3 / 5 / 5 |
| 0.5 | 13 | 2 / 5 / 10% | 0 / 4 / 3 | 2 / 3 / 5 | 2 / 3 / 5 | 3 / 5 / 3 | 4 / 5 / 5 |
| 0.3 | 5 | 2 / 5 / 10% | 1 / 2 / 3 | 0 / 1 / 1 | 0 / 2 / 1 | 1 / 2 / 4 | 2 / 4 / 5 |
| 0.3 | 9 | 2 / 5 / 10% | 0 / 3 / 4 | 2 / 3 / 4 | 2 / 3 / 4 | 1 / 3 / 5 | 3 / 5 / 5 |
| 0.3 | 13 | 2 / 5 / 10% | 2 / 4 / 5 | 2 / 2 / 5 | 4 / 5 / 5 | 2 / 4 / 4 | 4 / 5 / 5 |

- **The confound is confirmed.** With the planted directions applied as time reversal and the *centred* fit read out as
  usual, the oracle passes 48 of 90 replay draws (timing bar not computed), against 38 for the standard read-out and 78
  for the uncentred oracle. Direction labels inside the centred pipeline are worth roughly a doubling of the
  eigenvalue, as the algebra says (5% aligned ≈ 10% balanced: 1/3/5 against 0/4/3 at 10%); the large margin of the
  uncentred oracle is its read-out: after alignment the replays' lags add coherently to the imaginary part while the
  random conjugation of the other events cancels their consistent latency lags, so the leading uncentred eigenvector
  reads the route from the phase modulation of the mean pattern. That mechanism needs the directions and is not
  available label-free on balanced data.
- **The pair read-out gains a little on planted data and fails the criterion:** all pairs 47 passes against 38 over the
  18 replay cells, adjacent pairs 40; no cell gained by 3 or more draws (adjacent pairs are not the two halves of a split
  pair when co-firing components interleave; all pairs pay for more outputs in the null). With the timing bar there is no
  gain (the shuffled refits are read out with as many outputs, so their best |ρ| is higher). The direction oracle at
  5 cells per replay remains out of reach: 0–2 of 5 for every label-free read-out against 4–5 of 5.
- **False positives** at share 0: standard 4/15, pairs 3/15 (2/15 with the timing bar), centred oracle 0/15.
- **Assemblies without order:** the pair read-outs pass the whole-track test in 5/5 draws (the circular combination of
  two co-firing contrasts is a seriation of the co-firing structure along the map; the standard read-out 3/5, the centred
  oracle 5/5); fragment + timing rejects all of them (0/5). The pair read-out must never be used without the timing bar.

**Check B, the HMM** (9 ordered cells per replay): 85–89% of events are called congruent at the 95% level, with false
alarms of 0.86–0.89 and an AUC of 0.36–0.43 against the planted labels (0.88 congruent on assemblies without order). On
events with assemblies and consistent per-cell latencies every event has a "sequence" under the transition shuffle, so
congruence does not separate replays here. It is not a competitor on this grid.

**Check C** (`results/spikein_direction/summary.md`, `figures/spikein_direction.png`; the balanced case reproduces
`results/spikein/` event for event). Draws passing of 10 in the two largest sessions:

| read-out | kind | 0% | 1% | 2% | 3% | 5% | 10% | 20% |
|---|---|---|---|---|---|---|---|---|
| standard | balanced | 2 | 1 | 0 | 0 | 1 | 9 | 10 |
| standard | forward-only | 2 | 1 | 0 | 1 | 0 | 9 | 10 |
| all pairs | balanced | 0 | 2 | 1 | 3 | 5 | 10 | 10 |
| all pairs | forward-only | 0 | 1 | 1 | 0 | 6 | 9 | 10 |
| adjacent pairs | balanced | 1 | 1 | 1 | 1 | 1 | 5 | 10 |
| oracle, uncentred | balanced | 0 | 2 | 5 | 6 | 10 | 10 | 10 |
| oracle, centred | balanced | 1 | 0 | 0 | 2 | 1 | 10 | 10 |

- **Direction balance costs nothing measurable on real hc-11 events for the centred pipeline:** forward-only replays
  give the same threshold as balanced ones (10%; 0–1 of 10 at 5%), and the centred oracle is no better (1/10 at 5%).
  On real events the limit of the centred, second-order fit is not the split pair; it is the covariance edge set by
  ~20 active cells with spread spike times per event. In the other six sessions the forward-only case is, if anything,
  weaker at 10% (2/30 against 5/30).
- **The uncentred oracle reaches 5% (10/10) in the two largest sessions and 13/30 balanced, 26/30 forward-only in the
  others,** for the mechanism above; the centred oracle does not. On real events, too, what direction labels would buy is the first-order read-out,
  not a better centred fit.
- **The all-pairs read-out is the one label-free gain:** 5/10 (balanced) and 6/10 (forward-only) at 5% against 1/10
  and 0/10, 0/10 false positives at 0%, and in the other six sessions 21/30 at 20% against 10/30; the strict threshold
  (4 of 5 in both large sessions) stays at 10%. Since the spike-in moves spike times only, participation and co-firing are
  unchanged from the 0% base, so these passes are timing-driven; on real data the read-out still needs the timing bar.
  Adjacent pairs are worse than the standard read-out on real events (5/10 at 10%) and pass 3/30 at 0% in the small
  sessions.

### Reading, revised

- What was called "cancellation" is a split: balanced plays leave Re(q·qᴴ), whose degenerate pair still spans q. Reading
  the pair recovers some of the read-out loss (a modest, consistent gain on real events at 5%; little on the planted grid)
  but not the halved eigenvalue, and it seriates co-firing as readily as timing, so it is only usable with the timing bar.
- Direction labels fed to the centred pipeline are worth about a factor of two in share on planted data and nothing
  measurable on real events. The direction oracle's large margin comes from a different read-out, which needs the
  directions to exist: the coherent imaginary part of the aligned, latency-cancelled lag matrix. Label-free, that read-out
  is available only where directions are imbalanced (the lead–lag component after a per-cell latency correction), which
  hc-11 sleep is not (forward share 0.47–0.66); awake rest in some sessions is (0.65–0.83).
- The forward-only spike-in shows that on real hc-11 events the centred pipeline's 10% threshold is not a direction
  problem at all. The remaining label-free levers are the first-order read-out where directions are imbalanced, the
  pair read-out as a small gain, and more ordered cells per event.
- G1 still fails for every candidate tested; the families not tested remain as listed above (PP-Seq, latent-trajectory
  models, statistics beyond pairs). The HMM was tested on a slice and does not apply.
