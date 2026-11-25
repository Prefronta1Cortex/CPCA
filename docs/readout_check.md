# A read-out check: is a component a sequence? (repair step 1)

Code: `cpca/readout_check.py`, `scripts/calibrate_readout.py`. Results: `results/readout_check/`. Goal:
`docs/goals.md`.

## Why

The read-out (`cpca.decode.readout`) turns any component into an "order", including a component that only contrasts
two co-firing groups (phases near 0 and π), which then becomes a two-step order. Co-firing therefore passes order
tests (`docs/maze_rest.md`). Under the goal, every output must be one sequence, so components that are not
sequences must be rejected without using the map.

## First design: circularity (fixed November 5 before running). It failed.

**Design.** Circularity |Σ w²| / Σ |w|² over the members: 1 for a real vector, small for phases spread around the
circle. The threshold is the 90th percentile among route-matching components, on `sim/segments.py` data (120
cells, about 13 ordered cells per replay, 6 draws).

**Result.** The check does not separate sequences from other components:

| Components | Median circularity |
|---|---|
| Route-matching | 0.83–0.85 (forward-only replays 0.54) |
| Not route-matching | 0.81–0.86 |

At the threshold (0.92 centred), 0–6% of non-matching components are rejected. All the components that match the
route by co-firing alone (map-ordered assemblies, 42 of them) are kept. Short partial replays played both ways give
nearly real components even when they are true sequences. The circularity check is dropped.

## Second design: a cross-validated timing check (fixed November 5 before running)

**Fit.** Complex PCA is fitted on one half of the events (random split); each kept component (PCA or ICA rotation)
is a candidate.

**Expression on the other half.** Per event, max(|⟨z, q⟩|², |⟨z̄, q⟩|²), with z the event's prepared arrows (the
same convention, prepared on the held-out events). E_real is the mean over held-out events.

**Null.** The same mean on held-out events with spike trains shuffled among each event's active cells (who fires
kept, who fires when destroyed), 50 shuffles. p = the share of shuffles with E at least E_real.

**A component counts as a sequence if p < 0.05.** Cross-validation keeps the fit from favouring the real events.

**Calibration**, same conditions as above (both folds, centred and uncentred, PCA and ICA rotation). The check
passes calibration if:
- it keeps at least 80% of route-matching components in the replay conditions;
- it rejects at least 90% of all components in the no-replay and map-ordered-assembly conditions (which include
  the co-firing components that match the route).

**Note.** Each cell's own latency within bursts is a timing pattern too, and this check keeps it; dense sequences
such as wheel runs or laps are exactly such patterns. On planted data the latency structure is part of the
background, so the no-replay condition shows how often it survives.

**Result of the second design: it fails.** It keeps 98–100% of all components, including those from data with no
replays (centred PCA, 100%). Every cell has its own latency within bursts, as in real data, so shuffling spike trains
between cells lowers every component's expression. The check shows that components carry timing, not that the
timing is a replay.

## Third design: the latency null (fixed November 5 before running; the last attempt for this step)

As the second design, but the null keeps each cell's own timing and destroys only the coordination within an event.
In each held-out event, every active cell keeps its spike count, and its spikes' relative times (t / T) are redrawn
from that cell's own relative times pooled over all held-out events (times = relative time × T of the event). A
component that is only per-cell latency or co-firing keeps its expression; a replay, whose order differs from event
to event, loses it.

The pass bar is unchanged (keep at least 80% of route-matching components in the replay conditions; reject at least
90% of components in the no-replay and map-ordered-assembly conditions). If this design fails, step 1 stops and is
reported as unsolved.

**Limitation, stated in advance.** A sequence played identically in every event (wheel runs, laps) is a per-cell
latency pattern, so this check would reject it. It is meant for reactivation events, where replays vary from event
to event.

**Result of the third design: it passes for three of the four versions.**

| Version | Route-matching components kept (replay conditions, pooled; per condition 10% / 20% / forward 20%) | No-replay and map-ordered-assembly components rejected |
|---|---|---|
| Centred, PCA | 91% (84 / 95 / 94%) | 98–100% |
| Centred, ICA | 86% (67 / 92 / 100%) | 100% |
| Uncentred, ICA | 86% (69 / 95 / 86%) | 97–100% |
| Uncentred, PCA | **76%** (69 / 96 / 66%): fails the 80% bar | 97–100% |

In the replay conditions the check also keeps 28–97% of the components that do not match the route. They carry timing
that varies from event to event, but they are not a clean route: they belong to the separation problem
(`docs/separation.md`), not to this check.

**Adopted:** the latency check for reactivation events, for centred PCA, centred ICA and uncentred ICA.
