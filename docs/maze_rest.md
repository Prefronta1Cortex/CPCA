# Awake rest on the maze: complex PCA and count PCA where replay is common

Code: `scripts/run_maze_rest.py`. Results: `results/maze_rest/`.

## Why

hc-11 sleep is below the replay share that complex PCA, or any label-free method tested, needs
(`docs/replay.md`, `docs/power.md`). Rest on the maze is where the paper's decoder finds the most replay: 16–19%
clear replays in the best sessions. If complex PCA recovers the map anywhere in hc-11, it should be here. This
is the positive real-data test of the paper plan.

## Design (written on October 29 before running)

**Events.**
- Population bursts (`cpca.events.population_bursts`, as for sleep) during immobility on the maze: 2-D head
  speed below 5 cm/s (`cpca.replay.immobile_intervals`), with spike windows centred on the burst peak.
- All such events are used.
- **No spike is in both the events and the answer key:** the key (`place_centres`, `place_maps`) uses only
  spikes while running faster than 5 cm/s.

**Methods.**
- **Complex PCA, centred:** the primary result. It is scored with the rerun's read-out, best component.
- **Complex PCA, uncentred:** secondary.
- **Seriation, the same read-out on both spaces.** It gives one order of all cells
  (`cpca.subspace.seriate`), from complex PCA's reduced space and from count PCA's (z-scored spike counts per
  event, components above the shuffle null).

**Scores.**
- **Order:** best output against the waking map, with the best-of-outputs null
  (`scripts/plot_hc11_sequences.best_map_match`).
- **Topology:** a Mantel test of each reduced space's cell-pair similarity against the similarity of the cells'
  place maps.

**Replay content of the events.** The paper's Bayesian analysis (`cpca.replay`) measures the share of clear
replays (significant under both shuffles) among the events it can test, and the number of place cells active
per event.

**Control.** The same number of NREM bursts from PRE, drawn at random and analysed identically.

**Expected.**
- **POST-level threshold:** complex PCA recovers the map where clear replays are at least about 10% of events
  and about 10 or more place cells fire per event. These are the conditions under which it succeeded in the
  power test (on 1500 events).
- **Event count:** with fewer events the threshold is higher, so sessions with only a few hundred rest events
  may fail even with a high replay share.
- **Control:** PRE at chance.
- **Count PCA:** if complex PCA's timing matters, count PCA seriation should recover the order less often than
  complex PCA seriation. Topology may pass for both, because co-firing of cells with nearby fields carries
  topology without order.

## Results (`results/maze_rest/summary.md`)

**The planned analyses.** There are 184–1,546 rest events per session, with a median of 3–18 place cells
active per event; 2–18% of the events are clear replays.

| Measure | MAZE rest | PRE control |
|---|---|---|
| Complex PCA, centred, best component | 5/8 | 0/8 |
| Complex PCA, uncentred | 3/8 | 1/8 |
| Seriation, complex PCA | 4/8 | 0/8 |
| Seriation, count PCA | 4/8 | 0/8 |
| Topology, complex PCA | 7/8 | 1/8 |
| Topology, count PCA | 7/8 | 0/8 |

- **The prediction held for the primary result.** The three sessions with at least about 10% clear replays and
  about 10 or more place cells per event all pass: Achilles 10/25, Achilles 11/01 and Gatsby 08/02, at
  p ≤ 0.002.
- **The sessions that fail** have too few place cells per event (Buddy, 3) or too few replays (Gatsby 08/28,
  Cicero 09/01).
- **Count PCA recovers the order as often as complex PCA under the same read-out,** and passes topology as
  often.

**The timing control (added after these results, because count PCA matched complex PCA).** The same rest
events are refitted 100 times with each event's spike trains shuffled among the cells active in it. Which cells
fire is kept; which cell fires when is destroyed.

The order test can be passed without timing:
- **Achilles 10/25:** timing-shuffled centred fits match the map in 98/100 shuffles, and uncentred fits in
  97/100.
- **Achilles 11/01:** 87/100 and 94/100.

The order test's null permutes field centres, so it asks whether an output matches the map. It does not ask
whether the match comes from timing. The likely mechanisms:
- **Centred fits:** components contrast groups of cells that fire in different events (for example the two
  ends of the track), because silent cells enter the centred covariance, and the read-out turns the sign
  contrast into early and late times.
- **Uncentred fits:** shuffling keeps each event's timing profile, so groups that co-fire in events with
  different profiles still differ in phase.

**Timing p** is the share of shuffled fits at least as good as the real one; small values mean the real order
is more than co-firing explains. Only these sessions show order beyond co-firing:

| Session | Centred | Uncentred |
|---|---|---|
| Gatsby 08/02 | 0.02 | 0.01 |
| Cicero 09/10 | 0.03 | 0.60 |
| Cicero 09/17 | 0.08 | 0.01 |

The two Achilles sessions do not (0.15 and 0.41 centred). In sleep after the maze, complex PCA's one pass
(Achilles 10/25, p = 0.005) has a timing p of 0.11, so it is not shown to come from timing.

## Reading

Complex PCA recovers the waking map from awake rest in 5 of 8 sessions, against 0 of 8 in sleep before the
maze. The order specific to within-event timing (sequence order beyond co-firing) is shown in 2–3 of 8
sessions: Gatsby 08/02 under both conventions, plus Cicero 09/10 or Cicero 09/17 under one. In the rest, the
map match is carried by which cells fire together.

**Consequences:**
- **The order test is not sufficient on real data.** It needs the timing control alongside it.
- **Earlier hc-11 order results need re-reading with this control.** These include the component tests after
  and before the maze, the per-session best outputs of all methods, and the subspace read-outs.
- **On planted whole-route replays and hc-5**, co-firing carries no order by design or by data (count PCA fails
  there, `docs/topology.md`), so the order results there are about timing. The timing control should still be
  run there to confirm it is calibrated.
