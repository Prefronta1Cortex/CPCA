# What any method could do on hc-11 sleep (Step 0), and how much replay each method needs (Step 1)

Code: `scripts/run_oracle_hc11.py`, `scripts/run_power_hc11.py`, `cpca/sparse.py`, `scripts/summarize_power.py`.
Results: `results/power/summary.md`; figure: `figures/power_hc11.png`. Follows `docs/replay.md`.

## Step 0: the bound from perfect event choice

**Question.** If a method knew exactly which sleep events are replays, and which way each one runs, how
often could it recover the map's order from the events every method was given?

**Design.**
- **Events:** per session and sleep epoch, the matched 1500-event subset.
- **Split:** the place cells (a field in our answer key or under the paper's criterion) are split at random
  into halves A and B, 10 times.
- **Selection, half A only:** the paper's Bayesian decoder, built from A's place cells, picks the clear
  replays (significant under both shuffles) and says which map each one replays and which way it runs.
- **Scoring, half B only:** each B cell's mean relative spike time is averaged over the chosen events,
  reversed events flipped, one order per map. The orders are scored by the test used for every method,
  against B's field centres. Averaging the chosen events is what a perfect method reduces to once it knows
  the events and their directions; on planted data it recovers the order in 92–100% of splits
  (`docs/ceiling.md`).
- **Comparisons, scored on the same B cells:**
  - a loose selection (significant against shuffled place fields only);
  - random events, the same number;
  - complex PCA fitted without the map.

No cell both selects and scores.

**Results** (splits passing out of 10, sleep after the maze):
- **Perfect selection:** at most 3/10 in any session (Achilles 10/25), 0–1/10 in the others.
- **Loose selection:** 6/10 in Achilles 10/25, at most 2/10 elsewhere.
- **Complex PCA without the map:** 5/10 in Achilles 10/25, 0–3/10 elsewhere.
- **Random events:** 0–2/10.
- **Before the maze:** at most 2/10 for any read-out.
- **Clear replays available:** 3–38 of the 1500 events per session.

**Reading.** On these events, knowing which ones are replays does not give a better order than complex PCA
already finds without the map. No method, and no added architecture, can do clearly better on hc-11 sleep.

*Qualification:* both the bound and complex PCA are scored on half of the place cells here, which costs
both of them power.

## Step 1: how much replay each method needs

**Benchmark.** The spike-in test of `docs/replay.md`: real PRE events, with a share f turned into ideal
replays of the map. Every place cell that fires keeps its spike count, and its spikes move to its field's
position, forward or reverse. There are 8 sessions, f = 0–20%, and 5 draws.

For the 1500-event subset, the spiked data sets are regenerated with the original seeds, so every method
saw exactly the data complex PCA saw. Scoring is the same for every method.

**Methods:**
- **Complex PCA on 1500 events** (as before).
- **Complex PCA on all PRE events**, which gives more events and a lower noise bound.
- **Rastermap**, at the settings of `docs/methods_hc11.md`.
- **PP-Seq, kept short:** the two sessions with the most place cells, f = 0, 5, 10 and 20%, one draw each.
  Its hyperparameters are fixed to the setting its own sweep chose on the same PRE events in the matched
  rerun, and it is refitted once per data set with the rerun's sampler schedule, about 16 min per fit on 4
  threads.
- **A new event-selecting variant** (`cpca/sparse.py`, described next).

**Event-sparse complex templates.** Complex PCA averages over all events, so a sequence played in a few
percent of them is diluted below the noise bound. The variant fits each template only to the events that
match it best:
- seeds are single events;
- each event is matched forward or reversed (conjugated), rotated to the template's phase, and the top share
  (3% or 10%) of events is kept;
- the template is refitted on those events, repeated until the kept events stop changing;
- several templates are returned and scored with the usual best-of-outputs null.

**On planted data:**
- **The variant lowered the threshold 2 to 4 times.** With 9 cells per replay, at 5% replays it recovered
  4/4 against complex PCA's 1/4. With 15 cells per replay, at 2% it recovered 4/4 against 1/4.
- **False positives:** 1 of 24 no-replay draws (4%).

**Results on hc-11** (draws passing, two largest sessions; complex PCA's values in brackets):

| Method | 5% | 10% | 20% |
|---|---|---|---|
| Complex PCA, 1500 events | 1/10 | 9/10 | 10/10 |
| Complex PCA, all events | 2/10 | 8/10 | 10/10 |
| Event-sparse, raw match | 3/10 | 2/10 | 8/10 |
| Event-sparse, cosine match | 1/10 | 0/10 | 1/10 |
| Rastermap | 0/10 | 1/10 | 3/10 |

- **Other six sessions, at 20%:**
  - complex PCA 10/30 on 1500 events, 14/30 on all events;
  - event-sparse, raw 11/30;
  - event-sparse, cosine 2/30;
  - Rastermap 1/30.
- **No replays added:** every method passed in at most 2/10 draws in the two largest sessions (complex PCA
  on 1500 events: 2/10) and in 0–1/30 in the other six.
- **PP-Seq:** 0 of 2 at every level, including 20%, in the two largest sessions, at about 13 min per fit. Its
  hyperparameters were fixed to the setting chosen on the same events before any replays were added, so a new
  sweep on the spiked data might suit it better; this short check cannot rule that out.

**Why the event-selecting variant fails on real events.**
- **Raw matching follows event size.** It keeps the largest population bursts (a median of 39–56 active
  cells, against 21 overall). Only 2–7% of what it keeps are planted replays, below the 10% base rate.
- **Cosine matching does not help.** With the size bias removed it still keeps 2–9% replays. It also loses
  its planted-data advantage, which therefore came from planted replay events being larger than background
  events.
- **The underlying reason is overlap.** Two real replay events share only a few active place cells, because
  each recruits its own small subset. Repeated co-firing groups in the background resemble each other more
  than two replays do. Complex PCA adds up the weak pairwise timing relations across all events, and so
  integrates partial overlaps that matching one event against another cannot.

The variant was not tuned further, to avoid fitting it to this benchmark.

## Reading

- **On hc-11 sleep, no method could do much better than complex PCA does.** Perfect knowledge of the
  replays gives no better order (Step 0).
- **Complex PCA has the lowest threshold** of the methods tested (Step 1). About 10% ideal replays are
  needed in the sessions with the most place cells, against a real excess of 0–3.4%.
- **More events help a little.** The all-events gain is clearest in the six smaller sessions at 20%.
- **Event selection doesn't help on real events.** It helps on planted data, where replays stand out as
  large events, but not here.
- **Rastermap needs far more replay** than complex PCA.

The limit on hc-11 is not specific to complex PCA. The threshold is shared by, or higher for, the other
label-free methods tested, and the data lie below it.

## Addendum (November 5): uncentred complex PCA on the same spiked events

`scripts/run_power_hc11.py --methods=cpca_uncentred` was run on exactly the data sets of the centred power test. It
asks whether the centring rule (centred for sleep events) cost sensitivity.

| Sessions | Version | 0% | 5% | 10% | 20% |
|---|---|---|---|---|---|
| Two largest | centred | 2/10 | 1/10 | 9/10 | 10/10 |
| Two largest | uncentred | 0/10 | 0/10 | 8/10 | 10/10 |
| Other six | centred | 0/30 | 0/30 | 5/30 | 10/30 |
| Other six | uncentred | 0/30 | 2/30 | 7/30 | 11/30 |

- **The threshold is the same,** about 10%.
- **Uncentred gives fewer false positives at 0%** (0/10 against 2/10 in Achilles 10/25) and is slightly better in
  the six smaller sessions.

**Reading.** Centring is not what limits complex PCA on sleep events, and uncentred is not worse there. Uncentred is
clearly better on hc-5 and on hc-11 waking laps. Centred is better only on the planted realistic rung L6, which
recovered 5/6 maps against 1/6. The paper should report both versions everywhere.
