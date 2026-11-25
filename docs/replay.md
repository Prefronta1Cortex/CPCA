# Is there replay in hc-11 sleep, and is there enough for complex PCA?

Code: `cpca/replay.py`, `scripts/run_replay_hc11.py`, `scripts/run_spikein_hc11.py`, `scripts/summarize_replay.py`.
Results: `results/replay/summary.md`; figure: `figures/replay_hc11.png`.

## Question

Every label-free method is at chance on hc-11 sleep (`docs/methods_hc11.md`), and our supervised ceiling finds
almost no single events in the map's order (`docs/ceiling.md`). That ceiling test is weak: it uses half of our
place cells and a rank correlation per event. It also conflicts with the original paper, which reports replay
of the maze in POST sleep. Two tests:

1. Does the paper's own analysis, run on our copy of the data, find replay? If it does not, our pipeline is
   suspect.
2. If it does, is the amount of replay enough for complex PCA? This is checked by adding known amounts of
   replay to real sleep events.

## 1. The paper's Bayesian replay analysis (Grosmark & Buzsáki 2016, Supplementary Materials)

The analysis is reproduced as written; `cpca/replay.py` lists every step. In brief:

- **Events:** population-synchrony events (1 ms bins, 15 ms Gaussian, mean + 3 SD of NREM in PRE and POST).
  They last 50–500 ms and at least 5 pyramidal cells fire in them.
- **Epochs:** NREM yoked to the shorter epoch, plus immobility on the maze (MAZE).
- **Place cells:** 2 cm bins, 5 cm smoothing, speed of at least 5 cm/s. A cell has a field if at least 5
  consecutive bins are above the 99th percentile of 5000 spike-redraw nulls, with a peak of at least 1 Hz.
- **Decoding:** events of at least 100 ms in which at least max(5, 10%) of the place cells fire. 20 ms bins,
  each linear-track direction decoded separately, the event's direction chosen by Pr(direction | spikes).
- **Score:** rZ against 1000 templates with circularly shifted fields; the secondary null shuffles the
  event's time bins.

**Deviations:**
- **No ripple condition.** The files contain no LFP, so the paper's requirement of an LFP ripple inside each
  synchrony event cannot be applied. In the paper, 50.6% of synchrony events passed it. Our event set is
  therefore larger, and its replay share is diluted by up to about a factor of two.
- **Yoking** is implemented as the last T of PRE and the first T of POST.

**Checks:**
- **Toy events.** Ordered toy events score r = ±0.97 and are all significant. Unordered ones are significant
  at 7% (place shuffle) and 3% (time shuffle). Their mean rZ is 0.34, not 0, which is a known property of this
  null; the paper therefore compares epochs, not rZ against 0.
- **Running.** Position decoded from running in 500 ms bins has a median error of 4–9 cm, against 32–104 cm
  with shuffled fields.

**Results:**
- **POST scores above PRE.** Mean rZ is higher in POST in 7 of 8 sessions (Wilcoxon p = 0.027), and for all
  events pooled (one-sided rank-sum p = 0.002).
- **Rest on the maze scores highest.**

This is the paper's finding, so the data and our event pipeline do contain replay of the maze. Our own burst
windows score about the same as the paper's events, so the events we gave every method do not lose sequence
content.

**How much:**
- **Clear replays** (p < 0.05 under both shuffles) make up 2–8% of POST events.
- **The excess of POST over PRE** is a median of 0.8 percentage points and at most 3.4 points: about 75
  events in Achilles 10/25, the one session where complex PCA passes. Achilles 11/01 has +2.5 and Cicero 09/01
  +2.7.
- **Rest on the maze** has 16–19% clear replays in the best sessions.

**Side findings:**
- **The paper's place-cell criterion keeps about 1.5 times as many cells as our answer key** (Achilles 10/25:
  78/84 against 57/63). Our order test is therefore less powerful than it could be.
- **Rotations on the circular track are unnecessary.** hc-11's circular-track coordinate is a full circle
  minus the 30 cm reward area, run in one direction. The rotation search in the order test costs power without
  being needed. Both side findings apply to every method alike.

## 2. How much replay complex PCA needs (spike-in on real events)

**Base:** the matched 1500 PRE events of each session, the same events every method was given. A share f of
them is turned into replays of the waking map in one of two ways:
- **Ideal:** each place cell firing in the event keeps its spike count, and its spikes move to the time of its
  field centre, forward or reverse, with 5% jitter.
- **Awake:** real clear replays from rest on the maze (from test 1), each used once.

Complex PCA (centred, the rerun's read-out) is then scored by the usual best-of-outputs test, with 5 draws
per session and level.

**Ideal replays:**
- **1–5% of events:** essentially never detected. Of 160 draws in total, 3 pass, about the rate at 0%.
- **10%:** reliable only in the two Achilles sessions, 5 of 5 and 4 of 5 draws (median 10–12 place cells
  active per event).
- **20%:** reliable in three of the four sessions with 7–12 active place cells per event. The fourth,
  Cicero 09/10, is on the circular track, where the unnecessary rotation search costs power. In sessions with
  a median of 3–5 it mostly fails even at 20%, and a short ordered list cannot beat the null.

**Real awake replays at 1–3%:** 0 of 5 in every session that has enough of them (Achilles 10/25,
Cicero 09/01, Cicero 09/17, Gatsby 08/02).

## Reading

The plateau on hc-11 comes mainly from the data, but in a specific sense: the data's replay content is well
below what a label-free method needs. There is replay of the maze in hc-11 sleep, but beyond the level before
the maze it is in at most about 3% of events, and in most sessions under 1%. Complex PCA needs about 10% clear
replays even under ideal conditions, in the sessions with the most place cells. That is 3 to 10 times more
than the data have. In small sessions it needs more than 20%, because too few place cells fire per event. The
Bayesian decoder finds the replays only because it is given the map.

**Qualifications:**
- **Missing ripple condition.** The real replay share is likely higher with ripple-gated events, up to about
  twice as high, which is still below 10%.
- **The power curve is optimistic.** Ideal replays are perfectly ordered; real ones are messier, as the awake
  replays show.
- **Scoring could be more powerful.** A looser answer key and no rotations on the circular track would lower
  the detection threshold somewhat for every method.
- **Only complex PCA was measured.** Whether another label-free method needs less replay was not measured; all
  of them were at chance on the real events.
- **This is hc-11 only.** DANDI 000552 (a familiar maze) has not been checked this way.

**What it implies:** testing sleep replay extraction without labels needs recordings where replay is common
and many place cells fire per event. Examples are rest on the maze (16–19% clear replays in the best
sessions), early POST sleep, or recordings with hundreds of place cells. hc-11 sleep cannot carry that test
for any label-free method.
