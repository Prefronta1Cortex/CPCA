# Discovering reproducible sequences in reactivations, then relating them to waking (draft design)

Status: draft for discussion, nothing run. Written November 5, 2026.

## Framing

The question changes from "does the method recover the waking map from sleep?" to "does sleep contain sequences the
method can discover and reproduce on held-out events, and how do they relate to waking?".
- **Why change it:** a sleep sequence that does not copy the waking map (remote replay, a pre-existing "rigid"
  sequence, a fragment, or a map retuned during sleep) counted as failure before; here it counts as a discovery.
- **What carries the weight instead:** without the waking map as the answer key, three internal controls have to.
- **Where the map comes in:** only in the last step, as a description.

## Step 1: discovery (training events)

- **Events:** per session and epoch (POST, PRE, MAZE rest), all bursts, split into training and held-out halves
  by alternating 10-minute blocks of the recording. Alternating blocks keep the halves apart in time, so slow
  drift or nearby events do not link them.
- **Templates:** complex PCA (centred, components above the shuffle null) on the training half. Each kept
  component, and each conjugate-pair combination (`cpca/pairs.py`), is a candidate template q, with members its
  cells with weight of at least 0.2 of the largest.

## Step 2: reproducibility with controls (held-out events)

**Expression** of a template in a held-out event: the cosine between the event's arrows on the template's members
and the template, forward or reversed (q or its conjugate), whichever is larger. Events with fewer than 4 active
members are skipped. The **statistic** is the mean expression over the held-out events in which the template is
expressed most: the top 10% (pre-specified), so that a sequence present in a minority of events can count.

A template is reproducible only if it beats all three nulls, each built from the held-out events (template and
events otherwise unchanged; 200 surrogates each):

1. **Membership-only null.** Spike trains shuffled among each event's active cells. Who fires is kept; who fires
   when is destroyed. It must be beaten, so the result is not co-firing.
2. **Latency null.** Each active cell's spike time redrawn from that same cell's own distribution of relative
   times in the training events. Each cell's typical early or late firing within bursts is kept; coordination
   within an event is destroyed. It must be beaten, so the result is not the latency structure common to every
   burst.
3. **Template null.** The same statistic for templates fitted to training events whose spike trains were shuffled
   as in null 1, against the real held-out events. It must be beaten, so fitting itself does not create
   reproducibility.

Multiplicity: the best template per session and epoch, with each null taking the same best over templates. A
session counts if p < 0.05 against all three.

## Step 3: relation to waking (descriptive, for reproducible templates only)

- **The waking map:** whole-track and fragment order tests, plus containment in the waking components
  (`cpca/containment.py`).
- **Across epochs:** a POST template's expression in PRE held-out events, and a PRE template's in POST, against
  the same nulls. This separates templates new after the experience from pre-existing ones.
- **Where they are expressed:** the share of the events with the strongest expression that are ripple events, and
  that are clear replays by the paper's decoder.

## Validation (planted, before any real data)

- **No replays, with assemblies and per-cell latency structure (`sim/segments.py`):** at most about 5% sessions
  reproducible.
- **Replays of a map different from the waking map:** reproducible, with failure on the waking-map order test.
  This is the case the old criterion could not see.
- **Replays of the waking map:** reproducible, and the map recovered in Step 3.
- **Shares of 2, 5, 10%; cells per replay of 5, 9, 13; both directions:** gives the detection threshold of
  reproducibility itself.

## Data

hc-11 (8 sessions), DANDI 001695 (15 sessions, CA1), DANDI 000552 (65 sessions). Primary: POST and PRE counted
separately. A PRE/POST difference is not required for "discovery", but it is reported.
