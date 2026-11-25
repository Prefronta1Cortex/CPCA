# Sequences that share cells

Open with `Cmd+Shift+V` for formulas. Code: `cpca/mixture.py`, `scripts/run_sharing.py`, `scripts/summarize_sharing.py`. Results: `results/sharing/summary.md`; figures: `figures/sharing_*.png`.

## Acceptance criteria (written on October 22 before the fresh runs; not changed afterwards)

The method was developed on draws with seeds 1000–1001 (`sim.routes` conditions and ladder rungs). Everything below is scored on data it has not seen: 10 fresh draws per condition (3000 events), and the difficulty ladder's own events (3 draws, 1500 events each), on which PP-Seq and seqNMF were already run.

Step 3's output is the set of templates that pass the held-out check; nothing else is scored (no fallback to the components). Per planted map, the score is the best template's |Spearman| with the planted positions, with the coverage of the map reported alongside.

1. **Two orders on the same cells, whole routes** (L1, L2, L4; E): centred, median |ρ| per map at least 0.95 on the fresh draws and on the ladder events (PP-Seq there: 0.99–1.00; complex PCA components: 0.80–0.88).
2. **No harm where sequences do not share cells** (A, B, C, D, F, L0): median |ρ| per map not more than 0.02 below the direction split (Step 2) on the same draws.
3. **No invention.** No-order control (L0 with each event's spike trains handed to a random permutation of the cells that fired): a validated template in at most 1 of 10 draws. In every condition the number of validated templates is reported against the number of planted maps; templates that match no planted map are counted as false.
4. **Partial replays** (G, H, L3): no target; reported with coverage.
5. **Variable pacing** (L5): reported with the fixed half-circle scale and with each event's own spike spread as the scale (development draws: the fixed scale collapses the two maps into one template, the per-event scale separates them).
6. **hc-11** (8 sessions, POST and PRE, centred): number of validated templates; each tested against both running directions' place maps (`cpca.evaluate.order_vs_place`). Expected if the method works: more map-matching templates in POST than in PRE, and in the sessions with 50+ place cells, separate templates for the two running directions. Direction labels of each validated template checked against each event's rank-order correlation with the map, as in Step 2.
7. **Cost**: run time per fit, against complex PCA alone and against PP-Seq.

hc-5 is not used: 7–25 runs per session, fewer than the events the held-out check needs.

## Why PCA mixes two sequences on the same cells

Two maps on the same 60 cells, each with independent positions, are not orthogonal patterns of arrows: with positions uniform on the half-circle, $\mathbb{E}\,e^{j\pi(p_1-p_2)} = (2/\pi)^2 \approx 0.4$. When both are replayed about equally often, the eigenvectors of the lag matrix are their sum and their differences, not the maps. On the first fresh L2 draw (`figures/sharing_synthetic.png`, panel a) the first component follows both maps at once (|ρ| 0.61 and 0.86) and the next ones each map partly (0.77 / 0.83, 0.87 / 0.12). The direction split of Step 2 fits the blend; on the ladder it even lowers L1 from 0.88 to 0.80. This is what the difficulty ladder pointed at: one map already reaches 1.00 after Step 2, two maps on the same cells stay at 0.80–0.88.

## The method (`cpca/mixture.py`)

PCA stays the first stage; Step 3 starts from its kept components.

1. **Cells and events.** The union of the components' members; events in which at least 3 of them fire; raw arrows.
2. **Starts.** $K$ templates, $K$ = the number of kept components, started from the components and from three complex ICA rotations of them (ICA's outputs are sparser and already close to single sequences).
3. **Assign and refit.** Each event goes to the template it matches best either way round, $\max(|\langle s,q\rangle|, |\langle s,\bar q\rangle|)$, and takes its direction; it counts only if this best match beats the 95th percentile of the same quantity after shuffling which cell got which arrow in the event. Each template is refitted on its events with the backward ones conjugated. Once the assignment has settled, templates with fewer than 20 events are dropped and two templates at least 0.8 alike (up to conjugation) are merged. Of the four starts, the one whose templates capture the most energy on their events is kept.
4. **Held-out check.** Steps 1–3 are rerun on each half of the events (PCA, ICA and all); each full-data template is matched one-to-one to a half's templates; it is kept only if in both halves its match is at least 0.8 alike and finds expressing events in the other half above the 5% expected without order (each held-out event judged by its own permutation p-value; binomial; Bonferroni over templates and halves).

Three corrections were needed during development, all on development draws: merging and dropping must wait until the assignment has settled (early on, the template that becomes one map starts with few events and close to the blend it is being separated from, and was removed in favour of the blend); and the energy that chooses between starts must count only gated events (otherwise the blend, which matches synchronous volleys well, wins). The held-out match was first "most similar template in the half"; a volley template on route cells then borrowed a route template's evidence, so the match is one-to-one with a 0.8 threshold.

## Results

**Planted data** (`figures/sharing_synthetic.png`, 10 fresh draws per condition).

- *Criterion 1 (met).* Two orders on the same cells: centred Step 3 recovers every map at median |ρ| 1.00 in E, L1, L2 and L4 (Step 2: 0.78–0.89), with two validated templates per fit (one extra in one L4 draw), 100% of each map's replays assigned to its template (88–98% of a template's events are its map's replays, the rest volleys that pass the gate) and every direction right. Uncentred it works for E, L1 and L4 and collapses in some L2 draws (median 0.74).
- *Criterion 2 (met).* Where sequences do not share cells (A–D, F, L0) Step 3 matches Step 2 (1.00); A centred keeps no component (the route is in every event), as before.
- *Criterion 3 (met for the control; not for partial replays).* No validated template in any of the 10 no-order draws, either convention. With whole routes, 1 false template in 120 centred fits. With partial replays, more: 1 in G, 9 in H and 20 in L3 over 10 draws each (templates that mix segments of the two maps).
- *Criterion 4.* Partial replays: centred G 0.45 → 0.73, H 0.51 → 0.80, L3 0.58 → 0.64; but each map comes back as several segment templates covering 62–76% of it, and recall per template is about 25–30%. Uncentred L3 gets worse (0.53 → 0.22).
- *Criterion 5.* Variable pacing (L5): 0.80 → 0.90 with the fixed half-circle scale, 1.5 validated templates of 2 per fit; with each event's own spike spread as the scale, 1.00 and 2 of 2. The fixed 100 ms window of the planted events is what breaks it; hc-11 already uses each event's duration.

**Difficulty ladder, same events as PP-Seq and seqNMF** (`figures/sharing_ladder.png`). Centred, median |ρ| per map (maps recovered):

| rung | components | Step 3 | PP-Seq | seqNMF (K = 15, L = 120 ms) |
|---|---|---|---|---|
| L0 one map, both ways | 0.90 (3/3) | 1.00 (3/3) | 1.00 (3/3) | 0.97 (2/3) |
| L1 two maps, same cells, forward | 0.88 (6/6) | 1.00 (6/6) | 1.00 (6/6) | 0.99 (4/6) |
| L2 two maps, same cells, both ways | 0.83 (6/6) | 1.00 (6/6) | 1.00 (6/6) | 0.47 (4/6) |
| L3 + partial replays | 0.59 (4/6) | 0.70 (6/6) | 0.35 (3/6) | 0.08 (0/6) |
| L4 + assemblies | 0.85 (6/6) | 1.00 (6/6) | 1.00 (6/6) | 0.36 (4/6) |
| L5 + variable pacing | 0.81 (6/6) | 1.00 (5/6) | 0.99 (6/6) | 0.26 (3/6) |
| L6 + partial + assemblies (= H) | 0.48 (5/6) | 0.81 (6/6) | 0.39 (3/6) | 0.15 (0/6) |

On whole routes complex PCA with Step 3 now matches PP-Seq; on partial replays it is ahead of it (with the coverage caveat above: its best output is a segment). Median run time per fit: complex PCA 1.4 s, with Step 3 3.5 s (2.4–13 s), seqNMF's grid 50 min, PP-Seq's sweep and refit 116 min.

**hc-11** (`figures/sharing_hc11.png`). Step 3 adds almost nothing on these recordings. Across 16 session-epochs it fits 2–9 templates, of which 3 pass the held-out check: one in Achilles 10/25 POST, which matches both running directions' maps (ρ +0.66 and +0.50 against nulls of 0.30 and 0.25) and labels the direction of all 34 of its significant replays in agreement with the map; one in Achilles 10/25 PRE and one in Cicero 09/10 PRE, neither on the map. Most templates fail the significance part of the check (84 of 88), about half of them while reproducing in shape across halves (similarity ≥ 0.8): held-out sleep events rarely express a template with significant order, as in Step 2, where 3–8% of held-out events expressed the whole-population template against 5% chance. With so few cells active per event, single hc-11 events carry too little order for a per-event test. The components themselves still match the map in 18 of 60 POST components and 2 of 52 PRE (either running direction).

## What this does and does not show

It shows that the gap between complex PCA and PP-Seq on planted whole routes was the mixing of non-orthogonal sequences that share cells, and that a step after PCA (one template per sequence, events assigned to them, checked on held-out events) closes it at a few seconds per fit, against an hour or two for PP-Seq. It does not make partial replays a solved problem (segments, not maps, and false templates mixing them), and it does not change the hc-11 result: on these recordings the evidence remains at the level of components and of the reduced space, not of individual sequences. The cost in simplicity is real: Step 3 is an iterative fit with four starts, a gate, a merge rule and a held-out refit, where complex PCA alone is one eigen-decomposition.
