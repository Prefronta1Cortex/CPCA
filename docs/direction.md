# Forward and backward plays inside one component

Open with `Cmd+Shift+V` for formulas. Code: `cpca/direction.py` (the split and its tests), `scripts/run_direction.py` (planted data and hc-11, parallel), `scripts/summarize_direction.py` (figures, table). Results: `results/direction/summary.md`; figures: `figures/direction_synthetic.png`, `figures/direction_hc11.png`.

## Why

Hippocampal replays run a route forward or backward. In the half-circle encoding a spike at time $t$ of an event of length $T$ is the arrow $e^{j\pi t/T}$; played backwards it falls at $T - t$, whose arrow is

$$
e^{j\pi (T-t)/T} = -\,\overline{e^{j\pi t/T}} .
$$

**Time reversal is complex conjugation**, up to a sign that is a common phase of the event and cancels in every lag. So an event $s$ (its members' arrows) matches a template $q$ forward with strength $F = |\langle s, q\rangle|$ and backward with $B = |\langle s, \bar q\rangle|$, both unchanged by the event's clock. One template describes both directions.

What goes wrong without this: after centring, forward and backward plays of one route are opposite deviations from the same mean (with a 60/40 mix, $0.4\,(s_f - s_b)$ and $-0.6\,(s_f - s_b)$) and land on one component whose phases partly cancel. On planted data with one route played 60/40 both ways, the centred component's order reaches $|\rho|$ 0.74 of the route; uncentred components keep the two directions apart only when the route fills the half-circle.

## The split (per kept component)

1. **Members and events.** Members are the cells with weight at least 20% of the component's largest; the raw (uncentred) arrows of the members are used, in the events where at least 3 members fire.
2. **Template.** Alternate until the labels stop changing: label each event forward if $F \ge B$, conjugate the backward events, and take the leading eigenvector of their lag matrix as the template. Start from the leading eigenvector without flips.
3. **Gate.** An event expresses the template if $\max(F, B)$ exceeds the 95th percentile of the same quantity after shuffling which member got which arrow in that event (50 shuffles). The shuffle keeps who fired and the set of spike times, so a synchronous volley does not pass; only an event whose members fire in the template's order, either way round, does. The template is refitted on the expressing events, and directions are reported for those events only.
4. **Held-out test.** The template is fitted (steps 2–3) on one random half of the events; each event of the other half gets its own permutation p-value against it, with the same within-event shuffle. Without order among the members a held-out event reaches $p \le 0.05$ with probability at most 5%, independently given the template, so the count is tested against Binomial($n$, 0.05). Both halves take a turn and the smaller p is doubled.
5. **Guard.** The template replaces the component's order only if the two orders agree (|Spearman| over shared members at least 0.6). The split may sharpen an order but not overturn it: with partial replays one per-cell template is the wrong model (a 40% segment stretched over the whole event puts its cells 2.5 times further apart in angle than the template does), and the fit converges to something else. The threshold was set on development draws; every number below comes from fresh draws.

Which direction is called "forward" is arbitrary: $q$ and $\bar q$ describe the same pair of orders.

On hc-11 a second form is run as well: one template on all cells, with no component and therefore no guard (steps 1–4 with every cell a member). It is needed because, on hc-11, centred components spread their weight over 70–116 cells and the per-component templates collapse onto the session's dominant route (below).

## Planted data (`figures/direction_synthetic.png`)

80 cells, 3000 events, 10 fresh draws per condition, both conventions; per planted route the best component's order before and after the split, and for that component the labels of its own expressing events against the planted direction.

- **Where one route is played both ways the split restores the order.** Centred, median $|\rho|$ 0.74 → 1.00 (B, C), 0.81 → 1.00 (L0), 0.97 → 1.00 (F); uncentred L0 0.81 → 1.00. With two maps on the same cells (L2, L4, L5) the order moves from 0.72–0.80 to 0.80–0.81, never down.
- **Labels are right where the model applies.** Every event of the route labelled in its planted direction (median 100%) in B, C, F, L0, L2, L4, L5, both conventions; the reverse share is estimated within 0.01 (planted 0.40 → 0.40; 0.47–0.49 → 0.48–0.49).
- **Partial replays are outside the model.** In G, H and L3 (replays of 40% segments) labels are 54–60% correct, close to chance, and the guard keeps the component's order (0.46, 0.57, 0.61). This is the limit already seen in the difficulty ladder.
- **No reverse group in forward-only data.** D, E, L1: estimated reverse share 0.04–0.05 against 0 planted, except one L1 draw (two routes, both at 0.49): there the return map is close to the outbound map reversed, so "map 2 played forward" and "map 1 played backward" look the same and the split pairs the two maps as one route played both ways. This ambiguity is real and not specific to the method.
- **No order, nothing found.** In the no-order control (L0 with each event's spike trains handed to a random permutation of the cells that fired) 3.7% of held-out events express (5% chance), no component passes the held-out test and none is accepted.

## hc-11 (`figures/direction_hc11.png`)

Centred complex PCA on all population bursts of each session (as in `scripts/run_hc11.py`), POST and PRE. The waking place map is used only to score: per event, the rank-order correlation of its members' firing times with their field centres against a cell-identity shuffle (the standard replay score); a template's map is the running direction it matches best, and its labels are correct if a forward label goes with the template's own sign and a backward label with the opposite.

- **Example (Achilles 10/25 POST, panel a).** The whole-population template matches the map (ρ 0.46 against a null of 0.26); among its expressing events that are significant replays, 43 of 46 are labelled in the map's direction, and the reverse share is 0.29 by complex PCA (all expressing events) against 11/46 = 0.24 by the map (significant replays only).
- **Labels are right when the template is a good order** (panel c, the template that best matches the map per session): 49/51 in Achilles 10/25 POST, 7/7 Gatsby 08/02 POST, 7/9 Gatsby 08/28 POST; but 14/26 in Achilles 11/01 and 13/23 in Cicero 09/10 POST, where the template beats its null only narrowly and its sign relative to the map is unreliable. No template matches the map in three POST sessions (13–23 cells with a field per direction, null 95th percentiles of 0.5–0.8).
- **The two groups run opposite ways** (panel d). Templates fitted separately on the forward and on the backward events have opposite signs against the map for 23 of the 26 templates that match it and have 20 events in each group (for example +0.61 and −0.56 in Achilles 10/25 POST); the three exceptions are components of Achilles 11/01, the session whose labels are at chance.
- **The gate picks out replays** (panel e). For the best template of each session, expressing events are significant replays of the map 2–5 times as often as other events (Achilles 10/25 POST 15% against 3%); the exception is Gatsby 08/02 POST (3% against 4%).
- **What PRE shows.** The held-out test passes in the same three sessions in PRE as in POST (Achilles 10/25, Cicero 09/10, Gatsby 08/28), so part of the reproducible order in sleep is not specific to the maze. In Achilles 10/25 PRE a component's template also matches the map with 48/51 labels correct (3 of 12 PRE templates match the map there, against 8 of 11 in POST); the components themselves, before the split, match it in 0 of 12. I report this without interpreting it as preplay: one session, and the templates are fitted on 70–116 members.
- **The guard rejects nearly every hc-11 split** (it accepts 2 of 60 components in POST, 5 of 52 in PRE): the templates are not refinements of their components but the dominant route among broad member sets, so the component's order mostly stands (components matching the map in either direction: 17 of 60 in POST before and after the split; 2 and 3 of 52 in PRE). On hc-11 the information is in the templates, not in improved components.
- **Reverse shares are meaningful only where the template is real.** Where the held-out test fails, the gate still passes 4–8% of events and their labels split 0.3–0.5 either way; that number then describes noise. The earlier direction split by the sign of $\mathrm{Im}(a_1\overline{a_2})$ (`lessons_learned.md`) failed exactly this way, without a null.

## Not used: hc-5

The wheel runs, where time only runs forward, would be a real one-direction control, but each hc-5 session has 7–25 runs, fewer than the 20 events the split needs.

## What this does and does not show

It shows that one complex template and its conjugate describe a route played both ways, that the split recovers the order and the direction of every event when whole routes are replayed, that it invents neither a reverse group nor an order where there is none, and that on the best hc-11 session the labels agree with the map-based replay score on 43 of 46 replays without using the map. It does not handle partial replays, it cannot tell "map 2 forward" from "map 1 backward" when the two maps mirror each other, and on hc-11 it is limited, like every analysis here, by the number of place cells: only the two Achilles sessions have 50 or more per running direction.
