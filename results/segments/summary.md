# G1 with the direction oracle: constant-speed partial replays

Produced by `scripts/summarize_segments.py` from `scripts/run_segments.py`; design in `docs/segments.md`. Each cell: draws (of 5) whose best output passes the whole-track order test (p < 0.05) against the planted positions. Share 0 is shared between the two direction balances.

## Reverse fraction 0.5: whole-track order test, draws passing

| ordered cells per replay | share | n | complex PCA, centred | complex PCA, uncentred | lead-lag component | direction EM | direction oracle | event + direction oracle |
|---|---|---|---|---|---|---|---|---|
| 5 (measured 0.0) | 0% | 5 | 0/5 | 0/5 | 1/5 | 1/5 | 0/5 | 0/5 |
| 5 (measured 5.0) | 2% | 5 | 1/5 | 0/5 | 0/5 | 0/5 | 4/5 | 5/5 |
| 5 (measured 5.0) | 5% | 5 | 0/5 | 0/5 | 0/5 | 1/5 | 4/5 | 5/5 |
| 5 (measured 5.0) | 10% | 5 | 0/5 | 1/5 | 0/5 | 1/5 | 5/5 | 5/5 |
| 9 (measured 0.0) | 0% | 5 | 2/5 | 0/5 | 0/5 | 0/5 | 1/5 | 0/5 |
| 9 (measured 8.9) | 2% | 5 | 0/5 | 1/5 | 0/5 | 0/5 | 3/5 | 5/5 |
| 9 (measured 9.5) | 5% | 5 | 2/5 | 0/5 | 0/5 | 0/5 | 5/5 | 5/5 |
| 9 (measured 9.5) | 10% | 5 | 4/5 | 1/5 | 1/5 | 0/5 | 5/5 | 5/5 |
| 13 (measured 0.0) | 0% | 5 | 2/5 | 1/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 13 (measured 12.4) | 2% | 5 | 0/5 | 0/5 | 0/5 | 0/5 | 4/5 | 5/5 |
| 13 (measured 12.8) | 5% | 5 | 4/5 | 0/5 | 0/5 | 1/5 | 5/5 | 5/5 |
| 13 (measured 13.4) | 10% | 5 | 3/5 | 0/5 | 1/5 | 0/5 | 5/5 | 5/5 |

## Reverse fraction 0.3: whole-track order test, draws passing

| ordered cells per replay | share | n | complex PCA, centred | complex PCA, uncentred | lead-lag component | direction EM | direction oracle | event + direction oracle |
|---|---|---|---|---|---|---|---|---|
| 5 (measured 0.0) | 0% | 5 | 0/5 | 0/5 | 1/5 | 1/5 | 0/5 | 0/5 |
| 5 (measured 4.8) | 2% | 5 | 1/5 | 0/5 | 0/5 | 1/5 | 2/5 | 5/5 |
| 5 (measured 5.5) | 5% | 5 | 2/5 | 0/5 | 2/5 | 0/5 | 4/5 | 5/5 |
| 5 (measured 5.3) | 10% | 5 | 3/5 | 0/5 | 1/5 | 0/5 | 5/5 | 5/5 |
| 9 (measured 0.0) | 0% | 5 | 2/5 | 0/5 | 0/5 | 0/5 | 1/5 | 0/5 |
| 9 (measured 9.0) | 2% | 5 | 0/5 | 0/5 | 1/5 | 0/5 | 3/5 | 5/5 |
| 9 (measured 8.7) | 5% | 5 | 3/5 | 0/5 | 1/5 | 1/5 | 5/5 | 5/5 |
| 9 (measured 9.0) | 10% | 5 | 4/5 | 0/5 | 0/5 | 1/5 | 5/5 | 5/5 |
| 13 (measured 0.0) | 0% | 5 | 2/5 | 1/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 13 (measured 13.2) | 2% | 5 | 2/5 | 0/5 | 1/5 | 2/5 | 4/5 | 5/5 |
| 13 (measured 12.6) | 5% | 5 | 4/5 | 0/5 | 1/5 | 1/5 | 5/5 | 5/5 |
| 13 (measured 13.3) | 10% | 5 | 5/5 | 1/5 | 0/5 | 1/5 | 5/5 | 5/5 |

## Fragment criterion, bar 1 (a 40% stretch, at least 8 cells), draws passing

| reverse | cells | share | complex PCA, centred | complex PCA, uncentred | lead-lag component | direction EM | direction oracle | event + direction oracle |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 5 | 0% | 0/5 | 0/5 | 1/5 | 1/5 | 0/5 | 0/5 |
| 0.5 | 5 | 2% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 5/5 |
| 0.5 | 5 | 5% | 0/5 | 0/5 | 0/5 | 0/5 | 1/5 | 5/5 |
| 0.5 | 5 | 10% | 1/5 | 0/5 | 0/5 | 0/5 | 3/5 | 5/5 |
| 0.5 | 9 | 0% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 0.5 | 9 | 2% | 1/5 | 0/5 | 0/5 | 0/5 | 1/5 | 5/5 |
| 0.5 | 9 | 5% | 2/5 | 0/5 | 0/5 | 0/5 | 2/5 | 5/5 |
| 0.5 | 9 | 10% | 3/5 | 1/5 | 1/5 | 1/5 | 5/5 | 5/5 |
| 0.5 | 13 | 0% | 0/5 | 0/5 | 1/5 | 1/5 | 0/5 | 0/5 |
| 0.5 | 13 | 2% | 0/5 | 0/5 | 0/5 | 0/5 | 2/5 | 5/5 |
| 0.5 | 13 | 5% | 4/5 | 1/5 | 1/5 | 1/5 | 5/5 | 5/5 |
| 0.5 | 13 | 10% | 3/5 | 3/5 | 0/5 | 3/5 | 5/5 | 5/5 |
| 0.3 | 5 | 2% | 0/5 | 0/5 | 0/5 | 0/5 | 2/5 | 4/5 |
| 0.3 | 5 | 5% | 2/5 | 1/5 | 0/5 | 1/5 | 2/5 | 4/5 |
| 0.3 | 5 | 10% | 3/5 | 0/5 | 0/5 | 0/5 | 1/5 | 5/5 |
| 0.3 | 9 | 2% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 5/5 |
| 0.3 | 9 | 5% | 4/5 | 1/5 | 0/5 | 1/5 | 2/5 | 5/5 |
| 0.3 | 9 | 10% | 5/5 | 5/5 | 0/5 | 1/5 | 4/5 | 5/5 |
| 0.3 | 13 | 2% | 2/5 | 1/5 | 2/5 | 0/5 | 3/5 | 5/5 |
| 0.3 | 13 | 5% | 5/5 | 0/5 | 0/5 | 0/5 | 4/5 | 5/5 |
| 0.3 | 13 | 10% | 5/5 | 5/5 | 1/5 | 1/5 | 5/5 | 5/5 |

## Timing control (30 shuffled refits): draws passing whole-track + timing, and fragment bar 1 + timing

| reverse | cells | share | cPCA whole+timing | cPCA fragment+timing | lead-lag whole+timing | lead-lag fragment+timing | median timing p cPCA / lead-lag |
|---|---|---|---|---|---|---|---|
| 0.5 | 5 | 0% | 0/5 | 0/5 | 1/5 | 0/5 | 0.84 / 0.03 |
| 0.5 | 5 | 2% | 0/5 | 0/5 | 0/5 | 0/5 | 0.81 / 0.37 |
| 0.5 | 5 | 5% | 0/5 | 0/5 | 0/5 | 0/5 | 0.52 / 0.68 |
| 0.5 | 5 | 10% | 0/5 | 0/5 | 0/5 | 0/5 | 0.35 / 0.45 |
| 0.5 | 9 | 0% | 1/5 | 0/5 | 0/5 | 0/5 | 0.48 / 0.06 |
| 0.5 | 9 | 2% | 0/5 | 0/5 | 0/5 | 0/5 | 0.68 / 0.03 |
| 0.5 | 9 | 5% | 2/5 | 1/5 | 0/5 | 0/5 | 0.03 / 0.35 |
| 0.5 | 9 | 10% | 3/5 | 1/5 | 1/5 | 1/5 | 0.03 / 0.32 |
| 0.5 | 13 | 0% | 0/5 | 0/5 | 0/5 | 0/5 | 0.35 / 0.39 |
| 0.5 | 13 | 2% | 0/5 | 0/5 | 0/5 | 0/5 | 0.61 / 0.55 |
| 0.5 | 13 | 5% | 4/5 | 3/5 | 0/5 | 1/5 | 0.03 / 0.29 |
| 0.5 | 13 | 10% | 3/5 | 3/5 | 0/5 | 0/5 | 0.03 / 0.13 |
| 0.3 | 5 | 2% | 0/5 | 0/5 | 0/5 | 0/5 | 0.45 / 0.13 |
| 0.3 | 5 | 5% | 2/5 | 2/5 | 2/5 | 0/5 | 0.26 / 0.18 |
| 0.3 | 5 | 10% | 3/5 | 2/5 | 1/5 | 0/5 | 0.03 / 0.10 |
| 0.3 | 9 | 2% | 0/5 | 0/5 | 1/5 | 0/5 | 0.29 / 0.03 |
| 0.3 | 9 | 5% | 3/5 | 3/5 | 1/5 | 0/5 | 0.03 / 0.13 |
| 0.3 | 9 | 10% | 4/5 | 4/5 | 0/5 | 0/5 | 0.03 / 0.42 |
| 0.3 | 13 | 2% | 0/5 | 0/5 | 1/5 | 1/5 | 0.32 / 0.45 |
| 0.3 | 13 | 5% | 4/5 | 4/5 | 1/5 | 0/5 | 0.03 / 0.35 |
| 0.3 | 13 | 10% | 4/5 | 4/5 | 0/5 | 0/5 | 0.03 / 0.81 |

## Per-event direction labels on true replays (accuracy, median over draws)

| reverse | cells | share | direction EM (label-free) | labels from the oracle template | EM forward share |
|---|---|---|---|---|---|
| 0.5 | 5 | 2% | 0.59 | 0.67 | 0.95 |
| 0.5 | 5 | 5% | 0.52 | 0.75 | 0.94 |
| 0.5 | 5 | 10% | 0.55 | 0.79 | 0.95 |
| 0.5 | 9 | 2% | 0.56 | 0.76 | 0.96 |
| 0.5 | 9 | 5% | 0.57 | 0.82 | 0.95 |
| 0.5 | 9 | 10% | 0.50 | 0.85 | 0.95 |
| 0.5 | 13 | 2% | 0.62 | 0.76 | 0.95 |
| 0.5 | 13 | 5% | 0.53 | 0.88 | 0.94 |
| 0.5 | 13 | 10% | 0.56 | 0.88 | 0.89 |
| 0.3 | 5 | 2% | 0.69 | 0.64 | 0.96 |
| 0.3 | 5 | 5% | 0.67 | 0.71 | 0.95 |
| 0.3 | 5 | 10% | 0.65 | 0.82 | 0.95 |
| 0.3 | 9 | 2% | 0.64 | 0.77 | 0.96 |
| 0.3 | 9 | 5% | 0.64 | 0.85 | 0.95 |
| 0.3 | 9 | 10% | 0.70 | 0.87 | 0.95 |
| 0.3 | 13 | 2% | 0.63 | 0.77 | 0.94 |
| 0.3 | 13 | 5% | 0.65 | 0.88 | 0.94 |
| 0.3 | 13 | 10% | 0.69 | 0.89 | 0.96 |

## Controls (9 ordered cells per replay)

| control | n | complex PCA, centred | complex PCA, uncentred | lead-lag component | direction EM | direction oracle | event + direction oracle | cPCA fragment+timing | lead-lag fragment+timing |
|---|---|---|---|---|---|---|---|---|---|
| assemblies_no_order | 5 | 3/5 | 5/5 | 4/5 | 0/5 | 0/5 | 0/5 | 0/5 | 1/5 |
| forward_only_5pct | 5 | 4/5 | 1/5 | 0/5 | 2/5 | 5/5 | 5/5 | 4/5 | 0/5 |

## Pass criteria of docs/segments.md

- False positives at share 0, complex PCA, centred: 4/15 whole-track passes (criterion: at most 2 of 15).
- False positives at share 0, complex PCA, uncentred: 1/15 whole-track passes (criterion: at most 2 of 15).
- False positives at share 0, lead-lag component: 1/15 whole-track passes (criterion: at most 2 of 15).
- False positives at share 0, direction EM: 1/15 whole-track passes (criterion: at most 2 of 15).
- False positives at share 0, direction oracle: 1/15 whole-track passes (criterion: at most 2 of 15).
- False positives at share 0, event + direction oracle: 0/15 whole-track passes (criterion: at most 2 of 15).
- complex PCA, uncentred: does not beat complex PCA. Cells with +3 or more draws in the low-share/few-cell region: none; cells with -3 or fewer: [(0.5, 9, 0.1, -3), (0.5, 13, 0.05, -4), (0.5, 13, 0.1, -3), (0.3, 5, 0.1, -3), (0.3, 9, 0.05, -3), (0.3, 9, 0.1, -4), (0.3, 13, 0.05, -4), (0.3, 13, 0.1, -4)].
- lead-lag component: does not beat complex PCA. Cells with +3 or more draws in the low-share/few-cell region: none; cells with -3 or fewer: [(0.5, 9, 0.1, -3), (0.5, 13, 0.05, -4), (0.3, 9, 0.1, -4), (0.3, 13, 0.05, -3), (0.3, 13, 0.1, -5)].
- direction EM: does not beat complex PCA. Cells with +3 or more draws in the low-share/few-cell region: none; cells with -3 or fewer: [(0.5, 9, 0.1, -4), (0.5, 13, 0.05, -3), (0.5, 13, 0.1, -3), (0.3, 5, 0.1, -3), (0.3, 9, 0.1, -3), (0.3, 13, 0.05, -3), (0.3, 13, 0.1, -4)].
