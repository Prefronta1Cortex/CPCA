# First-Principles Exploration of Complex PCA (`exploration/`)

All exploratory code, fresh-seed confirmation results, and figures live exclusively under `exploration/`. No original files in `cpca/`, `sim/`, `scripts/`, or `docs/` were modified.

## Root Causes of the Roadblock

1. **Two conjugation bugs in `cpca/ica.py`** (lines 42 and 71):
   - Line 42 computed `(X.T @ (np.conj(Y) * g)) / M` ($\propto W^*$) instead of `(X.conj().T @ (Y * g)) / M` ($\propto W$), preventing Fixed-Point Complex FastICA from converging to independent sources.
   - Line 71 reconstructed `V @ (sqrt(lam)[:, None] * W)` instead of `V @ (sqrt(lam)[:, None] * np.conj(W))`.
   - Because Complex ICA returned scrambled mixtures (0/90 on Condition A), the project abandoned ICA and built a 6-stage heuristic pipeline (`cpca/separate.py`: quartimax → Otsu → direction template → refine → merge → explain_away → K-order mixture split with mode-specific spectral starts & held-out reproduction bars → 200-surrogate Bonferroni check).

2. **Global column centring leaks arbitrary co-firing covariance into $C$**:
   - Under `X - X.mean(axis=0)` (`cpca/decompose.py`), silent zeros become $-\mu_k$, so any co-firing assembly without timing contributes
     $$\mathrm{Cov}(a_k, a_l)\,\mu_k^{\text{act}}\overline{\mu_l^{\text{act}}} \neq 0.$$
   - At $\alpha = \pi$, $|\mu^{\text{act}}|^2 \approx 0.54$, flooding the top spectrum with background assemblies (planted A–F) and driving 87–98% timing-shuffled false positives on hc-11 maze rest.

3. **Cell-space quartimax (`cpca/rotation.py`) actively destroys complex phase order**:
   - On the 2D subspace $\mathrm{span}(q, \bar{q})$ of a balanced route, maximizing $\sum_k |w_k|^4$ prefers real vectors ($\mathbb{E}[\cos^4\theta] = \frac{3}{8} > \frac{1}{4}$), collapsing complex sequences into real pairs $(c, s)$ with phases $\{0, \pi\}$.

4. **Half-circle arc ($\alpha = \pi$) inflates map overlap and background mean**:
   - Unrelated maps on the same cells have overlap $(2/\pi)^2 = 0.41$ (up to $0.57$ with 10% window margins), and subtracting $\mu^{\text{act}} \approx 0.74 j$ squashes middle-of-track amplitudes.

5. **Per-event circular phase skew**:
   - Burst-peak timing jitter $\delta_i$ shifts all active cells in event $i$ together; without per-event circular mean phase alignment prior to active-centring, this leaks a spurious `k = 1` component even in timing-shuffled real bursts.

## Unified Principled Solution (`exploration/cpca_v2.py`)

One clean framework (`fit_cpca_v2`), needing **zero held-out data** to fit or separate, with locked settings ($\alpha = 1.5\pi$, Riemannian contrast power $p = 1.5$, 100 shuffles at $q = 99\%$, original `member_frac = 0.2`):

1. **Wider phase arc ($\alpha = 1.5\pi$)**: cuts unrelated-map overlap from $0.41$ to $0.09$ while keeping a $90^\circ$ gap for gap-cut unwrapping.
2. **Event-aligned active-centring + active-phase shuffle null (`regime="reactivation"`)**:
   - Align each event's circular mean phase to 0.
   - Subtract each cell's mean active phasor *only on active entries* (silent cells remain 0) and scale to unit RMS (`global_rms`). Algebraically guarantees $\mathbb{E}[\tilde{C}_{kl}] = 0$ for arbitrary co-firing and arbitrary per-cell latency.
   - Active-phase shuffle null permutes each cell's nonzero entries only among its active events (exact co-firing + latency null at the eigenvalue step).
3. **Uncentred + circular time-shift null (`regime="waking"`)**:
   - Component 0 tested against `null[:, 0]` (rejects no-sequence controls); components $j \ge 1$ tested against the noise edge `null[:, 1]`.
4. **Riemannian Complex ICA in whitened event space**:
   - Maximizes $\sum_j \mathbb{E}[|(ZW)_{ij}|^{2p}]$ ($p = 1.5$) directly over $W \in U(K)$ with polar retraction — zero circularity assumption, immune to background-noise dominance of `log(0.1+u)` and single-burst outliers of $p=2$, exact mixing patterns $P = V \Lambda^{1/2} W^*$.
5. **Pairwise route-symmetry quotient (`regime="reactivation"` only)**:
   - Merges conjugate pairs $(q, \bar{q})$, Takagi-combines co-occurring real pairs $(c, s)$, and folds exclusive partial-replay harmonics back into their fundamental mode in $C$.

## Key Confirmation Results (Fresh Untouched Seeds & Real Data)

Reproduced by:
- `exploration/run_confirm_part1.py` → `exploration/results/confirm_part1.pkl`
- `exploration/run_confirm_part2.py` → `exploration/results/confirm_part2.pkl`
- `exploration/make_figures.py` → `exploration/figures/*.png`

### Planted Reactivations (fresh confirmation seeds 9000 & 15000)
- **A–F (seed 9000)**: A 10/10, B 10/10, C 7/10, D 8/10 orig (10/10 share), E 10/10, F 10/10 (`k=0`, zero surrogates). Coverage 0.95–0.97 (vs 0.82–0.90 for `separate.py`).
- **Stress G–M (seed 15000)**: **64/70 orig / 65/70 share (93% clean)** vs `separate.py`'s **45/70 (64%)**:
  - **I (3% + 3%, diff cells)**: **9/10** (vs `separate.py` **1/10**)
  - **K (5% + 5%, same cells, both ways)**: **9/10** (vs `separate.py` **4/10**)
  - **L (3 maps, same cells, 7% each)**: **9/10 share / 8/10 orig** (vs `separate.py` **3/10 orig / 5/10 corr**)
  - **G (3 routes, diff cells)**: **10/10**; **H (5%+5%)**: **9/10**; **J (25 cells)**: **9/10**; **M (1 map 5%)**: **10/10**.

### Waking Traversals (zero `separate.py`, zero held-out split)
- **Planted W1–W5 (fresh seed 21000)**: **50/50 (100% share) / 47/50 partial** (`W5` control 10/10 clean, ~200 ms/fit).
- **hc-11 linear laps**: **6/10 clean** (Achilles 10/25, Cicero 09/01, Gatsby 08/02 all 2/2 clean; beats `separate.py`'s 5/10 partial / 4/10 share); circular 5/6; control **16/16**.
- **hc-5 wheel**: aligned **11/12** (vs `separate.py` 9/12), unaligned **12/12**, shuffled control **12/12**.
- **DANDI 001695 laps**: **15/30 share / 13/30 partial** (47/54 outputs pass; beats `separate.py`'s 14/30 / 12/30).
- **DANDI 000552 figure-eight (65 sessions)**: $k \ge 2$ in 54/65 sessions; **unsupervised Left vs Right trial classification accuracy = 100.0% median (97.8% mean, $\ge 90\%$ in 50/54 sessions)**; cosine similarity to supervised Oracle patterns = **0.997 median**. (Even the 100% supervised Oracle fails `judge_figure8` 0/15 because arm cells fire during return/stem/delay segments of opposite trials.)

### Real Reactivations & Co-firing Immunity
- **hc-11 spike-in power (2 largest sessions)**: **0%: 0/10, 1%: 2/10, 2%: 3/10, 3%: 4/10, 5%: 10/10, 10%: 10/10, 20%: 10/10** — matches the cheating uncentred direction oracle at 5% (vs existing cPCA: 0%: 2/10, 3%: 0/10, 5%: 1/10, 10%: 9/10).
- **hc-11 spike-in power (6 smaller sessions)**: **10%: 16/30 (53%), 20%: 20/30 (67%)** vs existing cPCA **5/30 (17%), 10/30 (33%)**.
- **hc-11 maze rest (50 timing shuffles/session)**: timing-shuffled false positives dropped from **98% (Achilles 10/25) and 87% (Achilles 11/01)** to **1/400 (0.2%) pooled**; Achilles 10/25 (`p=0.001, p_tim=0.020`) and Gatsby 08/02 (`p=0.031, p_tim=0.020`) pass both order and timing.
- **hc-11 sleep POST vs PRE (50 timing shuffles)**: **Achilles 10/25 POST passes BOTH order and timing (`|rho|=0.50, p=0.017, p_tim=0.020`)**, PRE 0/8. (Existing cPCA failed timing on Achilles 10/25 POST at `p_tim=0.11`.)
- **DANDI 001695 CA1**: maze rest improves from 0/15 to **2/15** order + timing (0.0% shuffled FP); sleep POST 0/15 and PRE 0/15 (consistent with measured 3.4% POST vs 3.3% PRE clear replays — zero excess).
