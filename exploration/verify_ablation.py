"""Which part of pillar 2 / 4 gives the spike-in gain and the timing-shuffle immunity (arc pi)?

Variants (patched into exploration.cpca_v2 inside each worker):
  full       as written
  no_ica     principal components only (ica=False)
  no_align   active-centring without per-event phase alignment
  no_centre  aligned, global RMS scaling only (no subtraction), same active-phase shuffle null
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exploration.verify_adoption import HC11, maze_rest, spikein  # noqa: E402


def patch(variant):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
    import exploration.cpca_v2 as m
    m.fit_cpca_v2 = partial(m.fit_cpca_v2, alpha=np.pi)
    if variant == "no_ica":
        m.fit_cpca_v2 = partial(m.fit_cpca_v2, ica=False)
    elif variant == "no_align":
        m.align_event_phase = lambda S: S
    elif variant == "no_centre":
        def scale_only(S, scale="global_rms"):
            a = np.abs(S) > 0
            return S / np.sqrt(np.mean(np.abs(S[a]) ** 2))
        m.active_centre_scale = scale_only


if __name__ == "__main__":
    for variant in sys.argv[1:]:
        with ProcessPoolExecutor(9, initializer=patch, initargs=(variant,)) as pool:
            spikes = list(pool.map(spikein, [(n, f, d) for n in HC11[:2] for f in (0.0, 0.03, 0.05) for d in range(5)]))
            maze = list(pool.map(maze_rest, [(n,) for n in HC11]))
        sp = "  ".join(f"{f:.0%}: {sum(s[3] for s in spikes if s[2] == f)}/10" for f in (0.0, 0.03, 0.05))
        fp = sum(m[4] for m in maze)
        real = sum(np.isfinite(m[2]) and m[2] < 0.05 for m in maze)
        print(f"{variant:10} spike-in {sp} | maze rest: shuffled fits passing order {fp}/160, real sessions passing {real}/8", flush=True)
