"""Components kept on hc-11 PRE events (matched 1500, no spike-in) by column vs active centring (cpca.fit, arc pi)."""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exploration.verify_adoption import HC11, _pre_events, seed  # noqa: E402


def job(name):
    from cpca.pipeline import fit
    from scripts.plot_hc11_sequences import best_map_match
    base, T, centres, run = _pre_events(name)
    out = [name]
    for center in (True, "active"):
        r = fit(base, T, rng=np.random.default_rng(seed("kept", name)), center=center, scale=True, ica=False)
        orders = [rd["times_ms"] for rd in r["readouts"]]
        p = best_map_match(orders, centres, run["circular"], run["length"], np.random.default_rng(seed("keptsc", name)), n_null=500)["p"] if orders else np.nan
        out += [r["n_kept"], p]
    return out


if __name__ == "__main__":
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
    with ProcessPoolExecutor(8) as pool:
        for name, kc, pc, ka, pa in pool.map(job, HC11):
            print(f"{name[:18]:18} column centred: kept {kc} best order p {pc:.3f} | active: kept {ka} best order p {pa:.3f}")
