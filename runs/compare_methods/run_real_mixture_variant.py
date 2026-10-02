"""Add the membership-blind mixture (similarity='phase_both') to each real-session result."""

import glob
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

from cpca import encode  # noqa: E402
from cpca.cluster import fit_mixture, split_half_stability  # noqa: E402


def main():
    for f in sorted(glob.glob(str(H.RESULTS / "real_*.pkl"))):
        out = pickle.load(open(f, "rb"))
        rng = np.random.default_rng(1)
        path = next(p for p in sorted(H.MAT_DIR.glob("*_sessInfo.mat")) if p.name.startswith(out["name"]))
        d = H.session_data(path)
        times = H.peak_window_times(d)
        S = encode(times, H.WINDOW_MS, how="mean", t0=0.0)
        by_K = {}
        for K in (2, 3, 4, 6):
            fit = fit_mixture(S, K, rng, n_init=8, similarity="phase_both")
            stab = split_half_stability(S, K, rng, n_init=4, similarity="phase_both")
            by_K[K] = H.evaluate_orders([H.template_to_times(fit["templates_delta"][:, k], H.WINDOW_MS) for k in range(K)], d, rng) | {"stability": stab, "sizes": fit["sizes"].tolist()}
        Kbest = max(by_K, key=lambda K: by_K[K]["stability"])
        out["mixture_phase_by_K"] = by_K
        out["mixture_phase_K"] = Kbest
        out["cpca_mixture_phase"] = by_K[Kbest]
        pickle.dump(out, open(f, "wb"))
        print(out["name"], {K: f"{v['best']:.2f} [{v['null95']:.2f}] stab {v['stability']:.2f}" for K, v in by_K.items()}, "->", Kbest, flush=True)


if __name__ == "__main__":
    main()
