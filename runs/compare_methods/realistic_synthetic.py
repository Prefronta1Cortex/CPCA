"""Synthetic POST sleep with the properties of the real recordings.

Compared with run_synthetic.py, each event here has: (1) assembly structure
in WHICH cells fire (membership is not random), (2) direction-specific place
maps and replays that cover only a segment of the track, (3) cells with no
place field that fire anyway. Each property can be switched off to see which
one removes the recoverable order. Results in results/realistic_synthetic.pkl.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

from cpca import encode  # noqa: E402
from cpca.cluster import fit_mixture, split_half_stability  # noqa: E402

T = 100.0


def make_session(rng, n_cells=120, n_place=60, n_events=3000, replay_frac=0.15, participation=0.15,
                 segment=0.4, jitter_ms=8.0, assemblies=True, direction_maps=True, nonplace=True):
    place = np.zeros(n_cells, dtype=bool)
    place[:n_place] = True
    if not nonplace:
        n_cells = n_place
        place = np.ones(n_cells, dtype=bool)
    centres = {1.0: np.full(n_cells, np.nan), -1.0: np.full(n_cells, np.nan)}
    centres[1.0][place] = rng.uniform(0, 1, size=place.sum())
    centres[-1.0][place] = rng.uniform(0, 1, size=place.sum()) if direction_maps else centres[1.0][place]
    n_assemblies = 6
    membership = rng.integers(0, n_assemblies, size=n_cells)
    times = np.full((n_events, n_cells, 3), np.nan)
    label = np.zeros(n_events, dtype=int)
    for i in range(n_events):
        if assemblies:
            a = rng.integers(n_assemblies)
            prob = np.where(membership == a, 0.45, 0.07)
        else:
            prob = np.full(n_cells, participation)
        fire = rng.random(n_cells) < prob
        if rng.random() < replay_frac:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            reverse = rng.random() < 0.5
            label[i] = 1
            start = rng.uniform(0, 1 - segment)
            cen = centres[sign]
            in_seg = place & (cen >= start) & (cen <= start + segment)
            frac = (cen - start) / segment
            if reverse:
                frac = 1 - frac
            t_seq = 10 + 80 * frac
            seq_fire = in_seg & (rng.random(n_cells) < 0.6)
            times[i, seq_fire, 0] = t_seq[seq_fire] + rng.normal(0, jitter_ms, size=int(seq_fire.sum()))
            other = fire & ~seq_fire
            times[i, other, 0] = rng.normal(50, 12, size=int(other.sum()))
        else:
            times[i, fire, 0] = rng.normal(50, jitter_ms, size=int(fire.sum()))
    times = np.clip(times, 0, T - 1e-6)
    return {"name": "realistic", "times": times, "label": label, "T": T, "centres": centres, "circular": False,
            "length": 1.0, "cell_ids": np.arange(n_cells)}


CONDITIONS = {
    "R0 all three properties":            dict(),
    "R1 no assembly structure":           dict(assemblies=False),
    "R2 one map for both directions":     dict(direction_maps=False),
    "R3 only place cells recorded":       dict(nonplace=False),
    "R4 none of the three":               dict(assemblies=False, direction_maps=False, nonplace=False),
    "R5 all three, 40% replay":           dict(replay_frac=0.4),
}


def run(name, kwargs, rng):
    d = make_session(rng, **kwargs)
    S = encode(d["times"], T, how="mean", t0=0.0)
    active = np.isfinite(d["times"]).any(axis=-1)
    out = {"condition": name, "fraction_replay": float(np.mean(d["label"] > 0))}
    single = H.method_cpca_single(S, active, T)
    out["cpca_C-C0"] = H.evaluate_orders(single["C-C0"], d, rng)
    for sim in ("cosine", "phase"):
        by_K = {}
        for K in (2, 3, 4, 6):
            fit = fit_mixture(S, K, rng, n_init=6, similarity=sim)
            stab = split_half_stability(S, K, rng, n_init=3, similarity=sim)
            by_K[K] = H.evaluate_orders([H.template_to_times(fit["templates_delta"][:, k], T) for k in range(K)], d, rng) | {"stability": stab}
        Kbest = max(by_K, key=lambda K: by_K[K]["stability"])
        out[f"mixture_{sim}"] = by_K[Kbest] | {"K": Kbest, "by_K": {K: (v["best"], v["null95"], v["stability"]) for K, v in by_K.items()}}
    sq = H.method_seqnmf(H.synthetic_blocks(d["times"], T, 0.010, pad=12), 0.010, rng)
    out["seqnmf"] = H.evaluate_orders(sq["orders"], d, rng)
    hm = H.method_hmm(H.synthetic_blocks(d["times"], T, 0.020), rng, n_states_options=(20,), n_iter=40)
    out["hmm"] = H.evaluate_orders(hm["orders"], d, rng)
    return out


def main():
    rng = np.random.default_rng(0)
    results = []
    for name, kwargs in CONDITIONS.items():
        t0 = time.perf_counter()
        res = run(name, kwargs, rng)
        results.append(res)
        print(f"{name}: replay {100*res['fraction_replay']:.0f}% | single C-C0 {res['cpca_C-C0']['best']:.2f} [{res['cpca_C-C0']['null95']:.2f}] | "
              f"mixture cosine K={res['mixture_cosine']['K']} {res['mixture_cosine']['best']:.2f} [{res['mixture_cosine']['null95']:.2f}] | "
              f"mixture phase K={res['mixture_phase']['K']} {res['mixture_phase']['best']:.2f} [{res['mixture_phase']['null95']:.2f}] | "
              f"seqNMF {res['seqnmf']['best']:.2f} [{res['seqnmf']['null95']:.2f}] | HMM {res['hmm']['best']:.2f} [{res['hmm']['null95']:.2f}] | {time.perf_counter()-t0:.0f}s", flush=True)
    with open(H.RESULTS / "realistic_synthetic.pkl", "wb") as fh:
        pickle.dump(results, fh)


if __name__ == "__main__":
    main()
