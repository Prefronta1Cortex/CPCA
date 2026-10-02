"""Validate every method on synthetic events with a planted order.

If a method recovers the planted order here, it is being used correctly; its
behaviour on real data then reflects the data. Results go to results/synthetic.pkl.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

from cpca import encode  # noqa: E402

CONDITIONS = {
    "S1 one route, all events replay":            dict(replay_frac=1.0, reverse_frac=0.0, participation=0.30, jitter_ms=5.0),
    "S2 two directions 50/50, all events replay": dict(replay_frac=1.0, reverse_frac=0.5, participation=0.30, jitter_ms=5.0),
    "S3 two directions, 40% replay, 20% cells":   dict(replay_frac=0.4, reverse_frac=0.5, participation=0.20, jitter_ms=6.0),
    "S4 two directions, 15% replay, 15% cells":   dict(replay_frac=0.15, reverse_frac=0.5, participation=0.15, jitter_ms=8.0, background_rate=0.03),
}


def run_condition(name, kwargs, rng):
    d = H.synthetic_session(rng, **kwargs)
    T = d["T"]
    S = encode(d["times"], T, how="mean", t0=0.0)
    active = np.isfinite(d["times"]).any(axis=-1)
    out = {"condition": name, "kwargs": kwargs, "n_events": d["times"].shape[0],
           "fraction_replay": float(np.mean(d["label"] > 0))}
    single = H.method_cpca_single(S, active, T)
    for key in ("C", "C-C0", "selected"):
        out[f"cpca_{key}"] = H.evaluate_orders(single[key], d, rng)
    mix = H.method_cpca_mixture(S, T, rng)
    out["mixture_by_K"] = {K: H.evaluate_orders(v["orders_delta"], d, rng) | {"stability": v["stability"], "sizes": v["sizes"]} for K, v in mix["by_K"].items()}
    out["mixture_K"] = mix.get("K_stability")
    out["cpca_mixture"] = out["mixture_by_K"][out["mixture_K"]] if out["mixture_K"] else {"best": np.nan, "null95": np.nan}
    blocks10 = H.synthetic_blocks(d["times"], T, 0.010, pad=12)
    sq = H.method_seqnmf(blocks10, 0.010, rng)
    out["seqnmf"] = H.evaluate_orders(sq["orders"], d, rng) | {"spans_ms": sq["spans_ms"], "n_cells": sq["n_cells"], "runtime_s": sq["runtime_s"]}
    blocks20 = H.synthetic_blocks(d["times"], T, 0.020)
    hm = H.method_hmm(blocks20, rng)
    out["hmm"] = H.evaluate_orders(hm["orders"], d, rng) | {"n_states": hm["n_states"], "chain_ratio": hm["chain_ratio"], "runtime_s": hm["runtime_s"]}
    ica = H.method_count_ica(blocks20, d, rng)
    out["count_ica"] = {k: v for k, v in ica.items() if k != "weights"}
    out["runtimes"] = {"cpca_single": single["runtime_s"], "cpca_mixture": mix["runtime_s"], "seqnmf": sq["runtime_s"], "hmm": hm["runtime_s"], "count_ica": ica["runtime_s"]}
    return out


def main():
    rng = np.random.default_rng(0)
    H.RESULTS.mkdir(exist_ok=True)
    results = []
    for name, kwargs in CONDITIONS.items():
        t0 = time.perf_counter()
        res = run_condition(name, kwargs, rng)
        results.append(res)
        print(f"{name}: replay {100*res['fraction_replay']:.0f}% | single C-C0 {res['cpca_C-C0']['best']:.2f} [{res['cpca_C-C0']['null95']:.2f}] | "
              f"mixture K={res['mixture_K']} {res['cpca_mixture']['best']:.2f} [{res['cpca_mixture']['null95']:.2f}] | "
              f"seqNMF {res['seqnmf']['best']:.2f} [{res['seqnmf']['null95']:.2f}] | HMM(M={res['hmm']['n_states']}) {res['hmm']['best']:.2f} [{res['hmm']['null95']:.2f}] | "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    with open(H.RESULTS / "synthetic.pkl", "wb") as fh:
        pickle.dump(results, fh)


if __name__ == "__main__":
    main()
