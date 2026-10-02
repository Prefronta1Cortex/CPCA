"""Run every label-free method on one hc-11 session (POST NREM, wide events).

Usage: python run_real.py Achilles_10252013
Writes results/real_<session>.pkl. The place map is used only for scoring.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

from cpca import encode  # noqa: E402


def main(name):
    rng = np.random.default_rng(0)
    H.RESULTS.mkdir(exist_ok=True)
    path = next(p for p in sorted(H.MAT_DIR.glob("*_sessInfo.mat")) if p.name.startswith(name))
    t_all = time.perf_counter()
    d = H.session_data(path)
    out = {"name": d["name"], "maze": d["maze"], "circular": d["circular"], "n_cells": len(d["cell_ids"]),
           "n_place": {k: int(np.isfinite(v).sum()) for k, v in d["centres"].items()},
           "n_events": len(d["peaks"]), "event_dur_ms": float(np.median(np.diff(d["bursts"], axis=1)) * 1000)}
    print(f"[{d['name']}] {d['maze']} | cells {out['n_cells']} | place cells {out['n_place']} | events {out['n_events']} (median {out['event_dur_ms']:.0f} ms)", flush=True)

    times = H.peak_window_times(d)
    active = np.isfinite(times).any(axis=-1)
    S = encode(times, H.WINDOW_MS, how="mean", t0=0.0)
    single = H.method_cpca_single(S, active, H.WINDOW_MS)
    for key in ("C", "C-C0", "selected"):
        out[f"cpca_{key}"] = H.evaluate_orders(single[key], d, rng)
    print(f"  cPCA single: C {out['cpca_C']['best']:.2f} [{out['cpca_C']['null95']:.2f}] | C-C0 {out['cpca_C-C0']['best']:.2f} [{out['cpca_C-C0']['null95']:.2f}] | selected {out['cpca_selected']['best']:.2f}", flush=True)

    mix = H.method_cpca_mixture(S, H.WINDOW_MS, rng)
    out["mixture_by_K"] = {K: H.evaluate_orders(v["orders_delta"], d, rng) | {"stability": v["stability"], "sizes": v["sizes"], "objective": v["objective"]} for K, v in mix["by_K"].items()}
    out["mixture_K"] = mix.get("K_stability")
    out["cpca_mixture"] = out["mixture_by_K"][out["mixture_K"]]
    print("  cPCA mixture:", {K: f"{v['best']:.2f} [{v['null95']:.2f}] stab {v['stability']:.2f}" for K, v in out["mixture_by_K"].items()}, f"-> K={out['mixture_K']} by stability | {mix['runtime_s']:.0f}s", flush=True)

    blocks10 = H.event_bins(d, 0.010, pad=12)
    sq = H.method_seqnmf(blocks10, 0.010, rng)
    out["seqnmf"] = H.evaluate_orders(sq["orders"], d, rng) | {"spans_ms": sq["spans_ms"], "n_cells": sq["n_cells"], "runtime_s": sq["runtime_s"]}
    print(f"  seqNMF: {out['seqnmf']['best']:.2f} [{out['seqnmf']['null95']:.2f}] factors cells {sq['n_cells']} spans {np.round(sq['spans_ms'])} | {sq['runtime_s']:.0f}s", flush=True)

    blocks20 = H.event_bins(d, 0.020)
    hm = H.method_hmm(blocks20, rng)
    out["hmm"] = H.evaluate_orders(hm["orders"], d, rng) | {"n_states": hm["n_states"], "chain_ratio": hm["chain_ratio"], "heldout_ll": hm["heldout_ll_per_bin"], "runtime_s": hm["runtime_s"]}
    print(f"  HMM: {out['hmm']['best']:.2f} [{out['hmm']['null95']:.2f}] states {hm['n_states']} chain ratio {hm['chain_ratio']:.2f} | {hm['runtime_s']:.0f}s", flush=True)

    ica = H.method_count_ica(blocks20, d, rng)
    out["count_ica"] = {k: v for k, v in ica.items() if k != "weights"}
    print(f"  count PCA/ICA: {ica['n_assemblies']} assemblies; member field spread {ica.get('member_field_spread', np.nan):.2f} vs random {ica.get('member_field_spread_null', np.nan):.2f}", flush=True)

    out["runtimes"] = {"cpca_single": single["runtime_s"], "cpca_mixture": mix["runtime_s"], "seqnmf": sq["runtime_s"], "hmm": hm["runtime_s"], "count_ica": ica["runtime_s"]}
    out["total_s"] = time.perf_counter() - t_all
    with open(H.RESULTS / f"real_{d['name']}.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print(f"[{d['name']}] done in {out['total_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
