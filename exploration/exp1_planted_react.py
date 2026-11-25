"""Experiment 1: Planted reactivations (sim/two_routes.py A-F, stress G-M, spatial assemblies control).

Tests clean cPCA + active-centring + active-phase shuffle null + Riemannian Complex ICA
+ conjugate-pair quotienting, WITHOUT any 6-stage separate.py machinery.
"""

import sys
import time
import zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exploration.core import fit_clean
from sim.two_routes import generate
from sim.segments import spatial_assemblies
from scripts.run_separation import CONDITIONS, judge
from scripts.run_pure_capabilities import judge_share


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


STRESS = {
    "G 3 routes diff 7%": dict(rates=(0.07, 0.07, 0.07), shared=False),
    "H diff 5% + 5%": dict(rates=(0.05, 0.05), shared=False),
    "I diff 3% + 3%": dict(rates=(0.03, 0.03), shared=False),
    "J diff 25 cells 10%+10%": dict(rates=(0.10, 0.10), shared=False, n_route=25),
    "K same 5% + 5% both": dict(rates=(0.05, 0.05), shared=True),
    "L 3 maps same 7% each": dict(rates=(0.07, 0.07, 0.07), shared=True),
    "M 1 map 5%": dict(rates=(0.05,), shared=False),
}


def eval_config(name, cfg, conds, seed_base=7000, n_draws=5, folds=(0, 1)):
    print(f"\n=== {name} ===")
    print(f"{'Condition':<38} | {'k_pca':>5} | {'n_out':>5} | {'pass(orig)':>13} | {'clean(orig)':>11} | {'clean(share)':>12} | {'cov':>5}")
    print("-" * 105)
    for ci, (cname, kw) in enumerate(conds.items()):
        ks, nouts, covs = [], [], []
        n_pass_o, n_tot, clean_o, clean_s = 0, 0, 0, 0
        n_fits = 0
        for draw in range(n_draws):
            d = generate(np.random.default_rng(seed_base + 97 * ci + draw), **kw)
            perm = np.random.default_rng(seed("exp1-split", seed_base, cname, draw)).permutation(d["times"].shape[0])
            halves = (perm[: perm.size // 2], perm[perm.size // 2:])
            maps = d["maps"] if sum(kw["rates"]) > 0 else []
            for fold in folds:
                tr, te = halves if fold == 0 else halves[::-1]
                rng = np.random.default_rng(seed("exp1-fit", cname, draw, fold))
                res = fit_clean(d["times"][tr], d["T"][tr], rng, **cfg)
                ks.append(res["n_kept"])
                orders = res["orders"]
                nouts.append(len(orders))
                n_fits += 1
                jrng = np.random.default_rng(seed("exp1-judge", cname, draw, fold))
                if not maps:
                    n_tot += len(orders)
                    if len(orders) == 0:
                        clean_o += 1
                        clean_s += 1
                    continue
                v_orig, v_share = [], []
                for o in orders:
                    ok_o, k_o, det = judge(o, maps, kw["shared"], jrng)
                    v_orig.append((ok_o, k_o))
                    if kw["shared"]:
                        ok_s, k_s = judge_share(o, maps, jrng)
                    else:
                        ok_s, k_s = ok_o, k_o
                    v_share.append((ok_s, k_s))
                    if ok_o:
                        covs.append(float(np.sum(np.isfinite(o) & np.isfinite(maps[k_o])) / np.sum(np.isfinite(maps[k_o]))))
                n_pass_o += sum(ok for ok, _ in v_orig)
                n_tot += len(v_orig)
                if len({k for ok, k in v_orig if ok}) == len(maps) and all(ok for ok, _ in v_orig):
                    clean_o += 1
                if len({k for ok, k in v_share if ok}) == len(maps) and all(ok for ok, _ in v_share):
                    clean_s += 1
        pct = f"{n_pass_o}/{n_tot} ({100*n_pass_o/max(n_tot,1):3.0f}%)" if n_tot else "0/0 ( - )"
        cov_s = f"{np.median(covs):.2f}" if covs else "  -  "
        print(f"{cname[:38]:<38} | {np.median(ks):5.1f} | {np.median(nouts):5.1f} | {pct:>13} | {clean_o:2d}/{n_fits:<8d} | {clean_s:2d}/{n_fits:<9d} | {cov_s:>5}")

    # Spatial assemblies control (map-ordered co-firing, NO replays!)
    ks_sp, nouts_sp = [], []
    for draw in range(5):
        rng_d = np.random.default_rng(7700 + draw)
        n_cells, n_route = 120, 40
        route = rng_d.permutation(n_cells)[:n_route]
        pos = np.full(n_cells, np.nan)
        pos[route] = rng_d.uniform(0, 1, n_route)
        mem = spatial_assemblies(rng_d, n_cells, route, pos, n_groups=6)
        d = generate(rng_d, rates=(0.0,), shared=False, assembly_membership=mem)
        perm = rng_d.permutation(d["times"].shape[0])
        tr = perm[: perm.size // 2]
        res = fit_clean(d["times"][tr], d["T"][tr], np.random.default_rng(7800 + draw), **cfg)
        ks_sp.append(res["n_kept"])
        nouts_sp.append(len(res["orders"]))
    print(f"{'Control: spatial assemblies (no rep)':<38} | {np.median(ks_sp):5.1f} | {np.median(nouts_sp):5.1f} | {'0/0' if sum(nouts_sp)==0 else f'0/{sum(nouts_sp)}':>13} | {sum(n==0 for n in nouts_sp):2d}/5        | {sum(n==0 for n in nouts_sp):2d}/5         |   -  ")


if __name__ == "__main__":
    t0 = time.perf_counter()
    # Compare configurations systematically on dev seeds (7000)
    configs = [
        ("1. Active-centred, alpha=pi, event-T, active-rms, ICA+merge",
         dict(alpha=np.pi, mode="active_centred", scale="active_rms", null_kind="active",
              ica=True, contrast="log", a=0.1, merge_conj=True, merge_thresh=0.5, member_rule="otsu")),
        ("2. Active-centred, alpha=1.6*pi, event-T, active-rms, ICA+merge",
         dict(alpha=1.6*np.pi, mode="active_centred", scale="active_rms", null_kind="active",
              ica=True, contrast="log", a=0.1, merge_conj=True, merge_thresh=0.5, member_rule="otsu")),
        ("3. Active-centred, alpha=1.6*pi, fixed-T=250, active-rms, ICA+merge",
         dict(alpha=1.6*np.pi, mode="active_centred", scale="active_rms", null_kind="active",
              fixed_T=250.0, ica=True, contrast="log", a=0.1, merge_conj=True, merge_thresh=0.5, member_rule="otsu")),
        ("4. Active-centred, alpha=1.6*pi, fixed-T=250, unscaled, ICA+merge",
         dict(alpha=1.6*np.pi, mode="active_centred", scale="none", null_kind="active",
              fixed_T=250.0, ica=True, contrast="log", a=0.1, merge_conj=True, merge_thresh=0.5, member_rule="otsu")),
        ("5. Active-centred, alpha=1.6*pi, fixed-T=250, global-rms, ICA+merge",
         dict(alpha=1.6*np.pi, mode="active_centred", scale="global_rms", null_kind="active",
              fixed_T=250.0, ica=True, contrast="log", a=0.1, merge_conj=True, merge_thresh=0.5, member_rule="otsu")),
    ]
    for name, cfg in configs:
        eval_config(name, cfg, CONDITIONS, seed_base=7000, n_draws=5, folds=(0, 1))

    # Also test the best one on STRESS conditions!
    print("\n\n>>> STRESS TEST (dev seed 13000) for Configs 2, 3, 4:")
    for name, cfg in configs[1:4]:
        eval_config(name + " [STRESS]", cfg, STRESS, seed_base=13000, n_draws=5, folds=(0, 1))
    print(f"\nDone in {time.perf_counter() - t0:.1f}s")
