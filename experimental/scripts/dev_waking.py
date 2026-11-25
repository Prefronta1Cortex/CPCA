"""Waking un-mixing on planted laps (docs/unmixing.md, waking design): development and confirmation harness.

Usage: python scripts/dev_waking.py <variant> [--phase=dev|confirm] [--workers=9]
Prints per-output tables; results/unmixing/waking_<phase>_<variant>.pkl
"""

import os
import pickle
import sys
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "unmixing"
SEED_BASE = {"dev": 20000, "confirm": 21000}
CONDITIONS = {
    "W1 70% cells both ways, positions rho 0.2, 30 laps each": dict(both=0.7, rho=0.2),
    "W2 70% both ways, rho 0.7": dict(both=0.7, rho=0.7),
    "W3 70% both ways, rho 0.2, 10 laps each": dict(both=0.7, rho=0.2, laps_per_direction=(10, 10)),
    "W4 one direction, 40 laps": dict(both=0.0, laps_per_direction=(40, 0)),
    "W5 control: spikes circularly shifted": dict(both=0.7, rho=0.2, shuffle=True),
}
VARIANTS = {
    "forward": dict(center=False, explain=True, conjugate=False, min_events=4),
    "forward_events": dict(center=False, explain=True, conjugate=False, min_events=4, event_starts=True),
    "forward_events_trim": dict(center=False, explain=True, conjugate=False, min_events=4, event_starts=True, trim=True),
    "waking": dict(center=False, explain=True, conjugate=False, min_events=4, event_starts=True, trim=True, adaptive=True),
    "forward_noexplain": dict(center=False, explain=False, conjugate=False, min_events=4),
    "forward_centred": dict(center=True, explain=True, conjugate=False, min_events=4),
    "with_reversal": dict(center=False, explain=True, conjugate=True, min_events=4),
}
N_SUR = 200


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def circular_shift(times, T, rng):
    shift = rng.uniform(0, 1, times.shape[:2])[..., None] * T[:, None, None]
    out = np.mod(times + shift, T[:, None, None])
    return np.sort(out, axis=-1)


def waking_p(q, te_times, te_T, rng, n_sur=N_SUR):
    """Held-out forward expression of q against circularly shifted surrogates of the held-out events."""
    from cpca.encode import encode
    q = q / max(np.linalg.norm(q), 1e-300)
    e = float(np.mean(np.abs(encode(te_times, te_T) @ np.conj(q)) ** 2))
    null = np.array([np.mean(np.abs(encode(circular_shift(te_times, te_T, rng), te_T) @ np.conj(q)) ** 2) for _ in range(n_sur)])
    return float((1 + np.sum(null >= e)) / (1 + n_sur))


def waking_outputs(tr_times, tr_T, te_times, te_T, rng, center=False, explain=True, conjugate=False, min_events=4,
                   split="k", reproduce=0.97, event_starts=False, trim=False, adaptive=False):
    """[(order, p_waking (Bonferroni over outputs), members)] for one training/held-out split."""
    from cpca import fit
    from cpca.decode import readout
    from cpca.encode import encode
    from experimental.separate import separate
    r = fit(tr_times, tr_T, rng=rng, ica=False, center=center, scale=center, n_shuffles=50)
    if not r["n_kept"]:
        return []
    temps = separate(r, encode(tr_times, tr_T), rng, explain=explain, split=split, validate=encode(te_times, te_T),
                     reproduce=reproduce, min_events=min_events, conjugate=conjugate, event_starts=event_starts, trim=trim, adaptive=adaptive)
    out = []
    for t in temps:
        q = t["template"]
        p = min(1.0, waking_p(q, te_times, te_T, rng) * len(temps))
        _, times, _, inc, _ = readout(q, r["T_read_ms"], min_weight=1e-9)
        out.append((np.where(inc & t["members"], times, np.nan), p, t["members"]))
    return out


def judge_both(order, maps, rng):
    from experimental.scripts.dev_unmixing import judge_partial, judge_share
    from scripts.run_separation import judge
    ok, k, _ = judge(order, maps, True, rng)
    ok2, k2 = judge_partial(order, maps, rng)
    ok3, k3 = judge_share(order, maps, rng)
    return ok, k, ok2, k2, ok3, k3


def job(variant, cond, draw, phase):
    from sim.laps import generate
    kw = CONDITIONS[cond]
    d = generate(np.random.default_rng(SEED_BASE[phase] + 97 * list(CONDITIONS).index(cond) + draw), **kw)
    maps = [m for m in d["maps"] if np.isfinite(m).sum() >= 6 and (kw.get("laps_per_direction", (30, 30))[0 if m is d["maps"][0] else 1] > 0)]
    if kw.get("shuffle"):
        maps = []
    rng = np.random.default_rng(seed("half", phase, cond, draw))
    dirs = d["direction"]
    half = np.zeros(dirs.size, bool)
    for s in np.unique(dirs):
        idx = rng.permutation(np.flatnonzero(dirs == s))
        half[idx[: idx.size // 2]] = True
    rows = []
    for fold, train in enumerate((half, ~half)):
        tr, te = np.flatnonzero(train), np.flatnonzero(~train)
        frng = np.random.default_rng(seed("fit", phase, cond, draw, fold, variant))
        outs = waking_outputs(d["times"][tr], d["T"][tr], d["times"][te], d["T"][te], frng, **VARIANTS[variant])
        jrng = np.random.default_rng(seed("judge", phase, cond, draw, fold))
        judged = []
        for order, p, mask in outs:
            if maps:
                ok, k, ok2, k2, ok3, k3 = judge_both(order, maps, jrng)
            else:
                ok = ok2 = ok3 = False
                k = k2 = k3 = -1
            cov = lambda kk: float((mask & np.isfinite(maps[kk])).sum() / np.isfinite(maps[kk]).sum()) if kk >= 0 else np.nan
            judged.append({"p": p, "passes_check": p < 0.05, "passes": ok, "map": k, "coverage": cov(k) if ok else np.nan,
                           "passes_partial": ok2, "map_partial": k2, "coverage_partial": cov(k2) if ok2 else np.nan,
                           "passes_share": ok3, "map_share": k3, "coverage_share": cov(k3) if ok3 else np.nan,
                           "n_members": int(mask.sum())})
        rows.append({"condition": cond, "draw": draw, "fold": fold, "outputs": judged, "n_maps": len(maps)})
    return rows


def table(rows, criterion=""):
    P, K, C = "passes" + criterion, "map" + criterion, "coverage" + criterion
    lines = ["| condition | outputs per fit | outputs passing | clean fits | coverage | fits with outputs |", "|---|---|---|---|---|---|"]
    for cond in CONDITIONS:
        rs = [r for r in rows if r["condition"] == cond]
        if not rs:
            continue
        outs = [[o for o in r["outputs"] if o["passes_check"]] for r in rs]
        n = [len(o) for o in outs]
        p = sum(o[P] for oo in outs for o in oo)
        if rs[0]["n_maps"] == 0:
            clean = sum(x == 0 for x in n)
        else:
            clean = sum(len({o[K] for o in oo if o[P]}) == rs[0]["n_maps"] and all(o[P] for o in oo) for oo in outs)
        covs = [o[C] for oo in outs for o in oo if o[P]]
        cov = f"{np.median(covs):.2f}" if covs else "–"
        lines.append(f"| {cond[:44]} | {np.median(n):.0f} | {p}/{sum(n)} ({p / max(sum(n), 1):.0%}) | {clean}/{len(rs)} | {cov} | {sum(x > 0 for x in n)}/{len(rs)} |")
    return "\n".join(lines)


def main(argv):
    variant = argv[0]
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    phase = next((a.split("=", 1)[1] for a in argv if a.startswith("--phase=")), "dev")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(job, variant, c, d, phase): (c, d) for c in CONDITIONS for d in range(5)}
        for f in as_completed(futs):
            try:
                rows += f.result()
            except Exception:
                print(f"FAILED {futs[f]}:\n{traceback.format_exc()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(rows, open(OUT / f"waking_{phase}_{variant}.pkl", "wb"))
    print(f"## {variant} ({phase}), variance-share criterion\n" + table(rows, "_share"))
    print(f"\n## {variant} ({phase}), partial-correlation criterion\n" + table(rows, "_partial"))
    print(f"\n## {variant} ({phase}), original criterion\n" + table(rows))


if __name__ == "__main__":
    main(sys.argv[1:])
