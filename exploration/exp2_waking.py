"""Experiment 2: Waking data (planted laps W1-W5, hc-11 laps, hc-5 wheel, DANDI 001695 laps, DANDI 000552 figure-eight).

Tests PURE cPCA + Riemannian Complex ICA (NO 6-stage separate.py, NO event_groups_start,
NO held-out splitting, NO 0.97 threshold!).
"""

import sys
import time
import zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exploration.core import fit_clean
from sim.laps import generate as gen_laps
from scripts.run_separation import judge
from scripts.run_pure_capabilities import judge_share, halves_by, LAPS, HC11
from scripts.plot_hc11_sequences import best_map_match


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def test_planted_laps(name, cfg, n_draws=5, seed_base=20000):
    print(f"\n=== Planted Laps: {name} ===")
    for ci, (cname, kw) in enumerate(LAPS.items()):
        ks, clean_o, clean_s, pass_s, tot = [], 0, 0, 0, 0
        for draw in range(n_draws):
            d = gen_laps(np.random.default_rng(seed_base + 97 * ci + draw), **kw)
            half = halves_by(d["direction"], np.random.default_rng(seed("exp2-laps", cname, draw)))
            n_dir = kw.get("laps_per_direction", (30, 30))
            maps = [] if kw.get("shuffle") else [m for m, n in zip(d["maps"], n_dir) if n > 0]
            for fold, train in enumerate((half, ~half)):
                tr = np.flatnonzero(train)
                rng = np.random.default_rng(seed("exp2-laps-fit", cname, draw, fold))
                res = fit_clean(d["times"][tr], d["T"][tr], rng, **cfg)
                ks.append(res["n_kept"])
                orders = res["orders"]
                jrng = np.random.default_rng(seed("exp2-laps-judge", cname, draw, fold))
                if not maps:
                    tot += len(orders)
                    if len(orders) == 0:
                        clean_o += 1
                        clean_s += 1
                    continue
                vo, vs = [], []
                for o in orders:
                    oko, ko, _ = judge(o, maps, True, jrng)
                    oks, ks_i = judge_share(o, maps, jrng)
                    vo.append((oko, ko))
                    vs.append((oks, ks_i))
                pass_s += sum(ok for ok, _ in vs)
                tot += len(vs)
                if len({k for ok, k in vo if ok}) == len(maps) and all(ok for ok, _ in vo):
                    clean_o += 1
                if len({k for ok, k in vs if ok}) == len(maps) and all(ok for ok, _ in vs):
                    clean_s += 1
        print(f"  {cname:<28} | med k={np.median(ks):.1f} | pass(share)={pass_s:2d}/{tot:<3d} | clean(orig)={clean_o:2d}/10 | clean(share)={clean_s:2d}/10")


def test_real_waking(name, cfg, run_fig8=True):
    from cpca import event_spike_times
    from cpca.hc11 import laps, load_session as load_hc11, place_centres, pyramidal_spikes, running, session_paths as hc11_paths
    import cpca.hc5 as hc5
    from scripts.run_hc5 import training_data
    from scripts.run_laps_hc11 import heldout_run
    import cpca.dandi001695 as dd
    import cpca.huszar as hz
    from experimental.scripts.run_initial_real import figure8_events, judge_figure8

    print(f"\n=== Real Waking: {name} ===")

    # 1. hc-11 laps (linear & circular) + circularly shifted control
    for kind in ("linear", "circular", "control_linear"):
        clean_s, pass_s, tot, n_fits = 0, 0, 0, 0
        details = []
        for sname in HC11:
            s = load_hc11(next(p for p in hc11_paths() if p.name.startswith(sname)))
            run = running(s)
            if kind == "linear" and run["circular"]:
                continue
            if kind == "circular" and not run["circular"]:
                continue
            if kind == "control_linear" and run["circular"]:
                continue
            L = laps(run)
            starts, ends, dirs = (np.array([x[i] for x in L]) for i in range(3))
            half = halves_by(dirs, np.random.default_rng(seed("waking-sep", sname)))
            by_cell, ids, _ = pyramidal_spikes(s, np.column_stack([starts, ends]))
            for fold in (0, 1):
                train = half if fold == 0 else ~half
                tr, te = np.flatnonzero(train), np.flatnonzero(~train)
                X, _ = event_spike_times(by_cell, ids, starts[tr], ends[tr], max_spikes=400)
                TX = (ends[tr] - starts[tr]) * 1000
                if kind == "control_linear":
                    crng = np.random.default_rng(seed("ctrl", sname, fold))
                    shift = crng.uniform(0, 1, X.shape[:2])[..., None] * TX[:, None, None]
                    X = np.sort(np.mod(X + shift, TX[:, None, None]), axis=-1)
                c = place_centres(s, ids, heldout_run(run, np.column_stack([starts[te], ends[te]])), min_spikes=20)
                maps = [] if kind == "control_linear" else [c[sg] for sg in (1.0, -1.0) if np.isfinite(c[sg]).sum() >= 6 and np.sum(dirs[te] == sg) >= 3]
                res = fit_clean(X, TX, np.random.default_rng(seed("exp2-hc11", sname, fold)), **cfg)
                orders = res["orders"]
                jrng = np.random.default_rng(seed("exp2-hc11-j", sname, fold))
                n_fits += 1
                if not maps:
                    tot += len(orders)
                    if len(orders) == 0:
                        clean_s += 1
                    continue
                if run["circular"]:
                    vs = [(bool(np.sum(np.isfinite(o) & np.isfinite(maps[0])) >= 6 and
                                best_map_match([o], {1.0: maps[0]}, True, run["length"], jrng, n_null=500)["p"] < 0.05), 0)
                          for o in orders]
                else:
                    vs = [judge_share(o, maps, jrng) for o in orders]
                pass_s += sum(ok for ok, _ in vs)
                tot += len(vs)
                ok_fit = len({k for ok, k in vs if ok}) == len(maps) and all(ok for ok, _ in vs)
                if ok_fit:
                    clean_s += 1
                details.append(f"{sname[:7]}f{fold}:k={res['n_kept']}:{[k if ok else 'X' for ok, k in vs]}")
        print(f"  hc-11 {kind:<15} | pass={pass_s:2d}/{tot:<3d} | clean={clean_s:2d}/{n_fits:<2d} | {' '.join(details[:6])}")

    # 2. hc-5 wheel (aligned)
    clean_s, pass_s, tot, n_fits = 0, 0, 0, 0
    for sname in hc5.session_names():
        sess = hc5.load_session(sname)
        for fold in (0, 1):
            X, T, train, test, _ = training_data(sess, "aligned", 0, fold)
            TX = np.full(X.shape[0], T) if np.ndim(T) == 0 else np.asarray(T)
            f = hc5.time_fields(sess, test)
            maps = [f] if np.isfinite(f).sum() >= 6 else []
            res = fit_clean(X, TX, np.random.default_rng(seed("exp2-hc5", sname, fold)), **cfg)
            jrng = np.random.default_rng(seed("exp2-hc5-j", sname, fold))
            vs = [judge_share(o, maps, jrng) for o in res["orders"]]
            pass_s += sum(ok for ok, _ in vs)
            tot += len(vs)
            n_fits += 1
            if len({k for ok, k in vs if ok}) == len(maps) and all(ok for ok, _ in vs):
                clean_s += 1
    print(f"  hc-5 wheel            | pass={pass_s:2d}/{tot:<3d} | clean={clean_s:2d}/{n_fits:<2d}")

    # 3. DANDI 001695 laps
    clean_s, pass_s, tot, n_fits = 0, 0, 0, 0
    for path in dd.session_paths():
        s = dd.load_session(path)
        run = running(s)
        L = laps(run)
        starts, ends, dirs = (np.array([x[i] for x in L]) for i in range(3))
        half = halves_by(dirs, np.random.default_rng(seed("dandi-laps-half", path.name)))
        by_cell, ids, _ = pyramidal_spikes(s, np.column_stack([starts, ends]))
        for fold in (0, 1):
            train = half if fold == 0 else ~half
            tr, te = np.flatnonzero(train), np.flatnonzero(~train)
            X, _ = event_spike_times(by_cell, ids, starts[tr], ends[tr], max_spikes=400)
            TX = (ends[tr] - starts[tr]) * 1000
            c = place_centres(s, ids, heldout_run(run, np.column_stack([starts[te], ends[te]])), min_spikes=20)
            maps = [c[sg] for sg in (1.0, -1.0) if np.isfinite(c[sg]).sum() >= 6 and np.sum(dirs[te] == sg) >= 3]
            res = fit_clean(X, TX, np.random.default_rng(seed("exp2-dd", path.name, fold)), **cfg)
            jrng = np.random.default_rng(seed("exp2-dd-j", path.name, fold))
            vs = [judge_share(o, maps, jrng) for o in res["orders"]]
            pass_s += sum(ok for ok, _ in vs)
            tot += len(vs)
            n_fits += 1
            if maps and len({k for ok, k in vs if ok}) == len(maps) and all(ok for ok, _ in vs):
                clean_s += 1
    print(f"  001695 laps           | pass={pass_s:2d}/{tot:<3d} | clean={clean_s:2d}/{n_fits:<2d}")

    # 4. DANDI 000552 figure-eight (first 20 sessions = 40 fits for speed)
    if run_fig8:
        paths = hz.session_paths()[:20]
        v_all = []
        both_c, clean_c, n_fits = 0, 0, 0
        for path in paths:
            s = hz.load_session(path)
            task_w = hz.task_window(s)[1]
            trials = hz.task_trials(s, task_w)
            arms = s["arm"][trials]
            cells = hz.putative_pyramidal(s)
            half = halves_by(arms, np.random.default_rng(seed("fig8-half", s["name"])))
            times, T = figure8_events(s, trials, cells)
            for fold in (0, 1):
                train = half if fold == 0 else ~half
                tr = np.flatnonzero(train)
                held = dict(s)
                held["arm"] = s["arm"].copy()
                held["arm"][trials[train]] = "none"
                cen = hz.place_centres(*hz.arm_maps(held, cells, task_w))
                res = fit_clean(times[tr], T[tr], np.random.default_rng(seed("exp2-f8", s["name"], fold)), **cfg)
                jrng = np.random.default_rng(seed("exp2-f8-j", s["name"], fold))
                vs = [judge_figure8(o, cen[1.0], cen[-1.0], jrng) for o in res["orders"]]
                v_all.extend(vs)
                n_fits += 1
                if {"left", "right"} <= set(vs):
                    both_c += 1
                    if "fail" not in vs:
                        clean_c += 1
        print(f"  000552 fig8 (20 sess) | L={v_all.count('left')} R={v_all.count('right')} S={v_all.count('stem')} fail={v_all.count('fail')} | both={both_c}/{n_fits} | clean={clean_c}/{n_fits}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    configs = [
        ("A. Uncentred, alpha=pi, circ_phase null, log-ICA, frac=0.2",
         dict(alpha=np.pi, mode="uncentred", scale="none", null_kind="circ_phase",
              ica=True, contrast="log", a=0.1, merge_conj=False, member_rule="frac", member_frac=0.2)),
        ("B. Uncentred, alpha=1.6*pi, circ_phase null, log-ICA, frac=0.2",
         dict(alpha=1.6*np.pi, mode="uncentred", scale="none", null_kind="circ_phase",
              ica=True, contrast="log", a=0.1, merge_conj=False, member_rule="frac", member_frac=0.2)),
        ("C. Uncentred, alpha=1.6*pi, circ_phase null, log-ICA, otsu",
         dict(alpha=1.6*np.pi, mode="uncentred", scale="none", null_kind="circ_phase",
              ica=True, contrast="log", a=0.1, merge_conj=False, member_rule="otsu")),
        ("D. Uncentred, alpha=1.6*pi, circ_phase null, kurt-ICA, otsu",
         dict(alpha=1.6*np.pi, mode="uncentred", scale="none", null_kind="circ_phase",
              ica=True, contrast="kurt", a=0.1, merge_conj=False, member_rule="otsu")),
    ]
    for name, cfg in configs:
        test_planted_laps(name, cfg)
    for name, cfg in configs[0:3]:
        test_real_waking(name, cfg, run_fig8=True)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s")
